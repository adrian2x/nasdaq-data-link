use ::zip::ZipArchive;
use anyhow::{Result, anyhow};
use polars::prelude::*;
use std::fs::File;
use std::io::copy;
use std::time::Instant;
const DUCKDB_FILENAME: &str = "nasdaq_duckdb.db";
const OUTPUT_DIR: &str = "output";
/// Write a Polars DataFrame to `output/<table>.parquet` then load it into DuckDB.
/// Returns the DataFrame so callers can continue chaining.
pub async fn write_df_to_duckdb(df: DataFrame, table_name: &str) -> Result<DataFrame> {
    let table_name = table_name.trim().to_string();
    if table_name.is_empty() {
        return Err(anyhow!("table_name cannot be empty"));
    }
    let df = tokio::task::spawn_blocking(move || {
        let mut df = df;
        df_write_duckdb(&mut df, &table_name)?;
        Ok::<DataFrame, anyhow::Error>(df)
    })
    .await??;
    Ok(df)
}

fn df_write_duckdb(df: &mut DataFrame, table: &str) -> Result<()> {
    let nrows = df.height();
    println!("Writing {nrows} rows to '{DUCKDB_FILENAME}' table '{table}'");
    let start = Instant::now();

    std::fs::create_dir_all(OUTPUT_DIR)?;
    let parquet_path = format!("{OUTPUT_DIR}/{table}.parquet");
    ParquetWriter::new(File::create(&parquet_path)?).finish(df)?;

    let conn = duckdb::Connection::open(DUCKDB_FILENAME)?;
    let qt = quote_ident(table);
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE {qt} AS SELECT * FROM read_parquet('{parquet_path}');"
    ))?;

    println!(
        "Wrote {nrows} rows to table '{table}' in {:.2?}",
        start.elapsed()
    );
    Ok(())
}

fn quote_ident(s: &str) -> String {
    format!("\"{}\"", s.replace('\"', "\"\""))
}

fn rolling_opts(window: usize) -> RollingOptionsFixedWindow {
    RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        ..Default::default()
    }
}

/// SMA expression over ticker groups.
fn sma_over_ticker(source: &str, period: usize) -> Expr {
    f(source)
        .rolling_mean(rolling_opts(period))
        .round(6)
        .over([col("ticker")])
}

/// EMA expression over ticker groups (span-based: α = 2/(span+1)).
fn ema_over_ticker(source: &str, span: usize) -> Expr {
    f(source)
        .ewm_mean(EWMOptions {
            alpha: 2.0 / (span as f64 + 1.0),
            min_periods: span,
            ..Default::default()
        })
        .round(6)
        .over([col("ticker")])
}

/// Percentage change expression over ticker groups.
fn chg_over_ticker(source: &str, period: i64) -> Expr {
    chg_expr(source, period).over([col("ticker")])
}

/// Realized volatility (annualized population std of log returns) computed
/// from a pre-existing log-return column. Avoids recomputing the log return
/// once per window length when used for multiple windows.
///
/// `log_ret_col` must already be the per-ticker log return (e.g. computed
/// with `(close / close.shift(1)).ln().over([ticker])`).
///
/// Returns annualized volatility as a percentage rounded to 2 dp.
fn rv_from_logret(log_ret_col: &str, window: usize, trading_periods: usize) -> Expr {
    let log_ret = col(log_ret_col);
    let mean = log_ret.clone().rolling_mean(rolling_opts(window));
    let mean_sq = (log_ret.clone() * log_ret).rolling_mean(rolling_opts(window));
    // Population std: sqrt(E[X²] - E[X]²), abs() guards against float rounding
    let pop_std = (mean_sq - mean.clone() * mean).abs().pow(lit(0.5));
    (pop_std * lit((trading_periods as f64).sqrt()) * lit(100.0))
        .round(2)
        .over([col("ticker")])
}

/// Load a CSV from a zip archive into a DataFrame.
pub fn load_csv_zip(path: &str) -> Result<DataFrame> {
    let csv_path = extract_zip_file(path)?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(csv_path.into()))?
        .finish()?;
    Ok(df)
}

/// Load company metadata, keeping only active (non-delisted) companies and
/// dropping columns that are not useful for analysis.
pub fn load_companies_meta(path: &str) -> Result<DataFrame> {
    let df = load_csv_zip(path)?
        .lazy()
        .filter(col("isdelisted").eq(lit("N")))
        .drop([
            "table",
            "permaticker",
            "lastupdated",
            "isdelisted",
            "cusips",
            "siccode",
            "sicsector",
            "sicindustry",
            "famasector",
            "famaindustry",
            "scalemarketcap",
            "scalerevenue",
            "relatedtickers",
            "firstadded",
            "firstpricedate",
            "lastpricedate",
            "firstquarter",
            "lastquarter",
        ])
        .collect()?;
    Ok(df)
}

/// Extract a zip file to a new file with the same name minus the .zip extension
pub fn extract_zip_file(zip_filename: &str) -> Result<String> {
    let output_filename = zip_filename.strip_suffix(".zip").unwrap_or(zip_filename);
    let zip_file = File::open(zip_filename)?;
    let mut archive = ZipArchive::new(zip_file)?;
    let mut csv_file = archive.by_index(0)?;
    let mut output_file = File::create(output_filename)?;
    copy(&mut csv_file, &mut output_file)?;
    Ok(output_filename.to_string())
}

/// Read a CSV zip file and load it into a Polars DataFrame, adjusting OHLC by closeadj.
///
/// Uses an explicit schema overwrite at parse time rather than post-load casts.
/// This skips Polars' type inference for the critical numeric columns, fails
/// loudly on schema drift, and is slightly faster (no inference pass + no
/// second cast pass).
///
/// Output dtypes:
///   - open, high, low, close: Float64 — no domain assumption about prices.
///   - volume: Float64 — flexible enough to handle share counts today and
///     potentially fractional/dollar volumes in the future without schema
///     changes downstream.
pub fn adjust_prices(path: &str) -> Result<DataFrame> {
    println!("adjust_prices...");
    let start = Instant::now();

    let csv_path = extract_zip_file(path)?;

    // Force the dtypes we care about at parse time. Other columns (ticker,
    // date, closeunadj, lastupdated) keep their inferred types — we don't
    // care about those.
    let schema = Schema::from_iter([
        Field::new("open".into(), DataType::Float64),
        Field::new("high".into(), DataType::Float64),
        Field::new("low".into(), DataType::Float64),
        Field::new("close".into(), DataType::Float64),
        Field::new("closeadj".into(), DataType::Float64),
        Field::new("volume".into(), DataType::Float64),
    ]);

    let adjustment_factor = col("closeadj") / col("close");
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_schema_overwrite(Some(std::sync::Arc::new(schema)))
        .try_into_reader_with_file_path(Some(csv_path.into()))?
        .finish()?
        .lazy()
        .with_columns([
            (col("low") * adjustment_factor.clone()).alias("low"),
            (col("high") * adjustment_factor.clone()).alias("high"),
            (col("open") * adjustment_factor).alias("open"),
        ])
        .drop(["close", "closeunadj", "lastupdated"])
        .rename(["closeadj"], ["close"], false)
        .sort(["ticker", "date"], Default::default())
        .collect()?;

    println!("adjust_prices completed in {:.2?}", start.elapsed());
    Ok(df)
}

/// Read a fundamentals CSV zip file and load it into a Polars DataFrame.
pub fn adjust_fundamentals(path: &str) -> Result<DataFrame> {
    println!("adjust_fundamentals...");
    let start = Instant::now();

    let csv_path = extract_zip_file(path)?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(csv_path.into()))?
        .finish()?
        .lazy()
        .rename(["calendardate"], ["date"], false)
        .drop(["dimension", "lastupdated"])
        // All columns derived purely from raw input
        .with_columns([
            (f("sharesbas") * f("sharefactor")).round(0).alias("shares"),
            (f("dps") / f("fxusd")).round(3).alias("dpsusd"),
            (f("eps") / f("fxusd")).round(3).alias("epsusd"),
            (f("ncfo") / f("fxusd")).round(3).alias("ncfousd"),
            (f("fcf") / f("fxusd")).round(3).alias("fcfusd"),
            (f("assets") - f("liabilities")).round(0).alias("equity"),
            ((f("debt") - f("cashneq")) / f("fxusd"))
                .round(0)
                .alias("netdebtusd"),
            (lit(100.0) * f("roe")).round(2).alias("roe"),
            (lit(100.0) * f("roic")).round(2).alias("roic"),
            (lit(100.0) * f("roa")).round(2).alias("roa"),
            (f("ncfo") / f("opinc")).round(3).alias("cfc"),
            (f("ebit") / f("intexp")).round(3).alias("icr"),
            (lit(100.0) * f("ebit") / (f("assets") - f("liabilitiesc")))
                .round(3)
                .alias("roce"),
            ((f("netinc") - f("netincdis")) / f("fxusd"))
                .round(0)
                .alias("netincadj"),
            ((lit(-1.0) * f("ncfcommon") / f("fxusd")) / f("marketcap") * lit(100.0))
                .round(2)
                .alias("bbyield"),
            (lit(100.0) * f("divyield")).round(2).alias("divyield"),
            ((lit(-1.0) * f("ncfdiv") + lit(-1.0) * f("ncfcommon")) / f("fxusd"))
                .round(0)
                .alias("shreturnusd"),
            (lit(100.0) * f("gp") / f("revenue"))
                .round(2)
                .alias("grossmargin"),
            (lit(100.0) * f("ebitda") / f("revenue"))
                .round(2)
                .alias("ebitdamargin"),
            (lit(100.0) * f("ebit") / f("revenue"))
                .round(2)
                .alias("ebitmargin"),
        ])
        // Columns that depend on the batch above
        .with_columns([
            (f("debt") / f("equity")).round(3).alias("de"),
            (f("netincadj") / f("shares")).round(3).alias("epsadj"),
            (lit(100.0) * f("netincadj") / f("revenueusd"))
                .round(2)
                .alias("netmargin"),
            (lit(100.0) * f("shreturnusd") / f("marketcap"))
                .round(2)
                .alias("shyield"),
        ])
        .sort(["ticker", "date"], Default::default())
        // fcfadj + fcfpsadj (net_workingcapital and maintenance_capex inlined)
        .with_columns({
            let fcfadj = (f("ncfo")
                - f("netincdis")
                - f("depamor")
                - (f("workingcapital") - f("workingcapital").shift(lit(1))).over([col("ticker")])
                - f("sbcomp"))
                / f("fxusd");
            [
                fcfadj.clone().round(0).alias("fcfadj"),
                (fcfadj.round(0) / f("shares")).round(3).alias("fcfpsadj"),
            ]
        })
        // Growth metrics (all source columns now exist)
        .with_columns([
            pct_change("revenue", 4).alias("revenueyoy"),
            cagr("revenue", 2).alias("revenuecagr"),
            pct_change("ebitdausd", 4).alias("ebitdayoy"),
            cagr("ebitdausd", 2).alias("ebitdacagr"),
            pct_change("epsadj", 4).alias("epsyoy"),
            cagr("epsadj", 2).alias("epscagr"),
            pct_change("fcfpsadj", 4).alias("fcfpsyoy"),
            cagr("fcfpsadj", 2).alias("fcfpscagr"),
        ])
        .collect()?;

    println!("adjust_fundamentals completed in {:.2?}", start.elapsed());
    Ok(df)
}

/// Load, normalize, aggregate, and persist insider transactions.
pub fn update_insiders(path: &str) -> Result<DataFrame> {
    println!("update_insiders...");
    let start = Instant::now();

    let csv_path = extract_zip_file(path)?;
    let six_months_ago = (chrono::Utc::now().date_naive() - chrono::Duration::weeks(26))
        .format("%Y-%m-%d")
        .to_string();

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_schema_overwrite(Some(std::sync::Arc::new(Schema::from_iter([Field::new(
            "formtype".into(),
            DataType::String,
        )]))))
        .try_into_reader_with_file_path(Some(csv_path.into()))?
        .finish()?
        .lazy()
        .with_columns([
            col("transactiondate").alias("date"),
            col("transactionshares")
                .cast(DataType::Float64)
                .abs()
                .alias("_transactionshares_abs"),
        ])
        .filter(
            col("date")
                .gt_eq(lit(six_months_ago.as_str()))
                .and(f("transactionvalue").neq(lit(0.0))),
        )
        .group_by([
            col("ticker"),
            col("date"),
            col("issuername"),
            col("ownername"),
            col("transactioncode"),
            col("securityadcode"),
            col("securitytitle"),
            col("officertitle"),
            col("isofficer"),
            col("isdirector"),
            col("istenpercentowner"),
        ])
        .agg([
            col("transactionvalue").sum().alias("transactionvalue"),
            col("_transactionshares_abs")
                .sum()
                .alias("transactionshares"),
            col("transactionpricepershare")
                .mean()
                .alias("transactionpricepershare"),
        ])
        .sort(
            ["date", "transactionvalue"],
            SortMultipleOptions::default().with_order_descending_multi([true, true]),
        )
        .with_columns([
            col("transactionpricepershare")
                .round(2)
                .alias("transactionpricepershare"),
            when(col("isofficer").eq(lit("Y")))
                .then(col("officertitle").fill_null(lit("")))
                .when(col("isdirector").eq(lit("Y")))
                .then(lit("Director"))
                .when(col("istenpercentowner").eq(lit("Y")))
                .then(lit("10% Owner"))
                .otherwise(col("officertitle").fill_null(lit("")))
                .alias("officertitle"),
        ])
        .collect()?;

    println!("update_insiders completed in {:.2?}", start.elapsed());
    Ok(df)
}

/// Compute technical indicators on price/volume data.
///
/// Optimization notes:
///   - Computes log return ONCE in a hidden column, then derives all 5 realized
///     volatility windows from it. The original code computed log_ret independently
///     for each RV window, doing 4× redundant log() work on 25M+ rows.
pub fn technical_indicators(df: DataFrame) -> Result<DataFrame> {
    println!("technical_indicators...");
    let start = Instant::now();

    // Hurst exponent (handles its own sorting + parallelism internally).
    // All OHLCV columns arrive as Float64 from adjust_prices, satisfying
    // fractaltools' input contract directly.
    let df = crate::fractaltools::with_hurst(df, crate::fractaltools::HurstConfig::default())?;

    let range_opts = RollingOptionsFixedWindow {
        window_size: 250,
        min_periods: 2,
        ..Default::default()
    };

    let result = df
        .lazy()
        .sort(["ticker", "date"], Default::default())
        // Materialize the log return once. All 5 RV windows reference this
        // column instead of recomputing close.shift+div+ln per window.
        .with_columns([(f("close") / f("close").shift(lit(1)))
            .log(std::f64::consts::E)
            .over([col("ticker")])
            .alias("__log_ret")])
        .with_columns([
            // Price changes
            chg_over_ticker("close", 1).alias("pct"),
            chg_over_ticker("close", 5).alias("chg5"),
            chg_over_ticker("close", 10).alias("chg10"),
            chg_over_ticker("close", 21).alias("chg1m"),
            chg_over_ticker("close", 3 * 21).alias("chg1q"),
            chg_over_ticker("close", 6 * 21).alias("chg2q"),
            chg_over_ticker("close", 9 * 21).alias("chg3q"),
            chg_over_ticker("close", 12 * 21).alias("chg1y"),
            // Realized volatility — all derived from the shared __log_ret column
            rv_from_logret("__log_ret", 5, 252).alias("rv5"),
            rv_from_logret("__log_ret", 10, 252).alias("rv10"),
            rv_from_logret("__log_ret", 20, 252).alias("rv20"),
            rv_from_logret("__log_ret", 60, 252).alias("rv60"),
            rv_from_logret("__log_ret", 252, 252).alias("rv252"),
            // SMA
            sma_over_ticker("close", 5).alias("sma5"),
            sma_over_ticker("close", 10).alias("sma10"),
            sma_over_ticker("close", 20).alias("sma20"),
            sma_over_ticker("close", 50).alias("sma50"),
            sma_over_ticker("close", 100).alias("sma100"),
            sma_over_ticker("close", 150).alias("sma150"),
            sma_over_ticker("close", 200).alias("sma200"),
            sma_over_ticker("volume", 50).alias("avgvolume3m"),
            // EMA
            ema_over_ticker("close", 10).alias("ema10"),
            ema_over_ticker("close", 20).alias("ema20"),
            ema_over_ticker("close", 60).alias("ema60"),
            ema_over_ticker("close", 120).alias("ema120"),
            ema_over_ticker("close", 250).alias("ema250"),
            // Rolling close range
            f("close")
                .rolling_max(range_opts.clone())
                .over([col("ticker")])
                .alias("maxc250"),
            f("close")
                .rolling_min(range_opts)
                .over([col("ticker")])
                .alias("minc250"),
        ])
        // rs1y depends on chg columns computed above
        .with_columns([(lit(0.4) * col("chg1q")
            + lit(0.2) * col("chg2q")
            + lit(0.2) * col("chg3q")
            + lit(0.2) * col("chg1y"))
        .over([col("ticker")])
        .alias("rs1y")])
        // Drop the hidden helper column
        .drop(["__log_ret"])
        // Enforce final dtypes following industry-standard market data conventions:
        //   - Percentages, ratios, Hurst: Float32 (bounded domain, ~7 sig digits
        //     is plenty, halves storage and shrinks JSON payloads).
        //   - Moving averages and price ranges: Float64 (they're prices/volume —
        //     no domain assumption; matches OHLC upstream).
        //
        // OHLCV and Hurst keep their upstream dtypes (Float64 / Float32).
        .with_columns([
            // Percentage changes — bounded domain (±100% × some multiplier),
            // 2dp precision; Float32 fits comfortably.
            col("pct").cast(DataType::Float32),
            col("chg5").cast(DataType::Float32),
            col("chg10").cast(DataType::Float32),
            col("chg1m").cast(DataType::Float32),
            col("chg1q").cast(DataType::Float32),
            col("chg2q").cast(DataType::Float32),
            col("chg3q").cast(DataType::Float32),
            col("chg1y").cast(DataType::Float32),
            // Realized volatility (annualized %, 2dp) — also bounded.
            col("rv5").cast(DataType::Float32),
            col("rv10").cast(DataType::Float32),
            col("rv20").cast(DataType::Float32),
            col("rv60").cast(DataType::Float32),
            col("rv252").cast(DataType::Float32),
            // Moving averages and price ranges stay Float64 — they are prices,
            // and we make no domain assumption about price magnitudes (consistent
            // with adjust_prices keeping OHLC at Float64). Already rounded to 6dp
            // upstream which aids DuckDB ALP compression without precision loss.
            // avgvolume3m stays Float64 to match volume (no assumption about
            // whether volume is integer share count or fractional dollar volume).
            // Relative strength composite (a weighted sum of percentages —
            // bounded domain, fits Float32 comfortably).
            col("rs1y").cast(DataType::Float32),
        ])
        .collect()?;

    println!("technical_indicators completed in {:.2?}", start.elapsed());
    Ok(result)
}

/// Shorthand: reference a column cast to Float64 for safe arithmetic.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Helper function to create a percentage change expression.
fn chg_expr(column: &str, period: i64) -> Expr {
    (((f(column) / f(column).shift(lit(period))) - lit(1.0)) * lit(100.0)).round(2)
}

/// Build a percentage-change expression for a column, scoped to ticker groups.
pub fn pct_change(column: &str, periods: i64) -> Expr {
    (((f(column) / f(column).shift(lit(periods))) - lit(1.0)) * lit(100.0))
        .round(2)
        .over([col("ticker")])
}

/// Build a CAGR expression for a column, scoped to ticker groups.
pub fn cagr(column: &str, periods: i64) -> Expr {
    (((f(column) / f(column).shift(lit(periods * 4))).pow(lit(1.0 / periods as f64)) - lit(1.0))
        * lit(100.0))
    .round(2)
    .over([col("ticker")])
}

/// Inner-join pre-filtered company metadata with the most recent financial data per ticker.
///
/// Optimization: replaces the previous "global sort by date desc + unique keep-first"
/// pattern with a per-ticker `sort_by + first` aggregation. The original approach
/// forced a full sort of the entire fundamentals frame; the group-wise approach
/// only sorts within each ticker, which is dramatically cheaper on a wide
/// multi-million-row frame and parallelizable across groups.
pub fn join_company_financials(
    financials_ttm: DataFrame,
    companies_meta: DataFrame,
) -> Result<DataFrame> {
    println!("join_company_financials...");
    let start = Instant::now();

    // Project to just the columns we need BEFORE the per-ticker latest-row
    // selection. This shrinks the data Polars has to shuffle through the
    // group-by from the full fundamentals schema (~80 columns) to ~30.
    let projected = financials_ttm.lazy().select([
        col("ticker"),
        col("date"),
        col("shares"),
        col("fiscalperiod"),
        col("ebitdausd"),
        col("netincadj"),
        col("fcfadj"),
        col("netdebtusd"),
        col("de"),
        col("grossmargin"),
        col("ebitdamargin"),
        col("ebitmargin"),
        col("netmargin"),
        col("dpsusd"),
        col("epsadj"),
        col("fcfpsadj"),
        col("revenueyoy"),
        col("revenuecagr"),
        col("ebitdayoy"),
        col("ebitdacagr"),
        col("epsyoy"),
        col("epscagr"),
        col("fcfpsyoy"),
        col("fcfpscagr"),
        col("shyield"),
        col("roa"),
        col("roe"),
        col("roic"),
        col("roce"),
        col("icr"),
        col("cfc"),
    ]);

    // Per-ticker "latest row" via group-by + arg_max on date.
    // This avoids the full-frame sort that the previous global
    // `.sort(["date"], desc).unique(["ticker"], First)` required.
    let latest_financials = projected
        .group_by([col("ticker")])
        .agg([col("*")
            .sort_by(
                [col("date")],
                SortMultipleOptions::default().with_order_descending(true),
            )
            .first()])
        // Drop the date column we used for ordering (it was included in col("*"))
        .drop(["date"]);

    let result = companies_meta
        .lazy()
        .join(
            latest_financials,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        )
        .collect()?;

    let elapsed = start.elapsed();
    println!("join_company_financials completed in {:.2?}", elapsed);
    Ok(result)
}
