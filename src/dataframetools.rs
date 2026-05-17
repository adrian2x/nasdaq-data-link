use ::zip::ZipArchive;
use anyhow::{Result, anyhow};
use polars::prelude::*;
use std::fs::File;
use std::io::copy;
use std::time::Instant;
const DUCKDB_FILENAME: &str = "nasdaq.duckdb";
const OUTPUT_DIR: &str = "output";
/// Persist a DataFrame as a queryable DuckDB table.
///
/// Two outputs are produced:
///   - `output/<table>.parquet` — the parquet file used as the load source
///     (kept on disk as a portable export)
///   - DuckDB table `<table>` in the configured database file — created via
///     `CREATE OR REPLACE TABLE ... AS SELECT * FROM read_parquet(...)`
///
/// Synchronous: both writes block on the calling thread. The writer pipeline
/// is sequential and doesn't benefit from off-loading to a blocking pool.
///
/// Takes the DataFrame by reference. The parquet writer needs `&mut DataFrame`
/// internally, so a shallow clone (cheap Arc bump, no data copy) is made.
pub fn df_to_duckdb(df: &DataFrame, table: &str) -> Result<()> {
    let table = table.trim();
    if table.is_empty() {
        return Err(anyhow!("table name cannot be empty"));
    }

    let nrows = df.height();
    println!("Writing {nrows} rows to '{DUCKDB_FILENAME}' table '{table}'");
    let start = Instant::now();

    std::fs::create_dir_all(OUTPUT_DIR)?;
    let parquet_path = format!("{OUTPUT_DIR}/{table}.parquet");
    let mut df = df.clone();
    ParquetWriter::new(File::create(&parquet_path)?).finish(&mut df)?;

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

/// SMA expression. Returns an `Expr` over a single time series. Caller
/// is responsible for grouping (e.g. `.over([col("ticker")])`) and aliasing.
pub fn sma_expr(source: &str, period: usize) -> Expr {
    f(source).rolling_mean(rolling_opts(period))
}

/// EMA expression. Span-based EMA (α = 2/(span+1)). Caller handles grouping
/// and aliasing. Distinct from `wilder_smooth` which uses α = 1/n.
pub fn ema_expr(source: &str, span: usize) -> Expr {
    f(source).ewm_mean(EWMOptions {
        alpha: 2.0 / (span as f64 + 1.0),
        min_periods: span,
        ..Default::default()
    })
}

/// Percentage-change expression. ((current / prev) - 1) * 100.
/// Caller handles grouping and aliasing.
///
/// Accepts any `Expr`, so the change can be computed on a derived quantity
/// (e.g. `chg_expr(f("revenue") / f("shares"), 4)` for YoY revenue-per-share
/// growth) rather than just a single column.
pub fn chg_expr(source: Expr, period: i64) -> Expr {
    ((source.clone() / source.shift(lit(period))) - lit(1.0)) * lit(100.0)
}

/// Realized volatility expression. Annualized population std of a log-return
/// series, returned as a percentage.
///
/// `log_ret_col` must already be the log return of close (e.g. created via
/// `(close / close.shift(1)).ln()`). For multi-ticker data, that input
/// column should itself have been computed per ticker. Caller handles
/// grouping of this expression and aliasing.
pub fn rv_expr(log_ret_col: &str, window: usize, trading_periods: usize) -> Expr {
    let log_ret = col(log_ret_col);
    let mean = log_ret.clone().rolling_mean(rolling_opts(window));
    let mean_sq = (log_ret.clone() * log_ret).rolling_mean(rolling_opts(window));
    // Population std: sqrt(E[X²] - E[X]²), abs() guards against float rounding.
    let pop_std = (mean_sq - mean.clone() * mean).abs().pow(lit(0.5));
    pop_std * lit((trading_periods as f64).sqrt()) * lit(100.0)
}

/// Load a CSV from a zip archive into a DataFrame.
/// Read a CSV file from a zip archive into a Polars DataFrame.
///
/// `schema_overrides` lets the caller pin specific columns to known dtypes,
/// bypassing Polars' inference for those columns. Use this when:
///   - a column's type isn't clear from the first 100 rows (e.g. SHARADAR's
///     `formtype` starts with numeric codes "4"/"5" and then hits strings
///     like "RESTATED - 4" hundreds of rows in)
///   - you want to skip inference cost for columns whose types you know
///   - you want to force a specific dtype regardless of what inference picks
///
/// Inference still runs for any column not in the overrides slice.
pub fn load_csv_zip(
    path: &str,
    schema_overrides: Option<&[(&str, DataType)]>,
) -> Result<DataFrame> {
    let csv_path = extract_zip_file(path)?;
    let mut opts = CsvReadOptions::default().with_has_header(true);
    if let Some(overrides) = schema_overrides {
        let schema = Schema::from_iter(
            overrides
                .iter()
                .map(|(name, dtype)| Field::new((*name).into(), dtype.clone())),
        );
        opts = opts.with_schema_overwrite(Some(std::sync::Arc::new(schema)));
    }
    let df = opts
        .try_into_reader_with_file_path(Some(csv_path.into()))?
        .finish()?;
    Ok(df)
}

/// Filter company metadata to active (non-delisted) companies and drop
/// columns that aren't useful for analysis. Takes a raw company-metadata
/// DataFrame (e.g. from `load_csv_zip` on the SHARADAR TICKERS feed).
pub fn filter_companies_meta(df: DataFrame) -> Result<DataFrame> {
    let df = df
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
/// Apply close-adjustment to OHLC and clean up the prices DataFrame.
///
/// Takes a raw prices DataFrame (e.g. from `load_csv_zip` on the SHARADAR
/// stocks_eod feed) with columns: ticker, date, open, high, low, close,
/// closeadj, closeunadj, volume, lastupdated. Outputs a DataFrame with
/// `close` adjusted-for-splits-and-dividends, with the adjustment factor
/// (`closeadj / close`) applied to open/high/low as well so that intrabar
/// ranges remain coherent. Drops the unadjusted close and lastupdated.
///
/// OHLC + closeadj + volume are cast to Float64 to defend against CSV
/// inference mis-typing them as Int64 when sampled rows happen to be
/// whole numbers.
pub fn adjust_prices(df: DataFrame) -> Result<DataFrame> {
    println!("adjust_prices...");
    let start = Instant::now();

    let adjustment_factor = col("closeadj") / col("close");
    let df = df
        .lazy()
        // Ensure correct dtypes regardless of how CSV inference typed columns.
        // `date` becomes a true Date dtype so downstream operations like
        // `group_by_dynamic` (in `resample`) can use it as a temporal index.
        .with_columns([
            col("date").cast(DataType::Date),
            col("open").cast(DataType::Float64),
            col("high").cast(DataType::Float64),
            col("low").cast(DataType::Float64),
            col("close").cast(DataType::Float64),
            col("closeadj").cast(DataType::Float64),
            col("volume").cast(DataType::Float64),
        ])
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

/// Resample OHLCV bars to a coarser time interval, per ticker.
///
/// Aggregation rules (the standard OHLCV-to-period conventions used by every
/// charting platform and academic dataset):
///   - `open`   → first value in the period (Monday's open for weekly, etc.)
///   - `high`   → max of the period
///   - `low`    → min of the period
///   - `close`  → last value in the period
///   - `volume` → sum of the period
///
/// The `interval` parameter follows Polars duration syntax (the underlying
/// engine is `group_by_dynamic`):
///   - `"1w"`   weekly
///   - `"2w"`   biweekly
///   - `"1mo"`  monthly
///   - `"1q"`   quarterly
///   - `"1y"`   yearly
///   - `"5d"`   every 5 trading days
///
/// The resulting `date` column is the left edge of each period — the date
/// the period **opened**. For weekly bars that's the Monday; for monthly,
/// the first of the month. This pairs naturally with `open` (the value at
/// that date) and `close` (the value at the end of the period that started
/// on that date), so a bar reads as "the period opening on date X".
///
/// Holiday-shortened periods still produce one bar; that bar just aggregates
/// fewer underlying daily bars.
///
/// Expects an input DataFrame with exactly these columns: `ticker`, `date`,
/// `open`, `high`, `low`, `close`, `volume`. Any other columns are silently
/// dropped — this is intentional, because applying close-of-period semantics
/// to derived columns (Hurst, technical indicators) would produce silently
/// wrong values. To compute indicators on resampled bars, call `resample`
/// first and then `technical_indicators` on the result.
///
/// Requires the input DataFrame to be sorted by `(ticker, date)`. The
/// `date` column must be `Date` dtype.
pub fn resample(df: DataFrame, interval: &str) -> Result<DataFrame> {
    println!("resample({interval})...");
    let start = Instant::now();

    let df = df
        .lazy()
        // group_by_dynamic builds time buckets per ticker. `every` and `period`
        // both equal the interval, so each bucket is exactly one period wide
        // (no overlap, no gaps). `label: Left` puts the period-start date
        // on each bar (the date the bar "opened"). `closed_window: Left`
        // makes the period [start, start+interval), i.e. the start date is
        // included in its own bar and the next period's start is excluded —
        // matches how trading weeks/months are conventionally bucketed.
        .group_by_dynamic(
            col("date"),
            [col("ticker")],
            DynamicGroupOptions {
                every: Duration::parse(interval),
                period: Duration::parse(interval),
                offset: Duration::parse("0d"),
                label: Label::Left,
                closed_window: ClosedWindow::Left,
                include_boundaries: false,
                start_by: StartBy::WindowBound,
                ..Default::default()
            },
        )
        .agg([
            col("open").first().alias("open"),
            col("high").max().alias("high"),
            col("low").min().alias("low"),
            col("close").last().alias("close"),
            col("volume").sum().alias("volume"),
        ])
        .sort(["ticker", "date"], Default::default())
        .collect()?;

    println!(
        "resample completed in {:.2?} ({} bars)",
        start.elapsed(),
        df.height()
    );
    Ok(df)
}

/// Transform raw fundamentals into the analytical schema used downstream.
///
/// Takes a fundamentals DataFrame (e.g. from `load_csv_zip` on the SHARADAR
/// SF1 feed) and derives the per-share, USD-converted, and growth-metric
/// columns required by `join_company_financials` and the company snapshot.
pub fn adjust_fundamentals(df: DataFrame) -> Result<DataFrame> {
    println!("adjust_fundamentals...");
    let start = Instant::now();

    let df = df
        .lazy()
        .rename(["calendardate"], ["date"], false)
        .with_columns([col("date").cast(DataType::Date)])
        .drop(["dimension", "lastupdated"])
        // All columns derived purely from raw input
        .with_columns([
            (f("sharesbas") * f("sharefactor")).alias("shares"),
            (f("dps") / f("fxusd")).alias("dpsusd"),
            (f("eps") / f("fxusd")).alias("epsusd"),
            (f("ncfo") / f("fxusd")).alias("ncfousd"),
            (f("fcf") / f("fxusd")).alias("fcfusd"),
            (f("assets") - f("liabilities")).alias("equity"),
            ((f("debt") - f("cashneq")) / f("fxusd")).alias("netdebtusd"),
            (lit(100.0) * f("roe")).alias("roe"),
            (lit(100.0) * f("roic")).alias("roic"),
            (lit(100.0) * f("roa")).alias("roa"),
            (f("ncfo") / f("opinc")).alias("cfc"),
            (f("ebit") / f("intexp")).alias("icr"),
            (lit(100.0) * f("ebit") / (f("assets") - f("liabilitiesc"))).alias("roce"),
            ((f("netinc") - f("netincdis")) / f("fxusd")).alias("netincadj"),
            ((lit(-1.0) * f("ncfcommon") / f("fxusd")) / f("marketcap") * lit(100.0))
                .alias("bbyield"),
            (lit(100.0) * f("divyield")).alias("divyield"),
            ((lit(-1.0) * f("ncfdiv") + lit(-1.0) * f("ncfcommon")) / f("fxusd"))
                .alias("shreturnusd"),
            (lit(100.0) * f("gp") / f("revenue")).alias("grossmargin"),
            (lit(100.0) * f("ebitda") / f("revenue")).alias("ebitdamargin"),
            (lit(100.0) * f("ebit") / f("revenue")).alias("ebitmargin"),
        ])
        // Columns that depend on the batch above
        .with_columns([
            (f("debt") / f("equity")).alias("de"),
            (f("netincadj") / f("shares")).alias("epsadj"),
            (lit(100.0) * f("netincadj") / f("revenueusd")).alias("netmargin"),
            (lit(100.0) * f("shreturnusd") / f("marketcap")).alias("shyield"),
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
                fcfadj.clone().alias("fcfadj"),
                (fcfadj / f("shares")).alias("fcfpsadj"),
            ]
        })
        // Growth metrics. Revenue and EBITDA growth are computed on a
        // per-share basis (dividing by shares outstanding) so that buybacks
        // and issuances are properly reflected — a company that buys back
        // 20% of its float while revenue stays flat is meaningfully better
        // per-share than it looks at the headline level. epsadj and fcfpsadj
        // are already per-share so they use the raw column.
        .with_columns([
            // YoY (4-quarter) and 2y CAGR
            chg_expr(f("revenue") / f("shares"), 4)
                .over([col("ticker")])
                .alias("revenueyoy"),
            cagr_expr(f("revenue") / f("shares"), 2)
                .over([col("ticker")])
                .alias("revenuecagr"),
            chg_expr(f("ebitdausd") / f("shares"), 4)
                .over([col("ticker")])
                .alias("ebitdayoy"),
            cagr_expr(f("ebitdausd") / f("shares"), 2)
                .over([col("ticker")])
                .alias("ebitdacagr"),
            chg_expr(f("epsadj"), 4)
                .over([col("ticker")])
                .alias("epsyoy"),
            cagr_expr(f("epsadj"), 2)
                .over([col("ticker")])
                .alias("epscagr"),
            chg_expr(f("fcfpsadj"), 4)
                .over([col("ticker")])
                .alias("fcfpsyoy"),
            cagr_expr(f("fcfpsadj"), 2)
                .over([col("ticker")])
                .alias("fcfpscagr"),
            // 3y change (12 quarters back, ~per-share basis where applicable)
            chg_expr(f("revenue") / f("shares"), 4 * 3)
                .over([col("ticker")])
                .alias("revenue3y"),
            chg_expr(f("ebitdausd") / f("shares"), 4 * 3)
                .over([col("ticker")])
                .alias("ebitda3y"),
            chg_expr(f("epsadj"), 4 * 3)
                .over([col("ticker")])
                .alias("eps3y"),
            chg_expr(f("fcfpsadj"), 4 * 3)
                .over([col("ticker")])
                .alias("fcfps3y"),
        ])
        .collect()?;

    println!("adjust_fundamentals completed in {:.2?}", start.elapsed());
    Ok(df)
}

/// Transform raw insider transactions into the analytical schema.
///
/// Takes an insider-transactions DataFrame (e.g. from `load_csv_zip` on the
/// SHARADAR SF3 feed) and produces a per-(ticker, date, person) summary of
/// recent insider activity. Filters to the last 6 months.
///
/// `formtype` is defensively cast to String — Polars' CSV inference may type
/// SEC form codes (4, 5) as Int64 if the sampled rows happen to be numeric.
pub fn update_insiders(df: DataFrame) -> Result<DataFrame> {
    println!("update_insiders...");
    let start = Instant::now();

    let six_months_ago = chrono::Utc::now().date_naive() - chrono::Duration::weeks(26);

    let df = df
        .lazy()
        .with_columns([
            col("formtype").cast(DataType::String),
            col("transactiondate").cast(DataType::Date).alias("date"),
            col("transactionshares")
                .cast(DataType::Float64)
                .abs()
                .alias("_transactionshares_abs"),
        ])
        .filter(
            col("date")
                .gt_eq(lit(six_months_ago))
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
        .with_columns([when(col("isofficer").eq(lit("Y")))
            .then(col("officertitle").fill_null(lit("")))
            .when(col("isdirector").eq(lit("Y")))
            .then(lit("Director"))
            .when(col("istenpercentowner").eq(lit("Y")))
            .then(lit("10% Owner"))
            .otherwise(col("officertitle").fill_null(lit("")))
            .alias("officertitle")])
        .collect()?;

    println!("update_insiders completed in {:.2?}", start.elapsed());
    Ok(df)
}

/// Compute the daily indicator set on a daily-bars price DataFrame.
///
/// Hurst is NOT computed here — call `fractaltools::with_hurst(df, ...)`
/// before this function with timeframe-appropriate window. The reason is
/// that the Hurst window is meaningfully timeframe-dependent (500 daily bars
/// ≈ 2 years; 500 weekly bars ≈ 10 years), while the other indicators here
/// are either bar-count-conventional (Wilder's 14) or have semantically
/// honest column names that describe their bar count.
///
/// Daily-specific column conventions:
///   - SMA/EMA at 5/10/20/50/100/150/200 daily bars (the canonical "50 SMA"
///     and "200 SMA" cross used by every charting platform)
///   - ROC at 1d, 5d, 10d, 1m, 1q, 2q, 3q, 1y (mapped to 1, 5, 10, 21, 63,
///     126, 189, 252 bars respectively — trading-day approximations)
///   - RV at 5/10/20/60/252 bars (annualized using 252 trading periods)
///   - avgvolume3m (50-bar SMA of volume, ≈ 3 months)
///   - max1y/min1y (250-bar rolling extrema)
///   - rs1y (composite of roc1q, 2q, 3q, 1y)
///
/// Timeframe-invariant indicators (added by helper calls, same on weekly):
///   - MACD(12, 26, 9), Bollinger(20, 2), RSI(14), ATR(14), ADX(14)
///
/// Optimization notes:
///   - Computes log return ONCE in a hidden column, then derives all 5
///     realized volatility windows from it (saves 4× redundant log() work
///     on 45M+ rows).
pub fn technical_indicators_daily(df: DataFrame) -> Result<DataFrame> {
    println!("technical_indicators_daily...");
    let start = Instant::now();

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
            .alias("log_ret")])
        .with_columns([
            // Price changes (1d/5d/10d/1m/1q/2q/3q/1y in trading days)
            chg_expr(f("close"), 1).over([col("ticker")]).alias("roc1d"),
            chg_expr(f("close"), 5).over([col("ticker")]).alias("roc1w"),
            chg_expr(f("close"), 20)
                .over([col("ticker")])
                .alias("roc1m"),
            chg_expr(f("close"), 3 * 21)
                .over([col("ticker")])
                .alias("roc1q"),
            chg_expr(f("close"), 6 * 21)
                .over([col("ticker")])
                .alias("roc2q"),
            chg_expr(f("close"), 9 * 21)
                .over([col("ticker")])
                .alias("roc3q"),
            chg_expr(f("close"), 12 * 21)
                .over([col("ticker")])
                .alias("roc1y"),
            // Realized volatility — all derived from the shared log_ret column.
            // Annualized with 252 trading periods/year.
            rv_expr("log_ret", 5, 252)
                .over([col("ticker")])
                .alias("rv1w"),
            rv_expr("log_ret", 21, 252)
                .over([col("ticker")])
                .alias("rv1m"),
            rv_expr("log_ret", 63, 252)
                .over([col("ticker")])
                .alias("rv1q"),
            rv_expr("log_ret", 252, 252)
                .over([col("ticker")])
                .alias("rv1y"),
            // SMA at canonical daily windows
            sma_expr("close", 5).over([col("ticker")]).alias("sma5"),
            sma_expr("close", 10).over([col("ticker")]).alias("sma10"),
            sma_expr("close", 20).over([col("ticker")]).alias("sma20"),
            sma_expr("close", 50).over([col("ticker")]).alias("sma50"),
            sma_expr("close", 100).over([col("ticker")]).alias("sma100"),
            sma_expr("close", 150).over([col("ticker")]).alias("sma150"),
            sma_expr("close", 200).over([col("ticker")]).alias("sma200"),
            // Volume average — 50 trading days ≈ 3 months
            sma_expr("volume", 50)
                .over([col("ticker")])
                .alias("avgvolume3m"),
            // EMA at canonical daily windows
            ema_expr("close", 10).over([col("ticker")]).alias("ema10"),
            ema_expr("close", 20).over([col("ticker")]).alias("ema20"),
            ema_expr("close", 63).over([col("ticker")]).alias("ema63"),
            ema_expr("close", 126).over([col("ticker")]).alias("ema126"),
            ema_expr("close", 250).over([col("ticker")]).alias("ema250"),
            // Rolling 1-year max/min of close (250 trading days)
            f("close")
                .rolling_max(range_opts.clone())
                .over([col("ticker")])
                .alias("max1y"),
            f("close")
                .rolling_min(range_opts)
                .over([col("ticker")])
                .alias("min1y"),
        ])
        // rs1y depends on roc columns computed above
        .with_columns([(lit(0.4) * col("roc1q")
            + lit(0.2) * col("roc2q")
            + lit(0.2) * col("roc3q")
            + lit(0.2) * col("roc1y"))
        .over([col("ticker")])
        .alias("rs1y")])
        // Drop the hidden helper column
        .drop(["log_ret"])
        // Float32 for bounded-domain columns (percentages, ratios, composites).
        .with_columns([
            col("roc1").cast(DataType::Float32),
            col("roc1w").cast(DataType::Float32),
            col("roc1m").cast(DataType::Float32),
            col("roc1q").cast(DataType::Float32),
            col("roc2q").cast(DataType::Float32),
            col("roc3q").cast(DataType::Float32),
            col("roc1y").cast(DataType::Float32),
            col("rv1w").cast(DataType::Float32),
            col("rv1m").cast(DataType::Float32),
            col("rv1q").cast(DataType::Float32),
            col("rv1y").cast(DataType::Float32),
            col("rs1y").cast(DataType::Float32),
        ])
        .collect()?;

    // Timeframe-invariant indicators — same windows on any bar interval.
    let result = macd(result, 12, 26, 9)?;
    let result = bollinger_bands(result, 20, 2.0)?;
    let result = rsi(result, 14)?;
    let result = atr(result, 14)?;
    let result = adx(result, 14)?;

    println!(
        "technical_indicators_daily completed in {:.2?}",
        start.elapsed()
    );
    Ok(result)
}

/// Compute the weekly indicator set on a weekly-bars price DataFrame.
///
/// Hurst is NOT computed here — call `fractaltools::with_hurst(df, ...)` with
/// a weekly-appropriate window (~100 bars ≈ 2 years) before this function.
///
/// Weekly-specific column conventions:
///   - SMA at 4/10/20/40 weeks (Stan Weinstein's 10/40-week trend rules;
///     40-week is the canonical weekly long-term trend line)
///   - EMA at 10/20/40 weeks
///   - ROC at 1w/4w/13w/26w/52w (weekly, monthly, quarterly, half, year)
///   - RV at 4/13/52 weeks (annualized using 52 weeks/year)
///   - avgvolume13w (13-bar SMA of volume, ≈ 3 months)
///   - max52w/min52w (52-bar rolling extrema, ≈ 1 year)
///   - rs1y (composite of roc13w, 26w, 39w, 52w — the weekly analog of the
///     daily rs1y, with quarter/half/three-quarter/year semantics)
///
/// Timeframe-invariant indicators (added by helper calls, same on daily):
///   - MACD(12, 26, 9), Bollinger(20, 2), RSI(14), ATR(14), ADX(14)
pub fn technical_indicators_weekly(df: DataFrame) -> Result<DataFrame> {
    println!("technical_indicators_weekly...");
    let start = Instant::now();

    // 52-bar rolling extrema = 1 year of weekly bars
    let range_opts = RollingOptionsFixedWindow {
        window_size: 52,
        min_periods: 2,
        ..Default::default()
    };

    let result = df
        .lazy()
        .sort(["ticker", "date"], Default::default())
        .with_columns([(f("close") / f("close").shift(lit(1)))
            .log(std::f64::consts::E)
            .over([col("ticker")])
            .alias("log_ret")])
        .with_columns([
            // Price changes — honest weekly cadence
            chg_expr(f("close"), 1).over([col("ticker")]).alias("roc1w"),
            chg_expr(f("close"), 4).over([col("ticker")]).alias("roc1m"),
            chg_expr(f("close"), 13)
                .over([col("ticker")])
                .alias("roc1q"),
            chg_expr(f("close"), 26)
                .over([col("ticker")])
                .alias("roc2q"),
            chg_expr(f("close"), 39)
                .over([col("ticker")])
                .alias("roc3q"),
            chg_expr(f("close"), 52)
                .over([col("ticker")])
                .alias("roc1y"),
            // Realized volatility — annualized using 52 weeks/year
            rv_expr("log_ret", 4, 52)
                .over([col("ticker")])
                .alias("rv1m"),
            rv_expr("log_ret", 13, 52)
                .over([col("ticker")])
                .alias("rv1q"),
            rv_expr("log_ret", 52, 52)
                .over([col("ticker")])
                .alias("rv1y"),
            // SMA at canonical weekly windows. 10/40 are the Stan Weinstein
            sma_expr("close", 10).over([col("ticker")]).alias("sma10"),
            sma_expr("close", 30).over([col("ticker")]).alias("sma30"),
            sma_expr("close", 40).over([col("ticker")]).alias("sma40"),
            sma_expr("close", 200).over([col("ticker")]).alias("sma200"),
            // Volume average — 13 weeks ≈ 3 months
            sma_expr("volume", 13)
                .over([col("ticker")])
                .alias("avgvolume3m"),
            // EMA at canonical weekly windows
            ema_expr("close", 10).over([col("ticker")]).alias("ema10"),
            ema_expr("close", 30).over([col("ticker")]).alias("ema30"),
            ema_expr("close", 40).over([col("ticker")]).alias("ema40"),
            ema_expr("close", 200).over([col("ticker")]).alias("ema200"),
            // Rolling 1-year max/min of close (52 weeks)
            f("close")
                .rolling_max(range_opts.clone())
                .over([col("ticker")])
                .alias("max1y"),
            f("close")
                .rolling_min(range_opts)
                .over([col("ticker")])
                .alias("min1y"),
        ])
        // rs1y: same composite shape as daily, built from weekly ROCs at the
        // matching quarter/half/three-quarter/year horizons.
        .with_columns([(lit(0.4) * col("roc1q")
            + lit(0.2) * col("roc2q")
            + lit(0.2) * col("roc3q")
            + lit(0.2) * col("roc1y"))
        .over([col("ticker")])
        .alias("rs1y")])
        .drop(["log_ret"])
        .with_columns([
            col("roc1w").cast(DataType::Float32),
            col("roc1m").cast(DataType::Float32),
            col("roc1q").cast(DataType::Float32),
            col("roc2q").cast(DataType::Float32),
            col("roc3q").cast(DataType::Float32),
            col("roc1y").cast(DataType::Float32),
            col("rv1m").cast(DataType::Float32),
            col("rv1q").cast(DataType::Float32),
            col("rv1y").cast(DataType::Float32),
            col("rs1y").cast(DataType::Float32),
        ])
        .collect()?;

    // Timeframe-invariant indicators — same as daily.
    let result = macd(result, 12, 26, 9)?;
    let result = bollinger_bands(result, 20, 2.0)?;
    let result = rsi(result, 14)?;
    let result = atr(result, 14)?;
    let result = adx(result, 14)?;

    println!(
        "technical_indicators_weekly completed in {:.2?}",
        start.elapsed()
    );
    Ok(result)
}

/// Core MACD line expression — operates on a single time series.
///
/// Returns `EMA(close, fast) − EMA(close, slow)`. Caller handles grouping
/// and aliasing.
pub fn macd_line_expr(fast: usize, slow: usize) -> Expr {
    ema_expr("close", fast) - ema_expr("close", slow)
}

/// Multi-ticker adapter: compute the MACD family per ticker and append
/// the resulting columns to the DataFrame.
///
/// Adds four columns. Default parameters (12, 26, 9) produce canonical
/// names: `ema12`, `ema26`, `macd`, `macdsignal`. Non-default parameters
/// produce period-suffixed names: `ema{fast}`, `ema{slow}`, `macd_{fast}_{slow}`,
/// `macdsignal_{fast}_{slow}_{signal}`.
///
/// The MACD line is `EMA(close, fast) − EMA(close, slow)`. The signal line
/// (`macdsignal`) is a span-`signal` EMA of the MACD line; "signal" is the
/// standard name across charting platforms (Investopedia, TradingView, etc.).
///
/// The histogram (`macd - macdsignal`) is intentionally not produced;
/// derive it at read time as needed.
///
/// All EMAs use span-based α = 2/(span+1) and are scoped per ticker.
/// Output dtype is Float64.
///
/// Requires the input DataFrame to be sorted by `(ticker, date)`.
pub fn macd(df: DataFrame, fast: usize, slow: usize, signal: usize) -> Result<DataFrame> {
    println!("macd({fast},{slow},{signal})...");
    let start = Instant::now();

    // Use canonical names for the standard 12/26/9 configuration,
    // and parameter-suffixed names otherwise.
    let is_default = fast == 12 && slow == 26 && signal == 9;
    let (ema_fast_col, ema_slow_col, macd_col, signal_col): (String, String, String, String) =
        if is_default {
            (
                "ema12".into(),
                "ema26".into(),
                "macd".into(),
                "macdsignal".into(),
            )
        } else {
            (
                format!("ema{fast}"),
                format!("ema{slow}"),
                format!("macd_{fast}_{slow}"),
                format!("macdsignal_{fast}_{slow}_{signal}"),
            )
        };

    let result = df
        .lazy()
        // Wave 1: compute ema_fast, ema_slow, and the MACD line.
        .with_columns([
            ema_expr("close", fast)
                .over([col("ticker")])
                .alias(ema_fast_col.as_str()),
            ema_expr("close", slow)
                .over([col("ticker")])
                .alias(ema_slow_col.as_str()),
            macd_line_expr(fast, slow)
                .over([col("ticker")])
                .alias(macd_col.as_str()),
        ])
        // Wave 2: signal line is EMA(signal) of macd. Separate wave because
        // ewm_mean over a freshly-created column isn't reliable in Polars.
        .with_columns([ema_expr(macd_col.as_str(), signal)
            .over([col("ticker")])
            .alias(signal_col.as_str())])
        .collect()?;

    println!("macd completed in {:.2?}", start.elapsed());
    Ok(result)
}

/// Core Bollinger upper band expression. SMA + multiplier × stdev of close.
/// Caller handles grouping.
pub fn bbtop_expr(period: usize, multiplier: f64) -> Expr {
    let sma = f("close").rolling_mean(rolling_opts(period));
    let stdev = f("close").rolling_std(rolling_opts(period));
    sma + lit(multiplier) * stdev
}

/// Core Bollinger lower band expression. SMA − multiplier × stdev of close.
/// Caller handles grouping.
pub fn bbbot_expr(period: usize, multiplier: f64) -> Expr {
    let sma = f("close").rolling_mean(rolling_opts(period));
    let stdev = f("close").rolling_std(rolling_opts(period));
    sma - lit(multiplier) * stdev
}

/// Multi-ticker adapter: compute Bollinger Bands per ticker and append
/// the resulting columns to the DataFrame.
///
/// Adds two columns. Default parameters (period=20, multiplier=2.0) produce
/// canonical names: `bbtop`, `bbbot`. Non-default parameters produce
/// period-suffixed names: `bbtop_{period}_{multiplier}`, similarly for bbbot.
///
/// The middle band is intentionally not produced — it equals the SMA at
/// the same period, computable via `sma(close, period)` upstream.
///
/// Stdev is sample stdev (ddof=1), matching Bollinger's original definition
/// and the convention used by TradingView and most charting tools.
///
/// All computations are scoped per ticker. Output dtype is Float64.
///
/// Requires the input DataFrame to be sorted by `(ticker, date)`.
pub fn bollinger_bands(df: DataFrame, period: usize, multiplier: f64) -> Result<DataFrame> {
    println!("bollinger_bands({period},{multiplier})...");
    let start = Instant::now();

    let is_default = period == 20 && (multiplier - 2.0).abs() < 1e-9;
    let (top_col, bot_col): (String, String) = if is_default {
        ("bbtop".into(), "bbbot".into())
    } else {
        // Format multiplier compactly: 2.5 stays "2.5", 2.0 becomes "2"
        let mult_str = if (multiplier - multiplier.round()).abs() < 1e-9 {
            format!("{}", multiplier as i64)
        } else {
            format!("{multiplier}")
        };
        (
            format!("bbtop_{period}_{mult_str}"),
            format!("bbbot_{period}_{mult_str}"),
        )
    };

    let result = df
        .lazy()
        .with_columns([
            bbbot_expr(period, multiplier)
                .over([col("ticker")])
                .alias(bot_col.as_str()),
            bbtop_expr(period, multiplier)
                .over([col("ticker")])
                .alias(top_col.as_str()),
        ])
        .collect()?;

    println!("bollinger_bands completed in {:.2?}", start.elapsed());
    Ok(result)
}

/// Wilder's smoothing: α = 1/period EMA. Used by RSI, ATR, ADX.
///
/// Distinct from the span-based EMA in `ema_expr` (which uses
/// α = 2/(span+1)). Wilder's original 1978 specification of these
/// indicators uses α = 1/n; standard charting platforms (TradingView,
/// Bloomberg) match that convention. Using span-based EMA here would
/// produce visibly different numbers from every standard implementation.
fn wilder_smooth(source: Expr, period: usize) -> Expr {
    source.ewm_mean(EWMOptions {
        alpha: 1.0 / period as f64,
        min_periods: period,
        ..Default::default()
    })
}

/// Core RSI expression — operates on a single time series.
///
/// Returns an `Expr` that computes the n-period Wilder RSI of the `close`
/// column. Caller is responsible for grouping (e.g. wrapping in
/// `.over([col("ticker")])` for multi-ticker data) and for aliasing the
/// result column.
///
/// Composable: drop into any `with_columns([])` block. For multi-ticker
/// DataFrames use the `rsi` adapter instead.
pub fn rsi_expr(period: usize) -> Expr {
    let delta = f("close") - f("close").shift(lit(1));
    let gain = when(delta.clone().gt(lit(0.0)))
        .then(delta.clone())
        .otherwise(lit(0.0));
    let loss = when(delta.clone().lt(lit(0.0)))
        .then(-delta)
        .otherwise(lit(0.0));

    let avg_gain = wilder_smooth(gain, period);
    let avg_loss = wilder_smooth(loss, period);
    let rs = avg_gain / avg_loss;
    lit(100.0) - lit(100.0) / (lit(1.0) + rs)
}

/// Multi-ticker adapter: compute RSI per ticker and append the resulting
/// column to the DataFrame.
///
/// Adds one column named `rsi{period}` (e.g. `rsi14`, `rsi7`, `rsi21`).
///
/// Output is in [0, 100] where >70 is conventionally "overbought" and <30
/// "oversold" (these thresholds are convention, not law).
///
/// Uses Wilder's smoothing (α = 1/period), matching TradingView, Bloomberg,
/// and the original 1978 specification.
///
/// Requires the input DataFrame to be sorted by `(ticker, date)`.
pub fn rsi(df: DataFrame, period: usize) -> Result<DataFrame> {
    println!("rsi({period})...");
    let start = Instant::now();

    let col_name = format!("rsi{period}");
    let expr = rsi_expr(period)
        .over([col("ticker")])
        .alias(col_name.as_str());

    let result = df.lazy().with_columns([expr]).collect()?;

    println!("rsi completed in {:.2?}", start.elapsed());
    Ok(result)
}

/// Core True Range expression. Returns `Expr` for max(high - low,
/// |high - prev_close|, |low - prev_close|). Caller handles grouping.
///
/// Useful as a building block — ATR is `wilder_smooth(true_range_expr(), n)`,
/// but you might also want TR for other indicators or for raw inspection.
pub fn true_range_expr() -> Expr {
    let prev_close = f("close").shift(lit(1));
    let hl = f("high") - f("low");
    let hc = (f("high") - prev_close.clone()).abs();
    let lc = (f("low") - prev_close).abs();
    // Polars expressions don't have a vectorized n-ary max, so we nest two
    // 2-way max operations using when/otherwise.
    let max_hl_hc = when(hl.clone().gt_eq(hc.clone())).then(hl).otherwise(hc);
    when(max_hl_hc.clone().gt_eq(lc.clone()))
        .then(max_hl_hc)
        .otherwise(lc)
}

/// Core ATR expression — operates on a single time series.
///
/// Wilder-smoothed True Range. Caller handles grouping and aliasing.
pub fn atr_expr(period: usize) -> Expr {
    wilder_smooth(true_range_expr(), period)
}

/// Multi-ticker adapter: compute ATR per ticker and append the resulting
/// column to the DataFrame.
///
/// Adds one column named `atr{period}` (e.g. `atr14`, `atr20`).
///
/// True Range = max(high − low, |high − prev_close|, |low − prev_close|).
/// Smoothed using Wilder's method (α = 1/period).
///
/// ATR is in the same units as the underlying price (dollars for equities).
/// For cross-asset comparison, derive NATR as `atr / close * 100` at
/// query time.
///
/// Requires the input DataFrame to be sorted by `(ticker, date)`.
pub fn atr(df: DataFrame, period: usize) -> Result<DataFrame> {
    println!("atr({period})...");
    let start = Instant::now();

    let col_name = format!("atr{period}");
    let expr = atr_expr(period)
        .over([col("ticker")])
        .alias(col_name.as_str());

    let result = df.lazy().with_columns([expr]).collect()?;

    println!("atr completed in {:.2?}", start.elapsed());
    Ok(result)
}

/// Core +DI expression — operates on a single time series.
///
/// 100 × wilder_smooth(+DM) / wilder_smooth(TR). Caller handles grouping.
pub fn plus_di_expr(period: usize) -> Expr {
    // Wilder's +DM rule: +DM = up_move if up_move > down_move AND up_move > 0,
    // else 0. Where up_move = high − prev_high, down_move = prev_low − low.
    let up_move = f("high") - f("high").shift(lit(1));
    let down_move = f("low").shift(lit(1)) - f("low");
    let plus_dm = when(
        up_move
            .clone()
            .gt(down_move)
            .and(up_move.clone().gt(lit(0.0))),
    )
    .then(up_move)
    .otherwise(lit(0.0));

    lit(100.0) * wilder_smooth(plus_dm, period) / wilder_smooth(true_range_expr(), period)
}

/// Core -DI expression — operates on a single time series.
///
/// 100 × wilder_smooth(-DM) / wilder_smooth(TR). Caller handles grouping.
pub fn minus_di_expr(period: usize) -> Expr {
    // Wilder's -DM rule: -DM = down_move if down_move > up_move AND
    // down_move > 0, else 0.
    let up_move = f("high") - f("high").shift(lit(1));
    let down_move = f("low").shift(lit(1)) - f("low");
    let minus_dm = when(
        down_move
            .clone()
            .gt(up_move)
            .and(down_move.clone().gt(lit(0.0))),
    )
    .then(down_move)
    .otherwise(lit(0.0));

    lit(100.0) * wilder_smooth(minus_dm, period) / wilder_smooth(true_range_expr(), period)
}

/// Multi-ticker adapter: compute ADX per ticker and append the resulting
/// column to the DataFrame.
///
/// Adds one column named `adx{period}` (e.g. `adx14`).
///
/// ADX measures *trend strength* regardless of direction. Conventionally:
///   - ADX < 20: weak/absent trend
///   - ADX > 25: trending market
///   - ADX > 40: strong trend
///
/// +DI and -DI are computed internally as intermediate steps (DX depends
/// on them) but not persisted as columns. If directionality is needed
/// alongside trend strength, use `plus_di_expr` and `minus_di_expr`
/// directly to materialize them.
///
/// Note: ADX has a ~2×period bar warm-up before stabilizing (two cascaded
/// Wilder smoothings). Early values are unreliable.
///
/// Requires the input DataFrame to be sorted by `(ticker, date)`.
pub fn adx(df: DataFrame, period: usize) -> Result<DataFrame> {
    println!("adx({period})...");
    let start = Instant::now();

    let col_name = format!("adx{period}");
    // Parameterize temp column names so concurrent ADX calls with different
    // periods don't collide. These are dropped before the function returns.
    let plusdi_tmp = format!("plusdi_{period}");
    let minusdi_tmp = format!("minusdi_{period}");

    let result = df
        .lazy()
        // Wave 1: materialize +DI and -DI as temp columns. Cheaper than
        // inlining them into DX since wilder_smooth(TR) would otherwise
        // be evaluated twice in the DX expression tree.
        .with_columns([
            plus_di_expr(period)
                .over([col("ticker")])
                .alias(plusdi_tmp.as_str()),
            minus_di_expr(period)
                .over([col("ticker")])
                .alias(minusdi_tmp.as_str()),
        ])
        // Wave 2: DX = 100 * |+DI - -DI| / (+DI + -DI), then ADX as Wilder-
        // smoothed DX.
        .with_columns([{
            let dx = lit(100.0) * (col(plusdi_tmp.as_str()) - col(minusdi_tmp.as_str())).abs()
                / (col(plusdi_tmp.as_str()) + col(minusdi_tmp.as_str()));
            wilder_smooth(dx, period)
                .over([col("ticker")])
                .alias(col_name.as_str())
        }])
        .drop([plusdi_tmp.as_str(), minusdi_tmp.as_str()])
        .collect()?;

    println!("adx completed in {:.2?}", start.elapsed());
    Ok(result)
}

/// Shorthand: reference a column cast to Float64 for safe arithmetic.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// CAGR expression. Compounded growth over `periods` annual periods
/// (the shift looks back periods*4 rows, assuming quarterly data).
/// Returns percentage value. Caller handles grouping.
///
/// Accepts any `Expr`, so CAGR can be computed on a derived quantity
/// (e.g. `cagr_expr(f("revenue") / f("shares"), 2)` for 2y CAGR of
/// revenue per share).
pub fn cagr_expr(source: Expr, periods: i64) -> Expr {
    ((source.clone() / source.shift(lit(periods * 4))).pow(lit(1.0 / periods as f64)) - lit(1.0))
        * lit(100.0)
}

/// Inner-join pre-filtered company metadata with the most recent financial
/// data and the most recent technical indicators per ticker, then convert
/// `rs1y` to a cross-sectional percentile rank.
///
/// Inputs:
///   - `financials_ttm`: TTM fundamentals frame (multi-row per ticker).
///     The latest row per ticker is selected via group-wise sort.
///   - `companies_meta`: pre-filtered company metadata (one row per ticker).
///   - `stock_prices`: prices frame after `technical_indicators` has been
///     applied. The latest row per ticker is selected via group-wise sort
///     and joined in alongside the financials.
///
/// Output: one row per ticker, with meta + latest fundamentals + latest
/// technicals + `rs1y` recomputed as a cross-sectional percentile rank
/// (0-100, higher = stronger relative-strength performer).
pub fn join_company_financials(
    financials_ttm: DataFrame,
    companies_meta: DataFrame,
    stock_prices: DataFrame,
) -> Result<DataFrame> {
    println!("join_company_financials...");
    let start = Instant::now();

    // Project fundamentals to just the columns we need BEFORE the per-ticker
    // latest-row selection. Shrinks the data the group-by has to shuffle
    // through from the full fundamentals schema (~80 columns) to ~30.
    let projected_financials = financials_ttm.lazy().select([
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
        col("revenue3y"),
        col("ebitdayoy"),
        col("ebitdacagr"),
        col("ebitda3y"),
        col("epsyoy"),
        col("epscagr"),
        col("eps3y"),
        col("fcfpsyoy"),
        col("fcfpscagr"),
        col("fcfps3y"),
        col("shyield"),
        col("roa"),
        col("roe"),
        col("roic"),
        col("roce"),
        col("icr"),
        col("cfc"),
    ]);

    // Per-ticker latest fundamental row.
    let latest_financials = projected_financials
        .group_by([col("ticker")])
        .agg([col("*")
            .sort_by(
                [col("date")],
                SortMultipleOptions::default().with_order_descending(true),
            )
            .first()])
        .drop(["date"]);

    // Per-ticker latest prices+technicals row. No projection — we keep all
    // columns from technical_indicators so callers can screen on any of them
    // downstream. The only column dropped is `date`, which gets re-introduced
    // by the financials join (and would otherwise collide).
    let latest_prices = stock_prices
        .lazy()
        .group_by([col("ticker")])
        .agg([col("*")
            .sort_by(
                [col("date")],
                SortMultipleOptions::default().with_order_descending(true),
            )
            .first()])
        .drop(["date"]);

    // companies_meta ← latest_financials ← latest_prices, all inner-joined
    // on ticker. Companies missing either fundamentals or recent prices
    // are dropped (data quality issue worth surfacing, not silently filling).
    let joined = companies_meta
        .lazy()
        .join(
            latest_financials,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        )
        .join(
            latest_prices,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        )
        .collect()?;

    // Replace rs1y with its cross-sectional percentile rank across all
    // companies in the joined output. The raw rs1y is a weighted sum of
    // 1q/2q/3q/1y price returns which has no intuitive scale; the percentile
    // form puts every company on a comparable 0-100 scale where 50 is the
    // median performer and 100 is the strongest. Nulls remain null.
    //
    // Method: average ranks for ties (standard convention), mapped to
    // [0, 100] via (rank - 1) / (n - 1) * 100.
    // RankOptions::default() = average method, ascending. Both are what we want.
    // Using Default::default() avoids needing to import RankOptions/RankMethod
    // explicitly — the prelude already pulls in everything needed when the
    // `rank` feature is enabled in Cargo.toml (see docs note at top of file).
    let n = col("rs1y").count().cast(DataType::Float64);
    let rank = col("rs1y").rank(Default::default(), None);
    let percentile = ((rank - lit(1.0)) / (n - lit(1.0)) * lit(100.0))
        // .round(0)
        // .cast(DataType::Int32)
        .alias("rs1y");

    // Market cap = close × shares. Both columns arrive in the joined frame:
    // close from latest_prices, shares from latest_financials. Float64 to
    // accommodate the wide range (millions to multi-trillions).
    let marketcap = (col("close") * col("shares")).alias("marketcap");

    // EV = marketcap + net debt (in USD). Computed in a separate with_columns
    // wave because Polars can't reliably reference a column created in the
    // same with_columns call. The note about net debt: in adjust_fundamentals
    // we compute netdebtusd as (debt - cashneq) / fxusd, which is the standard
    // EV definition (debt net of cash). Companies with more cash than debt
    // get a negative netdebtusd and therefore an EV below their market cap,
    // which is the correct behavior.
    let ev = (col("marketcap") + col("netdebtusd")).alias("ev");

    let result = joined
        .lazy()
        .with_columns([percentile, marketcap])
        .with_columns([ev])
        .collect()?;

    let elapsed = start.elapsed();
    println!("join_company_financials completed in {:.2?}", elapsed);
    Ok(result)
}
