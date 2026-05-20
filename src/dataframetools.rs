use ::zip::ZipArchive;
use anyhow::{Result, anyhow};
use polars::prelude::*;
use std::fs::File;
use std::io::copy;
use std::path::Path;

const OUTPUT_DIR: &str = "output";

/// Stream a CSV file to a Parquet file.
///
/// `csv_path`     — path to a plain (unzipped) CSV.
/// `parquet_path` — path to write the Parquet output to.
/// `schema_overrides` — columns pinned to known dtypes at parse time; all
///                      other columns are inferred.
///
/// The frame is never collected: `sink_parquet` streams the scan in batches,
/// so peak memory is one batch regardless of total file size.
///
/// Re-runs overwrite in place — `File::create` truncates any existing file;
/// nothing is deleted.
pub fn csv_to_parquet<P: AsRef<Path>, Q: AsRef<Path>>(
    csv_path: P,
    parquet_path: Q,
    schema_overrides: Option<&[(&str, DataType)]>,
) -> Result<()> {
    // Parse-time dtype schema: only the named columns are pinned.
    let overrides_schema = schema_overrides.map(|pairs| {
        let fields = pairs
            .iter()
            .map(|(name, dtype)| Field::new((*name).into(), dtype.clone()));
        Arc::new(Schema::from_iter(fields))
    });

    let lf = LazyCsvReader::new(csv_path.as_ref())
        .with_has_header(true)
        // Pins the named columns during parsing; no post-parse cast pass.
        .with_dtype_overwrite(overrides_schema)
        .finish()?;

    // Streaming sink: a plain projection/cast scan has no `.over()` chains,
    // so the streaming engine handles this plan and memory stays flat
    // regardless of how large the CSV is.
    //
    // Polars 0.46 `sink_parquet` signature: (path, ParquetWriteOptions,
    // Option<CloudOptions>). The path param is `&dyn AsRef<Path>` — a trait
    // object reference — so `parquet_path` is passed by reference (`&`).
    lf.sink_parquet(&parquet_path, ParquetWriteOptions::default(), None)?;

    Ok(())
}

/// Extract the first entry of a zip to a file alongside it (suffix stripped),
/// e.g. `foo.csv.zip` -> `foo.csv`. Returns the extracted path.
///
/// `File::create` truncates, so re-runs overwrite the extracted file.
pub fn extract_zip_file<P: AsRef<Path>>(zip_path: P) -> Result<String> {
    let zip_path = zip_path.as_ref();
    let zip_str = zip_path.to_string_lossy();
    let output_filename = zip_str
        .strip_suffix(".zip")
        .map(str::to_string)
        .unwrap_or_else(|| zip_str.to_string());

    let zip_file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(zip_file)?;
    let mut csv_file = archive.by_index(0)?;
    let mut output_file = File::create(&output_filename)?;
    copy(&mut csv_file, &mut output_file)?;
    Ok(output_filename)
}

/// Collect a LazyFrame, write it as parquet, and register it as a DuckDB table.
///
/// Streaming sink isn't viable in Polars 0.46 for our heavy `.over()` chains
/// (the streaming engine doesn't support those plan shapes yet), so we
/// materialize and write. Everything upstream stays lazy so the optimizer
/// sees the full pipeline; only this final step is eager.
pub fn lf_to_duckdb(lf: LazyFrame, table: &str) -> Result<()> {
    let table = table.trim();
    if table.is_empty() {
        return Err(anyhow!("table name cannot be empty"));
    }

    std::fs::create_dir_all(OUTPUT_DIR)?;
    let parquet_path = format!("{OUTPUT_DIR}/{table}.parquet");

    if std::env::var("EXPLAIN_PLAN").is_ok() {
        eprintln!("=== plan for {table} ===");
        eprintln!("{}", lf.clone().explain(true)?);
        eprintln!("=== end plan ===");
    }

    let mut df = lf.collect()?;
    ParquetWriter::new(File::create(&parquet_path)?).finish(&mut df)?;

    let qt = quote_ident(table);
    crate::sqltools::execute_sql(
        &format!("create_{table}"),
        &format!("CREATE OR REPLACE TABLE {qt} AS SELECT * FROM read_parquet('{parquet_path}');"),
    )?;

    Ok(())
}

fn quote_ident(s: &str) -> String {
    format!("\"{}\"", s.replace('\"', "\"\""))
}

// ---------------------------------------------------------------------------
// Expression building blocks
// ---------------------------------------------------------------------------

fn rolling_opts(window: usize) -> RollingOptionsFixedWindow {
    RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        ..Default::default()
    }
}

/// Shorthand: reference a column cast to Float64 for safe arithmetic.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// SMA expression. Caller handles grouping (e.g. `.over([col("ticker")])`).
pub fn sma_expr(source: &str, period: usize) -> Expr {
    f(source).rolling_mean(rolling_opts(period))
}

/// Span-based EMA expression (α = 2/(span+1)). Caller handles grouping.
/// Distinct from `wilder_smooth` which uses α = 1/n.
pub fn ema_expr(source: &str, span: usize) -> Expr {
    f(source).ewm_mean(EWMOptions {
        alpha: 2.0 / (span as f64 + 1.0),
        min_periods: span,
        ..Default::default()
    })
}

/// Percentage-change expression. ((current / prev) - 1) * 100.
pub fn chg_expr(source: Expr, period: i64) -> Expr {
    ((source.clone() / source.shift(lit(period))) - lit(1.0)) * lit(100.0)
}

/// CAGR expression. Compounded growth over `periods` annual periods
/// (shift looks back periods*4 rows, assuming quarterly data).
pub fn cagr_expr(source: Expr, periods: i64) -> Expr {
    ((source.clone() / source.shift(lit(periods * 4))).pow(lit(1.0 / periods as f64)) - lit(1.0))
        * lit(100.0)
}

/// Realized volatility expression. Annualized population std of a log-return
/// series, returned as a percentage.
pub fn rv_expr(log_ret_col: &str, window: usize, trading_periods: usize) -> Expr {
    let log_ret = col(log_ret_col);
    let mean = log_ret.clone().rolling_mean(rolling_opts(window));
    let mean_sq = (log_ret.clone() * log_ret).rolling_mean(rolling_opts(window));
    let pop_std = (mean_sq - mean.clone() * mean).abs().pow(lit(0.5));
    pop_std * lit((trading_periods as f64).sqrt()) * lit(100.0)
}

/// Wilder's smoothing: α = 1/period EMA. Used by RSI, ATR, ADX.
/// Distinct from `ema_expr` (which uses span-based α = 2/(span+1)).
fn wilder_smooth(source: Expr, period: usize) -> Expr {
    source.ewm_mean(EWMOptions {
        alpha: 1.0 / period as f64,
        min_periods: period,
        ..Default::default()
    })
}

/// MACD line expression: EMA(close, fast) − EMA(close, slow). Caller groups.
pub fn macd_line_expr(fast: usize, slow: usize) -> Expr {
    ema_expr("close", fast) - ema_expr("close", slow)
}

/// Bollinger upper band expression. Caller groups.
pub fn bbtop_expr(period: usize, multiplier: f64) -> Expr {
    let sma = f("close").rolling_mean(rolling_opts(period));
    let stdev = f("close").rolling_std(rolling_opts(period));
    sma + lit(multiplier) * stdev
}

/// Bollinger lower band expression. Caller groups.
pub fn bbbot_expr(period: usize, multiplier: f64) -> Expr {
    let sma = f("close").rolling_mean(rolling_opts(period));
    let stdev = f("close").rolling_std(rolling_opts(period));
    sma - lit(multiplier) * stdev
}

/// True Range expression. Composable for ad-hoc use.
pub fn true_range_expr() -> Expr {
    let prev_close = f("close").shift(lit(1));
    let hl = f("high") - f("low");
    let hc = (f("high") - prev_close.clone()).abs();
    let lc = (f("low") - prev_close).abs();
    let max_hl_hc = when(hl.clone().gt_eq(hc.clone())).then(hl).otherwise(hc);
    when(max_hl_hc.clone().gt_eq(lc.clone()))
        .then(max_hl_hc)
        .otherwise(lc)
}

// ---------------------------------------------------------------------------
// Lazy indicator helpers
//
// Each `with_*` takes a LazyFrame, appends the indicator columns, and returns
// the LazyFrame. They never collect — the optimizer sees the full pipeline
// and fuses redundant per-ticker partitioning across expressions.
//
// Wilder-smoothed indicators (RSI, ATR, ADX) use a hoisted-ternary pattern:
// inputs to `when().then().otherwise()` are materialized as flat per-ticker
// columns in a setup wave, so the ternaries themselves don't run under
// `.over()` (which would trigger expensive list aggregation). Each function
// is self-contained — call any subset, in any order.
// ---------------------------------------------------------------------------

pub fn with_macd(lf: LazyFrame, fast: usize, slow: usize, signal: usize) -> LazyFrame {
    let is_default = fast == 12 && slow == 26 && signal == 9;
    let (ema_fast_col, ema_slow_col, macd_col, signal_col) = if is_default {
        (
            "ema12".to_string(),
            "ema26".to_string(),
            "macd".to_string(),
            "macdsignal".to_string(),
        )
    } else {
        (
            format!("ema{fast}"),
            format!("ema{slow}"),
            format!("macd_{fast}_{slow}"),
            format!("macdsignal_{fast}_{slow}_{signal}"),
        )
    };

    // Two waves: signal line references the just-created macd column by name,
    // and ewm_mean over a freshly-created column isn't reliable in a single
    // with_columns wave.
    lf.with_columns([
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
    .with_columns([ema_expr(macd_col.as_str(), signal)
        .over([col("ticker")])
        .alias(signal_col.as_str())])
}

pub fn with_bollinger(lf: LazyFrame, period: usize, multiplier: f64) -> LazyFrame {
    let is_default = period == 20 && (multiplier - 2.0).abs() < 1e-9;
    let (top_col, bot_col) = if is_default {
        ("bbtop".to_string(), "bbbot".to_string())
    } else {
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

    lf.with_columns([
        bbbot_expr(period, multiplier)
            .over([col("ticker")])
            .alias(bot_col.as_str()),
        bbtop_expr(period, multiplier)
            .over([col("ticker")])
            .alias(top_col.as_str()),
    ])
}

pub fn with_rsi(lf: LazyFrame, period: usize) -> LazyFrame {
    let rsi_col = format!("rsi{period}");

    lf.with_columns([(f("close") - f("close").shift(lit(1)))
        .over([col("ticker")])
        .alias("_delta")])
        .with_columns([
            when(col("_delta").gt(lit(0.0)))
                .then(col("_delta"))
                .otherwise(lit(0.0))
                .alias("_gain"),
            when(col("_delta").lt(lit(0.0)))
                .then(-col("_delta"))
                .otherwise(lit(0.0))
                .alias("_loss"),
        ])
        .with_columns([
            wilder_smooth(col("_gain"), period)
                .over([col("ticker")])
                .alias("_avg_gain"),
            wilder_smooth(col("_loss"), period)
                .over([col("ticker")])
                .alias("_avg_loss"),
        ])
        .with_columns([(lit(100.0)
            - lit(100.0) / (lit(1.0) + col("_avg_gain") / col("_avg_loss")))
        .alias(rsi_col.as_str())])
        .drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"])
}

pub fn with_atr(lf: LazyFrame, period: usize) -> LazyFrame {
    let atr_col = format!("atr{period}");

    lf.with_columns([f("close")
        .shift(lit(1))
        .over([col("ticker")])
        .alias("_prev_close")])
        .with_columns([{
            let hl = f("high") - f("low");
            let hc = (f("high") - col("_prev_close")).abs();
            let lc = (f("low") - col("_prev_close")).abs();
            let max_hl_hc = when(hl.clone().gt_eq(hc.clone())).then(hl).otherwise(hc);
            when(max_hl_hc.clone().gt_eq(lc.clone()))
                .then(max_hl_hc)
                .otherwise(lc)
                .alias("_tr")
        }])
        .with_columns([wilder_smooth(col("_tr"), period)
            .over([col("ticker")])
            .alias(atr_col.as_str())])
        .drop(["_prev_close", "_tr"])
}

pub fn with_adx(lf: LazyFrame, period: usize) -> LazyFrame {
    let adx_col = format!("adx{period}");

    lf.with_columns([
        (f("high") - f("high").shift(lit(1)))
            .over([col("ticker")])
            .alias("_up_move"),
        (f("low").shift(lit(1)) - f("low"))
            .over([col("ticker")])
            .alias("_down_move"),
        f("close")
            .shift(lit(1))
            .over([col("ticker")])
            .alias("_prev_close"),
    ])
    .with_columns([
        {
            let hl = f("high") - f("low");
            let hc = (f("high") - col("_prev_close")).abs();
            let lc = (f("low") - col("_prev_close")).abs();
            let max_hl_hc = when(hl.clone().gt_eq(hc.clone())).then(hl).otherwise(hc);
            when(max_hl_hc.clone().gt_eq(lc.clone()))
                .then(max_hl_hc)
                .otherwise(lc)
                .alias("_tr")
        },
        when(
            col("_up_move")
                .gt(col("_down_move"))
                .and(col("_up_move").gt(lit(0.0))),
        )
        .then(col("_up_move"))
        .otherwise(lit(0.0))
        .alias("_plus_dm"),
        when(
            col("_down_move")
                .gt(col("_up_move"))
                .and(col("_down_move").gt(lit(0.0))),
        )
        .then(col("_down_move"))
        .otherwise(lit(0.0))
        .alias("_minus_dm"),
    ])
    .with_columns([
        wilder_smooth(col("_tr"), period)
            .over([col("ticker")])
            .alias("_smooth_tr"),
        wilder_smooth(col("_plus_dm"), period)
            .over([col("ticker")])
            .alias("_smooth_plus_dm"),
        wilder_smooth(col("_minus_dm"), period)
            .over([col("ticker")])
            .alias("_smooth_minus_dm"),
    ])
    .with_columns([
        (lit(100.0) * col("_smooth_plus_dm") / col("_smooth_tr")).alias("_plus_di"),
        (lit(100.0) * col("_smooth_minus_dm") / col("_smooth_tr")).alias("_minus_di"),
    ])
    .with_columns([wilder_smooth(
        lit(100.0) * (col("_plus_di") - col("_minus_di")).abs()
            / (col("_plus_di") + col("_minus_di")),
        period,
    )
    .over([col("ticker")])
    .alias(adx_col.as_str())])
    .drop([
        "_up_move",
        "_down_move",
        "_prev_close",
        "_tr",
        "_plus_dm",
        "_minus_dm",
        "_smooth_tr",
        "_smooth_plus_dm",
        "_smooth_minus_dm",
        "_plus_di",
        "_minus_di",
    ])
}

// ---------------------------------------------------------------------------
// Pipeline stages — all return LazyFrame
// ---------------------------------------------------------------------------

/// Apply close-adjustment to OHLC and clean up the prices frame.
///
/// Collapsed into a single `select` so the optimizer sees the full final
/// shape immediately — columns we drop never get parsed.
pub fn adjust_prices(lf: LazyFrame) -> LazyFrame {
    let adjustment_factor =
        col("closeadj").cast(DataType::Float64) / col("close").cast(DataType::Float64);

    lf.select([
        col("ticker"),
        col("date").cast(DataType::Date),
        (col("open").cast(DataType::Float64) * adjustment_factor.clone()).alias("open"),
        (col("high").cast(DataType::Float64) * adjustment_factor.clone()).alias("high"),
        (col("low").cast(DataType::Float64) * adjustment_factor).alias("low"),
        col("closeadj").cast(DataType::Float64).alias("close"),
        col("volume").cast(DataType::Float64),
    ])
    .sort(["ticker", "date"], Default::default())
}

/// Resample OHLCV bars to a coarser interval, per ticker.
///
/// `interval` follows Polars duration syntax: `"1w"`, `"1mo"`, `"1q"`, etc.
/// Output `date` is the left edge of each period.
///
/// Expects columns: ticker, date (Date dtype), open, high, low, close, volume.
/// Other columns are silently dropped — applying close-of-period semantics to
/// derived columns (Hurst, technical indicators) would produce silently wrong
/// values. Compute indicators on resampled bars downstream, not before.
pub fn resample(lf: LazyFrame, interval: &str) -> LazyFrame {
    lf.group_by_dynamic(
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
}

/// Transform raw fundamentals into the analytical schema.
pub fn adjust_fundamentals(lf: LazyFrame) -> LazyFrame {
    lf.with_columns([col("calendardate").cast(DataType::Date)])
        .drop(["dimension", "lastupdated"])
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
        // Depends on the previous wave by alias name.
        .with_columns([
            (f("debt") / f("equity")).alias("de"),
            (f("netincadj") / f("shares")).alias("epsadj"),
            (lit(100.0) * f("netincadj") / f("revenueusd")).alias("netmargin"),
            (lit(100.0) * f("shreturnusd") / f("marketcap")).alias("shyield"),
        ])
        .sort(["ticker", "calendardate"], Default::default())
        // fcfadj uses .shift() so must run after the sort.
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
        // Growth metrics. Revenue/EBITDA growth are per-share so buybacks
        // and issuances are reflected. epsadj/fcfpsadj are already per-share.
        .with_columns([
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
}

/// Transform raw insider transactions: per-(ticker, date, person) summary
/// filtered to the last 6 months.
pub fn update_insiders(lf: LazyFrame) -> LazyFrame {
    let six_months_ago = chrono::Utc::now().date_naive() - chrono::Duration::weeks(26);

    lf.with_columns([
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
}

/// Compute the full daily indicator set on a daily-bars price LazyFrame.
///
/// Hurst must be applied upstream (via `fractaltools::with_hurst`).
///
/// Daily-specific conventions:
///   - SMA/EMA at 5/10/20/50/100/150/200 daily bars
///   - ROC at 1d, 5d, 20d, 63d, 126d, 189d, 252d
///   - RV at 5/20/63/252 bars (annualized using 252 trading periods)
///   - avgvolume3m (50-bar SMA of volume)
///   - max1y/min1y (250-bar rolling extrema)
///   - rs1y (weighted composite of roc1q/2q/3q/1y)
///
/// Timeframe-invariant: MACD(12,26,9), Bollinger(20,2), RSI(14), ATR(14),
/// ADX(14).
pub fn technical_indicators_daily(lf: LazyFrame) -> LazyFrame {
    let range_opts = RollingOptionsFixedWindow {
        window_size: 250,
        min_periods: 2,
        ..Default::default()
    };

    let lf = lf
        .sort(["ticker", "date"], Default::default())
        .with_columns([(f("close") / f("close").shift(lit(1)))
            .log(std::f64::consts::E)
            .over([col("ticker")])
            .alias("log_ret")])
        .with_columns([
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
            sma_expr("close", 5).over([col("ticker")]).alias("sma5"),
            sma_expr("close", 10).over([col("ticker")]).alias("sma10"),
            sma_expr("close", 20).over([col("ticker")]).alias("sma20"),
            sma_expr("close", 50).over([col("ticker")]).alias("sma50"),
            sma_expr("close", 100).over([col("ticker")]).alias("sma100"),
            sma_expr("close", 150).over([col("ticker")]).alias("sma150"),
            sma_expr("close", 200).over([col("ticker")]).alias("sma200"),
            sma_expr("volume", 50)
                .over([col("ticker")])
                .alias("avgvolume3m"),
            ema_expr("close", 10).over([col("ticker")]).alias("ema10"),
            ema_expr("close", 20).over([col("ticker")]).alias("ema20"),
            ema_expr("close", 63).over([col("ticker")]).alias("ema63"),
            ema_expr("close", 126).over([col("ticker")]).alias("ema126"),
            ema_expr("close", 250).over([col("ticker")]).alias("ema250"),
            f("close")
                .rolling_max(range_opts.clone())
                .over([col("ticker")])
                .alias("max1y"),
            f("close")
                .rolling_min(range_opts)
                .over([col("ticker")])
                .alias("min1y"),
        ])
        // rs1y: pure arithmetic on per-ticker columns. No `.over()`.
        .with_columns([(lit(0.4) * col("roc1q")
            + lit(0.2) * col("roc2q")
            + lit(0.2) * col("roc3q")
            + lit(0.2) * col("roc1y"))
        .alias("rs1y")])
        .drop(["log_ret"])
        .with_columns([
            col("roc1d").cast(DataType::Float32),
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
        ]);

    let lf = with_macd(lf, 12, 26, 9);
    let lf = with_bollinger(lf, 20, 2.0);
    let lf = with_rsi(lf, 14);
    let lf = with_atr(lf, 14);
    with_adx(lf, 14)
}

/// Compute the full weekly indicator set on a weekly-bars price LazyFrame.
///
/// Hurst must be applied upstream with a weekly-appropriate window
/// (~100 bars ≈ 2 years).
///
/// Weekly-specific conventions:
///   - SMA/EMA at 10/30/40/200 weeks (Stan Weinstein's 10/40-week trend rules)
///   - ROC at 1w, 4w, 13w, 26w, 39w, 52w
///   - RV at 4/13/52 weeks (annualized using 52 weeks/year)
///   - avgvolume3m (13-bar SMA of volume)
///   - max1y/min1y (52-bar rolling extrema)
///   - rs1y (weighted composite of roc1q/2q/3q/1y at weekly cadence)
///
/// Timeframe-invariant: same MACD/Bollinger/RSI/ATR/ADX as daily.
pub fn technical_indicators_weekly(lf: LazyFrame) -> LazyFrame {
    let range_opts = RollingOptionsFixedWindow {
        window_size: 52,
        min_periods: 2,
        ..Default::default()
    };

    let lf = lf
        .sort(["ticker", "date"], Default::default())
        .with_columns([(f("close") / f("close").shift(lit(1)))
            .log(std::f64::consts::E)
            .over([col("ticker")])
            .alias("log_ret")])
        .with_columns([
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
            rv_expr("log_ret", 4, 52)
                .over([col("ticker")])
                .alias("rv1m"),
            rv_expr("log_ret", 13, 52)
                .over([col("ticker")])
                .alias("rv1q"),
            rv_expr("log_ret", 52, 52)
                .over([col("ticker")])
                .alias("rv1y"),
            sma_expr("close", 10).over([col("ticker")]).alias("sma10"),
            sma_expr("close", 30).over([col("ticker")]).alias("sma30"),
            sma_expr("close", 40).over([col("ticker")]).alias("sma40"),
            sma_expr("close", 200).over([col("ticker")]).alias("sma200"),
            sma_expr("volume", 13)
                .over([col("ticker")])
                .alias("avgvolume3m"),
            ema_expr("close", 10).over([col("ticker")]).alias("ema10"),
            ema_expr("close", 30).over([col("ticker")]).alias("ema30"),
            ema_expr("close", 40).over([col("ticker")]).alias("ema40"),
            ema_expr("close", 200).over([col("ticker")]).alias("ema200"),
            f("close")
                .rolling_max(range_opts.clone())
                .over([col("ticker")])
                .alias("max1y"),
            f("close")
                .rolling_min(range_opts)
                .over([col("ticker")])
                .alias("min1y"),
        ])
        .with_columns([(lit(0.4) * col("roc1q")
            + lit(0.2) * col("roc2q")
            + lit(0.2) * col("roc3q")
            + lit(0.2) * col("roc1y"))
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
        ]);

    let lf = with_macd(lf, 12, 26, 9);
    let lf = with_bollinger(lf, 20, 2.0);
    let lf = with_rsi(lf, 14);
    let lf = with_atr(lf, 14);
    with_adx(lf, 14)
}
