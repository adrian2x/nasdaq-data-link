//! Realized (historical) volatility indicator.
//!
//! Annualized standard deviation of daily log-returns over rolling windows.
//! Self-contained.

use polars::prelude::*;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Fixed rolling window requiring the full period before emitting a value.
fn rolling_opts(window: usize) -> RollingOptionsFixedWindow {
    RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        ..Default::default()
    }
}

/// Annualized realized volatility (in percent) of `log_ret_col` over a
/// rolling `window`, scaled by `sqrt(trading_periods)`.
pub fn rv_expr(log_ret_col: &str, window: usize, trading_periods: usize) -> Expr {
    let log_ret = col(log_ret_col);
    let mean = log_ret.clone().rolling_mean(rolling_opts(window));
    let mean_sq = (log_ret.clone() * log_ret).rolling_mean(rolling_opts(window));
    let pop_std = (mean_sq - mean.clone() * mean).abs().pow(lit(0.5));
    pop_std * lit((trading_periods as f64).sqrt()) * lit(100.0)
}

/// Add realized-volatility columns to `lf`, one per window: `rv10`, `rv21`,
/// ...
///
/// Per-ticker log-returns are computed from `source` internally via a
/// scratch `_log_ret` column, which is dropped before returning — the caller
/// does not need to materialize or clean up a log-return column. Output
/// columns are cast to Float32.
pub fn realized_volatility(
    lf: LazyFrame,
    source: &str,
    trading_periods: usize,
    windows: &[usize],
) -> LazyFrame {
    let lf = lf.with_columns([(f(source) / f(source).shift(lit(1)))
        .log(std::f64::consts::E)
        .over([col("ticker")])
        .alias("_log_ret")]);
    let cols: Vec<Expr> = windows
        .iter()
        .map(|&w| {
            rv_expr("_log_ret", w, trading_periods)
                .over([col("ticker")])
                .alias(format!("rv{w}"))
                .cast(DataType::Float32)
        })
        .collect();
    lf.with_columns(cols).drop(["_log_ret"])
}
