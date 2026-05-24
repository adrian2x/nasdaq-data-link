//! RSI (relative strength index) indicator.
//!
//! Self-contained. Carries its own copy of Wilder smoothing — the `atr` and
//! `adx` modules carry their own; this duplication is deliberate, so each
//! indicator module stays fully isolated.

use polars::prelude::*;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Wilder's smoothing: an EWM with `alpha = 1 / period`.
fn wilder_smooth(source: Expr, period: usize) -> Expr {
    source.ewm_mean(EWMOptions {
        alpha: 1.0 / period as f64,
        min_periods: period,
        ..Default::default()
    })
}

/// Add an RSI column to `lf`, named `rsi{period}` (e.g. `rsi14`).
pub fn rsi(lf: LazyFrame, period: usize) -> LazyFrame {
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
