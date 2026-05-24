//! MACD (moving average convergence divergence) indicator.
//!
//! Self-contained — including its own EMA expression, so it shares nothing
//! with the `ema` module.

use polars::prelude::*;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Span-based exponential moving average expression.
fn ema_expr(source: &str, span: usize) -> Expr {
    f(source).ewm_mean(EWMOptions {
        alpha: 2.0 / (span as f64 + 1.0),
        min_periods: span,
        ..Default::default()
    })
}

/// MACD line: fast EMA minus slow EMA of close.
pub fn macd_line_expr(fast: usize, slow: usize) -> Expr {
    ema_expr("close", fast) - ema_expr("close", slow)
}

/// Add MACD columns to `lf`.
///
/// With the default (12, 26, 9) parameters the columns are `ema12`, `ema26`,
/// `macd`, `macdsignal`; non-default parameters get suffixed names so several
/// MACD configurations can coexist.
pub fn macd(lf: LazyFrame, fast: usize, slow: usize, signal: usize) -> LazyFrame {
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

    // Signal is computed in a second wave because it depends on `macd_col`.
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
