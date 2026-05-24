//! Exponential moving average indicator.
//!
//! Self-contained: shares no code with sibling indicator modules.

use polars::prelude::*;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Partition `expr` per ticker and alias it.
fn per_ticker(expr: Expr, name: &str) -> Expr {
    expr.over([col("ticker")]).alias(name)
}

/// Exponential moving average expression using span-based alpha
/// (`alpha = 2 / (span + 1)`).
pub fn ema_expr(source: &str, span: usize) -> Expr {
    f(source).ewm_mean(EWMOptions {
        alpha: 2.0 / (span as f64 + 1.0),
        min_periods: span,
        ..Default::default()
    })
}

/// Add EMA columns to `lf`, one per span: `ema8`, `ema12`, ...
pub fn ema(lf: LazyFrame, source: &str, spans: &[usize]) -> LazyFrame {
    let cols: Vec<Expr> = spans
        .iter()
        .map(|&s| per_ticker(ema_expr(source, s), &format!("ema{s}")))
        .collect();
    lf.with_columns(cols)
}
