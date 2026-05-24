//! Simple moving average indicator.
//!
//! Self-contained: this module shares no code with sibling indicator
//! modules. The small primitives (`f`, `rolling_opts`, `per_ticker`) are
//! intentionally re-declared here rather than imported, so the module can be
//! read, changed, or removed in isolation.

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

/// Partition `expr` per ticker and alias it.
fn per_ticker(expr: Expr, name: &str) -> Expr {
    expr.over([col("ticker")]).alias(name)
}

/// Simple moving average expression over `period` rows of `source`.
pub fn sma_expr(source: &str, period: usize) -> Expr {
    f(source).rolling_mean(rolling_opts(period))
}

/// Add SMA columns to `lf`, one per period: `sma5`, `sma10`, ...
pub fn sma(lf: LazyFrame, source: &str, periods: &[usize]) -> LazyFrame {
    let cols: Vec<Expr> = periods
        .iter()
        .map(|&p| per_ticker(sma_expr(source, p), &format!("sma{p}")))
        .collect();
    lf.with_columns(cols)
}
