//! MACD (moving average convergence divergence) indicator.
//!
//! Self-contained — including its own EMA expression, so it shares nothing
//! with the `ema` module.

use polars::prelude::*;

/// Casts the column `name` to `Float64`.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Builds a span-based exponential moving average expression
/// (`alpha = 2 / (span + 1)`).
fn ema_expr(source: Expr, span: usize) -> Expr {
    source.ewm_mean(EWMOptions {
        alpha: 2.0 / (span as f64 + 1.0),
        min_periods: span,
        ..Default::default()
    })
}

/// Adds the MACD columns `macd` and `macd_signal` to `lf`.
///
/// `macd` is the fast-minus-slow EMA of close; `macd_signal` is the EMA of
/// the MACD line over `signal` periods. The fast and slow EMAs are computed
/// internally and not materialized as columns.
pub fn macd(lf: LazyFrame, fast: usize, slow: usize, signal: usize) -> LazyFrame {
    // MACD line: difference of the fast and slow close EMAs. The two EMAs
    // exist only inside this expression — they are never their own columns.
    let macd_line = ema_expr(f("close"), fast) - ema_expr(f("close"), slow);

    lf
        // Wave 1: materialize the MACD line. It must be a real column because
        // the signal line below is an EMA computed *from* it.
        .with_columns([macd_line.over([col("ticker")]).alias("macd")])
        // Wave 2: signal line — EMA of the now-materialized `macd` column.
        .with_columns([ema_expr(col("macd"), signal)
            .over([col("ticker")])
            .alias("macd_signal")])
}
