//! ADR (average daily range) indicator.
//!
//! Measures the rolling average of daily intraday range in percent:
//! `(high / low - 1) * 100`. Self-contained.

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

/// Add an ADR column to `lf`.
///
/// With the default 20-day period, the output column is `adr`; otherwise
/// the column name is `adr{period}` (e.g. `adr14`).
pub fn adr(lf: LazyFrame, period: usize) -> LazyFrame {
    let adr_col = if period == 20 {
        "adr".to_string()
    } else {
        format!("adr{period}")
    };
    lf.with_columns([(((f("high") / f("low")) - lit(1.0)) * lit(100.0))
        .rolling_mean(rolling_opts(period))
        .over([col("ticker")])
        .alias(adr_col.as_str())
        .cast(DataType::Float32)])
}
