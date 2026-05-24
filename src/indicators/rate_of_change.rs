//! Rate-of-change indicator.
//!
//! Time-series momentum: each column is the percent price change over a
//! fixed number of trading days. Self-contained.

use polars::prelude::*;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Percent change of `source` over `period` rows: `end / start - 1`.
///
/// Assumes a strictly positive `start` — correct for price series. Not
/// suitable for values that can go negative (use a sign-safe growth measure
/// for fundamentals instead).
pub fn chg_expr(source: Expr, period: i64) -> Expr {
    ((source.clone() / source.shift(lit(period))) - lit(1.0)) * lit(100.0)
}

/// Add percent-change columns to `lf`, one per period: `pct1`, `pct5`, ...
///
/// Output columns are cast to Float32 — the bounded range makes the reduced
/// precision sufficient and halves stored width.
pub fn rate_of_change(lf: LazyFrame, source: &str, periods: &[i64]) -> LazyFrame {
    let cols: Vec<Expr> = periods
        .iter()
        .map(|&p| {
            chg_expr(f(source), p)
                .over([col("ticker")])
                .alias(format!("pct{p}"))
                .cast(DataType::Float32)
        })
        .collect();
    lf.with_columns(cols)
}
