//! Donchian breakout channels.
//!
//! The upper band is the rolling maximum of intraday highs, the lower band
//! the rolling minimum of intraday lows. Each band covers the *prior* `w`
//! bars (shifted one bar back) so the current bar cannot break its own
//! channel — the columns are usable directly as breakout signal levels.
//!
//! This is distinct from a 52-week high/low *proximity* feature: a breakout
//! channel is a signal level the next bar is tested against, and the
//! canonical breakout lengths are short (10/20/55). The trailing 1-year
//! high/low is a different concept and lives in the `range52w` module.
//!
//! Self-contained.

use polars::prelude::*;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Partition `expr` per ticker and alias it.
fn per_ticker(expr: Expr, name: &str) -> Expr {
    expr.over([col("ticker")]).alias(name)
}

/// Add Donchian channel columns to `lf`, two per window: `min20`/`max20`,
/// `min55`/`max55`, ...
///
/// Requires the full window: a band is null until the ticker has `w` bars of
/// history, so a recent listing yields null rather than a partial-window
/// channel mislabeled as a `w`-bar one.
pub fn donchian(lf: LazyFrame, windows: &[usize]) -> LazyFrame {
    let cols: Vec<Expr> = windows
        .iter()
        .flat_map(|&w| {
            let opts = RollingOptionsFixedWindow {
                window_size: w,
                min_periods: w,
                ..Default::default()
            };
            [
                per_ticker(
                    f("low").rolling_min(opts.clone()).shift(lit(1)),
                    &format!("min{w}"),
                ),
                per_ticker(
                    f("high").rolling_max(opts).shift(lit(1)),
                    &format!("max{w}"),
                ),
            ]
        })
        .collect();
    lf.with_columns(cols)
}
