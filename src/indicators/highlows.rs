//! Rolling intraday high/low extremes.
//!
//! Produces `highN` / `lowN` columns — the rolling maximum of intraday
//! `high` and minimum of intraday `low` — for one or more windows `N`.
//!
//! The canonical window is 250 (one trading year): proximity to the 52-week
//! high is a studied outperformance signal (George & Hwang 2004, attributed
//! to anchoring bias, robust internationally and non-reversing). Any derived
//! ratio such as price-to-high is intentionally NOT computed here — it
//! depends on which window is treated as "the" 52-week one, a choice that
//! belongs to the caller.
//!
//! This is deliberately NOT a Donchian breakout channel and is kept in its
//! own module to avoid conflating the two: the window here is current-bar
//! inclusive (no `.shift(1)`), because proximity asks "is the stock at its
//! high *today*", which needs today's bar — whereas a breakout channel must
//! exclude the current bar so it cannot break its own level.
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

/// Add rolling intraday high/low columns to `lf`, two per window:
/// `high250`/`low250`, `high120`/`low120`, ...
///
/// The upper column is the rolling max of intraday `high`, the lower the
/// rolling min of intraday `low`. The window is current-bar inclusive (no
/// shift) — these are range/proximity features, not breakout signal levels.
///
/// Requires the full window: a column is null until the ticker has `w` bars
/// of history, so a recent listing yields null rather than a partial-window
/// extreme mislabeled as a `w`-bar one.
pub fn highlows(lf: LazyFrame, windows: &[usize]) -> LazyFrame {
    let cols: Vec<Expr> = windows
        .iter()
        .flat_map(|&w| {
            let opts = RollingOptionsFixedWindow {
                window_size: w,
                min_periods: w,
                ..Default::default()
            };
            [
                per_ticker(f("low").rolling_min(opts.clone()), &format!("low{w}")),
                per_ticker(f("high").rolling_max(opts), &format!("high{w}")),
            ]
        })
        .collect();
    lf.with_columns(cols)
}
