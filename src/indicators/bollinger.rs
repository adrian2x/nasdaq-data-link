//! Bollinger Bands indicator.
//!
//! Self-contained.

use polars::prelude::*;
use std::borrow::Cow;

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

/// Add Bollinger band columns to `lf`.
///
/// With the default (20, 2.0) parameters the columns are `bbtop` / `bbbot`;
/// non-default parameters get suffixed names so several configurations can
/// coexist.
pub fn bollinger(lf: LazyFrame, period: usize, multiplier: f64) -> LazyFrame {
    let is_default = period == 20 && (multiplier - 2.0).abs() < 1e-9;
    let (top_col, bot_col) = if is_default {
        (Cow::Borrowed("bbtop"), Cow::Borrowed("bbbot"))
    } else {
        let mult_str = if (multiplier - multiplier.round()).abs() < 1e-9 {
            format!("{}", multiplier as i64)
        } else {
            format!("{multiplier}")
        };
        (
            Cow::Owned(format!("bbtop_{period}_{mult_str}")),
            Cow::Owned(format!("bbbot_{period}_{mult_str}")),
        )
    };
    let opts = rolling_opts(period);

    lf.with_columns([
        f("close")
            .rolling_mean(opts.clone())
            .over([col("ticker")])
            .alias("_bb_sma"),
        f("close")
            .rolling_std(opts)
            .over([col("ticker")])
            .alias("_bb_std"),
    ])
    .with_columns([
        (col("_bb_sma") - lit(multiplier) * col("_bb_std")).alias(bot_col.as_ref()),
        (col("_bb_sma") + lit(multiplier) * col("_bb_std")).alias(top_col.as_ref()),
    ])
    .drop(["_bb_sma", "_bb_std"])
}
