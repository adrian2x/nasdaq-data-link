//! Bollinger Bands indicator.
//!
//! Self-contained.

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

/// Upper Bollinger band: `SMA + multiplier * rolling stdev`.
pub fn bbtop_expr(period: usize, multiplier: f64) -> Expr {
    let sma = f("close").rolling_mean(rolling_opts(period));
    let stdev = f("close").rolling_std(rolling_opts(period));
    sma + lit(multiplier) * stdev
}

/// Lower Bollinger band: `SMA - multiplier * rolling stdev`.
pub fn bbbot_expr(period: usize, multiplier: f64) -> Expr {
    let sma = f("close").rolling_mean(rolling_opts(period));
    let stdev = f("close").rolling_std(rolling_opts(period));
    sma - lit(multiplier) * stdev
}

/// Add Bollinger band columns to `lf`.
///
/// With the default (20, 2.0) parameters the columns are `bbtop` / `bbbot`;
/// non-default parameters get suffixed names so several configurations can
/// coexist.
pub fn bollinger(lf: LazyFrame, period: usize, multiplier: f64) -> LazyFrame {
    let is_default = period == 20 && (multiplier - 2.0).abs() < 1e-9;
    let (top_col, bot_col) = if is_default {
        ("bbtop".to_string(), "bbbot".to_string())
    } else {
        let mult_str = if (multiplier - multiplier.round()).abs() < 1e-9 {
            format!("{}", multiplier as i64)
        } else {
            format!("{multiplier}")
        };
        (
            format!("bbtop_{period}_{mult_str}"),
            format!("bbbot_{period}_{mult_str}"),
        )
    };

    lf.with_columns([
        bbbot_expr(period, multiplier)
            .over([col("ticker")])
            .alias(bot_col.as_str()),
        bbtop_expr(period, multiplier)
            .over([col("ticker")])
            .alias(top_col.as_str()),
    ])
}
