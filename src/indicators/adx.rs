//! ADX (average directional index) indicator.
//!
//! Self-contained. Carries its own copy of Wilder smoothing and of the
//! true-range expression — the duplication across `rsi`/`atr`/`adx` is
//! deliberate, so each indicator module stays fully isolated.

use polars::prelude::*;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Wilder's smoothing: an EWM with `alpha = 1 / period`.
fn wilder_smooth(source: Expr, period: usize) -> Expr {
    source.ewm_mean(EWMOptions {
        alpha: 1.0 / period as f64,
        min_periods: period,
        ..Default::default()
    })
}

/// Add an ADX column to `lf`, named `adx{period}` (e.g. `adx14`).
pub fn adx(lf: LazyFrame, period: usize) -> LazyFrame {
    let adx_col = format!("adx{period}");

    lf.with_columns([
        (f("high") - f("high").shift(lit(1)))
            .over([col("ticker")])
            .alias("_up_move"),
        (f("low").shift(lit(1)) - f("low"))
            .over([col("ticker")])
            .alias("_down_move"),
        f("close")
            .shift(lit(1))
            .over([col("ticker")])
            .alias("_prev_close"),
    ])
    .with_columns([
        {
            let hl = f("high") - f("low");
            let hc = (f("high") - col("_prev_close")).abs();
            let lc = (f("low") - col("_prev_close")).abs();
            let max_hl_hc = when(hl.clone().gt_eq(hc.clone())).then(hl).otherwise(hc);
            when(max_hl_hc.clone().gt_eq(lc.clone()))
                .then(max_hl_hc)
                .otherwise(lc)
                .alias("_tr")
        },
        when(
            col("_up_move")
                .gt(col("_down_move"))
                .and(col("_up_move").gt(lit(0.0))),
        )
        .then(col("_up_move"))
        .otherwise(lit(0.0))
        .alias("_plus_dm"),
        when(
            col("_down_move")
                .gt(col("_up_move"))
                .and(col("_down_move").gt(lit(0.0))),
        )
        .then(col("_down_move"))
        .otherwise(lit(0.0))
        .alias("_minus_dm"),
    ])
    .with_columns([
        wilder_smooth(col("_tr"), period)
            .over([col("ticker")])
            .alias("_smooth_tr"),
        wilder_smooth(col("_plus_dm"), period)
            .over([col("ticker")])
            .alias("_smooth_plus_dm"),
        wilder_smooth(col("_minus_dm"), period)
            .over([col("ticker")])
            .alias("_smooth_minus_dm"),
    ])
    .with_columns([
        (lit(100.0) * col("_smooth_plus_dm") / col("_smooth_tr")).alias("_plus_di"),
        (lit(100.0) * col("_smooth_minus_dm") / col("_smooth_tr")).alias("_minus_di"),
    ])
    .with_columns([wilder_smooth(
        lit(100.0) * (col("_plus_di") - col("_minus_di")).abs()
            / (col("_plus_di") + col("_minus_di")),
        period,
    )
    .over([col("ticker")])
    .alias(adx_col.as_str())])
    .drop([
        "_up_move",
        "_down_move",
        "_prev_close",
        "_tr",
        "_plus_dm",
        "_minus_dm",
        "_smooth_tr",
        "_smooth_plus_dm",
        "_smooth_minus_dm",
        "_plus_di",
        "_minus_di",
    ])
}
