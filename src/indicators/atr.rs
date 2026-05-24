//! ATR (average true range) indicator.
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

/// True range expression — depends on a precomputed `_prev_close` column
/// (per-ticker previous close).
fn true_range() -> Expr {
    let hl = f("high") - f("low");
    let hc = (f("high") - col("_prev_close")).abs();
    let lc = (f("low") - col("_prev_close")).abs();
    let max_hl_hc = when(hl.clone().gt_eq(hc.clone())).then(hl).otherwise(hc);
    when(max_hl_hc.clone().gt_eq(lc.clone()))
        .then(max_hl_hc)
        .otherwise(lc)
}

/// Add an ATR column to `lf`, named `atr{period}` (e.g. `atr14`).
pub fn atr(lf: LazyFrame, period: usize) -> LazyFrame {
    let atr_col = format!("atr{period}");

    lf.with_columns([f("close")
        .shift(lit(1))
        .over([col("ticker")])
        .alias("_prev_close")])
        .with_columns([true_range().alias("_tr")])
        .with_columns([wilder_smooth(col("_tr"), period)
            .over([col("ticker")])
            .alias(atr_col.as_str())])
        .drop(["_prev_close", "_tr"])
}
