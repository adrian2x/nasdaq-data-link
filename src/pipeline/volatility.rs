//! Volatility pipeline entry point.
//!
//! Applies the per-ticker volatility estimators from `crate::indicators` to a
//! multi-ticker OHLCV price frame. The estimators themselves live in their own
//! files (`crate::indicators::yang_zhang`, `crate::indicators::ewma_volatility`);
//! this module only wires them together and exposes the parameter surface.
//!
//! Expects a `LazyFrame` with columns
//! `ticker, date, open, high, low, close` (a `volume` column is passed
//! through untouched if present). Rows MUST be sorted by `date` within each
//! ticker before being passed in — the estimators do not sort, since `over`
//! partitions but does not order.

use polars::prelude::*;

use crate::indicators::ewma_volatility::ewma_vol;
use crate::indicators::yang_zhang::yang_zhang;

/// Parameters for the volatility calculations. `Default` gives the standard
/// daily-data values: a 30-day Yang-Zhang window, RiskMetrics lambda 0.94,
/// and 252-day annualization.
#[derive(Debug, Clone, Copy)]
pub struct VolParams {
    /// Rolling lookback for Yang-Zhang, in trading days.
    pub yz_window: usize,
    /// EWMA decay factor (RiskMetrics daily standard is 0.94).
    pub ewma_lambda: f64,
    /// Periods per year for annualizing both estimators. Use 1.0 to keep
    /// the outputs as daily figures.
    pub annualize_factor: f64,
}

impl Default for VolParams {
    fn default() -> Self {
        Self {
            yz_window: 30,
            ewma_lambda: 0.94,
            annualize_factor: 252.0,
        }
    }
}

/// Apply both volatility estimators to a multi-ticker OHLCV frame.
///
/// Returns the input frame with four added columns:
/// `yz_variance`, `yz_vol`, `ewma_variance`, `ewma_vol`.
///
/// The caller is responsible for passing a frame sorted by `date` within
/// each ticker.
pub fn apply_volatility(lf: LazyFrame, params: VolParams) -> PolarsResult<LazyFrame> {
    let lf = yang_zhang(lf, params.yz_window, params.annualize_factor)?;
    let lf = ewma_vol(lf, params.ewma_lambda, params.annualize_factor)?;
    Ok(lf)
}