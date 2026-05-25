use polars::prelude::*;

/// Compute the EWMA (RiskMetrics-style) volatility estimator per ticker.
///
/// Expects columns: `ticker`, `close` (other columns are passed through).
/// Rows must be sorted by date within each ticker.
///
/// `lambda` is the RiskMetrics decay (0.94 is the standard daily value).
/// `annualize_factor` is periods per year for the output (252.0), or 1.0 for
/// a daily figure.
///
/// Returns the frame with added columns `ewma_variance` and `ewma_vol`.
/// The EWMA is seeded from the first available return per ticker, so early
/// values are unstable until the window effectively fills (~60+ rows at 0.94).
pub fn ewma_vol(lf: LazyFrame, lambda: f64, annualize_factor: f64) -> PolarsResult<LazyFrame> {
    // EWM in terms of alpha: variance recursion is
    //   v_t = lambda * v_{t-1} + (1 - lambda) * r_t^2
    // which is an EWM mean of squared returns with alpha = 1 - lambda.
    let alpha = 1.0 - lambda;

    let ewm_opts = EWMOptions {
        alpha,
        adjust: false, // recursive form, not the bias-corrected finite-sample form
        bias: false,
        min_periods: 1,
        ignore_nulls: true,
    };

    // Log return, partitioned per ticker so the first row of each is null.
    let log_ret = (col("close") / col("close").shift(lit(1)))
        .log(std::f64::consts::E)
        .over([col("ticker")]);

    let lf = lf
        .with_columns([log_ret.alias("__ewma_ret")])
        .with_columns([
            // EWM mean of squared returns = the EWMA variance estimate.
            col("__ewma_ret")
                .pow(2)
                .ewm_mean(ewm_opts)
                .over([col("ticker")])
                .alias("ewma_variance_raw"),
        ])
        .with_columns([(col("ewma_variance_raw") * lit(annualize_factor)).alias("ewma_variance")])
        .with_columns([col("ewma_variance").sqrt().alias("ewma_vol")])
        .drop(["__ewma_ret", "ewma_variance_raw"]);

    Ok(lf)
}
