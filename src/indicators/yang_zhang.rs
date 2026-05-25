use polars::prelude::*;

/// Rolling Yang-Zhang volatility estimator per ticker.
///
/// Expects columns: `ticker`, `open`, `high`, `low`, `close`.
/// Rows must be sorted by date within each ticker.
///
/// `window` is the rolling lookback in trading days. `annualize_factor` is
/// periods per year (252.0), or 1.0 for a daily figure.
///
/// Returns the frame with added columns `yz_variance` and `yz_vol`.
/// The first `window` rows per ticker are null.
pub fn yang_zhang(lf: LazyFrame, window: usize, annualize_factor: f64) -> PolarsResult<LazyFrame> {
    let n = window as f64;
    let k = 0.34 / (1.0 + (n + 1.0) / (n - 1.0));

    // ddof defaults to 1 for rolling_var (the sample denominator YZ wants);
    // leaving fn_params: None keeps that default.
    let rolling_opts = RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        weights: None,
        center: false,
        fn_params: None,
    };

    let prev_close = col("close").shift(lit(1)).over([col("ticker")]);

    let overnight = (col("open") / prev_close).log(std::f64::consts::E);
    let open_to_close = (col("close") / col("open")).log(std::f64::consts::E);

    let ln_ho = (col("high") / col("open")).log(std::f64::consts::E);
    let ln_hc = (col("high") / col("close")).log(std::f64::consts::E);
    let ln_lo = (col("low") / col("open")).log(std::f64::consts::E);
    let ln_lc = (col("low") / col("close")).log(std::f64::consts::E);

    // Rogers-Satchell per-day term (a mean over the window, not a variance).
    let rs_term = ln_hc * ln_ho + ln_lc * ln_lo;

    let lf = lf
        .with_columns([
            overnight.alias("__yz_on"),
            open_to_close.alias("__yz_oc"),
            rs_term.alias("__yz_rs"),
        ])
        .with_columns([
            col("__yz_on")
                .rolling_var(rolling_opts.clone())
                .over([col("ticker")])
                .alias("__yz_var_on"),
            col("__yz_oc")
                .rolling_var(rolling_opts.clone())
                .over([col("ticker")])
                .alias("__yz_var_oc"),
            col("__yz_rs")
                .rolling_mean(rolling_opts.clone())
                .over([col("ticker")])
                .alias("__yz_var_rs"),
        ])
        .with_columns([((col("__yz_var_on")
            + lit(k) * col("__yz_var_oc")
            + lit(1.0 - k) * col("__yz_var_rs"))
            * lit(annualize_factor))
        .alias("yz_variance")])
        .with_columns([col("yz_variance").sqrt().alias("yz_vol")])
        .drop([
            "__yz_on",
            "__yz_oc",
            "__yz_rs",
            "__yz_var_on",
            "__yz_var_oc",
            "__yz_var_rs",
        ]);

    Ok(lf)
}
