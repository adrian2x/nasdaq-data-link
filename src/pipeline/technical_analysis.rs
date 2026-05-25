//! Daily technical-indicator computation pipeline.

use polars::prelude::*;

use crate::indicators::{adr, atr, bollinger, highlows, percentile, rate_of_change, sma, sma_expr};

/// Compute all daily technical indicator columns.
/// ...
pub fn technical_indicators_daily(lf: LazyFrame) -> LazyFrame {
    let lf = lf.sort(["ticker", "date"], Default::default());

    // Liquidity.
    let lf = lf.with_columns([sma_expr("volume", 60)
        .over([col("ticker")])
        .alias("avgvolume60")]);

    // Price-range features.
    let lf = highlows(lf, &[20, 55, 252]).with_columns([
        col("close")
            .gt(col("max252").shift(lit(1)))
            .alias("high252"),
        col("close").lt(col("min252").shift(lit(1))).alias("low252"),
    ]);

    // Volatility measures
    let lf = adr(lf, 20);
    let lf = atr(lf, 20);
    let lf = bollinger(lf, 20, 2.0);

    // Time-series and cross-sectional momentum
    let lf = rate_of_change(lf, "close", &[1, 5, 21, 3 * 21, 6 * 21, 9 * 21, 12 * 21]);
    // Composite relative-strength score ...
    let lf = lf.with_columns([(lit(0.4) * col("pct63")
        + lit(0.2) * col("pct126")
        + lit(0.2) * col("pct189")
        + lit(0.2) * col("pct252"))
    .alias("rs1y")
    .cast(DataType::Float32)]);
    let lf = percentile(lf, "rs1y", "date", true, "rs1y");

    // Trend / oscillator indicators.
    sma(lf, "close", &[10, 20, 50, 100, 150, 200])
}
