//! Daily technical-indicator computation pipeline.

use polars::prelude::*;

use crate::indicators::{
    adx, atr, bollinger, donchian, highlows, macd, percentile, rate_of_change, rsi, sma, sma_expr,
};

/// Compute all daily technical indicator columns.
/// ...
pub fn technical_indicators_daily(lf: LazyFrame) -> LazyFrame {
    let lf = lf.sort(["ticker", "date"], Default::default());

    // Liquidity.
    let lf = lf.with_columns([
        sma_expr("volume", 20)
            .over([col("ticker")])
            .alias("avgvolume1m"),
        sma_expr("volume", 50)
            .over([col("ticker")])
            .alias("avgvolume3m"),
    ]);

    // Price-range features.
    let lf = highlows(lf, &[252]);
    let lf = donchian(lf, &[20, 55]);

    // Volatility measures
    let lf = atr(lf, 14);
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
    let lf = sma(lf, "close", &[10, 20, 50, 100, 150, 200]);
    let lf = macd(lf, 12, 26, 9);
    let lf = rsi(lf, 14);
    adx(lf, 14)
}
