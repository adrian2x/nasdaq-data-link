//! Daily technical-indicator computation pipeline.

use polars::prelude::*;

use crate::indicators::{
    adr, atr, bollinger, highlows, macd, percentile, rate_of_change, rsi, sma, sma_expr,
};

/// Computes technical indicator columns for each individual ticker
pub fn technical_indicators_daily(lf: LazyFrame) -> LazyFrame {
    let lf = lf.sort(["ticker", "date"], Default::default());

    // Average volume of 3 months is often cited as liquidity filter
    // Institutions buy shares in "blocks" of 100k units or more
    let lf = lf.with_columns([sma_expr("volume", 60)
        .over([col("ticker")])
        .alias("avgvolume60")]);

    // Computes the price range (high/low) equivalent to approx 1m, 3m, and 1y
    let lf = highlows(lf, &[20, 55, 252]).with_columns([
        // Adds a signal (true/false) value if the stock made new 52-week highs or lows
        col("close")
            .gt(col("max252").shift(lit(1)))
            .alias("high252"),
        col("close").lt(col("min252").shift(lit(1))).alias("low252"),
    ]);

    // Volatility measures
    let lf = adr(lf, 20);
    let lf = atr(lf, 20);
    let lf = bollinger(lf, 20, 2.0);

    // Rate of change calculates Time-series momentum
    // These are percentage values of price changes over 1d, 5d, 1m, 1q, 2q, 3q, 1y periods
    let lf = rate_of_change(lf, "close", &[1, 5, 21, 3 * 21, 6 * 21, 9 * 21, 12 * 21]);
    // Composite relative-strength score value
    let lf = lf.with_columns([(lit(0.4) * col("pct63")
        + lit(0.2) * col("pct126")
        + lit(0.2) * col("pct189")
        + lit(0.2) * col("pct252"))
    .alias("rs1y")
    .cast(DataType::Float32)]);

    // Calculates the cross-sectional (relative) momentum on the same date as a percentile value
    // Expressed as a percentile (0-100) where higher is better
    // Note - this replaces the previous rs1y value composite
    let lf = percentile(lf, "rs1y", "date", true, "rs1y");

    // Trend / oscillator indicators.
    let lf = sma(lf, "close", &[10, 20, 50, 100, 150, 200]);
    let lf = macd(lf, 12, 26, 9);
    rsi(lf, 14)
}
