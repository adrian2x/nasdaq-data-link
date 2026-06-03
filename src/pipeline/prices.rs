//! Price-data preparation.
//!
//! Split-/dividend-adjusts raw OHLCV bars. Self-contained.
use anyhow::Result;

use crate::filetools::read_csv;
use polars::prelude::*;

/// Load raw stock prices and adjust OHLC values using `closeadj / close`.
/// Keeps only the adjusted OHLCV columns used downstream.
pub fn load_prices_adjusted() -> Result<LazyFrame> {
    let lf = read_csv(
        "stocks_prices",
        Some(&[
            ("open", DataType::Float64),
            ("high", DataType::Float64),
            ("low", DataType::Float64),
            ("close", DataType::Float64),
            ("closeadj", DataType::Float64),
            ("volume", DataType::Float64),
        ]),
    )?;
    // Adjusts the historical prices by splits and corporate events
    let adjustment_factor =
        col("closeadj").cast(DataType::Float64) / col("close").cast(DataType::Float64);
    // Returns the adjusted historical prices with typed values
    Ok(lf.select([
        col("ticker"),
        col("date").cast(DataType::Date),
        (col("open").cast(DataType::Float64) * adjustment_factor.clone()).alias("open"),
        (col("high").cast(DataType::Float64) * adjustment_factor.clone()).alias("high"),
        (col("low").cast(DataType::Float64) * adjustment_factor).alias("low"),
        col("closeadj").cast(DataType::Float64).alias("close"),
        col("volume").cast(DataType::Float64),
    ]))
}

/// Resamples sorted OHLCV bars into larger time buckets.
///
/// The returned bar is labeled by the first trading date in the bucket because
/// the bar opens on that session. OHLCV aggregation follows market convention:
/// first open, maximum high, minimum low, last close, summed volume.
pub fn resample_ohlcv(lf: LazyFrame, interval: &str) -> LazyFrame {
    lf.sort(["ticker", "date"], Default::default())
        .with_column(col("date").dt().truncate(lit(interval)).alias("__period"))
        .group_by_stable([col("ticker"), col("__period")])
        .agg([
            col("date").first().alias("date"),
            col("open").first().alias("open"),
            col("high").max().alias("high"),
            col("low").min().alias("low"),
            col("close").last().alias("close"),
            col("volume").sum().alias("volume"),
        ])
        .drop(["__period"])
        .sort(["ticker", "date"], Default::default())
}
