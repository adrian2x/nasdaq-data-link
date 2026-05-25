//! Price-data preparation.
//!
//! Split-/dividend-adjusts raw OHLCV bars. Self-contained.
use anyhow::Result;

use polars::prelude::*;
use crate::filetools::read_csv;

/// Load raw stock prices and adjust OHLC values using `closeadj / close`.
/// Keeps only the adjusted OHLCV columns used downstream.
pub fn load_prices_adjusted() -> Result<LazyFrame> {
    let lf = read_csv(
        "stocks_eod",
        Some(&[
            ("open", DataType::Float64),
            ("high", DataType::Float64),
            ("low", DataType::Float64),
            ("close", DataType::Float64),
            ("closeadj", DataType::Float64),
            ("volume", DataType::Float64),
        ]),
    )?;
    let adjustment_factor =
        col("closeadj").cast(DataType::Float64) / col("close").cast(DataType::Float64);
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
