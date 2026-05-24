//! Price-data preparation.
//!
//! Split-/dividend-adjusts raw OHLCV bars. Self-contained.

use polars::prelude::*;

/// Adjust raw OHLC values using the `closeadj / close` factor and keep only
/// the adjusted OHLCV columns used downstream.
pub fn adjust_prices(lf: LazyFrame) -> LazyFrame {
    let adjustment_factor =
        col("closeadj").cast(DataType::Float64) / col("close").cast(DataType::Float64);

    lf.select([
        col("ticker"),
        col("date").cast(DataType::Date),
        (col("open").cast(DataType::Float64) * adjustment_factor.clone()).alias("open"),
        (col("high").cast(DataType::Float64) * adjustment_factor.clone()).alias("high"),
        (col("low").cast(DataType::Float64) * adjustment_factor).alias("low"),
        col("closeadj").cast(DataType::Float64).alias("close"),
        col("volume").cast(DataType::Float64),
    ])
}
