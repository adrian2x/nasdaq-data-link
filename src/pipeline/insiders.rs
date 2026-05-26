//! Insider-transaction preparation.
//!
//! Filters and aggregates Form 4 insider transactions over a trailing
//! ~6-month window. Self-contained.

use polars::prelude::*;

/// Aggregate insider transactions over the last ~6 months.
pub fn update_insiders(lf: LazyFrame) -> LazyFrame {
    let six_months_ago = chrono::Utc::now().date_naive() - chrono::Duration::weeks(26);

    lf.with_columns([
        // formtype sometimes starts with a digit ("4 - Form S"), which can
        // confuse the CSV parser's type inference — force String.
        col("formtype").cast(DataType::String),
        col("transactiondate").cast(DataType::Date).alias("date"),
        // Y/N flag columns -> Boolean, done once up front so both the
        // group_by below and the officertitle logic operate on real bools.
        col("isofficer").eq(lit("Y")).alias("isofficer"),
        col("isdirector").eq(lit("Y")).alias("isdirector"),
        col("istenpercentowner")
            .eq(lit("Y"))
            .alias("istenpercentowner"),
    ])
    .filter(
        col("date").gt_eq(lit(six_months_ago)).and(
            col("transactionvalue")
                .cast(DataType::Float64)
                .neq(lit(0.0)),
        ),
    )
    // Aggregate multiple transactions on the same date for the same ownername.
    .group_by([
        col("ticker"),
        col("date"),
        col("issuername"),
        col("ownername"),
        col("transactioncode"),
        col("securityadcode"),
        col("securitytitle"),
        col("officertitle"),
        col("isofficer"),
        col("isdirector"),
        col("istenpercentowner"),
    ])
    .agg([
        // transactionvalue is negative for sales, positive for buys.
        col("transactionvalue").sum().alias("transactionvalue"),
        // abs() applied inside the agg — no pre-group scratch column needed.
        col("transactionshares")
            .cast(DataType::Float64)
            .abs()
            .sum()
            .alias("transactionshares"),
        col("transactionpricepershare")
            .mean()
            .alias("transactionpricepershare"),
    ])
    .sort(
        ["date", "transactionvalue"],
        SortMultipleOptions::default().with_order_descending_multi([true, true]),
    )
    // Normalize officertitle so empty values still carry a useful role label.
    .with_columns([when(col("isofficer"))
        .then(col("officertitle").fill_null(lit("")))
        .when(col("isdirector"))
        .then(lit("Director"))
        .when(col("istenpercentowner"))
        .then(lit("10% Owner"))
        .otherwise(col("officertitle").fill_null(lit("")))
        .alias("officertitle")])
}
