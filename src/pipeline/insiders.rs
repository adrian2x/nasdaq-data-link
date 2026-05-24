//! Insider-transaction preparation.
//!
//! Filters and aggregates Form 4 insider transactions over a trailing
//! ~6-month window. Self-contained.

use polars::prelude::*;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Aggregate insider transactions over the last ~6 months.
pub fn update_insiders(lf: LazyFrame) -> LazyFrame {
    let six_months_ago = chrono::Utc::now().date_naive() - chrono::Duration::weeks(26);

    lf.with_columns([
        col("formtype").cast(DataType::String),
        col("transactiondate").cast(DataType::Date).alias("date"),
        col("transactionshares")
            .cast(DataType::Float64)
            .abs()
            .alias("_transactionshares_abs"),
    ])
    .filter(
        col("date")
            .gt_eq(lit(six_months_ago))
            .and(f("transactionvalue").neq(lit(0.0))),
    )
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
        col("transactionvalue").sum().alias("transactionvalue"),
        col("_transactionshares_abs")
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
    .with_columns([when(col("isofficer").eq(lit("Y")))
        .then(col("officertitle").fill_null(lit("")))
        .when(col("isdirector").eq(lit("Y")))
        .then(lit("Director"))
        .when(col("istenpercentowner").eq(lit("Y")))
        .then(lit("10% Owner"))
        .otherwise(col("officertitle").fill_null(lit("")))
        .alias("officertitle")])
}
