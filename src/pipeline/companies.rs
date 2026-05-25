use polars::prelude::*;

pub fn build_company_snapshot(
    companies: LazyFrame,
    prices: LazyFrame,
    financials: LazyFrame,
) -> LazyFrame {
    let active = companies.filter(col("isdelisted").eq(lit("N"))).drop([
        "table",
        "permaticker",
        "isdelisted",
        "cusips",
        "sicsector",
        "sicindustry",
        "figi",
        "famaindustry",
        "scalemarketcap",
        "scalerevenue",
        "relatedtickers",
        "lastupdated",
        "firstadded",
        "firstpricedate",
        "lastpricedate",
        "firstquarter",
        "lastquarter",
    ]);

    let latest_financials = financials
        .with_column(
            col("calendardate")
                .max()
                .over([col("ticker")])
                .alias("__max_calendardate"),
        )
        .filter(col("calendardate").eq(col("__max_calendardate")))
        .drop(["__max_calendardate"])
        .group_by([col("ticker")])
        .agg([all().last()]);

    let latest_prices = prices
        .with_column(col("date").max().over([lit(1)]).alias("__max_date"))
        .filter(col("date").eq(col("__max_date")))
        .drop(["__max_date"])
        .group_by([col("ticker")])
        .agg([all().last()]);

    active
        .join(
            latest_financials,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        )
        .join(
            latest_prices,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        )
        .with_columns([
            (col("shares") * col("close")).alias("marketcap"),
            (col("shares") * col("close") + col("netdebtusd")).alias("ev"),
        ])
}