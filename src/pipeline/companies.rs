//! Builds company snapshots by joining active companies with latest prices and fundamentals.
use polars::prelude::*;

/// Builds the `companies` snapshot with latest fundamental and price-derived fields.
pub fn build_company_snapshot(
    companies: LazyFrame,
    prices: LazyFrame,
    financials: LazyFrame,
) -> LazyFrame {
    let companies = companies.filter(col("isdelisted").eq(lit("N"))).select([
        col("ticker"),
        col("name"),
        col("companysite"),
        col("location"),
        col("currency"),
        col("exchange"),
        col("category"),
        col("sector"),
        col("industry"),
        col("secfilings"),
        col("cusips"),
    ]);

    let latest_financials = financials
        .select([
            col("ticker"),
            col("calendardate"),
            col("reportperiod"),
            col("fiscalperiod"),
            // Income Statement
            col("revenue"),
            col("cor"),
            col("sgna"),
            col("rnd"),
            col("opex"),
            col("intexp"),
            col("taxexp"),
            col("netincdis"),
            col("netinc"),
            col("netincadj"),
            col("netinccmn"),
            col("sharesbas"),
            col("sharesdil"),
            col("dps"),
            // Cash Flow Statement
            col("capex"),
            col("ncfbus"),
            col("ncfinv"),
            col("ncff"),
            col("ncfdebt"),
            col("ncfcommon"),
            col("ncfdiv"),
            col("ncfi"),
            col("ncfo"),
            col("ncfx"),
            col("ncf"),
            col("sbcomp"),
            col("depamor"),
            col("fcfadj"),
            // Balance Sheet
            col("assets"),
            col("cashneq"),
            col("investments"),
            col("investmentsc"),
            col("investmentsnc"),
            col("deferredrev"),
            col("deposits"),
            col("ppnenet"),
            col("inventory"),
            col("taxassets"),
            col("receivables"),
            col("payables"),
            col("intangibles"),
            col("liabilities"),
            col("equity"),
            col("retearn"),
            col("accoci"),
            col("assetsc"),
            col("liabilitiesc"),
            col("liabilitiesnc"),
            col("taxliabilities"),
            col("debt"),
            col("debtc"),
            col("debtnc"),
            col("netdebtusd"),
            // Metrics
            col("fxusd"),
            col("roe"),
            col("roa"),
            col("roic"),
            col("roce"),
            col("grossmargin"),
            col("ebitdausd"),
            col("ebitdamargin"),
            col("bbyield"),
            col("revenue1y"),
            col("revenuecagr3y"),
            col("revenue5y"),
            col("ebitda1y"),
            col("ebitdacagr3y"),
            col("eps1y"),
            col("epscagr3y"),
            col("fcf1y"),
            col("fcfcagr3y"),
            col("fsscore"),
        ])
        .with_column(
            col("calendardate")
                .max()
                .over([col("ticker")])
                .alias("maxcalendardate"),
        )
        .filter(col("calendardate").eq(col("maxcalendardate")))
        .drop(["maxcalendardate"])
        .group_by([col("ticker")])
        .agg([all().last()]);

    let latest_prices = prices
        .with_column(col("date").max().over([lit(1)]).alias("maxdate"))
        .filter(col("date").eq(col("maxdate")))
        .drop(["maxdate"])
        .group_by([col("ticker")])
        .agg([all().last()]);

    companies
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
            (col("sharesbas") * col("close")).alias("marketcap"),
            (col("sharesbas") * col("close") + col("netdebtusd")).alias("ev"),
        ])
}
