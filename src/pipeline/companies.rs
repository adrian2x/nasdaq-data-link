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
        .select([
            // Metadata
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
            col("consolinc"),
            col("netincnci"),
            col("netincadj"),
            col("prefdivis"),
            col("netinccmn"),
            col("shares"),
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
            col("ebitdamargin"),
            col("shyield"),
            col("revenue1y"),
            col("revenue3y"),
            col("revenue5y"),
            col("revenuecagr3y"),
            col("revenuecagr5y"),
            col("ebitda1y"),
            col("ebitda3y"),
            col("ebitda5y"),
            col("ebitdacagr3y"),
            col("ebitdacagr5y"),
            col("eps1y"),
            col("eps3y"),
            col("eps5y"),
            col("epscagr3y"),
            col("epscagr5y"),
            col("fcf1y"),
            col("fcf3y"),
            col("fcf5y"),
            col("fcfcagr3y"),
            col("fcfcagr5y"),
            col("fsscore"),
        ])
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
