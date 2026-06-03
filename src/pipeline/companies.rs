//! Builds company snapshots by joining active companies with latest prices and fundamentals.
//!
//! This is where ticker-level ranks become a tradable "current snapshot".
//! Fundamentals do not include industry, so industry and composite ranks are
//! added after joining the latest company metadata, latest financial report,
//! and latest price row.
//!
//! Research references for the snapshot-only ranks:
//! - IBD Composite Rating publicly combines EPS, RS, SMR, Accumulation/
//!   Distribution, industry group rank, and percent off the 52-week high, with
//!   extra weight on EPS and RS.
//!   https://origin.williamoneilindia.com/proprietary-ratings-and-rankings/
//! - MarketSmith/O'Neil materials emphasize EPS, RS, A/D, and leading
//!   industries as core historical-winner traits.
//!   https://www.marketsmith.hk/overview/details-tab/
use polars::prelude::*;

use super::rankings::{
    add_canslim_snapshot_ranks, add_composite_inputs, add_composite_rank,
    add_fundamental_momentum_snapshot_rank, add_momentum_rank, add_overall_score,
    add_rafi_snapshot_ranks, rafi_feature_columns,
};

/// Builds the `companies` snapshot with latest fundamental and price-derived fields.
///
/// Output semantics:
/// - one row per active, non-delisted ticker;
/// - latest fundamental row by `calendardate`;
/// - latest quarterly feature row by `calendardate`;
/// - latest price row by `date`;
/// - market capitalization = basic shares * latest close;
/// - EV, or Enterprise Value = market capitalization + net debt;
/// - industryrank, momrank, comprank, and rankscore are calculated
///   only after industry metadata and latest price/fundamental ranks are
///   visible together.
pub fn build_company_snapshot(
    companies: LazyFrame,
    prices: LazyFrame,
    financials: LazyFrame,
    financials_quarter: LazyFrame,
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

    let mut latest_financial_columns = vec![
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
        col("fcf"),
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
        col("de"),
        col("debtc"),
        col("debtnc"),
        col("netdebt"),
        // Metrics
        col("fxusd"),
        col("roe"),
        col("roa"),
        col("roic"),
        col("roce"),
        col("pretaxmargin"),
        col("grossmargin"),
        col("netmargin"),
        col("ebitda"),
        col("ebitdamargin"),
        col("revenueyoy"),
        col("revenuecagr3y"),
        col("revenue5y"),
        col("ebitda1y"),
        col("ebitdacagr3y"),
        col("epsyoy"),
        col("epscagr3y"),
        col("fcfyoy"),
        col("fcfcagr3y"),
        col("fsscore"),
        col("dpscagr5y"),
        col("bbyield"),
    ];
    latest_financial_columns.extend(rafi_feature_columns().iter().map(|name| col(*name)));

    let latest_financials = financials
        .select(latest_financial_columns)
        // `adjust_fundamentals` sorts by ticker/date before materialization.
        // Taking the last row per ticker avoids a full max-date window scan.
        .group_by_stable([col("ticker")])
        .agg([all().last()]);

    let latest_quarterly_rankings = financials_quarter
        .select([
            col("ticker"),
            col("revenueqtryoy"),
            col("revenueqtraccel"),
            col("netmargin").alias("netmarginqtr"),
            col("epsadj").alias("epsqtr"),
            col("epsqtryoy"),
            col("epsqtrchg"),
            col("epsqtr1qagoyoy"),
            col("epsqtr2qagoyoy"),
            col("epsqtraccel"),
            col("grossmarginqtrexp"),
            col("ebitdamarginqtrexp"),
            col("roicqtrexp"),
        ])
        .group_by_stable([col("ticker")])
        .agg([all().last()]);

    let latest_prices = prices
        // `write_stocks` returns full price history; the company snapshot owns
        // reducing that history to one current row per ticker.
        .group_by_stable([col("ticker")])
        .agg([all().last()]);

    let snapshot = companies
        .join(
            latest_prices,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        )
        .join(
            latest_financials,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        )
        .join(
            latest_quarterly_rankings,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        )
        .with_columns([
            (col("sharesbas") * col("close")).alias("marketcap"),
            (col("sharesbas") * col("close") + col("netdebt")).alias("ev"),
        ]);

    let snapshot = add_rafi_snapshot_ranks(snapshot);
    let snapshot = add_canslim_snapshot_ranks(snapshot);
    let snapshot = add_fundamental_momentum_snapshot_rank(snapshot).cache();

    let snapshot = add_composite_inputs(snapshot);
    let snapshot = add_momentum_rank(snapshot);
    let snapshot = add_composite_rank(snapshot);
    add_overall_score(snapshot)
}
