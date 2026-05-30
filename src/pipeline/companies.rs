//! Builds company snapshots by joining active companies with latest prices and fundamentals.
//!
//! This is where ticker-level ranks become a tradable "current snapshot".
//! Fundamentals do not include industry, so industry and composite ranks are
//! calculated here after joining the latest company metadata, latest financial
//! report, and latest price row.
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

const MIN_SCORE_WEIGHT_COVERAGE: f64 = 0.5;

/// Cross-sectional percentile for the already-latest company snapshot.
///
/// This is simpler than the fundamentals `with_period_ranks` helper because
/// this frame has already been reduced to one row per active ticker. Formula:
/// `(rank - 1) / (N - 1) * 100`, rounded to an integer. `ascending=true` means
/// high raw values get high scores; `ascending=false` means low raw values get
/// high scores. Nulls are excluded from `N`.
fn percentile_expr(source: &'static str, ascending: bool, alias: &'static str) -> Expr {
    let opts = RankOptions {
        method: RankMethod::Average,
        descending: !ascending,
    };
    let n = col(source).count().over([lit(1)]);
    let rank = col(source).rank(opts, None).over([lit(1)]);

    when(n.clone().gt(lit(1)))
        .then(((rank - lit(1.0)) / (n.clone() - lit(1.0))) * lit(100.0))
        .otherwise(when(n.eq(lit(1))).then(lit(100.0)).otherwise(lit(NULL)))
        .round(0)
        .cast(DataType::Int32)
        .alias(alias)
}

/// Weighted blend for latest-snapshot ranks.
///
/// All inputs are 0..100 ranks where higher is better. The denominator uses
/// only non-null components, so a missing A/D or industry rank does not force
/// the composite to zero. At least 50% of the intended weight must be present;
/// thinner records return null instead of receiving a sparse-data score.
fn weighted_score(weighted: &[(&str, f64)], alias: &'static str) -> Expr {
    let mut numerator = lit(0.0);
    let mut denominator = lit(0.0);
    let min_weight =
        weighted.iter().map(|(_, weight)| *weight).sum::<f64>() * MIN_SCORE_WEIGHT_COVERAGE;
    for (name, weight) in weighted {
        numerator = numerator + col(*name).fill_null(lit(0.0)) * lit(*weight);
        denominator = denominator + col(*name).is_not_null().cast(DataType::Float64) * lit(*weight);
    }

    when(denominator.clone().gt_eq(lit(min_weight)))
        .then((numerator / denominator).round(0))
        .otherwise(lit(NULL))
        .cast(DataType::Int32)
        .alias(alias)
}

/// Builds the `companies` snapshot with latest fundamental and price-derived fields.
///
/// Output semantics:
/// - one row per active, non-delisted ticker;
/// - latest fundamental row by `calendardate`;
/// - latest price row by `date`;
/// - market capitalization = basic shares * latest close;
/// - EV, or Enterprise Value = market capitalization + net debt;
/// - industryrank, momentumrank, and compositerank are calculated only after
///   industry metadata and latest price/fundamental ranks are visible together.
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
            col("ebitdaps"),
            col("ebitda1y"),
            col("ebitdagrowth1y"),
            col("revenueaccel"),
            col("ebitdaaccel"),
            col("grossmarginexpansion"),
            col("ebitdamarginexpansion"),
            col("roicexpansion"),
            col("ebitdacagr3y"),
            col("eps1y"),
            col("epscagr3y"),
            col("fcf1y"),
            col("fcfcagr3y"),
            col("fsscore"),
            col("qualrank"),
            col("valuerank"),
            col("epsrank"),
            col("smrrank"),
            col("fmomrank"),
            // Fundamental composite component ranks. These are kept public so
            // downstream consumers can reconstitute or reweight the composites.
            col("roicrank"),
            col("rocerank"),
            col("roerank"),
            col("grossmarginrank"),
            col("ebitdamarginrank"),
            col("ebitmarginrank"),
            col("revenue1yrank"),
            col("revenuecagr3yrank"),
            col("revenuecagr5yrank"),
            col("salesassetscagr3yrank"),
            col("grossprofitassetscagr3yrank"),
            col("opprofitassetscagr3yrank"),
            col("rndintensityrank"),
            col("rndcagr3yrank"),
            col("cfcrank"),
            col("fcfassetsrank"),
            col("cashaccrualsrank"),
            col("icrrank"),
            col("derank"),
            col("fsscorerank"),
            col("cashassetsrank"),
            col("intangiblesassetsrank"),
            col("evebitdarank"),
            col("evsalesrank"),
            col("ebitdapegrank"),
            col("ebitda1yrank"),
            col("ebitdaaccelrank"),
            col("ebitdacagr3yrank"),
            col("ebitdacagr5yrank"),
            col("revenueaccelrank"),
            col("grossmarginexpansionrank"),
            col("ebitdamarginexpansionrank"),
            col("roicexpansionrank"),
        ])
        // `adjust_fundamentals` sorts by ticker/date before materialization.
        // Taking the last row per ticker avoids a full max-date window scan.
        .group_by_stable([col("ticker")])
        .agg([all().last()]);

    let latest_prices = prices
        // `write_stocks` passes a latest-row snapshot, but this keeps the
        // function correct for sorted full-history callers without a global
        // max-date window over the entire price history.
        .group_by_stable([col("ticker")])
        .agg([all().last()]);

    let snapshot = companies
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
        .cache();

    // Industry momentum rank.
    //
    // We approximate IBD's Industry Group Relative Strength by ranking each
    // industry on the average 6-month price return (`pct126`) of its current
    // constituents. O'Neil/MarketSmith ranks industry groups over a similar
    // six-month window, so the final composite treats group leadership as its
    // own signal instead of hiding it inside individual `rsrank`.
    let industry_ranks = snapshot
        .clone()
        .group_by([col("industry")])
        .agg([col("pct126").mean().alias("__industry_momentum")])
        .with_column(percentile_expr("__industry_momentum", true, "industryrank"))
        .select([col("industry"), col("industryrank")]);

    // IBD-inspired composite rank.
    //
    // Inputs:
    // - epsrank: EBITDA-per-share growth/acceleration proxy for EPS Rating.
    // - rsrank: Relative Strength, a cross-sectional price momentum rank.
    // - smrrank: Sales + Margins + Returns.
    // - adrank: Accumulation/Distribution, price/volume buying pressure.
    // - industryrank: industry group momentum.
    // - highrank: percentile of close / 52-week high. O'Neil-style screens
    //   prefer leaders near highs because breakouts generally start from
    //   strength, not from statistically cheap weakness.
    //
    // IBD discloses that EPS and RS receive the extra weight, but not the exact
    // proprietary formula. This proxy keeps those two dominant, then uses SMR,
    // A/D, industry rank, and 52-week-high proximity as confirmation.
    let composite_weighted = [
        ("epsrank", 0.3),
        ("rsrank", 0.3),
        ("smrrank", 0.15),
        ("adrank", 0.1),
        ("industryrank", 0.1),
        ("highrank", 0.05),
    ];

    // AQR/Twin-Momentum-inspired rank.
    //
    // Components are left as public columns:
    // - fmomrank: fundamental momentum.
    // - rsrank: price momentum.
    // - volconfirmrank: price momentum confirmed by expanding traded value.
    let momentum_weighted = [
        ("fmomrank", 0.45),
        ("rsrank", 0.45),
        ("volconfirmrank", 0.1),
    ];

    snapshot
        .join(
            industry_ranks,
            [col("industry")],
            [col("industry")],
            JoinArgs::new(JoinType::Left),
        )
        .with_column(
            when(col("max252").cast(DataType::Float64).gt(lit(0.0)))
                .then(col("close").cast(DataType::Float64) / col("max252").cast(DataType::Float64))
                .otherwise(lit(NULL))
                .alias("__high_proximity"),
        )
        .with_column(percentile_expr("__high_proximity", true, "highrank"))
        .with_column(weighted_score(&momentum_weighted, "momentumrank"))
        .with_column(weighted_score(&composite_weighted, "compositerank"))
        .drop(["__high_proximity"])
}
