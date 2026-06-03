//! CAN SLIM/IBD-inspired EPS, SMR, and composite rankings.

use polars::prelude::*;

use super::{DEFAULT_MIN_SCORE_WEIGHT_COVERAGE, snapshot_percentile_expr, weighted_rank_score};

pub(in crate::pipeline) fn add_canslim_snapshot_ranks(snapshot: LazyFrame) -> LazyFrame {
    // EPS rank proxy.
    //
    // IBD's EPS Rating measures earnings growth versus all stocks with emphasis
    // on recent acceleration. The largest weights use the same-quarter EPS
    // improvement from ARQ rows, scaled by report-period close:
    // `(quarter_t - quarter_t-4) / close`. That keeps the delta/acceleration
    // flavor without rewarding high nominal EPS or high-price names merely for
    // having larger per-share dollar changes. The multi-year EPS CAGR leg is
    // sourced from the latest TTM/MRT fundamentals on the joined snapshot.
    let eps_weighted = [
        ("__e_epsqchg", 0.4),
        ("__e_epsqaccel", 0.3),
        ("__e_eps1y", 0.2),
        ("__e_epscagr3y", 0.1),
    ];

    // SMR rank proxy: Sales + Margins + Returns.
    //
    // Public IBD/O'Neil descriptions define SMR as Sales growth, Profit
    // Margins, and Return on Equity. Sales growth and after-tax/net margin stay
    // quarterly; pretax margin and ROE use their TTM/MRT columns so their
    // levels are not sourced from single-quarter ARQ rows.
    let smr_weighted = [
        ("__s_revenue1y", 0.25),
        ("__s_netmargin", 0.25),
        ("__s_pretaxmargin", 0.25),
        ("__s_roe", 0.25),
    ];

    snapshot
        .with_columns([
            snapshot_percentile_expr("epsqtrchg", true, "__e_epsqchg"),
            snapshot_percentile_expr("epsqtraccel", true, "__e_epsqaccel"),
            snapshot_percentile_expr("epsqtryoy", true, "__e_eps1y"),
            snapshot_percentile_expr("epscagr3y", true, "__e_epscagr3y"),
            snapshot_percentile_expr("revenueqtryoy", true, "__s_revenue1y"),
            snapshot_percentile_expr("netmarginqtr", true, "__s_netmargin"),
            snapshot_percentile_expr("pretaxmargin", true, "__s_pretaxmargin"),
            snapshot_percentile_expr("roe", true, "__s_roe"),
        ])
        .with_columns([
            weighted_rank_score(&eps_weighted, "epsrank", DEFAULT_MIN_SCORE_WEIGHT_COVERAGE),
            weighted_rank_score(&smr_weighted, "smrqrank", DEFAULT_MIN_SCORE_WEIGHT_COVERAGE),
        ])
        .drop([
            "__e_epsqchg",
            "__e_epsqaccel",
            "__e_eps1y",
            "__e_epscagr3y",
            "__s_revenue1y",
            "__s_netmargin",
            "__s_pretaxmargin",
            "__s_roe",
        ])
}

pub(in crate::pipeline) fn add_composite_inputs(snapshot: LazyFrame) -> LazyFrame {
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
        .with_column(snapshot_percentile_expr(
            "__industry_momentum",
            true,
            "industryrank",
        ))
        .select([col("industry"), col("industryrank")]);

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
        .with_column(snapshot_percentile_expr(
            "__high_proximity",
            true,
            "highrank",
        ))
}

pub(in crate::pipeline) fn add_composite_rank(snapshot: LazyFrame) -> LazyFrame {
    // IBD-inspired composite rank.
    //
    // Inputs:
    // - epsrank: quarterly EPS growth/acceleration plus EPS CAGR.
    // - rsrank: Relative Strength, a cross-sectional price momentum rank.
    // - smrqrank: quarterly sales/net margin plus TTM pretax margin/ROE.
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
        ("smrqrank", 0.15),
        ("adrank", 0.1),
        ("industryrank", 0.1),
        ("highrank", 0.05),
    ];

    snapshot
        .with_columns([
            snapshot_percentile_expr("epsrank", true, "__c_epsrank"),
            snapshot_percentile_expr("smrqrank", true, "__c_smrqrank"),
        ])
        .with_column(weighted_rank_score(
            &[
                ("__c_epsrank", 0.3),
                ("rsrank", 0.3),
                ("__c_smrqrank", 0.15),
                ("adrank", 0.1),
                ("industryrank", 0.1),
                ("highrank", 0.05),
            ],
            "comprank",
            DEFAULT_MIN_SCORE_WEIGHT_COVERAGE,
        ))
        .drop(["__c_epsrank", "__c_smrqrank", "__high_proximity"])
}
