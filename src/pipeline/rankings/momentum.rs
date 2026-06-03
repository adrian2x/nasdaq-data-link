//! Fundamental and price-confirmed momentum rankings.

use polars::prelude::*;

use crate::pipeline::rankings::weighted_rank_score_neutral;

use super::{DEFAULT_MIN_SCORE_WEIGHT_COVERAGE, snapshot_percentile_expr, weighted_rank_score};

pub(in crate::pipeline) fn add_fundamental_momentum_snapshot_rank(
    snapshot: LazyFrame,
) -> LazyFrame {
    // Fundamental momentum rank.
    //
    // This is the business-metric counterpart to price momentum. It asks:
    // are growth rates accelerating, are margins expanding, and is capital
    // productivity improving? Revenue and EPS acceleration are the "growth
    // surprise" legs; EPS acceleration is scaled by price to avoid nominal EPS
    // bias; margin and ROIC expansion are the "quality of growth" legs. It
    // stays separate from the IBD-style `comprank` because the published
    // Composite inputs do not include a distinct fundamental-momentum factor.
    let fundamental_momentum_weighted = [
        ("__m_revenueaccel", 0.25),
        ("__m_epsqaccel", 0.25),
        ("__m_grossmarginqexp", 0.15),
        ("__m_ebitdamarginqexp", 0.2),
        ("__m_roicqexp", 0.15),
    ];

    snapshot
        .with_columns([
            snapshot_percentile_expr("revenueqtraccel", true, "__m_revenueaccel"),
            snapshot_percentile_expr("epsqtraccel", true, "__m_epsqaccel"),
            snapshot_percentile_expr("grossmarginqtrexp", true, "__m_grossmarginqexp"),
            snapshot_percentile_expr("ebitdamarginqtrexp", true, "__m_ebitdamarginqexp"),
            snapshot_percentile_expr("roicqtrexp", true, "__m_roicqexp"),
        ])
        .with_column(weighted_rank_score(
            &fundamental_momentum_weighted,
            "growthrank",
            DEFAULT_MIN_SCORE_WEIGHT_COVERAGE,
        ))
        .drop([
            "__m_revenueaccel",
            "__m_epsqaccel",
            "__m_grossmarginqexp",
            "__m_ebitdamarginqexp",
            "__m_roicqexp",
        ])
}

pub(in crate::pipeline) fn add_momentum_rank(snapshot: LazyFrame) -> LazyFrame {
    // AQR/Twin-Momentum-inspired rank.
    //
    // Components are left as public columns:
    // - growthrank: quarterly fundamental momentum.
    // - rsrank: price momentum.
    // - volconfirmrank: price momentum confirmed by expanding traded value.
    let momentum_weighted = [
        ("growthrank", 0.45),
        ("rsrank", 0.45),
        ("volconfirmrank", 0.1),
    ];

    snapshot.with_column(weighted_rank_score_neutral(
        &momentum_weighted,
        "momrank",
        DEFAULT_MIN_SCORE_WEIGHT_COVERAGE,
    ))
}
