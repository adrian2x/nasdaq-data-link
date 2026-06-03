//! Overall snapshot ranking.

use polars::prelude::*;

use crate::pipeline::rankings::weighted_rank_score_neutral;

use super::{DEFAULT_MIN_SCORE_WEIGHT_COVERAGE, snapshot_percentile_expr, weighted_rank_score};

/// Adds the headline 0-100 overall rating to the company snapshot.
///
/// Coverage-weighted blend of the three evidence-backed pillars — momentum
/// (primary), quality (durable returns/margins), EPS surprise/acceleration
/// (confirmation, deliberately small) — plus a light industry-leadership and
/// 52-week-high-proximity tilt carrying the only O'Neil signals the three
/// pillars miss. Industry vs high keeps the composite's 2:1 emphasis.
///
/// Because `weighted_rank_score` returns a convex combination of 0-100
/// percentiles, the output is bounded to [0, 100] and rescales to 0-10 by a
/// plain `/ 10`, which preserves ordering. Note it is a continuous *score*,
/// not a decile: a blend of percentiles bunches toward the middle, so the
/// 0-10 result will not be uniformly distributed. If you ever want a uniform
/// 1-10 (10% per band) instead, that requires a `snapshot_percentile_expr`
/// re-rank, not this function.
pub(in crate::pipeline) fn add_overall_score(snapshot: LazyFrame) -> LazyFrame {
    // Confirmation factors, neutral-imputed (missing -> 50). Relative weights
    // 0.35 / 0.40 / 0.15; value (0.10) is blended on top only when present.
    let confirmation_weighted = [
        ("momrank", 0.35),
        ("qualrank", 0.40),
        ("epsrank", 0.15),
    ];
    const VALUE_WEIGHT: f64 = 0.10;

    snapshot
        .with_column(weighted_rank_score_neutral(&confirmation_weighted, "__conf", 0.5))
        // Value present -> blend at 0.10; value null -> fall back to __conf,
        // which redistributes value's weight across momentum/quality/EPS.
        .with_column(
            when(col("valuerank").is_not_null())
                .then(
                    lit(1.0 - VALUE_WEIGHT) * col("__conf")
                        + lit(VALUE_WEIGHT) * col("valuerank").cast(DataType::Float64),
                )
                .otherwise(col("__conf").cast(DataType::Float64))
                .alias("__core"),
        )
        // Stabilizer: O'Neil positioning only, neutral (50) when missing so it
        // can never null out an otherwise-scoreable row.
        .with_column(
            ((col("industryrank").fill_null(lit(50)) + col("highrank").fill_null(lit(50)))
                / lit(2.0))
            .alias("__stab"),
        )
        .with_column((col("__core") + lit(0.03) * col("__stab")).alias("__core_adj"))
        .with_column(snapshot_percentile_expr("__core_adj", true, "rankscore"))
        .drop(["__conf", "__core", "__stab", "__core_adj"])
}
