//! Cross-sectional percentile rank.
//!
//! Ranks every ticker against its peers on the same date and maps the rank
//! to a 0..100 percentile. Self-contained.

use polars::prelude::*;

/// Add a cross-sectional percentile-rank column named `alias`.
///
/// For each distinct value of `date_col`, ranks all ticker rows by `source`
/// and maps the rank to a 0–100 percentile. The ranking universe is whatever
/// rows are in the frame at call time — rank against the whole market by
/// calling on the full table, or against a peer group by calling after a
/// filter.
///
/// `ascending` follows the pandas `Series.rank` convention: whether the
/// elements are ranked in ascending order of `source`.
///   - `true`: smallest `source` value ranks first → percentile 0; largest
///     → percentile 100. Use for "bigger is better" inputs like momentum.
///   - `false`: largest `source` value ranks first → percentile 0.
///
/// Ties take the average rank. Null `source` values produce a null rank and
/// are excluded from the per-date count, so surviving rows rank only against
/// each other.
pub fn percentile(
    lf: LazyFrame,
    source: &str,
    date_col: &str,
    ascending: bool,
    alias: &str,
) -> LazyFrame {
    let opts = RankOptions {
        descending: !ascending,
        ..Default::default()
    };

    // n = count of non-null `source` values within the date group.
    let n = col(source).count().over([col(date_col)]);
    let rank = col(source).rank(opts, None).over([col(date_col)]);

    lf.with_columns([(((rank - lit(1.0)) / (n - lit(1.0))) * lit(100.0))
        .alias(alias)
        .cast(DataType::Float32)])
}
