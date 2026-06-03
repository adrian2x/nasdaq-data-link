//! Ranking formulas used by fundamentals and company snapshots.
//!
//! Research map:
//! - RAFI Fundamental/Growth Index rulebooks: rank businesses by multiple
//!   real economic fundamentals instead of one accounting line.
//!   https://www.rafi.com/index-strategies/rafi-fundamental-indices
//! - Fundsmith/Terry Smith quality lens: prefer high returns on capital,
//!   strong cash conversion, low leverage, and durable reinvestment.
//!   https://www.fundsmith.co.uk/media/t5vjdlkm/fsf-owners-manual.pdf
//! - Novy-Marx fundamental momentum: improving fundamentals, especially
//!   earnings/profitability momentum, help explain price momentum.
//!   https://www.nber.org/papers/w20984
//! - IBD/CAN SLIM ratings: EPS, SMR, RS, A/D, industry strength, and 52-week
//!   high proximity are combined into a broad growth-stock composite.
//!   Public examples of the disclosed components:
//!   https://finance.yahoo.com/news/composite-rating-helps-spot-highest-220400010.html

mod canslim;
mod momentum;
mod overall;
mod rafi;

use polars::prelude::*;

pub(super) use canslim::{add_canslim_snapshot_ranks, add_composite_inputs, add_composite_rank};
pub(super) use momentum::{add_fundamental_momentum_snapshot_rank, add_momentum_rank};
pub(super) use overall::add_overall_score;
pub(super) use rafi::{add_rafi_features, add_rafi_snapshot_ranks, rafi_feature_columns};

const TICKER: &str = "ticker";
const REPORT_PERIOD: &str = "reportperiod";
const RANK_JOIN_KEY: &str = "__rank_join_key";
const DEFAULT_MIN_SCORE_WEIGHT_COVERAGE: f64 = 0.5;

/// A requested report-period cross-sectional ranking.
///
/// `source` is any Polars expression, not just a physical column. That lets
/// callers rank formulas such as `EV / EBITDA` without materializing temporary
/// public columns. `alias` is the temporary or final rank column to create.
/// `ascending` deliberately follows pandas `rank`: when `true`, larger values
/// end up with larger ranks/percentiles; when `false`, lower values are better.
/// `percentile` controls whether the output is a 0..100 score or a raw 1..N
/// rank. The production scores all use percentiles so formulas with different
/// units can be blended safely.
struct PeriodRankSpec {
    source: Expr,
    alias: &'static str,
    ascending: bool,
    percentile: bool,
}

/// Casts a numeric input to `Float64` before arithmetic.
///
/// Financial statement fields arrive with mixed integer/float physical types.
/// Ranking formulas compare many ratios across companies, so all intermediate
/// calculations use `Float64` to avoid integer division and reduce rounding
/// noise. Final user-facing scores are rounded/cast separately.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Rounds `expr` to `dp` decimals and stores it as `Float32`.
fn round(expr: Expr, dp: u32) -> Expr {
    expr.round(dp).cast(DataType::Float32)
}

/// Replaces non-finite results (`NaN` and `inf`, e.g. from division by zero)
/// with null, so absence is represented as null rather than a sentinel float.
fn finite(expr: Expr) -> Expr {
    when(expr.clone().is_finite())
        .then(expr)
        .otherwise(lit(NULL))
}

/// Divides two expressions and keeps undefined math as null.
fn safe_div(numerator: Expr, denominator: Expr) -> Expr {
    finite(numerator / denominator)
}

/// Rescales `fraction` to a percentage (`100 * fraction`) and finalizes it at
/// 1 decimal place. The argument may itself be a division; non-finite results
/// become null.
fn pct(fraction: Expr) -> Expr {
    round(finite(lit(100.0) * fraction), 1)
}

/// Divides `numerator` by `denominator` and finalizes the result at 3 decimal
/// places. Non-finite results become null.
fn ratio(numerator: Expr, denominator: Expr) -> Expr {
    round(safe_div(numerator, denominator), 3)
}

/// Computes per-ticker compound annual growth rate:
/// `(end / start)^(1 / years) - 1`, reported as a percent.
///
/// The source frame is quarterly trailing-twelve-month (TTM) data. A three-year
/// CAGR therefore compares the current TTM row with the same fiscal-quarter
/// TTM row 12 observations earlier (`years * 4`). We require strictly positive
/// start and end values because a CAGR is not mathematically meaningful across
/// zero or sign changes; those cases are left null and excluded from the rank
/// denominator.
fn cagr(source: Expr, years: i64) -> Expr {
    let source = finite(source);
    let start = source.clone().shift(lit(years * 4));
    let r = safe_div(source.clone(), start.clone());
    let value = (r.pow(lit(1.0 / years as f64)) - lit(1.0)) * lit(100.0);
    let guarded = when(start.gt(lit(0.0)).and(source.clone().gt(lit(0.0))))
        .then(finite(value))
        .otherwise(lit(NULL))
        .over([col(TICKER)]);
    round(guarded, 1)
}

/// Describes one rank request without executing it.
///
/// `with_period_ranks` accepts many of these specs and builds all ranks from
/// one as-of report snapshot. That matters for runtime: the expensive part is
/// forming each report-date cross-section, so batching quality, value, EPS,
/// SMR, and fundamental-momentum signals avoids repeating the same join.
fn period_rank(
    source: Expr,
    alias: &'static str,
    ascending: bool,
    percentile: bool,
) -> PeriodRankSpec {
    PeriodRankSpec {
        source,
        alias,
        ascending,
        percentile,
    }
}

/// Converts a temporary value column into a report-period rank expression.
///
/// Formula for percentile ranks:
/// `(rank - 1) / (non_null_count - 1) * 100`.
///
/// This makes every signal read the same way: 0 is the weakest company in the
/// cross-section and 100 is the strongest. Ties use average rank, which avoids
/// arbitrary ordering when many companies share a coarse accounting value.
/// Null source values do not count in `non_null_count`, so missing metrics do
/// not dilute the live peer universe. A one-stock universe receives 100 because
/// the only observable company is, by definition, best in that tiny set.
fn period_rank_expr(value: &str, alias: &'static str, ascending: bool, percentile: bool) -> Expr {
    let opts = RankOptions {
        method: RankMethod::Average,
        descending: !ascending,
    };
    let n = col(value).count().over([col(REPORT_PERIOD)]);
    let rank = col(value).rank(opts, None).over([col(REPORT_PERIOD)]);

    if percentile {
        when(n.clone().gt(lit(1)))
            .then(((rank - lit(1.0)) / (n.clone() - lit(1.0))) * lit(100.0))
            .otherwise(when(n.eq(lit(1))).then(lit(100.0)).otherwise(lit(NULL)))
            .round(2)
            .cast(DataType::Float32)
            .alias(alias)
    } else {
        rank.cast(DataType::Float32).alias(alias)
    }
}

/// Cross-sectional percentile for an already-latest company snapshot.
///
/// This is simpler than the report-period helper because the frame has already
/// been reduced to one row per active ticker. Formula:
/// `(rank - 1) / (N - 1) * 100`, rounded to an integer. `ascending=true` means
/// high raw values get high scores; `ascending=false` means low raw values get
/// high scores. Nulls are excluded from `N`.
fn snapshot_percentile_expr(source: &'static str, ascending: bool, alias: &'static str) -> Expr {
    let opts = RankOptions {
        method: RankMethod::Average,
        descending: !ascending,
    };
    let n = col(source).count().over([lit(1)]);
    let rank = col(source).rank(opts, None).over([lit(1)]);

    when(n.clone().gt(lit(1)))
        .then(((rank - lit(1.0)) / (n.clone() - lit(1.0))) * lit(100.0))
        .otherwise(when(n.eq(lit(1))).then(lit(100.0)).otherwise(lit(NULL)))
        .round(2)
        .cast(DataType::Float32)
        .alias(alias)
}

/// Blends 0..100 rank components into one 0..100 score.
///
/// Each input is already a cross-sectional percentile where higher is better.
/// The formula is:
/// `sum(component_percentile * weight) / sum(weight for non-null components)`.
///
/// The denominator is intentionally dynamic. If a company lacks, say, R&D
/// history, A/D, or a meaningful EV multiple, the available signals are
/// reweighted instead of forcing a zero into the score. `min_coverage` is the
/// minimum share of intended weight that must be present; thinner records
/// return null rather than being flattered by sparse coverage.
fn weighted_rank_score(weighted: &[(&str, f64)], alias: &'static str, min_coverage: f64) -> Expr {
    let mut numerator = lit(0.0);
    let mut denominator = lit(0.0);
    let min_weight = weighted.iter().map(|(_, weight)| *weight).sum::<f64>() * min_coverage;
    for (name, weight) in weighted {
        numerator = numerator + col(*name).fill_null(lit(0.0)) * lit(*weight);
        denominator = denominator + col(*name).is_not_null().cast(DataType::Float64) * lit(*weight);
    }

    when(denominator.clone().gt_eq(lit(min_weight)))
        .then((numerator / denominator).round(2))
        .otherwise(lit(NULL))
        .cast(DataType::Float32)
        .alias(alias)
}

/// Blend of 0–100 percentile components where a MISSING component is imputed to
/// a neutral 50 rather than having its weight redistributed to the present ones.
///
/// Use where "missing" means "unconfirmed" (a momentum name with no fundamental
/// or earnings data): redistributing the weight lets the present signal inherit
/// the full score, so an unconfirmed name ranks as if fully confirmed. Imputing
/// to neutral pulls it toward the middle. Keep the default `weighted_rank_score`
/// where present components legitimately represent the whole construct.
fn weighted_rank_score_neutral(
    weighted: &[(&str, f64)],
    alias: &'static str,
    min_coverage: f64,
) -> Expr {
    let total_weight: f64 = weighted.iter().map(|(_, weight)| *weight).sum();
    let min_weight = total_weight * min_coverage;

    let mut numerator = lit(0.0);
    let mut present = lit(0.0);
    for (name, weight) in weighted {
        let value = when(col(*name).is_not_null())
            .then(col(*name).cast(DataType::Float64))
            .otherwise(lit(50.0));
        numerator = numerator + value * lit(*weight);
        present = present + col(*name).is_not_null().cast(DataType::Float64) * lit(*weight);
    }

    when(present.gt_eq(lit(min_weight)))
        .then((numerator / lit(total_weight)).round(2)) // divide by TOTAL, not present
        .otherwise(lit(NULL))
        .cast(DataType::Float32)
        .alias(alias)
}

/// Adds report-period cross-sectional rank columns.
///
/// For each `reportperiod` anchor date, the ranking universe is each ticker's
/// most recent report from the prior 366 days. This avoids look-ahead while
/// keeping companies with non-standard fiscal dates in the same cross-section.
/// `ascending=true` follows pandas rank semantics: higher values receive
/// higher rank scores. Percentiles are 0..100, with higher always better.
///
/// Why the as-of design:
/// - Fundamentals are not released on one standardized date. Apple, Costco,
///   and banks can all have different fiscal calendars.
/// - A strict `group_by(reportperiod)` would rank only companies with exactly
///   matching report dates, which is too narrow.
/// - A global latest-row rank would leak future information when scoring
///   historical reports.
/// - The as-of join creates, for every anchor `reportperiod`, the latest known
///   report per ticker within 366 days. That is the closest lazy Polars
///   equivalent of "rank all currently known companies at this reporting date."
///
/// Implementation sketch:
/// 1. Build all anchor report dates.
/// 2. Cross them with all tickers to define every desired `(date, ticker)`.
/// 3. As-of join backward by ticker to fetch the latest report not after the
///    anchor date and no more than one year stale.
/// 4. Rank each requested expression inside the anchor-date universe.
/// 5. Join those rank columns back to the original rows.
fn with_period_ranks(lf: LazyFrame, ranks: &[PeriodRankSpec]) -> LazyFrame {
    let lf = lf.cache();
    if ranks.is_empty() {
        return lf;
    }

    // Evaluate all rank input formulas once in the right-hand report table.
    // The names are deliberately private because callers only care about the
    // final rank aliases, not the temporary ratios used to get there.
    let value_aliases = ranks
        .iter()
        .enumerate()
        .map(|(i, _)| format!("__rank_value_{i}"))
        .collect::<Vec<_>>();
    let mut value_exprs = Vec::with_capacity(ranks.len() + 2);
    value_exprs.push(col(TICKER));
    value_exprs.push(col(REPORT_PERIOD));
    for (rank, value_alias) in ranks.iter().zip(value_aliases.iter()) {
        value_exprs.push(
            rank.source
                .clone()
                .cast(DataType::Float64)
                .alias(value_alias),
        );
    }

    let values = lf
        .clone()
        .select(value_exprs)
        // Enforce one candidate report per `(ticker, reportperiod)`. The
        // as-of join below then guarantees one matched report per ticker in
        // each anchor-date ranking group.
        .group_by_stable([col(TICKER), col(REPORT_PERIOD)])
        .agg(
            value_aliases
                .iter()
                .map(|value_alias| col(value_alias.as_str()).last())
                .collect::<Vec<_>>(),
        )
        .sort([TICKER, REPORT_PERIOD], Default::default());

    let has_rank_value = value_aliases.iter().skip(1).fold(
        col(value_aliases[0].as_str()).is_not_null(),
        |acc, value_alias| acc.or(col(value_alias.as_str()).is_not_null()),
    );

    // Anchor dates are the dates for which we want a full cross-section.
    // Ticker ranges stop the cross join from manufacturing impossible
    // `(date, ticker)` pairs before a company's first report or long after its
    // final report; those rows could only become null ranks.
    let anchors = lf
        .clone()
        .select([col(REPORT_PERIOD)])
        .unique(None, UniqueKeepStrategy::First)
        .with_column(lit(1i32).alias(RANK_JOIN_KEY));
    let ticker_ranges = lf
        .clone()
        .select([col(TICKER), col(REPORT_PERIOD)])
        .group_by([col(TICKER)])
        .agg([
            col(REPORT_PERIOD).min().alias("__first_reportperiod"),
            col(REPORT_PERIOD).max().alias("__last_reportperiod"),
        ])
        .with_column(lit(1i32).alias(RANK_JOIN_KEY));
    let anchor_day = col(REPORT_PERIOD).cast(DataType::Int32);
    let first_day = col("__first_reportperiod").cast(DataType::Int32);
    let last_valid_day = col("__last_reportperiod").cast(DataType::Int32) + lit(366i32);
    let universe = anchors
        .join(
            ticker_ranges,
            [col(RANK_JOIN_KEY)],
            [col(RANK_JOIN_KEY)],
            JoinArgs::new(JoinType::Inner),
        )
        .drop([RANK_JOIN_KEY])
        .filter(
            anchor_day
                .clone()
                .gt_eq(first_day)
                .and(anchor_day.lt_eq(last_valid_day)),
        )
        .drop(["__first_reportperiod", "__last_reportperiod"])
        .sort([TICKER, REPORT_PERIOD], Default::default());

    // `JoinType::AsOf` is the core of the "latest filing within one year"
    // behavior. `Backward` means the matched report must be at or before the
    // anchor date. `left_by/right_by = ticker` prevents one company's report
    // from filling another company's slot. `tolerance = 366` admits late fiscal
    // calendars without letting stale, multi-year-old fundamentals survive.
    let snapshots = universe
        .join(
            values,
            [col(REPORT_PERIOD)],
            [col(REPORT_PERIOD)],
            JoinArgs::new(JoinType::AsOf(AsOfOptions {
                strategy: AsofStrategy::Backward,
                tolerance: Some(AnyValue::Int32(366)),
                tolerance_str: None,
                left_by: Some(vec![TICKER.into()]),
                right_by: Some(vec![TICKER.into()]),
                allow_eq: true,
                check_sortedness: false,
            })),
        )
        // If the matched latest report has no rankable values at all, every
        // rank output would be null. Filtering those rows before the rank
        // windows shrinks the work without changing the left-joined result.
        .filter(has_rank_value);

    let mut rank_exprs = Vec::with_capacity(ranks.len());
    for (rank, value_alias) in ranks.iter().zip(value_aliases.iter()) {
        rank_exprs.push(period_rank_expr(
            value_alias,
            rank.alias,
            rank.ascending,
            rank.percentile,
        ));
    }

    let mut select_exprs = Vec::with_capacity(ranks.len() + 2);
    select_exprs.push(col(TICKER));
    select_exprs.push(col(REPORT_PERIOD));
    select_exprs.extend(ranks.iter().map(|rank| col(rank.alias)));

    let ranked = snapshots.with_columns(rank_exprs).select(select_exprs);

    lf.join(
        ranked,
        [col(TICKER), col(REPORT_PERIOD)],
        [col(TICKER), col(REPORT_PERIOD)],
        JoinArgs::new(JoinType::Left),
    )
}

fn weighted_period_rank(
    lf: LazyFrame,
    ranks: &[PeriodRankSpec],
    weighted: &[(&str, f64)],
    alias: &'static str,
    min_coverage: f64,
) -> LazyFrame {
    with_period_ranks(lf, ranks)
        .with_column(weighted_rank_score(weighted, alias, min_coverage))
        .drop(ranks.iter().map(|rank| rank.alias).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn period_rank_uses_latest_report_within_one_year() -> PolarsResult<()> {
        let df = df![
            "ticker" => ["A", "A", "B", "B", "C"],
            "reportperiod" => [1000i32, 1400, 900, 1100, 600],
            "value" => [10.0, 20.0, 100.0, 30.0, 999.0],
        ]?;
        let out = with_period_ranks(df.lazy(), &[
            period_rank(col("value"), "higher_pct", true, true),
            period_rank(col("value"), "lower_rank", false, false),
        ])
        .filter(
            col("ticker")
                .eq(lit("A"))
                .and(col("reportperiod").eq(lit(1400i32))),
        )
        .select([col("higher_pct"), col("lower_rank")])
        .collect()?;

        assert_eq!(Vec::from(out.column("higher_pct")?.f32()?), &[Some(0.0)]);
        assert_eq!(Vec::from(out.column("lower_rank")?.f32()?), &[Some(2.0)]);

        Ok(())
    }

    #[test]
    fn weighted_score_respects_min_coverage() -> PolarsResult<()> {
        let df = df![
            "a" => [Some(80i32), Some(80), None, None, None],
            "b" => [Some(100i32), None, Some(100), None, None],
            "c" => [None, Some(100i32), None, Some(100i32), None],
        ]?;
        let out = df
            .lazy()
            .select([weighted_rank_score(
                &[("a", 0.4), ("b", 0.4), ("c", 0.2)],
                "score",
                DEFAULT_MIN_SCORE_WEIGHT_COVERAGE,
            )])
            .collect()?;

        assert_eq!(Vec::from(out.column("score")?.i32()?), &[
            Some(90),
            Some(87),
            None,
            None,
            None
        ]);

        Ok(())
    }

    #[test]
    fn equal_weight_value_score_computes_with_available_components() -> PolarsResult<()> {
        let df = df![
            "ev_ebitda" => [Some(90i32), None, Some(90), None],
            "ev_sales" => [Some(60i32), Some(60), Some(60), None],
            "ebitda_peg" => [None, None, Some(30i32), None],
        ]?;
        let out = df
            .lazy()
            .select([weighted_rank_score(
                &[
                    ("ev_ebitda", 1.0 / 3.0),
                    ("ev_sales", 1.0 / 3.0),
                    ("ebitda_peg", 1.0 / 3.0),
                ],
                "valuerank",
                1.0 / 3.0,
            )])
            .collect()?;

        assert_eq!(Vec::from(out.column("valuerank")?.i32()?), &[
            Some(75),
            Some(60),
            Some(60),
            None
        ]);

        Ok(())
    }

    #[test]
    fn composite_rank_renormalizes_missing_eps_and_high_rank() -> PolarsResult<()> {
        let df = df![
            "epsrank" => [None, Some(90i32), None, None],
            "rsrank" => [Some(90i32), Some(80), Some(100), None],
            "smrqrank" => [Some(80i32), Some(70), Some(80), None],
            "adrank" => [Some(70i32), Some(60), Some(60), None],
            "industryrank" => [Some(60i32), Some(50), Some(40), None],
            "highrank" => [Some(50i32), None, None, None],
            "__high_proximity" => [None::<f64>, None, None, None],
        ]?;
        let out = add_composite_rank(df.lazy())
            .select([col("comprank")])
            .collect()?;

        assert_eq!(Vec::from(out.column("comprank")?.i32()?), &[
            Some(84),
            Some(68),
            Some(86),
            None
        ]);

        Ok(())
    }
}
