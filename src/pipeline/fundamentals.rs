//! Fundamentals preparation.
//!
//! Transforms raw quarterly financial-statement rows into analysis columns,
//! including per-share growth metrics. Self-contained — the growth helpers
//! live here because fundamentals is their only consumer.
//!
//! Research map for the ranking columns built here:
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

use polars::prelude::*;

use crate::indicators::fs_score;

const TICKER: &str = "ticker";
const REPORT_PERIOD: &str = "reportperiod";
const RANK_JOIN_KEY: &str = "__rank_join_key";
const MIN_SCORE_WEIGHT_COVERAGE: f64 = 0.5;

/// A requested cross-sectional ranking.
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
/// noise. Final user-facing metrics are rounded/cast separately.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Rounds `expr` to `dp` decimals and stores it as `Float32`. All metric
/// finalization in this module funnels through this function.
fn round(expr: Expr, dp: u32) -> Expr {
    expr.round(dp).cast(DataType::Float32)
}

/// Replaces non-finite results (`NaN` and `±inf`, e.g. from division by
/// zero) with null, so absence is represented as null rather than a sentinel
/// float value.
fn finite(expr: Expr) -> Expr {
    when(expr.clone().is_finite())
        .then(expr)
        .otherwise(lit(NULL))
}

/// Divides two expressions and keeps undefined math as null.
fn safe_div(numerator: Expr, denominator: Expr) -> Expr {
    finite(numerator / denominator)
}

/// Uses `secondary` only when `primary` is null.
fn fallback(primary: Expr, secondary: Expr) -> Expr {
    when(primary.clone().is_not_null())
        .then(primary)
        .otherwise(secondary)
}

/// Rescales `fraction` to a percentage (`100 * fraction`) and finalizes it at
/// 1 decimal place. The argument may itself be a division; non-finite
/// results (e.g. division by zero) become null.
fn pct(fraction: Expr) -> Expr {
    round(finite(lit(100.0) * fraction), 1)
}

/// Divides `numerator` by `denominator` and finalizes the result at 3 decimal
/// places. Non-finite results (e.g. division by zero) become null.
fn ratio(numerator: Expr, denominator: Expr) -> Expr {
    round(safe_div(numerator, denominator), 3)
}

/// Computes per-ticker growth of `source` over `period` rows as
/// `(end - start) / |start| * 100`, finalized at 1 decimal place.
///
/// The `|start|` denominator keeps the sign correct for negative starts: a
/// shrinking loss reads positive, a deepening loss negative, and a
/// loss-to-profit turnaround strongly positive. Returns null when `start` is
/// zero or history is insufficient.
fn pct_change(source: Expr, period: i64) -> Expr {
    let source = finite(source);
    let start = source.clone().shift(lit(period));
    let growth = safe_div(source - start.clone(), start.abs()) * lit(100.0);
    round(growth.over([col("ticker")]), 1)
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
///
fn cagr(source: Expr, years: i64) -> Expr {
    let source = finite(source);
    let start = source.clone().shift(lit(years * 4));
    let r = safe_div(source.clone(), start.clone());
    let value = (r.pow(lit(1.0 / years as f64)) - lit(1.0)) * lit(100.0);
    let guarded = when(start.gt(lit(0.0)).and(source.clone().gt(lit(0.0))))
        .then(finite(value))
        .otherwise(lit(NULL))
        .over([col("ticker")]);
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

/// Converts a temporary value column into a rank expression.
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
fn rank_expr(value: &str, alias: &'static str, ascending: bool, percentile: bool) -> Expr {
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

/// Blends percentile components into one 0..100 score.
///
/// Each input is already a cross-sectional percentile where higher is better.
/// The formula is:
/// `sum(component_percentile * weight) / sum(weight for non-null components)`.
///
/// The denominator is intentionally dynamic. If a company lacks, say, R&D
/// history or a meaningful EV multiple, the available signals are reweighted
/// instead of forcing a zero into the score. At least 50% of the intended
/// weight must be present; thinner records return null rather than being
/// flattered by sparse coverage.
fn weighted_percentile_score(weighted: &[(&str, f64)], alias: &'static str) -> Expr {
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
        rank_exprs.push(rank_expr(
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

/// Adds the fundamental ranking family.
///
/// The philosophy is "rank the business, not just the stock chart." Every
/// component first becomes a report-date cross-sectional percentile via
/// `with_period_ranks`, then related components are blended into readable
/// scores. Component rank columns are retained in the output so consumers can
/// reconstitute or reweight any composite.
///
/// - `qualrank`: durable quality. High returns on capital, strong margins,
///   cash conversion, accrual quality, low leverage, liquidity, and reinvestment
///   signals. This is loosely RAFI/Fundsmith inspired: broad real business
///   fundamentals plus owner-quality tests.
/// - `valuerank`: cheapness. Uses standard lower-is-better multiples
///   `EV / EBITDA`, `EV / Sales`, and a modified PEG-style multiple
///   `(EV / EBITDA) / EBITDA growth 1Y`.
/// - `epsrank`: IBD EPS Rating proxy. IBD uses EPS growth and acceleration;
///   this code substitutes EBITDA per share because EBITDA is more comparable
///   across tax rates, capital structures, and non-cash accounting choices.
/// - `smrrank`: IBD SMR proxy. SMR means Sales + Margins + Return on Equity.
/// - `fmomrank`: fundamental momentum. Novy-Marx shows earnings/fundamental
///   momentum explains much of price momentum, so this blends acceleration and
///   margin/return expansion into a standalone business-momentum diagnostic.
///
/// Acronyms used below:
/// - EV: Enterprise Value, roughly market capitalization plus net debt.
/// - EBITDA: Earnings Before Interest, Taxes, Depreciation, and Amortization.
/// - PEG: Price/Earnings-to-Growth; here adapted to `EV / EBITDA` so the
///   numerator and denominator both operate at enterprise value scale.
/// - ROE: Return on Equity. ROIC: Return on Invested Capital.
/// - CFC: Cash Flow Conversion, `NCFO / operating income`.
/// - ICR: Interest Coverage Ratio, `EBIT / interest expense`.
fn fundamental_rankings(lf: LazyFrame) -> LazyFrame {
    // Guard enterprise-value multiples. Negative EV, negative EBITDA, and
    // negative sales can happen in raw data but do not represent meaningful
    // "cheapness" in standard multiple analysis, so they rank as null.
    let ev_safe = when(f("ev").gt(lit(0.0)))
        .then(f("ev"))
        .otherwise(lit(NULL));
    let ebitda_safe = when(f("ebitdausd").gt(lit(0.0)))
        .then(f("ebitdausd"))
        .otherwise(lit(NULL));
    let sales_safe = when(f("revenueusd").gt(lit(0.0)))
        .then(f("revenueusd"))
        .otherwise(lit(NULL));
    // Standard value multiples:
    // - EV / EBITDA: how many dollars of enterprise value the market pays for
    //   one dollar of operating cash-profit proxy.
    // - EV / Sales: how many dollars of enterprise value the market pays for
    //   one dollar of revenue. It is noisier than EV/EBITDA but useful for
    //   cyclicals and firms where current margins are temporarily depressed.
    // Lower multiples are better, so these rank with `ascending=false`.
    let ev_ebitda = safe_div(ev_safe.clone(), ebitda_safe);
    let ev_sales = safe_div(ev_safe, sales_safe);

    // Modified PEG:
    // `(EV / EBITDA) / EBITDA_growth_1Y`.
    //
    // A stock with a low multiple and strong EBITDA growth should score well.
    // Non-positive growth gets a very large multiple instead of null, because
    // a shrinking company should be penalized on this leg rather than excused.
    // Missing growth stays null. The growth input is a percent value, so this
    // is a ranking proxy, not a textbook PEG ratio meant for direct
    // interpretation.
    let ebitda_growth = f("ebitdagrowth1y");
    let ebitda_peg = when(
        f("ev")
            .gt(lit(0.0))
            .and(f("ebitdausd").gt(lit(0.0)))
            .and(ebitda_growth.clone().is_not_null()),
    )
    .then(
        when(ebitda_growth.clone().gt(lit(0.0)))
            .then(safe_div(ev_ebitda.clone(), ebitda_growth))
            .otherwise(lit(1.0e9)),
    )
    .otherwise(lit(NULL));

    let revenuecagr3y_source = fallback(f("revenuecagr3y"), f("revenue1y"));
    let revenuecagr5y_source = fallback(f("revenuecagr5y"), revenuecagr3y_source.clone());
    let ebitdacagr3y_source = fallback(f("ebitdacagr3y"), f("ebitda1y"));
    let ebitdacagr5y_source = fallback(f("ebitdacagr5y"), ebitdacagr3y_source.clone());
    let salesassetscagr3y_source = fallback(f("salesassetscagr3y"), f("salesassets1y"));
    let grossprofitassetscagr3y_source =
        fallback(f("grossprofitassetscagr3y"), f("grossprofitassets1y"));
    let opprofitassetscagr3y_source = fallback(f("opprofitassetscagr3y"), f("opprofitassets1y"));
    let rndcagr3y_source = fallback(f("rndcagr3y"), f("rnd1y"));

    // All raw ingredients are ranked together in one batched pass. Rank aliases
    // are public component columns so downstream consumers can reconstitute the
    // composites or apply their own weights.
    let ranks = [
        period_rank(f("roic"), "roicrank", true, true),
        period_rank(f("roce"), "rocerank", true, true),
        period_rank(f("roe"), "roerank", true, true),
        period_rank(f("grossmargin"), "grossmarginrank", true, true),
        period_rank(f("ebitdamargin"), "ebitdamarginrank", true, true),
        period_rank(f("ebitmargin"), "ebitmarginrank", true, true),
        period_rank(
            revenuecagr3y_source.clone(),
            "revenuecagr3yrank",
            true,
            true,
        ),
        period_rank(revenuecagr5y_source, "revenuecagr5yrank", true, true),
        period_rank(
            salesassetscagr3y_source,
            "salesassetscagr3yrank",
            true,
            true,
        ),
        period_rank(
            grossprofitassetscagr3y_source,
            "grossprofitassetscagr3yrank",
            true,
            true,
        ),
        period_rank(
            opprofitassetscagr3y_source,
            "opprofitassetscagr3yrank",
            true,
            true,
        ),
        period_rank(f("rndintensity"), "rndintensityrank", true, true),
        period_rank(rndcagr3y_source, "rndcagr3yrank", true, true),
        period_rank(f("cfc"), "cfcrank", true, true),
        period_rank(f("fcfassets"), "fcfassetsrank", true, true),
        period_rank(f("cashaccruals"), "cashaccrualsrank", true, true),
        period_rank(f("icr"), "icrrank", true, true),
        period_rank(f("de"), "derank", false, true),
        period_rank(f("fsscore"), "fsscorerank", true, true),
        period_rank(f("cashassets"), "cashassetsrank", true, true),
        period_rank(f("intangiblesassets"), "intangiblesassetsrank", true, true),
        period_rank(ev_ebitda, "evebitdarank", false, true),
        period_rank(ev_sales, "evsalesrank", false, true),
        period_rank(ebitda_peg, "ebitdapegrank", false, true),
        period_rank(f("ebitda1y"), "ebitda1yrank", true, true),
        period_rank(f("ebitdaaccel"), "ebitdaaccelrank", true, true),
        period_rank(ebitdacagr3y_source.clone(), "ebitdacagr3yrank", true, true),
        period_rank(ebitdacagr5y_source, "ebitdacagr5yrank", true, true),
        period_rank(f("revenue1y"), "revenue1yrank", true, true),
        period_rank(f("revenueaccel"), "revenueaccelrank", true, true),
        period_rank(
            f("grossmarginexpansion"),
            "grossmarginexpansionrank",
            true,
            true,
        ),
        period_rank(
            f("ebitdamarginexpansion"),
            "ebitdamarginexpansionrank",
            true,
            true,
        ),
        period_rank(f("roicexpansion"), "roicexpansionrank", true, true),
    ];
    let lf = with_period_ranks(lf, &ranks);

    // Quality rank weights.
    //
    // Profitability and returns on capital carry the most weight because high
    // returns sustained over time are the central "quality compounder" idea.
    // Growth is included, but only alongside cash conversion, accrual quality,
    // interest coverage, leverage, liquidity, and reinvestment proxies so the
    // rank does not simply become a growth-at-any-price score.
    let quality_weighted = [
        ("roicrank", 0.08),
        ("rocerank", 0.08),
        ("roerank", 0.06),
        ("grossmarginrank", 0.06),
        ("ebitmarginrank", 0.06),
        ("revenuecagr3yrank", 0.04),
        ("revenuecagr5yrank", 0.04),
        ("salesassetscagr3yrank", 0.04),
        ("grossprofitassetscagr3yrank", 0.04),
        ("opprofitassetscagr3yrank", 0.04),
        ("rndintensityrank", 0.025),
        ("rndcagr3yrank", 0.025),
        ("cfcrank", 0.05),
        ("fcfassetsrank", 0.05),
        ("cashaccrualsrank", 0.05),
        ("icrrank", 0.04),
        ("derank", 0.04),
        ("fsscorerank", 0.04),
        ("cashassetsrank", 0.03),
        ("intangiblesassetsrank", 0.03),
    ];

    // Value rank weights.
    //
    // Equal-weighting keeps the score robust. EV/EBITDA captures operating
    // profit yield, EV/Sales captures revenue cheapness, and the modified PEG
    // leg avoids giving the best rank to statistically cheap businesses whose
    // EBITDA is not growing.
    let value_weighted = [
        ("evebitdarank", 1.0 / 3.0),
        ("evsalesrank", 1.0 / 3.0),
        ("ebitdapegrank", 1.0 / 3.0),
    ];

    // EPS rank proxy.
    //
    // IBD's EPS Rating measures earnings growth versus all stocks with emphasis
    // on recent acceleration. We use EBITDA per share because it neutralizes
    // some tax, capital-structure, and depreciation differences while still
    // measuring operating earnings available per diluted share. The weight stack
    // intentionally favors 1Y growth and quarter-over-quarter acceleration over
    // older CAGR history.
    let eps_weighted = [
        ("ebitda1yrank", 0.4),
        ("ebitdaaccelrank", 0.3),
        ("ebitdacagr3yrank", 0.2),
        ("ebitdacagr5yrank", 0.1),
    ];

    // SMR rank proxy: Sales + Margins + Returns.
    //
    // Public IBD/O'Neil descriptions define SMR as Sales growth, Profit
    // Margins, and Return on Equity. We use revenue growth for Sales,
    // gross/EBITDA margins for Margins, and ROE for Returns.
    let smr_weighted = [
        ("revenue1yrank", 0.25),
        ("revenuecagr3yrank", 0.15),
        ("grossmarginrank", 0.15),
        ("ebitdamarginrank", 0.2),
        ("roerank", 0.25),
    ];

    // Fundamental momentum rank.
    //
    // This is the business-metric counterpart to price momentum. It asks:
    // are growth rates accelerating, are margins expanding, and is capital
    // productivity improving? Revenue and EBITDA acceleration are the "growth
    // surprise" legs; margin and ROIC expansion are the "quality of growth"
    // legs. It stays separate from the IBD-style `compositerank` because the
    // published Composite inputs do not include a distinct fundamental-momentum
    // factor.
    let fundamental_momentum_weighted = [
        ("revenueaccelrank", 0.25),
        ("ebitdaaccelrank", 0.25),
        ("grossmarginexpansionrank", 0.15),
        ("ebitdamarginexpansionrank", 0.2),
        ("roicexpansionrank", 0.15),
    ];

    lf.with_columns([
        weighted_percentile_score(&quality_weighted, "qualrank"),
        weighted_percentile_score(&value_weighted, "valuerank"),
        weighted_percentile_score(&eps_weighted, "epsrank"),
        weighted_percentile_score(&smr_weighted, "smrrank"),
        weighted_percentile_score(&fundamental_momentum_weighted, "fmomrank"),
    ])
}

/// Transforms raw fundamentals into analysis columns.
pub fn adjust_fundamentals(lf: LazyFrame) -> LazyFrame {
    let lf = lf
        .with_columns([
            col("calendardate").cast(DataType::Date),
            col(REPORT_PERIOD).cast(DataType::Date),
        ])
        .drop(["dimension", "lastupdated"])
        .with_columns([
            // Basic shares outstanding adjusted by the vendor share factor.
            // Basic shares are the current common-share count before dilution.
            (f("sharesbas") * f("sharefactor")).alias("sharesbas"),
            // Diluted shares adjusted by share factor. Diluted shares include
            // options, warrants, convertibles, and other claims that can become
            // common stock, so per-share growth uses this more conservative
            // denominator.
            (f("shareswadil") * f("sharefactor")).alias("sharesdil"),
            // NCFO = Net Cash Flow from Operations. `ncfousd = ncfo / fxusd`
            // normalizes operating cash flow into USD for cross-company ranks.
            safe_div(f("ncfo"), f("fxusd")).alias("ncfousd"),
            // FCF = Free Cash Flow. `fcfusd = fcf / fxusd` normalizes free
            // cash flow into USD.
            safe_div(f("fcf"), f("fxusd")).alias("fcfusd"),
            // Net debt in USD = (total debt - cash and equivalents) / fxusd.
            // Positive values increase Enterprise Value; negative values mean
            // the company has net cash.
            safe_div(f("debt") - f("cashneq"), f("fxusd")).alias("netdebtusd"),
            // ROE = Return on Equity. Vendor value is a fraction, converted to
            // percent so 0.20 becomes 20.0.
            pct(f("roe")).alias("roe"),
            // ROIC = Return on Invested Capital, also converted to percent.
            // This is the main quality-compounding signal: high ROIC suggests
            // reinvested capital can earn attractive incremental returns.
            pct(f("roic")).alias("roic"),
            // ROA = Return on Assets, converted to percent. It is used by the
            // Alpha Architect FS-Score module and retained as a diagnostic.
            pct(f("roa")).alias("roa"),
            // CFC = Cash Flow Conversion = NCFO / operating income.
            // Terry Smith/Fundsmith use cash conversion to separate accounting
            // profit from profit that actually turns into cash.
            ratio(f("ncfo"), f("opinc")).alias("cfc"),
            // ICR = Interest Coverage Ratio = EBIT / interest expense.
            // Higher coverage means the operating business has more room to
            // service debt before equity holders are at risk.
            ratio(f("ebit"), f("intexp")).alias("icr"),
            // ROCE = Return on Capital Employed.
            // Formula here: EBIT / (total assets - current liabilities).
            // It is another Fundsmith-style return-on-capital lens.
            pct(safe_div(f("ebit"), f("assets") - f("liabilitiesc"))).alias("roce"),
            // Adjusted net income = common net income - discontinued operations,
            // normalized to USD. This keeps EPS growth focused on ongoing
            // operations.
            safe_div(f("netinccmn") - f("netincdis"), f("fxusd")).alias("netincadj"),
            // Buyback yield = cash spent on common-share repurchases / market
            // cap. Sharadar reports repurchases as a negative financing cash
            // flow, so multiplying by -1 makes buybacks positive.
            pct(safe_div(
                safe_div(lit(-1.0) * f("ncfcommon"), f("fxusd")),
                f("marketcap"),
            ))
            .alias("bbyield"),
            // Dividend yield converted from fraction to percent.
            pct(f("divyield")).alias("divyield"),
            // Shareholder return in USD = dividends + buybacks. Both cash-flow
            // fields are outflows, so each is multiplied by -1.
            safe_div(
                lit(-1.0) * f("ncfdiv") + lit(-1.0) * f("ncfcommon"),
                f("fxusd"),
            )
            .alias("shreturnusd"),
            // Gross margin = gross profit / revenue, percent.
            pct(safe_div(f("gp"), f("revenue"))).alias("grossmargin"),
            // EBITDA margin = EBITDA / revenue, percent.
            pct(safe_div(f("ebitda"), f("revenue"))).alias("ebitdamargin"),
            // EBIT margin = earnings before interest and taxes / revenue,
            // percent. This captures operating profitability after D&A.
            pct(safe_div(f("ebit"), f("revenue"))).alias("ebitmargin"),
        ])
        .with_columns([
            // EPS adjusted for discontinued operations and dilution:
            // adjusted net income / diluted shares.
            safe_div(f("netincadj"), f("sharesdil")).alias("epsadj"),
        ])
        .sort(["ticker", "calendardate"], Default::default())
        // Adjusted FCF / "owner earnings" proxy in USD.
        //
        // Starting point is NCFO because it is already cash from operations.
        // The adjustment removes discontinued operations, depreciation/
        // amortization, stock-based compensation, and working-capital changes
        // to approximate a cleaner recurring owner-earnings base. This is not
        // a full Buffett maintenance-capex estimate; the comment below is a
        // deliberate reminder that maintenance capex could be layered on later
        // with a longer normalized capex model.
        .with_columns({
            let fcfadj = safe_div(
                f("ncfo")
                // Remove income from discontinued operations.
                - f("netincdis")
                // Remove non-cash charges: depreciation/amortization and stock
                // based compensation.
                - (f("depamor") + f("sbcomp"))
                // Remove quarter-over-quarter working-capital change for the
                // same ticker.
                - (f("workingcapital") - f("workingcapital").shift(lit(1))).over([col("ticker")]),
                f("fxusd"),
            );
            [
                // fcfadj = adjusted owner-earnings proxy in USD.
                fcfadj.clone().alias("fcfadj"),
                // fcfpsadj = adjusted FCF per diluted share.
                safe_div(fcfadj, f("sharesdil")).alias("fcfpsadj"),
            ]
        })
        .with_columns({
            // Per-share growth adjusts the business metrics for buybacks and
            // issuance. That is important for O'Neil/IBD-style growth ranks:
            // a company should not look like it is compounding owner value if
            // aggregate EBITDA rises only because the share count ballooned.
            let ebitdaps = safe_div(f("ebitdausd"), f("sharesdil"));
            let eps = f("epsadj");
            let fcfps = f("fcfpsadj");
            [
                // EBITDA per share = EBITDA in USD / diluted shares. This is
                // the EPS-rating proxy used for operating comparability.
                ebitdaps.clone().alias("ebitdaps"),
                // Revenue growth inputs:
                // - revenue1y = year-over-year sales per share growth.
                // - revenue3y/revenue5y = total growth over 3/5 years.
                // - revenuecagr3y/revenuecagr5y = annualized growth over
                //   3/5 years. These measure persistent demand expansion.
                // `sps` is sales per share, so it reflects dilution/buybacks.
                pct_change(col("sps"), 4).alias("revenue1y"),
                pct_change(col("sps"), 4 * 3).alias("revenue3y"),
                pct_change(col("sps"), 4 * 5).alias("revenue5y"),
                cagr(col("sps"), 3).alias("revenuecagr3y"),
                cagr(col("sps"), 5).alias("revenuecagr5y"),
                // EBITDA growth inputs:
                // - ebitda1y = year-over-year EBITDA per share growth.
                // - ebitdagrowth1y = year-over-year aggregate EBITDA growth,
                //   used in the modified PEG because EV/EBITDA is an
                //   enterprise-level multiple.
                // - ebitdacagr3y/5y = longer operating-profit compounding.
                pct_change(ebitdaps.clone(), 4).alias("ebitda1y"),
                pct_change(f("ebitdausd"), 4).alias("ebitdagrowth1y"),
                cagr(ebitdaps.clone(), 3).alias("ebitdacagr3y"),
                cagr(ebitdaps.clone(), 5).alias("ebitdacagr5y"),
                // EPS and FCF growth are retained as general diagnostics even
                // though the IBD-style rank prefers EBITDA per share. EPS is
                // accounting profit per diluted share; FCF is free cash flow
                // per diluted share after capital expenditure.
                pct_change(eps.clone(), 4).alias("eps1y"),
                cagr(eps.clone(), 3).alias("epscagr3y"),
                cagr(eps.clone(), 5).alias("epscagr5y"),
                pct_change(fcfps.clone(), 4).alias("fcf1y"),
                cagr(fcfps.clone(), 3).alias("fcfcagr3y"),
                cagr(fcfps.clone(), 5).alias("fcfcagr5y"),
            ]
        })
        .with_columns([
            // Fundamental momentum acceleration:
            // - revenueaccel = current revenue1y minus previous quarter's
            //   revenue1y. Positive means sales growth is speeding up.
            // - ebitdaaccel = current ebitda1y minus previous quarter's
            //   ebitda1y. This is the EBITDA-per-share version of O'Neil's
            //   earnings acceleration concept.
            (f("revenue1y") - f("revenue1y").shift(lit(1)).over([col("ticker")]))
                .alias("revenueaccel"),
            (f("ebitda1y") - f("ebitda1y").shift(lit(1)).over([col("ticker")]))
                .alias("ebitdaaccel"),
            // Margin/return expansion:
            // current margin or ROIC minus the same TTM metric one year ago.
            // These are percentage-point changes, not relative percent growth.
            // They reward companies whose growth is becoming more profitable,
            // which is the "business momentum" half of the AQR/Novy-Marx idea.
            (f("grossmargin") - f("grossmargin").shift(lit(4)).over([col("ticker")]))
                .alias("grossmarginexpansion"),
            (f("ebitdamargin") - f("ebitdamargin").shift(lit(4)).over([col("ticker")]))
                .alias("ebitdamarginexpansion"),
            (f("roic") - f("roic").shift(lit(4)).over([col("ticker")])).alias("roicexpansion"),
        ])
        .with_columns({
            // Balance-sheet and reinvestment ratios used by quality/fundamental
            // ranks. Zero-asset and zero-revenue denominators become null, so
            // impossible ratios do not produce infinite rank winners.
            let assets_safe = when(f("assets").eq(lit(0.0)))
                .then(lit(NULL))
                .otherwise(f("assets"));
            let revenue_safe = when(f("revenue").eq(lit(0.0)))
                .then(lit(NULL))
                .otherwise(f("revenue"));
            [
                // cashassets = cash and equivalents / total assets.
                // A liquidity cushion is useful, especially for quality screens.
                ratio(f("cashneq"), assets_safe.clone()).alias("cashassets"),
                // intangiblesassets = intangibles / total assets. This is a
                // rough moat/IP proxy, not a valuation adjustment.
                ratio(f("intangibles"), assets_safe.clone()).alias("intangiblesassets"),
                // fcfassets = free cash flow / total assets. It measures cash
                // return on the asset base.
                ratio(f("fcfusd"), assets_safe.clone()).alias("fcfassets"),
                // cashaccruals = (operating cash flow - EBIT) / assets.
                // Positive values mean cash earnings exceed accounting EBIT,
                // usually a higher-quality earnings signal.
                ratio(f("ncfousd") - f("ebitusd"), assets_safe.clone()).alias("cashaccruals"),
                // rndintensity = R&D / revenue. Used as an innovation/reinvestment
                // proxy; null when revenue is zero.
                pct(safe_div(f("rnd"), revenue_safe)).alias("rndintensity"),
                // Growth of productivity-style ratios. These borrow the RAFI
                // Growth idea of looking beyond plain revenue growth: sales,
                // gross profit, and operating profit should rise relative to
                // the asset base, not only by consuming more capital.
                pct_change(f("assetturnover"), 4).alias("salesassets1y"),
                cagr(f("assetturnover"), 3).alias("salesassetscagr3y"),
                pct_change(safe_div(f("gp"), assets_safe.clone()), 4).alias("grossprofitassets1y"),
                cagr(safe_div(f("gp"), assets_safe.clone()), 3).alias("grossprofitassetscagr3y"),
                pct_change(safe_div(f("opinc"), assets_safe.clone()), 4).alias("opprofitassets1y"),
                cagr(safe_div(f("opinc"), assets_safe.clone()), 3).alias("opprofitassetscagr3y"),
                pct_change(safe_div(f("rnd"), f("fxusd")), 4).alias("rnd1y"),
                cagr(safe_div(f("rnd"), f("fxusd")), 3).alias("rndcagr3y"),
            ]
        });

    fundamental_rankings(fs_score(lf))
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
        let out = with_period_ranks(
            df.lazy(),
            &[
                period_rank(col("value"), "higher_pct", true, true),
                period_rank(col("value"), "lower_rank", false, false),
            ],
        )
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
    fn safe_division_turns_zero_denominator_into_null() -> PolarsResult<()> {
        let df = df![
            "numerator" => [10.0, 10.0],
            "denominator" => [2.0, 0.0],
        ]?;
        let out = df
            .lazy()
            .select([safe_div(col("numerator"), col("denominator")).alias("value")])
            .collect()?;

        assert_eq!(Vec::from(out.column("value")?.f64()?), &[Some(5.0), None]);

        Ok(())
    }

    #[test]
    fn cagr_fallback_uses_5y_then_3y_then_1y() -> PolarsResult<()> {
        let df = df![
            "cagr5y" => [Some(5.0), None, None, None],
            "cagr3y" => [Some(3.0), Some(3.0), None, None],
            "growth1y" => [Some(1.0), Some(1.0), Some(1.0), None],
        ]?;
        let out = df
            .lazy()
            .select([fallback(f("cagr5y"), fallback(f("cagr3y"), f("growth1y"))).alias("value")])
            .collect()?;

        assert_eq!(
            Vec::from(out.column("value")?.f64()?),
            &[Some(5.0), Some(3.0), Some(1.0), None]
        );

        Ok(())
    }

    #[test]
    fn weighted_score_requires_50_percent_weight_coverage() -> PolarsResult<()> {
        let df = df![
            "a" => [Some(80i32), Some(80), None, None],
            "b" => [Some(100i32), None, Some(100), None],
            "c" => [None, Some(100i32), None, Some(100i32)],
        ]?;
        let out = df
            .lazy()
            .select([weighted_percentile_score(
                &[("a", 0.4), ("b", 0.4), ("c", 0.2)],
                "score",
            )])
            .collect()?;

        assert_eq!(
            Vec::from(out.column("score")?.i32()?),
            &[Some(90), Some(87), None, None]
        );

        Ok(())
    }

    #[test]
    fn equal_weight_value_score_computes_with_two_of_three_components() -> PolarsResult<()> {
        let df = df![
            "ev_ebitda" => [Some(90i32), Some(90), Some(90)],
            "ev_sales" => [Some(60i32), None, Some(60)],
            "ebitda_peg" => [None, None, Some(30i32)],
        ]?;
        let out = df
            .lazy()
            .select([weighted_percentile_score(
                &[
                    ("ev_ebitda", 1.0 / 3.0),
                    ("ev_sales", 1.0 / 3.0),
                    ("ebitda_peg", 1.0 / 3.0),
                ],
                "valuerank",
            )])
            .collect()?;

        assert_eq!(
            Vec::from(out.column("valuerank")?.i32()?),
            &[Some(75), None, Some(60)]
        );

        Ok(())
    }
}
