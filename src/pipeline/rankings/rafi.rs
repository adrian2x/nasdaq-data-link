//! RAFI/Fundsmith-inspired quality and value rankings.

use polars::prelude::*;

use super::{
    DEFAULT_MIN_SCORE_WEIGHT_COVERAGE, cagr, f, pct, ratio, safe_div, snapshot_percentile_expr,
    weighted_rank_score,
};

const VALUE_MIN_SCORE_WEIGHT_COVERAGE: f64 = 0.3;

const QUALITY_FEATURE_COLUMNS: &[&str] = &[
    "__q_ebitmargin_metric",
    "__q_revenuecagr5y_metric",
    "__q_salesassetscagr3y_metric",
    "__q_grossprofitassetscagr3y_metric",
    "__q_opprofitassetscagr3y_metric",
    "__q_rndintensity_metric",
    "__q_rndcagr3y_metric",
    "__q_cfc_metric",
    "__q_fcfassets_metric",
    "__q_cashaccruals_metric",
    "__q_icr_metric",
    "__q_cashassets_metric",
    "__q_intangiblesassets_metric",
    "__q_gpassets_metric",
];

const QUALITY_RANK_COLUMNS: &[&str] = &[
    "__q_roic",
    "__q_roce",
    "__q_gpassets",
    "__q_roe",
    "__q_grossmargin",
    "__q_ebitmargin",
    "__q_revenuecagr3y",
    "__q_revenuecagr5y",
    "__q_salesassetscagr3y",
    "__q_grossprofitassetscagr3y",
    "__q_opprofitassetscagr3y",
    "__q_rndintensity",
    "__q_rndcagr3y",
    "__q_cfc",
    "__q_fcfassets",
    "__q_cashaccruals",
    "__q_icr",
    "__q_de",
    "__q_fsscore",
    "__q_cashassets",
    "__q_intangiblesassets",
    "__q_bbyield",
];

const VALUE_RANK_COLUMNS: &[&str] = &["__v_ev_ebitda", "__v_ev_sales", "__v_ebitda_peg"];

const VALUE_FEATURE_COLUMNS: &[&str] = &[
    "__v_ev_ebitda_metric",
    "__v_ev_sales_metric",
    "__v_ebitda_peg_metric",
];

/// Returns the RAFI helper columns that must survive until snapshot ranking.
pub(in crate::pipeline) fn rafi_feature_columns() -> &'static [&'static str] {
    QUALITY_FEATURE_COLUMNS
}

/// Adds the history-dependent RAFI helper features used by snapshot ranks.
///
/// The quality rank consumes durable-business inputs: high returns on capital,
/// strong margins, cash conversion, accrual quality, low leverage, liquidity,
/// and reinvestment signals. Value-rank multiples depend on EV, so those
/// helper metrics are built after the latest price row is joined.
///
/// Acronyms used below:
/// - EV: Enterprise Value, roughly market capitalization plus net debt.
/// - EBITDA: Earnings Before Interest, Taxes, Depreciation, and Amortization.
/// - PEG: Price/Earnings-to-Growth; here adapted to `EV / EBITDA` so the
///   numerator and denominator both operate at enterprise value scale.
/// - ROE: Return on Equity. ROIC: Return on Invested Capital.
/// - CFC: Cash Flow Conversion, `NCFO / operating income`.
/// - ICR: Interest Coverage Ratio, `EBIT / interest expense`.
pub(in crate::pipeline) fn add_rafi_features(lf: LazyFrame) -> LazyFrame {
    lf.with_columns({
        let assets_safe = when(f("assets").eq(lit(0.0)))
            .then(lit(NULL))
            .otherwise(f("assets"));
        let revenue_safe = when(f("revenue").eq(lit(0.0)))
            .then(lit(NULL))
            .otherwise(f("revenue"));
        [
            pct(safe_div(f("ebitadj"), f("revenue"))).alias("__q_ebitmargin_metric"),
            cagr(col("sps"), 5).alias("__q_revenuecagr5y_metric"),
            cagr(f("assetturnover"), 3).alias("__q_salesassetscagr3y_metric"),
            cagr(safe_div(f("gp"), assets_safe.clone()), 3)
                .alias("__q_grossprofitassetscagr3y_metric"),
            cagr(safe_div(f("opinc"), assets_safe.clone()), 3)
                .alias("__q_opprofitassetscagr3y_metric"),
            pct(safe_div(f("rnd"), revenue_safe)).alias("__q_rndintensity_metric"),
            cagr(f("rnd"), 3).alias("__q_rndcagr3y_metric"),
            ratio(f("ncfo"), f("opinc")).alias("__q_cfc_metric"),
            ratio(f("fcf"), assets_safe.clone()).alias("__q_fcfassets_metric"),
            ratio(f("ncfo") - f("ebitadj"), assets_safe.clone()).alias("__q_cashaccruals_metric"),
            ratio(f("ebitadj"), f("intexp")).alias("__q_icr_metric"),
            ratio(f("cashneq"), assets_safe.clone()).alias("__q_cashassets_metric"),
            ratio(f("gp"), assets_safe.clone()).alias("__q_gpassets_metric"),
            ratio(f("intangibles"), assets_safe).alias("__q_intangiblesassets_metric"),
        ]
    })
}

/// Adds RAFI quality and value ranks to the current company snapshot.
pub(in crate::pipeline) fn add_rafi_snapshot_ranks(snapshot: LazyFrame) -> LazyFrame {
    add_value_snapshot_rank(add_quality_snapshot_rank(snapshot))
}

fn add_quality_snapshot_rank(snapshot: LazyFrame) -> LazyFrame {
    // Quality rank weights.
    //
    // Profitability and returns on capital carry the most weight because high
    // returns sustained over time are the central "quality compounder" idea.
    // Growth is included, but only alongside cash conversion, accrual quality,
    // interest coverage, leverage, liquidity, and reinvestment proxies so the
    // rank does not simply become a growth-at-any-price score.
    let quality_weighted = [
        ("__q_roic", 0.08),
        ("__q_roce", 0.08),
        ("__q_gpassets", 0.08),
        ("__q_roe", 0.03),
        ("__q_grossmargin", 0.06),
        ("__q_ebitmargin", 0.06),
        ("__q_revenuecagr3y", 0.04),
        ("__q_revenuecagr5y", 0.04),
        ("__q_salesassetscagr3y", 0.04),
        ("__q_grossprofitassetscagr3y", 0.04),
        ("__q_opprofitassetscagr3y", 0.04),
        ("__q_rndintensity", 0.025),
        ("__q_rndcagr3y", 0.025),
        ("__q_cfc", 0.05),
        ("__q_fcfassets", 0.05),
        ("__q_cashaccruals", 0.05),
        ("__q_icr", 0.04),
        ("__q_de", 0.04),
        ("__q_fsscore", 0.04),
        ("__q_cashassets", 0.03),
        ("__q_intangiblesassets", 0.03),
        ("__q_bbyield", 0.03),
    ];

    let drop_columns = QUALITY_RANK_COLUMNS
        .iter()
        .chain(QUALITY_FEATURE_COLUMNS.iter())
        .copied()
        .collect::<Vec<_>>();

    snapshot
        .with_columns([
            snapshot_percentile_expr("roic", true, "__q_roic"),
            snapshot_percentile_expr("roce", true, "__q_roce"),
            snapshot_percentile_expr("roe", true, "__q_roe"),
            snapshot_percentile_expr("grossmargin", true, "__q_grossmargin"),
            snapshot_percentile_expr("__q_ebitmargin_metric", true, "__q_ebitmargin"),
            snapshot_percentile_expr("revenuecagr3y", true, "__q_revenuecagr3y"),
            snapshot_percentile_expr("__q_revenuecagr5y_metric", true, "__q_revenuecagr5y"),
            snapshot_percentile_expr(
                "__q_salesassetscagr3y_metric",
                true,
                "__q_salesassetscagr3y",
            ),
            snapshot_percentile_expr(
                "__q_grossprofitassetscagr3y_metric",
                true,
                "__q_grossprofitassetscagr3y",
            ),
            snapshot_percentile_expr(
                "__q_opprofitassetscagr3y_metric",
                true,
                "__q_opprofitassetscagr3y",
            ),
            snapshot_percentile_expr("__q_rndintensity_metric", true, "__q_rndintensity"),
            snapshot_percentile_expr("__q_rndcagr3y_metric", true, "__q_rndcagr3y"),
            snapshot_percentile_expr("__q_cfc_metric", true, "__q_cfc"),
            snapshot_percentile_expr("__q_fcfassets_metric", true, "__q_fcfassets"),
            snapshot_percentile_expr("__q_cashaccruals_metric", true, "__q_cashaccruals"),
            snapshot_percentile_expr("__q_icr_metric", true, "__q_icr"),
            snapshot_percentile_expr("de", false, "__q_de"),
            snapshot_percentile_expr("fsscore", true, "__q_fsscore"),
            snapshot_percentile_expr("__q_cashassets_metric", true, "__q_cashassets"),
            snapshot_percentile_expr(
                "__q_intangiblesassets_metric",
                true,
                "__q_intangiblesassets",
            ),
            snapshot_percentile_expr("__q_gpassets_metric", true, "__q_gpassets"),
            snapshot_percentile_expr("bbyield", true, "__q_bbyield"),
        ])
        .with_column(weighted_rank_score(
            &quality_weighted,
            "qualrank",
            DEFAULT_MIN_SCORE_WEIGHT_COVERAGE,
        ))
        .drop(drop_columns)
}

fn add_value_snapshot_rank(snapshot: LazyFrame) -> LazyFrame {
    // Guard enterprise-value multiples. Negative EV, negative EBITDA, and
    // negative sales do not represent meaningful cheapness in standard multiple
    // analysis, so they rank as null.
    let ev_safe = when(f("ev").gt(lit(0.0)))
        .then(f("ev"))
        .otherwise(lit(NULL));
    let ebitda_safe = when(f("ebitda").gt(lit(0.0)))
        .then(f("ebitda"))
        .otherwise(lit(NULL));
    let sales_safe = when(f("revenue").gt(lit(0.0)))
        .then(f("revenue"))
        .otherwise(lit(NULL));
    let ev_ebitda = safe_div(ev_safe.clone(), ebitda_safe);
    let ebitda_growth = f("ebitda1y");
    let ebitda_peg = when(
        f("ev")
            .gt(lit(0.0))
            .and(f("ebitda").gt(lit(0.0)))
            .and(ebitda_growth.clone().is_not_null()),
    )
    .then(
        when(ebitda_growth.clone().gt(lit(0.0)))
            .then(safe_div(ev_ebitda.clone(), ebitda_growth))
            .otherwise(lit(1.0e9)),
    )
    .otherwise(lit(NULL));
    // Value rank weights.
    //
    // Equal-weighting keeps the score robust. EV/EBITDA captures operating
    // profit yield, EV/Sales captures revenue cheapness, and the modified PEG
    // leg avoids giving the best rank to statistically cheap businesses whose
    // EBITDA is not growing.
    let value_weighted = [
        ("__v_ev_ebitda", 1.0 / 3.0),
        ("__v_ev_sales", 1.0 / 3.0),
        ("__v_ebitda_peg", 1.0 / 3.0),
    ];

    let drop_columns = VALUE_RANK_COLUMNS
        .iter()
        .chain(VALUE_FEATURE_COLUMNS.iter())
        .copied()
        .collect::<Vec<_>>();

    snapshot
        .with_columns([
            ev_ebitda.alias("__v_ev_ebitda_metric"),
            safe_div(ev_safe, sales_safe).alias("__v_ev_sales_metric"),
            ebitda_peg.alias("__v_ebitda_peg_metric"),
        ])
        .with_columns([
            snapshot_percentile_expr("__v_ev_ebitda_metric", false, "__v_ev_ebitda"),
            snapshot_percentile_expr("__v_ev_sales_metric", false, "__v_ev_sales"),
            snapshot_percentile_expr("__v_ebitda_peg_metric", false, "__v_ebitda_peg"),
        ])
        .with_column(weighted_rank_score(
            &value_weighted,
            "valuerank",
            VALUE_MIN_SCORE_WEIGHT_COVERAGE,
        ))
        .with_column(
            col("valuerank")
                .round(2)
                .cast(DataType::Float32)
                .alias("valuerank"),
        )
        .drop(drop_columns)
}
