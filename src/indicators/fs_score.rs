//! Alpha Architect FS-Score.
//!
//! A 10-signal fundamental-quality score (0–10, higher is stronger) from
//! Alpha Architect's "Simple Methods to Improve the Piotroski F-Score"
//! (2015). It builds on Piotroski's 9-signal F-Score, keeping the same
//! three-category structure but tweaking three variables and regrouping:
//!
//!   - Current profitability — are the levels healthy now?
//!   - Stability            — is the balance sheet not deteriorating?
//!   - Recent operational improvements — are the trends improving?
//!
//! Key departures from Piotroski: cash-flow signals use free cash flow over
//! total assets (FCFTA) rather than operating cash flow, to account for the
//! drag of capital expenditure; ΔROA and ΔFCFTA are grouped as operational
//! improvements rather than profitability; and an issuance check is included.
//!
//! Assumes the input frame holds quarterly trailing-twelve-month rows — each
//! row's figures already span a year, and consecutive rows are one quarter
//! apart. The year-over-year tests therefore compare each row against the
//! row 4 quarters back.
//!
//! Self-contained.

use polars::prelude::*;

/// Quarters per year — the row shift for a year-over-year comparison on a
/// quarterly TTM frame.
const YOY_SHIFT: i64 = 4;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Value of `name` one year (4 quarterly TTM rows) ago for the same ticker.
fn lag(name: &str) -> Expr {
    col(name).shift(lit(YOY_SHIFT)).over([col("ticker")])
}

/// A single FS-Score test: a boolean expression scored as 0 or 1, with a
/// missing input counted as 0 (a failed test) rather than nulling the sum.
fn test(cond: Expr) -> Expr {
    cond.fill_null(lit(false)).cast(DataType::Int32)
}

/// Add the Alpha Architect FS-Score column `fsscore` (0–10).
///
/// Requires the input frame sorted by ticker then date — the year-over-year
/// tests depend on row order within each ticker.
pub fn fs_score(lf: LazyFrame) -> LazyFrame {
    // Free cash flow over total assets (FCFTA), current period and one year
    // ago. The denominator is guarded: zero assets yields null, so the
    // dependent tests score 0. AA uses FCF, not operating cash flow, so the
    // measure reflects cash generation net of capital expenditure.
    let assets_safe = when(f("assets").eq(lit(0.0)))
        .then(lit(NULL))
        .otherwise(f("assets"));
    let fcfta = f("fcf") / assets_safe;

    let assets_lag_safe = when(lag("assets").eq(lit(0.0)))
        .then(lit(NULL))
        .otherwise(lag("assets"));
    let fcfta_lag = lag("fcf") / assets_lag_safe;

    let score =
        // --- Current profitability: are the levels healthy today? ---

        // 1. ROA positive — the firm earns a profit on its asset base.
        test(f("roa").gt(lit(0.0)))

        // 2. FCFTA positive — operations generate cash after capex, not just
        //    accounting profit.
        + test(fcfta.clone().gt(lit(0.0)))

        // 3. Accruals: FCFTA exceeds ROA — earnings are backed by cash rather
        //    than accruals. Cash-based profit is higher quality and more
        //    persistent than accrual-based profit.
        + test(fcfta.clone().gt(f("roa")))

        // --- Stability: is the balance sheet not deteriorating? ---

        // 4. Leverage falling — debt-to-equity is lower than a year ago. A
        //    rising debt load is a negative signal on financial health.
        + test(f("de").lt(lag("de")))

        // 5. Liquidity rising — current ratio is higher than a year ago,
        //    indicating a stronger short-term solvency cushion.
        + test(f("currentratio").gt(lag("currentratio")))

        // 6. No net issuance — share count has not risen versus a year ago
        //    (a proxy for AA's net-equity-issuance variable). A distressed
        //    firm raising external equity signals it cannot fund itself
        //    internally.
        + test(f("shareswadil").lt_eq(lag("shareswadil")))

        // --- Recent operational improvements: are the trends improving? ---

        // 7. ROA improving — return on assets is higher than a year ago.
        + test(f("roa").gt(lag("roa")))

        // 8. FCFTA improving — cash return on assets is higher than a year
        //    ago; cash generation is strengthening, not just profit.
        + test(fcfta.gt(fcfta_lag))

        // 9. Gross margin improving — a wider margin signals pricing power
        //    or falling input costs.
        + test(f("grossmargin").gt(lag("grossmargin")))

        // 10. Asset turnover improving — more revenue per unit of assets;
        //     the firm is using its asset base more productively.
        + test(f("assetturnover").gt(lag("assetturnover")));

    lf.with_columns([score.alias("fsscore")])
}
