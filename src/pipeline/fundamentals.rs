//! Fundamentals preparation.
//!
//! Transforms raw quarterly financial-statement rows into analysis columns,
//! including per-share growth metrics. Self-contained — the growth helpers
//! live here because fundamentals is their only consumer.

use polars::prelude::*;

use crate::indicators::fs_score;

/// Casts the column `name` to `Float64`.
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

/// Rescales `fraction` to a percentage (`100 * fraction`) and finalizes it at
/// 1 decimal place. The argument may itself be a division; non-finite
/// results (e.g. division by zero) become null.
fn pct(fraction: Expr) -> Expr {
    round(finite(lit(100.0) * fraction), 1)
}

/// Divides `numerator` by `denominator` and finalizes the result at 3 decimal
/// places. Non-finite results (e.g. division by zero) become null.
fn ratio(numerator: Expr, denominator: Expr) -> Expr {
    round(finite(numerator / denominator), 3)
}

/// Computes per-ticker growth of `source` over `period` rows as
/// `(end - start) / |start| * 100`, finalized at 1 decimal place.
///
/// The `|start|` denominator keeps the sign correct for negative starts: a
/// shrinking loss reads positive, a deepening loss negative, and a
/// loss-to-profit turnaround strongly positive. Returns null when `start` is
/// zero or history is insufficient.
fn pct_change(source: Expr, period: i64) -> Expr {
    let start = source.clone().shift(lit(period));
    let growth = ((source - start.clone()) / start.abs()) * lit(100.0);
    round(finite(growth.over([col("ticker")])), 1)
}

/// Computes the per-ticker compound annual growth rate of `source` over
/// `years`, finalized at 1 decimal place. The cadence is quarterly, so the
/// row shift is `years * 4`.
///
/// Returns null unless both endpoints are strictly positive — CAGR is
/// undefined when the series is zero or crosses sign. Use `pct_change` for
/// those cases.
fn cagr(source: Expr, years: i64) -> Expr {
    let start = source.clone().shift(lit(years * 4));
    let r = source.clone() / start.clone();
    let value = (r.pow(lit(1.0 / years as f64)) - lit(1.0)) * lit(100.0);
    let guarded = when(start.gt(lit(0.0)).and(source.gt(lit(0.0))))
        .then(value)
        .otherwise(lit(NULL))
        .over([col("ticker")]);
    round(guarded, 1)
}

/// Transforms raw fundamentals into analysis columns.
pub fn adjust_fundamentals(lf: LazyFrame) -> LazyFrame {
    let lf = lf
        .with_columns([col("calendardate").cast(DataType::Date)])
        .drop(["dimension", "lastupdated"])
        .with_columns([
            // Basic shares are current outstanding shares
            (f("sharesbas") * f("sharefactor")).alias("sharesbas"),
            // Diluted shares also account for all warrants, options, and other shares
            (f("shareswadil") * f("sharefactor")).alias("sharesdil"),
            // Converts NCFO to USD
            (f("ncfo") / f("fxusd")).alias("ncfousd"),
            // Converts FCF to USD
            (f("fcf") / f("fxusd")).alias("fcfusd"),
            // Net Debt = Debt - Cash (in USD)
            ((f("debt") - f("cashneq")) / f("fxusd")).alias("netdebtusd"),
            // ROE as a percent value 0-100
            pct(f("roe")).alias("roe"),
            // ROIC as a percent value 0-100
            pct(f("roic")).alias("roic"),
            // ROA as a percent value 0-100
            pct(f("roa")).alias("roa"),
            // CFC (Terry Smith)
            ratio(f("ncfo"), f("opinc")).alias("cfc"),
            // ICR (Terry Smith)
            ratio(f("ebit"), f("intexp")).alias("icr"),
            // ROCE (Terry Smith)
            pct(f("ebit") / (f("assets") - f("liabilitiesc"))).alias("roce"),
            // Net Income adjusted for discontinued operations
            ((f("netinccmn") - f("netincdis")) / f("fxusd")).alias("netincadj"),
            // Buyback yield
            round(
                (lit(-1.0) * f("ncfcommon") / f("fxusd")) / f("marketcap") * lit(100.0),
                1,
            )
            .alias("bbyield"),
            // Dividend yield
            pct(f("divyield")).alias("divyield"),
            // Caash returned to shareholders via dividends and buybacks
            ((lit(-1.0) * f("ncfdiv") + lit(-1.0) * f("ncfcommon")) / f("fxusd"))
                .alias("shreturnusd"),
            // Gross margin as percent 0-100
            pct(f("gp") / f("revenue")).alias("grossmargin"),
            // EBITDA margin as percent 0-100
            pct(f("ebitda") / f("revenue")).alias("ebitdamargin"),
            // EBIT margin as percent 0-100
            pct(f("ebit") / f("revenue")).alias("ebitmargin"),
        ])
        .with_columns([
            // EPS adjusted for discontinued operations and diluted shares
            (f("netincadj") / f("sharesdil")).alias("epsadj"),
        ])
        .sort(["ticker", "calendardate"], Default::default())
        // FCF adjusted in USD (Owner Earnings by Buffett method)
        // Note - this should also deduct the 5Y avg CAGR figure, to arrive at "maintenance" capex
        // but leaving it out since it can be applied downstream
        .with_columns({
            let fcfadj = (f("ncfo")
                // Minus income from discontinued operations
                - f("netincdis")
                // Adjust for non-cash charges (Depreciation and Stock Based Comp)
                - (f("depamor") + f("sbcomp"))
                // Add back changes in working capital
                - (f("workingcapital") - f("workingcapital").shift(lit(1))).over([col("ticker")]))
                / f("fxusd");
            [
                fcfadj.clone().alias("fcfadj"),
                (fcfadj / f("sharesdil")).alias("fcfpsadj"),
            ]
        })
        .with_columns({
            // Note - per share values adjusts for share issuance and buybacks
            // EBITDA used as a proxy for operating earnings
            let ebitdaps = f("ebitdausd") / f("sharesdil");
            let eps = f("epsadj");
            let fcfps = f("fcfpsadj");
            [
                // Year-over-Year growth rate of sales
                // Note - sps adjusts for share issuance and buybacks
                pct_change(col("sps"), 4).alias("revenue1y"),
                // Total growth of sales over last 3Y
                pct_change(col("sps"), 4 * 3).alias("revenue3y"),
                // Total growth of sales over last 5Y
                pct_change(col("sps"), 4 * 5).alias("revenue5y"),
                // Annualized growth of sales per year over last 3Y
                cagr(col("sps"), 3).alias("revenuecagr3y"),
                // Annualized growth of sales per year over last 5Y
                cagr(col("sps"), 5).alias("revenuecagr5y"),
                // Year-over-Year growth rate of EBITDA
                pct_change(ebitdaps.clone(), 4).alias("ebitda1y"),
                // Annualized growth of EBITDA per year over last 3Y
                cagr(ebitdaps.clone(), 3).alias("ebitdacagr3y"),
                // Annualized growth of EBITDA per year over last 5Y
                cagr(ebitdaps.clone(), 5).alias("ebitdacagr5y"),
                // Year-over-Year growth rate of EPS
                pct_change(eps.clone(), 4).alias("eps1y"),
                // Annualized growth of EPS per year over last 3Y
                cagr(eps.clone(), 3).alias("epscagr3y"),
                // Annualized growth of EPS per year over last 5Y
                cagr(eps.clone(), 5).alias("epscagr5y"),
                // Year-over-Year growth rate of FCF
                pct_change(fcfps.clone(), 4).alias("fcf1y"),
                // Annualized growth of FCF per year over last 3Y
                cagr(fcfps.clone(), 3).alias("fcfcagr3y"),
                // Annualized growth of FCF per year over last 5Y
                cagr(fcfps.clone(), 5).alias("fcfcagr5y"),
            ]
        });

    fs_score(lf)
}
