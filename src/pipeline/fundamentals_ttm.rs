//! TTM fundamentals preparation.
//!
//! Transforms MRT financial-statement rows into TTM analysis columns. Ranking
//! formulas live in the sibling `rankings` module.

use polars::prelude::*;

use crate::indicators::fs_score;

use super::rankings::add_rafi_features;

const REPORT_PERIOD: &str = "reportperiod";

fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

fn round(expr: Expr, dp: u32) -> Expr {
    expr.round(dp).cast(DataType::Float32)
}

fn finite(expr: Expr) -> Expr {
    when(expr.clone().is_finite())
        .then(expr)
        .otherwise(lit(NULL))
}

fn safe_div(numerator: Expr, denominator: Expr) -> Expr {
    finite(numerator / denominator)
}

fn pct(fraction: Expr) -> Expr {
    round(finite(lit(100.0) * fraction), 1)
}

fn pct_change(source: Expr, period: i64) -> Expr {
    let source = finite(source);
    let start = source.clone().shift(lit(period));
    let growth = safe_div(source - start.clone(), start.abs()) * lit(100.0);
    round(growth.over([col("ticker")]), 1)
}

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

/// Normalizes raw MRT fundamentals for the persisted `financials_ttm` table.
pub fn adjust_financials_ttm(lf: LazyFrame) -> LazyFrame {
    lf.with_columns([
        col("calendardate").cast(DataType::Date),
        col(REPORT_PERIOD).cast(DataType::Date),
    ])
    .drop(["dimension", "lastupdated"])
    .with_columns([
        (f("revenue") / f("fxusd")).alias("revenue"),
        (f("cor") / f("fxusd")).alias("cor"),
        (f("sgna") / f("fxusd")).alias("sgna"),
        (f("rnd") / f("fxusd")).alias("rnd"),
        (f("opex") / f("fxusd")).alias("opex"),
        (f("intexp") / f("fxusd")).alias("intexp"),
        (f("taxexp") / f("fxusd")).alias("taxexp"),
        (f("netincdis") / f("fxusd")).alias("netincdis"),
        (f("consolinc") / f("fxusd")).alias("consolinc"),
        (f("netincnci") / f("fxusd")).alias("netincnci"),
        (f("netinc") / f("fxusd")).alias("netinc"),
        (f("prefdivis") / f("fxusd")).alias("prefdivis"),
        (f("netinccmn") / f("fxusd")).alias("netinccmn"),
        (f("eps") / f("fxusd")).alias("eps"),
        (f("epsdil") / f("fxusd")).alias("epsdil"),
        (f("capex") / f("fxusd")).alias("capex"),
        (f("ncfbus") / f("fxusd")).alias("ncfbus"),
        (f("ncfinv") / f("fxusd")).alias("ncfinv"),
        (f("workingcapital") / f("fxusd")).alias("workingcapital"),
        (f("ncff") / f("fxusd")).alias("ncff"),
        (f("ncfdebt") / f("fxusd")).alias("ncfdebt"),
        (f("ncfcommon") / f("fxusd")).alias("ncfcommon"),
        (f("ncfdiv") / f("fxusd")).alias("ncfdiv"),
        (f("ncfi") / f("fxusd")).alias("ncfi"),
        (f("ncfo") / f("fxusd")).alias("ncfo"),
        (f("ncfx") / f("fxusd")).alias("ncfx"),
        (f("ncf") / f("fxusd")).alias("ncf"),
        (f("sbcomp") / f("fxusd")).alias("sbcomp"),
        (f("depamor") / f("fxusd")).alias("depamor"),
        (f("assets") / f("fxusd")).alias("assets"),
        (f("cashneq") / f("fxusd")).alias("cashneq"),
        (f("investments") / f("fxusd")).alias("investments"),
        (f("investmentsc") / f("fxusd")).alias("investmentsc"),
        (f("investmentsnc") / f("fxusd")).alias("investmentsnc"),
        (f("deferredrev") / f("fxusd")).alias("deferredrev"),
        (f("deposits") / f("fxusd")).alias("deposits"),
        (f("ppnenet") / f("fxusd")).alias("ppnenet"),
        (f("inventory") / f("fxusd")).alias("inventory"),
        (f("taxassets") / f("fxusd")).alias("taxassets"),
        (f("receivables") / f("fxusd")).alias("receivables"),
        (f("payables") / f("fxusd")).alias("payables"),
        (f("intangibles") / f("fxusd")).alias("intangibles"),
        (f("liabilities") / f("fxusd")).alias("liabilities"),
        (f("equity") / f("fxusd")).alias("equity"),
        (f("retearn") / f("fxusd")).alias("retearn"),
        (f("accoci") / f("fxusd")).alias("accoci"),
        (f("assetsc") / f("fxusd")).alias("assetsc"),
        (f("assetsnc") / f("fxusd")).alias("assetsnc"),
        (f("liabilitiesc") / f("fxusd")).alias("liabilitiesc"),
        (f("liabilitiesnc") / f("fxusd")).alias("liabilitiesnc"),
        (f("taxliabilities") / f("fxusd")).alias("taxliabilities"),
        (f("debt") / f("fxusd")).alias("debt"),
        (f("debtc") / f("fxusd")).alias("debtc"),
        (f("debtnc") / f("fxusd")).alias("debtnc"),
        (f("ebt") / f("fxusd")).alias("ebt"),
        (f("ebit") / f("fxusd")).alias("ebit"),
        (f("ebitda") / f("fxusd")).alias("ebitda"),
        (f("invcap") / f("fxusd")).alias("invcap"),
        (f("equityavg") / f("fxusd")).alias("equityavg"),
        (f("assetsavg") / f("fxusd")).alias("assetsavg"),
        (f("invcapavg") / f("fxusd")).alias("invcapavg"),
        (f("tangibles") / f("fxusd")).alias("tangibles"),
        (f("fcf") / f("fxusd")).alias("fcf"),
        (f("gp") / f("fxusd")).alias("gp"),
        (f("opinc") / f("fxusd")).alias("opinc"),
        (f("fcfps") / f("fxusd")).alias("fcfps"),
        (f("bvps") / f("fxusd")).alias("bvps"),
        (f("tbvps") / f("fxusd")).alias("tbvps"),
        (f("debt") / f("equity")).round(3).alias("de"),
        pct(f("roe")).alias("roe"),
        pct(f("roa")).alias("roa"),
        pct(f("roic")).alias("roic"),
        pct(f("grossmargin")).alias("grossmargin"),
        pct(f("netmargin")).alias("netmargin"),
        pct(f("ebitdamargin")).alias("ebitdamargin"),
        pct(f("ebit") / f("revenue")).alias("ebitmargin"),
        pct(f("ros")).alias("ros"),
        pct(f("payoutratio")).alias("payoutratio"),
        pct(f("divyield")).alias("divyield"),
        (f("sharesbas") * f("sharefactor")).alias("sharesbas"),
        (f("shareswadil") * f("sharefactor")).alias("sharesdil"),
    ])
    .with_columns([
        (f("netinccmn") - f("netincdis"))
            .alias("netincadj")
            .round(2),
        (f("debt") - f("cashneq")).alias("netdebt").round(2),
    ])
    .with_columns([safe_div(f("netincadj"), f("sharesdil"))
        .alias("epsadj")
        .round(2)])
    .drop([
        "equityusd",
        "epsusd",
        "revenueusd",
        "netinccmnusd",
        "cashnequsd",
        "debtusd",
        "ebitusd",
        "ebitdausd",
    ])
    .sort(["ticker", "calendardate"], Default::default())
}

/// Adds derived TTM fundamentals used by company snapshots and snapshot ranks.
pub fn financials_ttm_metrics(lf: LazyFrame) -> LazyFrame {
    let lf = lf
        .sort(["ticker", "calendardate"], Default::default())
        .with_columns([
            (f("ebit") - f("netincdis")).alias("ebitadj").round(2),
        ])
        .with_columns([
            pct(safe_div(f("ebitadj"), f("assets") - f("liabilitiesc"))).alias("roce"),
            pct(safe_div(f("ebt"), f("revenue"))).alias("pretaxmargin"),
            pct(safe_div(lit(-1.0) * f("ncfcommon"), f("marketcap"))).alias("bbyield"),
            cagr(col("dps"), 5).alias("dpscagr5y"),
        ])
        .with_columns({
            let fcfadj = finite(
                f("ncfo")
                    - f("netincdis")
                    - (f("depamor") + f("sbcomp"))
                    - (f("workingcapital") - f("workingcapital").shift(lit(1)))
                        .over([col("ticker")]),
            );
            [
                fcfadj.clone().alias("fcfadj").round(2),
                safe_div(fcfadj, f("sharesdil")).alias("fcfpsadj").round(2),
            ]
        })
        .with_columns({
            let ebitdaps = safe_div(f("ebitda"), f("sharesdil"));
            let eps = f("epsadj");
            let fcfps = f("fcfpsadj");
            [
                ebitdaps.clone().alias("ebitdaps"),
                pct_change(col("sps"), 4).alias("revenueyoy"),
                pct_change(col("sps"), 4 * 5).alias("revenue5y"),
                cagr(col("sps"), 3).alias("revenuecagr3y"),
                cagr(col("sps"), 5).alias("revenuecagr5y"),
                pct_change(ebitdaps.clone(), 4).alias("ebitda1y"),
                cagr(ebitdaps.clone(), 3).alias("ebitdacagr3y"),
                cagr(ebitdaps, 5).alias("ebitdacagr5y"),
                pct_change(eps.clone(), 4).alias("epsyoy"),
                cagr(eps.clone(), 3).alias("epscagr3y"),
                cagr(eps.clone(), 5).alias("epscagr5y"),
                pct_change(fcfps.clone(), 4).alias("fcfyoy"),
                cagr(fcfps.clone(), 3).alias("fcfcagr3y"),
                cagr(fcfps, 5).alias("fcfcagr5y"),
            ]
        });

    add_rafi_features(fs_score(lf))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
