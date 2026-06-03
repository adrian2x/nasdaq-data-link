//! Quarterly fundamentals preparation.
//!
//! Transforms ARQ financial-statement rows into study-style quarterly
//! fundamental-momentum signals.

use polars::prelude::*;

const REPORT_PERIOD: &str = "reportperiod";
const PRICE_LOOKBACK_DAYS: i32 = 10;

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

fn eps_yield_yoy_delta(eps: Expr, close: Expr) -> Expr {
    let eps = finite(eps);
    let close = finite(close);
    let eps_delta = (eps.clone() - eps.shift(lit(4))).over([col("ticker")]);
    let scaled = safe_div(eps_delta, close.clone()) * lit(100.0);

    round(
        when(close.gt(lit(0.0)))
            .then(finite(scaled))
            .otherwise(lit(NULL)),
        3,
    )
}

fn join_report_closes(lf: LazyFrame, price_closes: LazyFrame) -> LazyFrame {
    let price_closes = price_closes
        .select([
            col("ticker"),
            col("date").cast(DataType::Date).alias(REPORT_PERIOD),
            col("close").cast(DataType::Float64).alias("__report_close"),
        ])
        .sort(["ticker", REPORT_PERIOD], Default::default());

    lf.sort(["ticker", REPORT_PERIOD], Default::default()).join(
        price_closes,
        [col(REPORT_PERIOD)],
        [col(REPORT_PERIOD)],
        JoinArgs::new(JoinType::AsOf(AsOfOptions {
            strategy: AsofStrategy::Backward,
            tolerance: Some(AnyValue::Int32(PRICE_LOOKBACK_DAYS)),
            tolerance_str: None,
            left_by: Some(vec!["ticker".into()]),
            right_by: Some(vec!["ticker".into()]),
            allow_eq: true,
            check_sortedness: false,
        })),
    )
}

/// Adds study-style quarterly signals from ARQ fundamentals.
pub fn compute_quarterly_fundamental_metrics(lf: LazyFrame, price_closes: LazyFrame) -> LazyFrame {
    join_report_closes(lf, price_closes)
        .sort(["ticker", "calendardate"], Default::default())
        .with_columns({
            let eps = f("epsadj");
            [
                pct_change(col("sps"), 4).alias("revenueqtryoy"),
                pct_change(eps.clone(), 4).alias("epsqtryoy"),
                eps_yield_yoy_delta(eps, f("__report_close")).alias("epsqtrchg"),
            ]
        })
        .with_columns({
            let epsqtr1qagoyoy = f("epsqtrchg").shift(lit(1)).over([col("ticker")]);
            let epsqtr2qagoyoy = f("epsqtrchg").shift(lit(2)).over([col("ticker")]);
            [
                (f("revenueqtryoy") - f("revenueqtryoy").shift(lit(1)).over([col("ticker")]))
                    .alias("revenueqtraccel"),
                epsqtr1qagoyoy.clone().alias("epsqtr1qagoyoy"),
                ((f("epsqtrchg") - epsqtr2qagoyoy.clone()) / lit(2.0)).alias("epsqtraccel"),
                epsqtr2qagoyoy.alias("epsqtr2qagoyoy"),
                (f("grossmargin") - f("grossmargin").shift(lit(4)).over([col("ticker")]))
                    .alias("grossmarginqtrexp"),
                (f("ebitdamargin") - f("ebitdamargin").shift(lit(4)).over([col("ticker")]))
                    .alias("ebitdamarginqtrexp"),
                (f("roic") - f("roic").shift(lit(4)).over([col("ticker")])).alias("roicqtrexp"),
            ]
        })
        .drop(["__report_close"])
}

/// Normalizes raw ARQ fundamentals for the persisted `financials_quarter` table.
pub fn adjust_fundamentals_quarter(lf: LazyFrame) -> LazyFrame {
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
        pct(f("roe")).alias("roe"),
        pct(f("roa")).alias("roa"),
        pct(f("roic")).alias("roic"),
        pct(f("grossmargin")).alias("grossmargin"),
        pct(f("netmargin")).alias("netmargin"),
        pct(f("ebitdamargin")).alias("ebitdamargin"),
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
        (f("debt") - f("cashneq")).alias("netdebt"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eps_yield_delta_scales_same_quarter_delta_by_price() -> PolarsResult<()> {
        let df = df![
            "ticker" => ["A", "A", "A", "A", "A"],
            "eps" => [1.0, 2.0, 3.0, 4.0, 1.5],
            "close" => [100.0, 100.0, 100.0, 100.0, 50.0],
        ]?;
        let out = df
            .lazy()
            .select([eps_yield_yoy_delta(col("eps"), col("close")).alias("delta")])
            .collect()?;

        assert_eq!(
            Vec::from(out.column("delta")?.f32()?),
            &[None, None, None, None, Some(1.0)]
        );
        Ok(())
    }
}
