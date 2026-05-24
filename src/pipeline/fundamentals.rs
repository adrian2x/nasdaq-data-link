//! Fundamentals preparation.
//!
//! Transforms raw quarterly financial-statement rows into analysis columns,
//! including per-share growth metrics. Self-contained — the growth/CAGR
//! expression helpers live here because fundamentals is their only consumer.

use polars::prelude::*;

/// Cast a column to Float64.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Growth of `source` over `period` rows, robust to negative start values.
///
/// `(end - start) / |start| * 100`. For positive `start` this is an ordinary
/// percent change. For negative `start` the sign is correct: a shrinking
/// loss reads positive, a deepening loss negative, a loss-to-profit
/// turnaround strongly positive. Null when `start` is zero or history is
/// insufficient.
pub fn growth_expr(source: Expr, period: i64) -> Expr {
    let start = source.clone().shift(lit(period));
    ((source - start.clone()) / start.abs()) * lit(100.0)
}

/// Compound annual growth rate of `source` over `years` (quarterly cadence:
/// `years * 4` row shift). Returns a true compounding rate in percent.
///
/// Null unless BOTH endpoints are strictly positive — CAGR is undefined when
/// the series is zero or crosses sign, so loss-makers and turnarounds yield
/// null here by design. Use `growth_expr` for a measure that covers those.
pub fn cagr_expr(source: Expr, years: i64) -> Expr {
    let start = source.clone().shift(lit(years * 4));
    let ratio = source.clone() / start.clone();
    let cagr = (ratio.pow(lit(1.0 / years as f64)) - lit(1.0)) * lit(100.0);
    when(start.gt(lit(0.0)).and(source.gt(lit(0.0))))
        .then(cagr)
        .otherwise(lit(NULL))
}

/// Transform raw fundamentals into analysis columns.
pub fn adjust_fundamentals(lf: LazyFrame) -> LazyFrame {
    lf.with_columns([col("calendardate").cast(DataType::Date)])
        .drop(["dimension", "lastupdated"])
        .with_columns([
            (f("sharesbas") * f("sharefactor")).alias("shares"),
            (f("dps") / f("fxusd")).alias("dpsusd"),
            (f("eps") / f("fxusd")).alias("epsusd"),
            (f("ncfo") / f("fxusd")).alias("ncfousd"),
            (f("fcf") / f("fxusd")).alias("fcfusd"),
            (f("assets") - f("liabilities")).alias("equity"),
            ((f("debt") - f("cashneq")) / f("fxusd")).alias("netdebtusd"),
            (lit(100.0) * f("roe")).alias("roe"),
            (lit(100.0) * f("roic")).alias("roic"),
            (lit(100.0) * f("roa")).alias("roa"),
            (f("ncfo") / f("opinc")).alias("cfc"),
            (f("ebit") / f("intexp")).alias("icr"),
            (lit(100.0) * f("ebit") / (f("assets") - f("liabilitiesc"))).alias("roce"),
            ((f("netinc") - f("netincdis")) / f("fxusd")).alias("netincadj"),
            ((lit(-1.0) * f("ncfcommon") / f("fxusd")) / f("marketcap") * lit(100.0))
                .alias("bbyield"),
            (lit(100.0) * f("divyield")).alias("divyield"),
            ((lit(-1.0) * f("ncfdiv") + lit(-1.0) * f("ncfcommon")) / f("fxusd"))
                .alias("shreturnusd"),
            (lit(100.0) * f("gp") / f("revenue")).alias("grossmargin"),
            (lit(100.0) * f("ebitda") / f("revenue")).alias("ebitdamargin"),
            (lit(100.0) * f("ebit") / f("revenue")).alias("ebitmargin"),
        ])
        .with_columns([
            (f("debt") / f("equity")).alias("de"),
            (f("netincadj") / f("shares")).alias("epsadj"),
            (lit(100.0) * f("netincadj") / f("revenueusd")).alias("netmargin"),
            (lit(100.0) * f("shreturnusd") / f("marketcap")).alias("shyield"),
        ])
        .sort(["ticker", "calendardate"], Default::default())
        // `shift`-based metrics require sorted time order.
        .with_columns({
            let fcfadj = (f("ncfo")
                - f("netincdis")
                - f("depamor")
                - (f("workingcapital") - f("workingcapital").shift(lit(1))).over([col("ticker")])
                - f("sbcomp"))
                / f("fxusd");
            [
                fcfadj.clone().alias("fcfadj"),
                (fcfadj / f("shares")).alias("fcfpsadj"),
            ]
        })
        // Per-share growth metrics. The per-share basis (dividing by shares)
        // strips out dilution and acquisition-via-stock effects, isolating
        // organic, per-shareholder growth — see AQR's use of sales-per-share
        // growth as a defined factor. Quarterly cadence: 1y = 4 rows.
        //
        // `{item}{n}y`     total growth, negative-safe (`growth_expr`).
        // `{item}cagr{n}y` true CAGR, null when an endpoint is <= 0
        //                  (`cagr_expr`); no 1y CAGR — it equals 1y growth.
        .with_columns({
            let revenue = f("revenue") / f("shares");
            let ebitda = f("ebitdausd") / f("shares");
            let eps = f("epsadj");
            let fcf = f("fcfpsadj");
            [
                growth_expr(revenue.clone(), 4)
                    .over([col("ticker")])
                    .alias("revenue1y"),
                growth_expr(revenue.clone(), 4 * 3)
                    .over([col("ticker")])
                    .alias("revenue3y"),
                growth_expr(revenue.clone(), 4 * 5)
                    .over([col("ticker")])
                    .alias("revenue5y"),
                cagr_expr(revenue.clone(), 3)
                    .over([col("ticker")])
                    .alias("revenuecagr3y"),
                cagr_expr(revenue, 5)
                    .over([col("ticker")])
                    .alias("revenuecagr5y"),
                growth_expr(ebitda.clone(), 4)
                    .over([col("ticker")])
                    .alias("ebitda1y"),
                growth_expr(ebitda.clone(), 4 * 3)
                    .over([col("ticker")])
                    .alias("ebitda3y"),
                growth_expr(ebitda.clone(), 4 * 5)
                    .over([col("ticker")])
                    .alias("ebitda5y"),
                cagr_expr(ebitda.clone(), 3)
                    .over([col("ticker")])
                    .alias("ebitdacagr3y"),
                cagr_expr(ebitda, 5)
                    .over([col("ticker")])
                    .alias("ebitdacagr5y"),
                growth_expr(eps.clone(), 4)
                    .over([col("ticker")])
                    .alias("eps1y"),
                growth_expr(eps.clone(), 4 * 3)
                    .over([col("ticker")])
                    .alias("eps3y"),
                growth_expr(eps.clone(), 4 * 5)
                    .over([col("ticker")])
                    .alias("eps5y"),
                cagr_expr(eps.clone(), 3)
                    .over([col("ticker")])
                    .alias("epscagr3y"),
                cagr_expr(eps, 5).over([col("ticker")]).alias("epscagr5y"),
                growth_expr(fcf.clone(), 4)
                    .over([col("ticker")])
                    .alias("fcf1y"),
                growth_expr(fcf.clone(), 4 * 3)
                    .over([col("ticker")])
                    .alias("fcf3y"),
                growth_expr(fcf.clone(), 4 * 5)
                    .over([col("ticker")])
                    .alias("fcf5y"),
                cagr_expr(fcf.clone(), 3)
                    .over([col("ticker")])
                    .alias("fcfcagr3y"),
                cagr_expr(fcf, 5).over([col("ticker")]).alias("fcfcagr5y"),
            ]
        })
}
