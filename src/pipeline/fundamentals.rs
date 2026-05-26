//! Fundamentals preparation.
//!
//! Transforms raw quarterly financial-statement rows into analysis columns,
//! including per-share growth metrics. Self-contained — the growth/CAGR
//! expression helpers live here because fundamentals is their only consumer.

use polars::prelude::*;

use crate::indicators::fs_score;

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
    let lf = lf
        .with_columns([col("calendardate").cast(DataType::Date)])
        .drop(["dimension", "lastupdated"])
        .with_columns([
            (f("sharesbas") * f("sharefactor")).alias("sharesbas"),
            (f("shareswadil") * f("sharefactor")).alias("sharesdil"),
            (f("ncfo") / f("fxusd")).alias("ncfousd"),
            (f("fcf") / f("fxusd")).alias("fcfusd"),
            (f("assets") - f("liabilities")).alias("equity"),
            ((f("debt") - f("cashneq")) / f("fxusd")).alias("netdebtusd"),
            (lit(100.0) * f("roe"))
                .round(1)
                .alias("roe")
                .cast(DataType::Float32),
            (lit(100.0) * f("roic"))
                .round(1)
                .alias("roic")
                .cast(DataType::Float32),
            (lit(100.0) * f("roa"))
                .round(1)
                .alias("roa")
                .cast(DataType::Float32),
            (f("ncfo") / f("opinc"))
                .round(3)
                .alias("cfc")
                .cast(DataType::Float32),
            (f("ebit") / f("intexp"))
                .round(3)
                .alias("icr")
                .cast(DataType::Float32),
            (lit(100.0) * f("ebit") / (f("assets") - f("liabilitiesc")))
                .round(1)
                .alias("roce")
                .cast(DataType::Float32),
            ((f("netinccmn") - f("netincdis")) / f("fxusd")).alias("netincadj"),
            ((lit(-1.0) * f("ncfcommon") / f("fxusd")) / f("marketcap") * lit(100.0))
                .round(1)
                .alias("bbyield")
                .cast(DataType::Float32),
            (lit(100.0) * f("divyield"))
                .round(1)
                .alias("divyield")
                .cast(DataType::Float32),
            ((lit(-1.0) * f("ncfdiv") + lit(-1.0) * f("ncfcommon")) / f("fxusd"))
                .alias("shreturnusd"),
            (lit(100.0) * f("gp") / f("revenue"))
                .round(1)
                .alias("grossmargin")
                .cast(DataType::Float32),
            (lit(100.0) * f("ebitda") / f("revenue"))
                .round(1)
                .alias("ebitdamargin")
                .cast(DataType::Float32),
            (lit(100.0) * f("ebit") / f("revenue"))
                .round(1)
                .alias("ebitmargin")
                .cast(DataType::Float32),
        ])
        .with_columns([
            (f("debt") / f("equity"))
                .round(3)
                .alias("de")
                .cast(DataType::Float32),
            (f("netincadj") / f("sharesdil")).alias("epsadj"),
            (lit(100.0) * f("shreturnusd") / f("marketcap"))
                .round(1)
                .alias("shyield")
                .cast(DataType::Float32),
        ])
        .sort(["ticker", "calendardate"], Default::default())
        // Calculate "Owner Earnings" using the Buffett method
        .with_columns({
            let fcfadj = (f("ncfo")
                // Discontinued operations adjustment
                - f("netincdis")
                // Adjust for non-cash charges adjustment
                - (f("depamor") + f("sbcomp"))
                // Add back changes in working capital adjustment
                - (f("workingcapital") - f("workingcapital").shift(lit(1))).over([col("ticker")]))
                / f("fxusd");
            [
                fcfadj.clone().alias("fcfadj"),
                (fcfadj / f("sharesdil")).alias("fcfpsadj"),
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
            let ebitda = f("ebitdausd") / f("sharesdil");
            let eps = f("epsadj");
            let fcfps = f("fcfpsadj");
            [
                growth_expr(col("sps"), 4)
                    .over([col("ticker")])
                    .round(1)
                    .alias("revenue1y")
                    .cast(DataType::Float32),
                growth_expr(col("sps"), 4 * 3)
                    .over([col("ticker")])
                    .round(1)
                    .alias("revenue3y")
                    .cast(DataType::Float32),
                growth_expr(col("sps"), 4 * 5)
                    .over([col("ticker")])
                    .round(1)
                    .alias("revenue5y")
                    .cast(DataType::Float32),
                cagr_expr(col("sps"), 3)
                    .over([col("ticker")])
                    .round(1)
                    .alias("revenuecagr3y")
                    .cast(DataType::Float32),
                cagr_expr(col("sps"), 5)
                    .over([col("ticker")])
                    .round(1)
                    .alias("revenuecagr5y")
                    .cast(DataType::Float32),
                growth_expr(ebitda.clone(), 4)
                    .over([col("ticker")])
                    .round(1)
                    .alias("ebitda1y")
                    .cast(DataType::Float32),
                growth_expr(ebitda.clone(), 4 * 3)
                    .over([col("ticker")])
                    .round(1)
                    .alias("ebitda3y")
                    .cast(DataType::Float32),
                growth_expr(ebitda.clone(), 4 * 5)
                    .over([col("ticker")])
                    .round(1)
                    .alias("ebitda5y")
                    .cast(DataType::Float32),
                cagr_expr(ebitda.clone(), 3)
                    .over([col("ticker")])
                    .round(1)
                    .alias("ebitdacagr3y")
                    .cast(DataType::Float32),
                cagr_expr(ebitda, 5)
                    .over([col("ticker")])
                    .round(1)
                    .alias("ebitdacagr5y")
                    .cast(DataType::Float32),
                growth_expr(eps.clone(), 4)
                    .over([col("ticker")])
                    .round(1)
                    .alias("eps1y")
                    .cast(DataType::Float32),
                growth_expr(eps.clone(), 4 * 3)
                    .over([col("ticker")])
                    .round(1)
                    .alias("eps3y")
                    .cast(DataType::Float32),
                growth_expr(eps.clone(), 4 * 5)
                    .over([col("ticker")])
                    .round(1)
                    .alias("eps5y")
                    .cast(DataType::Float32),
                cagr_expr(eps.clone(), 3)
                    .over([col("ticker")])
                    .round(1)
                    .alias("epscagr3y")
                    .cast(DataType::Float32),
                cagr_expr(eps, 5)
                    .over([col("ticker")])
                    .round(1)
                    .alias("epscagr5y")
                    .cast(DataType::Float32),
                growth_expr(fcfps.clone(), 4)
                    .over([col("ticker")])
                    .round(1)
                    .alias("fcf1y")
                    .cast(DataType::Float32),
                growth_expr(fcfps.clone(), 4 * 3)
                    .over([col("ticker")])
                    .round(1)
                    .alias("fcf3y")
                    .cast(DataType::Float32),
                growth_expr(fcfps.clone(), 4 * 5)
                    .over([col("ticker")])
                    .round(1)
                    .alias("fcf5y")
                    .cast(DataType::Float32),
                cagr_expr(fcfps.clone(), 3)
                    .over([col("ticker")])
                    .round(1)
                    .alias("fcfcagr3y")
                    .cast(DataType::Float32),
                cagr_expr(fcfps, 5)
                    .over([col("ticker")])
                    .round(1)
                    .alias("fcfcagr5y")
                    .cast(DataType::Float32),
            ]
        });
    let lf = fs_score(lf);
    lf
}
