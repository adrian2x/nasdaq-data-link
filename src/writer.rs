use anyhow::Result;
use polars::prelude::*;

use crate::indicators::technical_indicators_daily;
use crate::pipeline::{adjust_fundamentals, adjust_prices, update_insiders};
use crate::filetools::{lf_to_duckdb, scan_dataset, write_arrow_files};
use crate::fractaltools::{HurstConfig, with_hurst};
use crate::ui::with_spinner;

fn write_stocks() -> Result<LazyFrame> {
    let df = scan_dataset(
        "stocks_eod",
        Some(&[
            ("open", DataType::Float64),
            ("high", DataType::Float64),
            ("low", DataType::Float64),
            ("close", DataType::Float64),
            ("closeadj", DataType::Float64),
            ("volume", DataType::Float64),
        ]),
    )?;

    let prices = adjust_prices(df);
    let prices_with_hurst = with_spinner("computing hurst", || {
        let prices_df = prices.clone().collect()?;
        with_hurst(prices_df, HurstConfig::default()).map_err(anyhow::Error::from)
    })?;
    let df = technical_indicators_daily(prices_with_hurst.lazy());
    with_spinner("writing stock_prices", || {
        lf_to_duckdb(df.clone(), "stock_prices")
    })?;

    Ok(df)
}

fn write_financials() -> Result<LazyFrame> {
    let df = scan_dataset("financials_ttm", None)?;

    let financials_adj = adjust_fundamentals(df);
    with_spinner("writing financials_ttm", || {
        lf_to_duckdb(financials_adj.clone(), "financials_ttm")
    })?;
    Ok(financials_adj)
}

fn write_companies(prices: LazyFrame, financials: LazyFrame) -> Result<()> {
    let companies = scan_dataset("companies", None)?;

    let active = companies.filter(col("isdelisted").eq(lit("N"))).drop([
        "table",
        "permaticker",
        "isdelisted",
        "cusips",
        "sicsector",
        "sicindustry",
        "figi",
        "famaindustry",
        "scalemarketcap",
        "scalerevenue",
        "relatedtickers",
        "lastupdated",
        "firstadded",
        "firstpricedate",
        "lastpricedate",
        "firstquarter",
        "lastquarter",
    ]);

    let latest_financials = financials
        .with_column(
            col("calendardate")
                .max()
                .over([col("ticker")])
                .alias("__max_calendardate"),
        )
        .filter(col("calendardate").eq(col("__max_calendardate")))
        .drop(["__max_calendardate"])
        .group_by([col("ticker")])
        .agg([all().last()]);

    let latest_prices = prices
        .with_column(col("date").max().over([lit(1)]).alias("__max_date"))
        .filter(col("date").eq(col("__max_date")))
        .drop(["__max_date"])
        .group_by([col("ticker")])
        .agg([all().last()]);

    let joined = active
        .join(
            latest_financials,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        )
        .join(
            latest_prices,
            [col("ticker")],
            [col("ticker")],
            JoinArgs::new(JoinType::Inner),
        );

    let snapshot = joined.with_columns([
        (col("shares") * col("close")).alias("marketcap"),
        (col("shares") * col("close") + col("netdebtusd")).alias("ev"),
    ]);

    with_spinner("writing companies", || lf_to_duckdb(snapshot, "companies"))?;

    Ok(())
}

fn write_insiders() -> Result<()> {
    let df = scan_dataset("insiders", Some(&[("formtype", DataType::String)]))?;

    let insiders_adj = update_insiders(df);
    with_spinner("writing insiders", || {
        lf_to_duckdb(insiders_adj, "insiders")
    })?;
    Ok(())
}

/// Build all output tables from downloaded datasets.
pub fn run_writer() -> Result<()> {
    let prices = write_stocks()?;
    let financials = write_financials()?;
    write_companies(prices, financials)?;
    write_insiders()?;
    write_arrow_files()?;
    Ok(())
}
