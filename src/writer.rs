use anyhow::Result;
use polars::prelude::*;
use crate::indicators::realized_volatility;

use crate::pipeline::{
    adjust_fundamentals, adjust_prices, build_company_snapshot, technical_indicators_daily,
    update_insiders,
};
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
    let df = realized_volatility(df, "close", 252, &[5, 21, 63, 252]);
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
    let snapshot = build_company_snapshot(companies, prices, financials);

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
