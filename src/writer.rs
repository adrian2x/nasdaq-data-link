use crate::indicators::{ewma_vol, realized_volatility, yang_zhang};
use anyhow::Result;
use polars::prelude::*;

use crate::filetools::{df_to_duckdb, lf_to_duckdb, read_csv, write_arrow_files};
use crate::fractaltools::{HurstConfig, with_hurst};
use crate::pipeline::{
    adjust_fundamentals, build_company_snapshot, load_prices_adjusted, technical_indicators_daily,
    update_insiders,
};
use crate::ui::with_spinner;

fn write_stocks() -> Result<LazyFrame> {
    // load_prices_adjusted guarantees rows sorted by (ticker, date).
    // Downstream `over("ticker")` rolling computations depend on this.
    let prices = load_prices_adjusted()?;

    let df = technical_indicators_daily(prices);
    // let df = realized_volatility(df, "close", 252, &[5, 21, 63, 252]);
    // let df = yang_zhang(df, 21, 252.0)?;
    // let df = ewma_vol(df, 0.94, 252.0)?;

    let df = with_spinner("applying indicators", || {
        df.collect().map_err(anyhow::Error::from)
    })?;

    let mut df = with_spinner("computing hurst", || {
        with_hurst(df, HurstConfig::default()).map_err(anyhow::Error::from)
    })?;

    // with_hurst already returns a materialized DataFrame — write it
    // directly rather than re-lazying and forcing another collect.
    with_spinner("writing stock_prices", || {
        df_to_duckdb(&mut df, "stock_prices")
    })?;

    Ok(df.lazy())
}

fn write_financials() -> Result<LazyFrame> {
    let df = read_csv("financials_ttm", None)?;

    let financials_adj = adjust_fundamentals(df);
    with_spinner("writing financials_ttm", || {
        lf_to_duckdb(financials_adj.clone(), "financials_ttm")
    })?;
    Ok(financials_adj)
}

fn write_companies(prices: LazyFrame, financials: LazyFrame) -> Result<()> {
    let companies = read_csv("companies", None)?;
    let snapshot = build_company_snapshot(companies, prices, financials);

    with_spinner("writing companies", || lf_to_duckdb(snapshot, "companies"))?;

    Ok(())
}

fn write_insiders() -> Result<()> {
    let df = read_csv("insiders", Some(&[("formtype", DataType::String)]))?;

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
