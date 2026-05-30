//! Builds derived output tables and Arrow exports.
use anyhow::Result;
use polars::prelude::*;

use crate::filetools::{df_to_duckdb, lf_to_duckdb, read_csv, write_arrow_files};
use crate::fractaltools::{HurstConfig, with_hurst};
use crate::pipeline::{
    adjust_fundamentals, build_company_snapshot, load_prices_adjusted, resample_ohlcv,
    technical_indicators_daily, update_insiders,
};
use crate::ui::{spinner, timed};

/// Writes the stock_prices table with price history and derived indicators
fn write_stocks() -> Result<LazyFrame> {
    // load_prices_adjusted guarantees rows sorted by (ticker, date).
    // Downstream `over("ticker")` rolling computations depend on this.
    let priceslf = load_prices_adjusted()?;
    let priceslf = technical_indicators_daily(priceslf);
    let pricesdf = spinner!("applying indicators", priceslf.collect())?;

    // Computes the Hurst exponent and adds the "hurst" column to the df
    let mut pricesdf = timed!(
        "computing hurst",
        with_hurst(pricesdf, HurstConfig::default())
    )?;

    // Writes to the "stock_prices" table
    timed!(
        "writing stock_prices",
        df_to_duckdb(&mut pricesdf, "stock_prices")
    )?;

    // Companies only need the current price/indicator row. Returning this
    // small snapshot avoids carrying the full daily history into the company
    // join after `stock_prices` has already been persisted.
    let latest_prices = timed!(
        "preparing latest stock snapshot",
        pricesdf
            .lazy()
            .group_by_stable([col("ticker")])
            .agg([all().last()])
            .collect()
    )?;

    Ok(latest_prices.lazy())
}

/// Writes weekly stock prices with the same indicators as daily prices.
fn write_stocks_weekly() -> Result<()> {
    let priceslf = resample_ohlcv(load_prices_adjusted()?, "1w");

    let priceslf = technical_indicators_daily(priceslf);
    let pricesdf = spinner!("applying weekly indicators", priceslf.collect())?;

    let mut pricesdf = timed!(
        "computing weekly hurst",
        with_hurst(
            pricesdf,
            HurstConfig {
                window: 100,
                ..HurstConfig::default()
            },
        )
    )?;

    timed!(
        "writing stock_prices_weekly",
        df_to_duckdb(&mut pricesdf, "stock_prices_weekly")
    )?;

    Ok(())
}

/// Writes the financials_ttm table using
fn write_financials() -> Result<LazyFrame> {
    let financialslf = read_csv("financials_ttm", None)?;
    let financialslf = adjust_fundamentals(financialslf);

    let mut financialsdf = timed!("applying fundamentals", financialslf.collect())?;

    timed!(
        "writing financials_ttm",
        df_to_duckdb(&mut financialsdf, "financials_ttm")
    )?;

    Ok(financialsdf.lazy())
}

/// Writes the companies table including latest price and fundamental data
fn write_companies(prices: LazyFrame, financials: LazyFrame) -> Result<()> {
    let companieslf = read_csv("companies", None)?;
    let companieslf = build_company_snapshot(companieslf, prices, financials);

    timed!("writing companies", lf_to_duckdb(companieslf, "companies"))?;

    Ok(())
}

/// Writes the insider transactions table
fn write_insiders() -> Result<()> {
    let insiderslf = read_csv("insiders", Some(&[("formtype", DataType::String)]))?;
    let insiderslf = update_insiders(insiderslf);

    timed!("writing insiders", lf_to_duckdb(insiderslf, "insiders"))?;

    Ok(())
}

/// Builds all output tables from downloaded datasets.
///
/// # Failure
/// Returns an error if any pipeline stage or write step fails.
pub fn run_writer(export_arrow: bool) -> Result<()> {
    let prices = write_stocks()?;
    write_stocks_weekly()?;
    let financials = write_financials()?;
    write_companies(prices, financials)?;
    write_insiders()?;
    if export_arrow {
        write_arrow_files()?;
    }
    Ok(())
}
