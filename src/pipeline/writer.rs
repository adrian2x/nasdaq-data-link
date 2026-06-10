//! Builds derived output tables and Arrow exports.
use anyhow::Result;
use polars::prelude::*;

use crate::filetools::{df_to_duckdb, lf_to_duckdb, read_csv, write_arrow_files};
use crate::indicators::{with_hurst, HurstConfig};

use super::{
    adjust_financials_ttm, adjust_financials_qtr, build_company_snapshot,
    financials_ttm_metrics, financials_qtr_metrics, load_prices_adjusted,
    resample_ohlcv, technical_indicators_daily, update_insiders,
};
use crate::ui::{spinner, timed};

/// Writes the stock_prices table with price history and derived indicators
fn write_stocks(compute_hurst: bool) -> Result<DataFrame> {
    // load_prices_adjusted guarantees rows sorted by (ticker, date).
    // Downstream `over("ticker")` rolling computations depend on this.
    let priceslf = load_prices_adjusted()?;
    let priceslf = technical_indicators_daily(priceslf);
    let pricesdf = spinner!("applying indicators", priceslf.collect())?;

    let mut pricesdf = if compute_hurst {
        // Computes the Hurst exponent and adds the "hurst" column to the df.
        timed!(
            "computing hurst",
            with_hurst(pricesdf, HurstConfig::default())
        )?
    } else {
        pricesdf
    };

    // Writes to the "stock_prices" table
    timed!(
        "writing stock_prices",
        df_to_duckdb(&mut pricesdf, "stock_prices")
    )?;

    Ok(pricesdf)
}

/// Writes weekly stock prices with the same indicators as daily prices.
fn write_stocks_weekly(compute_hurst: bool) -> Result<DataFrame> {
    let priceslf = resample_ohlcv(load_prices_adjusted()?, "1w");

    let priceslf = technical_indicators_daily(priceslf);
    let pricesdf = spinner!("applying weekly indicators", priceslf.collect())?;

    let mut pricesdf = if compute_hurst {
        timed!(
            "computing weekly hurst",
            with_hurst(
                pricesdf,
                HurstConfig {
                    window: 100,
                    ..HurstConfig::default()
                },
            )
        )?
    } else {
        pricesdf
    };

    timed!(
        "writing stock_prices_weekly",
        df_to_duckdb(&mut pricesdf, "stock_prices_weekly")
    )?;

    Ok(pricesdf)
}

/// Writes the financials_ttm table with adjusted fundamentals and derived metrics.
fn write_financials_ttm() -> Result<LazyFrame> {
    let lf = read_csv("financials_ttm", None)?;
    let lf = adjust_financials_ttm(lf);
    let lf = financials_ttm_metrics(lf);

    timed!(
        "writing financials_ttm",
        lf_to_duckdb(lf.clone(), "financials_ttm")
    )?;

    Ok(lf)
}

/// Writes the financials_qtr table from ARQ fundamentals.
fn write_financials_qtr(prices: &DataFrame) -> Result<LazyFrame> {
    let lf = read_csv("financials_qtr", None)?;
    let lf = adjust_financials_qtr(lf);
    let price_closes = prices.select(["ticker", "date", "close"])?;
    let lf = financials_qtr_metrics(lf, price_closes.lazy());

    timed!(
        "writing financials_qtr",
        lf_to_duckdb(lf.clone(), "financials_qtr")
    )?;

    Ok(lf)
}

/// Writes the companies table including latest price and fundamental data
fn write_companies(
    prices: DataFrame,
    financials_ttm: LazyFrame,
    financials_qtr: LazyFrame,
) -> Result<()> {
    let companies = read_csv("companies", None)?;
    let companies =
        build_company_snapshot(companies, prices.lazy(), financials_ttm, financials_qtr);

    timed!("writing companies", lf_to_duckdb(companies, "companies"))?;

    Ok(())
}

/// Writes the insider transactions table
fn write_insiders() -> Result<()> {
    let lf = read_csv("insiders", Some(&[("formtype", DataType::String)]))?;
    let lf = update_insiders(lf);

    timed!("writing insiders", lf_to_duckdb(lf, "insiders"))?;

    Ok(())
}

/// Builds all output tables from downloaded datasets.
///
/// # Failure
/// Returns an error if any pipeline stage or write step fails.
pub fn run_writer(export_arrow: bool, write_weekly: bool, compute_hurst: bool) -> Result<()> {
    let prices = write_stocks(compute_hurst)?;
    if write_weekly {
        write_stocks_weekly(compute_hurst)?;
    }
    let financials = write_financials_ttm()?;
    let financials_qtr = write_financials_qtr(&prices)?;
    write_companies(prices, financials, financials_qtr)?;
    write_insiders()?;
    if export_arrow {
        write_arrow_files()?;
    }
    Ok(())
}
