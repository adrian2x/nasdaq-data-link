//! Builds derived output tables and Arrow exports.
use anyhow::Result;
use polars::prelude::*;

use crate::filetools::{df_to_duckdb, lf_to_duckdb, read_csv, write_arrow_files};
use crate::indicators::{HurstConfig, with_hurst};

use super::{
    adjust_fundamentals, adjust_fundamentals_quarter, build_company_snapshot,
    compute_fundamental_metrics, compute_quarterly_fundamental_metrics, load_prices_adjusted,
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

/// Writes the financials_ttm table using
fn write_financials() -> Result<LazyFrame> {
    let financialslf = read_csv("financials_ttm", None)?;
    let financialslf = adjust_fundamentals(financialslf);

    let mut financialsdf = timed!("adjusting fundamentals", financialslf.collect())?;

    timed!(
        "writing financials_ttm",
        df_to_duckdb(&mut financialsdf, "financials_ttm")
    )?;

    Ok(compute_fundamental_metrics(financialsdf.lazy()))
}

/// Writes the financials_quarter table from ARQ fundamentals.
fn write_financials_quarter(prices: &DataFrame) -> Result<LazyFrame> {
    let financialslf = read_csv("financials_quarter", None)?;
    let financialslf = adjust_fundamentals_quarter(financialslf);

    let mut financialsdf = timed!("adjusting quarterly fundamentals", financialslf.collect())?;

    timed!(
        "writing financials_quarter",
        df_to_duckdb(&mut financialsdf, "financials_quarter")
    )?;

    let price_closes = prices.select(["ticker", "date", "close"])?;

    Ok(compute_quarterly_fundamental_metrics(
        financialsdf.lazy(),
        price_closes.lazy(),
    ))
}

/// Writes the companies table including latest price and fundamental data
fn write_companies(
    prices: DataFrame,
    financials: LazyFrame,
    financials_quarter: LazyFrame,
) -> Result<()> {
    let companieslf = read_csv("companies", None)?;
    let companieslf =
        build_company_snapshot(companieslf, prices.lazy(), financials, financials_quarter);

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
pub fn run_writer(export_arrow: bool, write_weekly: bool, compute_hurst: bool) -> Result<()> {
    let prices = write_stocks(compute_hurst)?;
    if write_weekly {
        write_stocks_weekly(compute_hurst)?;
    }
    let financials = write_financials()?;
    let financials_quarter = write_financials_quarter(&prices)?;
    write_companies(prices, financials, financials_quarter)?;
    write_insiders()?;
    if export_arrow {
        write_arrow_files()?;
    }
    Ok(())
}
