use anyhow::Result;
use polars::prelude::{DataType, LazyFrame, ScanArgsParquet};

use crate::config::DOWNLOADS_DIR;
use crate::dataframetools::{
    adjust_fundamentals, adjust_prices, csv_to_parquet, extract_zip_file, lf_to_duckdb, resample,
    update_insiders,
};
use crate::sqltools::execute_sql_file;
use crate::ui::with_spinner;

const BUILD_COMPANIES_SQL: &str = "build_companies.sql";

/// Unzip `downloads/<name>.csv.zip` and stream it to `downloads/<name>.parquet`.
fn dataset_to_parquet(name: &str, schema_overrides: Option<&[(&str, DataType)]>) -> Result<String> {
    let zip_path = format!("{DOWNLOADS_DIR}/{name}.csv.zip");
    let parquet_path = format!("{DOWNLOADS_DIR}/{name}.parquet");

    let csv_path = with_spinner(&format!("extracting {name}"), || {
        extract_zip_file(&zip_path)
    })?;
    with_spinner(&format!("converting {name} to parquet"), || {
        csv_to_parquet(&csv_path, &parquet_path, schema_overrides)
    })?;
    Ok(parquet_path)
}

/// Convert the dataset to parquet, then lazily scan it.
fn scan_dataset(name: &str, schema_overrides: Option<&[(&str, DataType)]>) -> Result<LazyFrame> {
    let parquet_path = dataset_to_parquet(name, schema_overrides)?;
    let lf = with_spinner(&format!("scanning {name}"), || {
        LazyFrame::scan_parquet(&parquet_path, ScanArgsParquet::default())
            .map_err(anyhow::Error::from)
    })?;
    Ok(lf)
}

fn write_stocks() -> Result<()> {
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

    let daily = adjust_prices(df);
    with_spinner("writing stocks_daily", || {
        lf_to_duckdb(daily.clone(), "stocks_daily")
    })?;

    let weekly = resample(daily, "1w");
    with_spinner("writing stocks_weekly", || {
        lf_to_duckdb(weekly, "stocks_weekly")
    })?;

    Ok(())
}

fn write_financials() -> Result<()> {
    let df = scan_dataset("financials_ttm", None)?;

    let financials_adj = adjust_fundamentals(df);
    with_spinner("writing financials_ttm", || {
        lf_to_duckdb(financials_adj, "financials_ttm")
    })?;
    Ok(())
}

fn write_companies() -> Result<()> {
    dataset_to_parquet("companies", None)?;

    with_spinner("building companies snapshot", || {
        execute_sql_file(BUILD_COMPANIES_SQL)
    })?;

    Ok(())
}

fn write_insiders() -> Result<()> {
    // `formtype` pinned to String — it starts with numeric codes ("4", "5")
    // then hits strings like "RESTATED - 4" hundreds of rows in.
    let df = scan_dataset("insiders", Some(&[("formtype", DataType::String)]))?;

    let insiders_adj = update_insiders(df);
    with_spinner("writing insiders", || {
        lf_to_duckdb(insiders_adj, "insiders")
    })?;
    Ok(())
}

pub fn run_writer() -> Result<()> {
    write_stocks()?;
    write_financials()?;
    write_companies()?;
    write_insiders()?;
    Ok(())
}
