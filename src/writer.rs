use anyhow::Result;
use polars::prelude::*;

use crate::config::DOWNLOADS_DIR;
use crate::dataframetools::{
    adjust_fundamentals, adjust_prices, csv_to_parquet, extract_zip_file, lf_to_duckdb,
    technical_indicators_daily, update_insiders,
};
use crate::ui::with_spinner;

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

/// Build the enriched daily price table, write it, and return the LazyFrame
/// so downstream steps (build_companies) can consume it without re-scanning.
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

    // adjust_prices yields the adjusted OHLCV bars; technical_indicators_daily
    // enriches them with indicator columns. The result is a superset of the
    // price data, so it is written as the single `stock_prices` table — no
    // separate bars table is kept.
    let prices = adjust_prices(df);
    let enriched = technical_indicators_daily(prices);
    with_spinner("writing stock_prices", || {
        lf_to_duckdb(enriched.clone(), "stock_prices")
    })?;

    Ok(enriched)
}

/// Build the TTM fundamentals table, write it, and return the LazyFrame so
/// build_companies can consume it without re-scanning.
fn write_financials() -> Result<LazyFrame> {
    let df = scan_dataset("financials_ttm", None)?;

    let financials_adj = adjust_fundamentals(df);
    with_spinner("writing financials_ttm", || {
        lf_to_duckdb(financials_adj.clone(), "financials_ttm")
    })?;
    Ok(financials_adj)
}

/// Build the `companies` snapshot: active companies joined to their latest
/// fundamentals and latest price, with marketcap/ev recomputed at the latest
/// close. Replaces the former build_companies.sql — done in Polars so the
/// pipeline is engine-agnostic (only lf_to_duckdb touches the database).
///
/// The snapshot keeps every column from all three sources (companies,
/// financials_ttm, stock_prices). The only shared column is `ticker` (the
/// join key); marketcap/ev arrive raw from financials and are then
/// overwritten by the latest-close recompute below.
fn write_companies(prices: LazyFrame, financials: LazyFrame) -> Result<()> {
    let companies = scan_dataset("companies", None)?;

    // Active companies only.
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

    // Latest fundamentals row per ticker without full-frame sort/join.
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

    // Keep only the globally latest trading date first (strict active rule),
    // then one row per ticker. This prunes aggressively before grouping.
    let latest_prices = prices
        .with_column(col("date").max().over([lit(1)]).alias("__max_date"))
        .filter(col("date").eq(col("__max_date")))
        .drop(["__max_date"])
        .group_by([col("ticker")])
        .agg([all().last()]);

    // Inner-join active companies x latest fundamentals x latest price.
    // INNER drops tickers missing either fundamentals or a current price.
    // `ticker` is the only shared column, so no other column collides.
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

    // Recompute marketcap and ev at the latest close (overwriting the raw
    // Sharadar values): marketcap = shares*close, ev = marketcap + netdebt.
    let snapshot = joined.with_columns([
        (col("shares") * col("close")).alias("marketcap"),
        (col("shares") * col("close") + col("netdebtusd")).alias("ev"),
    ]);

    with_spinner("writing companies", || lf_to_duckdb(snapshot, "companies"))?;

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
    let prices = write_stocks()?;
    let financials = write_financials()?;
    write_companies(prices, financials)?;
    write_insiders()?;
    Ok(())
}
