use ::zip::ZipArchive;
use anyhow::{Result, anyhow};
use polars::prelude::*;
use std::fs::File;
use std::io::copy;
use std::path::Path;

use crate::config::{DOWNLOADS_DIR, OUTPUT_DIR};
use crate::ui::with_spinner;

/// Write bytes to a path, creating parent directories when needed.
pub fn save_file(data: &[u8], filepath: impl AsRef<Path>) -> Result<()> {
    let filepath = filepath.as_ref();
    let path = filepath;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow!(
                "Failed to create directories for '{}': {}",
                filepath.display(),
                e
            )
        })?;
    }

    std::fs::write(path, data)
        .map_err(|e| anyhow!("Failed to write file '{}': {}", filepath.display(), e))?;
    Ok(())
}

/// Ensure a directory exists.
pub fn ensure_directory(dir: impl AsRef<Path>) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)
        .map_err(|e| anyhow!("Failed to create directory '{}': {}", dir.display(), e))?;
    Ok(())
}

/// Stream CSV to parquet.
pub fn csv_to_parquet<P: AsRef<Path>, Q: AsRef<Path>>(
    csv_path: P,
    parquet_path: Q,
    schema_overrides: Option<&[(&str, DataType)]>,
) -> Result<()> {
    let overrides_schema = schema_overrides.map(|pairs| {
        let fields = pairs
            .iter()
            .map(|(name, dtype)| Field::new((*name).into(), dtype.clone()));
        Arc::new(Schema::from_iter(fields))
    });

    let lf = LazyCsvReader::new(csv_path.as_ref())
        .with_has_header(true)
        .with_dtype_overwrite(overrides_schema)
        .finish()?;

    lf.sink_parquet(&parquet_path, ParquetWriteOptions::default(), None)?;
    Ok(())
}

/// Extract the first ZIP entry next to the archive and return its path.
pub fn extract_zip_file<P: AsRef<Path>>(zip_path: P) -> Result<String> {
    let zip_path = zip_path.as_ref();
    let zip_str = zip_path.to_string_lossy();
    let output_filename = zip_str
        .strip_suffix(".zip")
        .map(str::to_string)
        .unwrap_or_else(|| zip_str.to_string());

    let zip_file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(zip_file)?;
    let mut csv_file = archive.by_index(0)?;
    let mut output_file = File::create(&output_filename)?;
    copy(&mut csv_file, &mut output_file)?;
    Ok(output_filename)
}

/// Convert `downloads/<name>.csv.zip` to `downloads/<name>.parquet`.
pub fn dataset_to_parquet(
    name: &str,
    schema_overrides: Option<&[(&str, DataType)]>,
) -> Result<String> {
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

/// Build parquet for a dataset and return a lazy parquet scan.
pub fn scan_dataset(
    name: &str,
    schema_overrides: Option<&[(&str, DataType)]>,
) -> Result<LazyFrame> {
    let parquet_path = dataset_to_parquet(name, schema_overrides)?;
    with_spinner(&format!("scanning {name}"), || {
        LazyFrame::scan_parquet(&parquet_path, ScanArgsParquet::default())
            .map_err(anyhow::Error::from)
    })
}

/// Materialize a LazyFrame to parquet and register it in DuckDB.
pub fn lf_to_duckdb(lf: LazyFrame, table: &str) -> Result<()> {
    let table = table.trim();
    if table.is_empty() {
        return Err(anyhow!("table name cannot be empty"));
    }

    std::fs::create_dir_all(OUTPUT_DIR)?;
    let parquet_path = format!("{OUTPUT_DIR}/{table}.parquet");

    if std::env::var("EXPLAIN_PLAN").is_ok() {
        eprintln!("=== plan for {table} ===");
        eprintln!("{}", lf.clone().explain(true)?);
        eprintln!("=== end plan ===");
    }

    // Eager boundary: upstream pipeline remains lazy.
    let mut df = lf.collect()?;
    ParquetWriter::new(File::create(&parquet_path)?).finish(&mut df)?;

    let qt = quote_ident(table);
    crate::sqltools::execute_sql(
        &format!("create_{table}"),
        &format!("CREATE OR REPLACE TABLE {qt} AS SELECT * FROM read_parquet('{parquet_path}');"),
    )?;

    Ok(())
}

fn quote_ident(s: &str) -> String {
    format!("\"{}\"", s.replace('\"', "\"\""))
}
