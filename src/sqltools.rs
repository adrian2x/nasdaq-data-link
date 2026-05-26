//! Provides DuckDB SQL execution helpers for inline scripts and SQL files.
use anyhow::{Context, Result};
use std::path::Path;

use crate::config::DUCKDB_FILENAME;

/// Executes an SQL batch against the configured DuckDB database.
///
/// # Failure
/// Returns an error if the database cannot be opened or the SQL batch fails.
pub fn execute_sql(label: &str, sql: &str) -> Result<()> {
    let conn = duckdb::Connection::open(DUCKDB_FILENAME)
        .with_context(|| format!("opening {}", DUCKDB_FILENAME))?;
    conn.execute_batch(sql)
        .with_context(|| format!("executing SQL script '{}'", label))?;
    Ok(())
}

/// Reads an SQL file and executes it against the configured DuckDB database.
///
/// # Failure
/// Returns an error if the SQL file cannot be read or execution fails.
pub fn execute_sql_file<P: AsRef<Path>>(sql_path: P) -> Result<()> {
    let sql_path = sql_path.as_ref();
    let label = sql_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "sql_script".to_string());
    let sql = std::fs::read_to_string(sql_path)
        .with_context(|| format!("reading SQL script {}", sql_path.display()))?;
    execute_sql(&label, &sql)
}
