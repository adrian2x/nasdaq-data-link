//! Provides DuckDB SQL execution helpers for inline scripts and SQL files.
use anyhow::{Context, Result};

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
