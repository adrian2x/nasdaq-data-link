use anyhow::{Context, Result};
use std::path::Path;

use crate::config::DUCKDB_FILENAME;

/// Execute a SQL script (one or more statements) against the local database.
///
/// `label` is a short human-readable name for the script (e.g.
/// "build_companies"). It surfaces in the error chain if the query fails,
/// turning an opaque DuckDB error into `executing SQL script 'X': <error>`.
pub fn execute_sql(label: &str, sql: &str) -> Result<()> {
    let conn = duckdb::Connection::open(DUCKDB_FILENAME)
        .with_context(|| format!("opening {}", DUCKDB_FILENAME))?;
    conn.execute_batch(sql)
        .with_context(|| format!("executing SQL script '{}'", label))?;
    Ok(())
}

/// Read a `.sql` file from disk and execute it as a batch.
///
/// Convenience wrapper over `execute_sql`. The file's stem is used as the
/// error-chain label (e.g. `build_companies.sql` -> label "build_companies").
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
