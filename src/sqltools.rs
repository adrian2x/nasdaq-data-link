use anyhow::{Result, anyhow};
use polars::prelude::{AnyValue, DataFrame, DataType as PolarsType};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Date64Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
    Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::{Array, DictionaryArray};
use arrow_ipc::reader::FileReader;
use arrow_schema::DataType as ArrowType;
use rusqlite::{types::Value as SqlValue, Connection};
use std::fs::File;
use std::path::Path;

/// SQLite's maximum host-parameter count (SQLite >= 3.32.0).
const MAX_SQL_PARAMS: usize = 32_766;

/// Read an Arrow IPC file and stream its record batches directly into a SQLite table.
///
/// Designed for maximum throughput on large files:
/// - Streams record batches one at a time — the full file is never held in memory.
/// - Wraps all inserts in a single transaction.
/// - Uses multi-row INSERT statements (up to `MAX_SQL_PARAMS / ncols` rows each).
/// - Applies aggressive bulk-load SQLite PRAGMAs.
/// - Runs blocking SQLite I/O on a dedicated thread via `spawn_blocking`.
///
/// # Arguments
/// * `arrow_file` - Source Arrow IPC file path
/// * `sqlite_db`  - Destination SQLite file (created if absent)
/// * `table_name` - Target table; defaults to the arrow file's stem
pub async fn arrow_to_sqlite(
    arrow_file: &str,
    sqlite_db: &str,
    table_name: Option<&str>,
) -> Result<()> {
    let derived;
    let table = match table_name {
        Some(n) => n,
        None => {
            derived = Path::new(arrow_file)
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow!("Cannot derive table name from: {arrow_file}"))?
                .to_string();
            derived.as_str()
        }
    };
    let table = table.trim().to_string();
    if table.is_empty() {
        return Err(anyhow!("table_name cannot be empty"));
    }

    let arrow_file = arrow_file.to_string();
    let sqlite_db = sqlite_db.to_string();

    tokio::task::spawn_blocking(move || write_blocking(&arrow_file, &sqlite_db, &table)).await??;
    Ok(())
}

fn write_blocking(arrow_file: &str, sqlite_db: &str, table: &str) -> Result<()> {
    println!("Reading {arrow_file}");
    let file = File::open(arrow_file)?;
    let mut reader =
        FileReader::try_new(file, None).map_err(|e| anyhow!("Arrow IPC open failed: {e}"))?;
    let schema = reader.schema();
    let ncols = schema.fields().len();

    // Pack as many rows as possible per INSERT without hitting the param limit.
    let batch_size = (MAX_SQL_PARAMS / ncols.max(1)).clamp(1, 100);

    println!(
        "Writing to '{}' table '{}' (batch_size={batch_size})",
        sqlite_db, table
    );

    let mut conn = Connection::open(sqlite_db)?;

    // -----------------------------------------------------------------------
    // Bulk-load pragma set — maximises write throughput.
    //
    // journal_mode = OFF  — no rollback journal; fastest for pure bulk loads.
    // synchronous  = OFF  — skip fsyncs; OS may lose data on crash (acceptable
    //                        for an import job that can simply be re-run).
    // cache_size       — 1 GiB page cache keeps hot pages off disk.
    // temp_store       — keep temp tables / indices in RAM.
    // mmap_size        — allow up to 30 GiB of memory-mapped I/O.
    // locking_mode     — EXCLUSIVE avoids repeated lock acquisitions.
    // page_size        — 64 KiB pages cut seek overhead on large sequential writes
    //                    (only effective before any tables are created).
    // -----------------------------------------------------------------------
    conn.execute_batch(
        "PRAGMA page_size    = 65536;
         PRAGMA journal_mode = OFF;
         PRAGMA synchronous  = OFF;
         PRAGMA cache_size   = -1048576;
         PRAGMA temp_store   = MEMORY;
         PRAGMA mmap_size    = 30000000000;
         PRAGMA locking_mode = EXCLUSIVE;",
    )?;

    let qt = quote_ident(table);
    let col_defs = schema
        .fields()
        .iter()
        .map(|f| format!("{} {}", quote_ident(f.name()), arrow_to_sql_type(f.data_type())))
        .collect::<Vec<_>>()
        .join(", ");
    conn.execute_batch(&format!(
        "DROP TABLE IF EXISTS {qt}; CREATE TABLE {qt} ({col_defs});"
    ))?;

    let qcols = schema
        .fields()
        .iter()
        .map(|f| quote_ident(f.name()))
        .collect::<Vec<_>>()
        .join(", ");

    // Pre-build the INSERT SQL for a full batch.
    let full_batch_sql = make_insert_sql(&qt, &qcols, ncols, batch_size);

    // Single transaction wraps the entire load.
    let tx = conn.transaction()?;
    let mut pending: Vec<SqlValue> = Vec::with_capacity(batch_size * ncols);
    let mut total: u64 = 0;

    {
        // Reuse the prepared statement for every full-size batch.
        let mut stmt = tx.prepare(&full_batch_sql)?;

        for batch_result in reader.by_ref() {
            let batch = batch_result.map_err(|e| anyhow!("Batch read error: {e}"))?;

            for row in 0..batch.num_rows() {
                for col_idx in 0..ncols {
                    pending.push(sql_value(batch.column(col_idx).as_ref(), row));
                }
                if pending.len() == batch_size * ncols {
                    stmt.execute(rusqlite::params_from_iter(pending.iter()))?;
                    total += batch_size as u64;
                    pending.clear();
                }
            }
        }
    } // stmt dropped — tx is usable again

    // Flush any partial trailing batch.
    if !pending.is_empty() {
        let rem = pending.len() / ncols;
        tx.execute(
            &make_insert_sql(&qt, &qcols, ncols, rem),
            rusqlite::params_from_iter(pending.iter()),
        )?;
        total += rem as u64;
    }

    tx.commit()?;
    println!("Wrote {total} rows to table '{table}'");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map an Arrow data type to the best-fit SQLite type affinity.
fn arrow_to_sql_type(dt: &ArrowType) -> &'static str {
    match dt {
        ArrowType::Boolean
        | ArrowType::Int8
        | ArrowType::Int16
        | ArrowType::Int32
        | ArrowType::Int64
        | ArrowType::UInt8
        | ArrowType::UInt16
        | ArrowType::UInt32
        | ArrowType::UInt64
        | ArrowType::Date32
        | ArrowType::Date64
        | ArrowType::Timestamp(_, _)
        | ArrowType::Time32(_)
        | ArrowType::Time64(_)
        | ArrowType::Duration(_) => "INTEGER",
        ArrowType::Float32 | ArrowType::Float64 => "REAL",
        ArrowType::Binary | ArrowType::LargeBinary | ArrowType::FixedSizeBinary(_) => "BLOB",
        _ => "TEXT",
    }
}

/// Extract a scalar `SqlValue` from an Arrow array at a given row index.
/// Uses typed array accessors — no boxing or dynamic dispatch per value.
fn sql_value(array: &dyn Array, idx: usize) -> SqlValue {
    use arrow_schema::TimeUnit::{Microsecond, Millisecond, Nanosecond, Second};

    if array.is_null(idx) {
        return SqlValue::Null;
    }
    match array.data_type() {
        ArrowType::Boolean => SqlValue::Integer(array.as_boolean().value(idx) as i64),
        ArrowType::Int8 => SqlValue::Integer(array.as_primitive::<Int8Type>().value(idx) as i64),
        ArrowType::Int16 => {
            SqlValue::Integer(array.as_primitive::<Int16Type>().value(idx) as i64)
        }
        ArrowType::Int32 => {
            SqlValue::Integer(array.as_primitive::<Int32Type>().value(idx) as i64)
        }
        ArrowType::Int64 => SqlValue::Integer(array.as_primitive::<Int64Type>().value(idx)),
        ArrowType::UInt8 => {
            SqlValue::Integer(array.as_primitive::<UInt8Type>().value(idx) as i64)
        }
        ArrowType::UInt16 => {
            SqlValue::Integer(array.as_primitive::<UInt16Type>().value(idx) as i64)
        }
        ArrowType::UInt32 => {
            SqlValue::Integer(array.as_primitive::<UInt32Type>().value(idx) as i64)
        }
        ArrowType::UInt64 => {
            let v = array.as_primitive::<UInt64Type>().value(idx);
            match i64::try_from(v) {
                Ok(i) => SqlValue::Integer(i),
                Err(_) => SqlValue::Text(v.to_string()),
            }
        }
        ArrowType::Float32 => {
            SqlValue::Real(array.as_primitive::<Float32Type>().value(idx) as f64)
        }
        ArrowType::Float64 => SqlValue::Real(array.as_primitive::<Float64Type>().value(idx)),
        ArrowType::Date32 => {
            SqlValue::Integer(array.as_primitive::<Date32Type>().value(idx) as i64)
        }
        ArrowType::Date64 => SqlValue::Integer(array.as_primitive::<Date64Type>().value(idx)),
        ArrowType::Timestamp(Second, _) => {
            SqlValue::Integer(array.as_primitive::<TimestampSecondType>().value(idx))
        }
        ArrowType::Timestamp(Millisecond, _) => {
            SqlValue::Integer(array.as_primitive::<TimestampMillisecondType>().value(idx))
        }
        ArrowType::Timestamp(Microsecond, _) => {
            SqlValue::Integer(array.as_primitive::<TimestampMicrosecondType>().value(idx))
        }
        ArrowType::Timestamp(Nanosecond, _) => {
            SqlValue::Integer(array.as_primitive::<TimestampNanosecondType>().value(idx))
        }
        ArrowType::Time32(Second) => {
            SqlValue::Integer(array.as_primitive::<Time32SecondType>().value(idx) as i64)
        }
        ArrowType::Time32(Millisecond) => {
            SqlValue::Integer(array.as_primitive::<Time32MillisecondType>().value(idx) as i64)
        }
        ArrowType::Time64(Microsecond) => {
            SqlValue::Integer(array.as_primitive::<Time64MicrosecondType>().value(idx))
        }
        ArrowType::Time64(Nanosecond) => {
            SqlValue::Integer(array.as_primitive::<Time64NanosecondType>().value(idx))
        }
        ArrowType::Utf8 => SqlValue::Text(array.as_string::<i32>().value(idx).to_string()),
        ArrowType::LargeUtf8 => SqlValue::Text(array.as_string::<i64>().value(idx).to_string()),
        ArrowType::Binary => SqlValue::Blob(array.as_binary::<i32>().value(idx).to_vec()),
        ArrowType::LargeBinary => SqlValue::Blob(array.as_binary::<i64>().value(idx).to_vec()),
        // Resolve dictionary key → value without needing arrow-cast / arrow-arith.
        ArrowType::Dictionary(key_type, _) => {
            macro_rules! dict_val {
                ($kt:ty) => {
                    if let Some(d) = array.as_any().downcast_ref::<DictionaryArray<$kt>>() {
                        let key = d.keys().value(idx) as usize;
                        return sql_value(d.values().as_ref(), key);
                    }
                };
            }
            match key_type.as_ref() {
                ArrowType::Int8   => dict_val!(Int8Type),
                ArrowType::Int16  => dict_val!(Int16Type),
                ArrowType::Int32  => dict_val!(Int32Type),
                ArrowType::Int64  => dict_val!(Int64Type),
                ArrowType::UInt8  => dict_val!(UInt8Type),
                ArrowType::UInt16 => dict_val!(UInt16Type),
                ArrowType::UInt32 => dict_val!(UInt32Type),
                ArrowType::UInt64 => dict_val!(UInt64Type),
                _ => {}
            }
            SqlValue::Null
        }
        _ => SqlValue::Text(format!("{:?}", array.data_type())),
    }
}

/// Write a Polars DataFrame directly into a SQLite table.
///
/// Skips any intermediate file format — values are extracted from typed
/// column arrays and inserted in multi-row batches inside a single transaction.
/// Applies the same aggressive bulk-load PRAGMAs as the other writers.
///
/// The destination table is always **dropped and recreated**, replacing its
/// entire contents on every call.
///
/// # Arguments
/// * `df`         - DataFrame to write (consumed)
/// * `sqlite_db`  - Destination SQLite file (created if absent)
/// * `table_name` - Target table name
pub async fn df_to_sqlite(df: DataFrame, sqlite_db: &str, table_name: &str) -> Result<()> {
    let sqlite_db = sqlite_db.to_string();
    let table_name = table_name.to_string();
    tokio::task::spawn_blocking(move || df_write_blocking(df, &sqlite_db, &table_name)).await??;
    Ok(())
}

fn df_write_blocking(df: DataFrame, sqlite_db: &str, table: &str) -> Result<()> {
    let ncols = df.width();
    let nrows = df.height();
    let batch_size = (MAX_SQL_PARAMS / ncols.max(1)).clamp(1, 100);

    println!("Writing {nrows} rows to '{sqlite_db}' table '{table}' (batch_size={batch_size})");

    let mut conn = Connection::open(sqlite_db)?;
    conn.execute_batch(
        "PRAGMA page_size    = 65536;
         PRAGMA journal_mode = OFF;
         PRAGMA synchronous  = OFF;
         PRAGMA cache_size   = -1048576;
         PRAGMA temp_store   = MEMORY;
         PRAGMA mmap_size    = 30000000000;
         PRAGMA locking_mode = EXCLUSIVE;",
    )?;

    let qt = quote_ident(table);
    let col_defs = df
        .get_columns()
        .iter()
        .map(|s| format!("{} {}", quote_ident(s.name()), polars_dtype_to_sql(s.dtype())))
        .collect::<Vec<_>>()
        .join(", ");
    conn.execute_batch(&format!(
        "DROP TABLE IF EXISTS {qt}; CREATE TABLE {qt} ({col_defs});"
    ))?;

    let qcols = df
        .get_columns()
        .iter()
        .map(|s| quote_ident(s.name()))
        .collect::<Vec<_>>()
        .join(", ");

    let full_batch_sql = make_insert_sql(&qt, &qcols, ncols, batch_size);
    let tx = conn.transaction()?;
    let mut pending: Vec<SqlValue> = Vec::with_capacity(batch_size * ncols);
    let mut total: u64 = 0;
    let cols = df.get_columns();

    {
        let mut stmt = tx.prepare(&full_batch_sql)?;

        for row in 0..nrows {
            for col in cols {
                pending.push(series_value_at(col, row));
            }
            if pending.len() == batch_size * ncols {
                stmt.execute(rusqlite::params_from_iter(pending.iter()))?;
                total += batch_size as u64;
                pending.clear();
            }
        }
    }

    if !pending.is_empty() {
        let rem = pending.len() / ncols;
        tx.execute(
            &make_insert_sql(&qt, &qcols, ncols, rem),
            rusqlite::params_from_iter(pending.iter()),
        )?;
        total += rem as u64;
    }

    tx.commit()?;
    println!("Wrote {total} rows to table '{table}'");
    Ok(())
}

/// Map a Polars DataType to the best-fit SQLite type affinity.
fn polars_dtype_to_sql(dt: &PolarsType) -> &'static str {
    match dt {
        PolarsType::Boolean
        | PolarsType::Int8
        | PolarsType::Int16
        | PolarsType::Int32
        | PolarsType::Int64
        | PolarsType::UInt8
        | PolarsType::UInt16
        | PolarsType::UInt32
        | PolarsType::UInt64
        | PolarsType::Date
        | PolarsType::Datetime(_, _)
        | PolarsType::Duration(_)
        | PolarsType::Time => "INTEGER",
        PolarsType::Float32 | PolarsType::Float64 => "REAL",
        PolarsType::Binary => "BLOB",
        _ => "TEXT",
    }
}

/// Extract a scalar `SqlValue` from a Polars Column at a given row index.
/// Uses `Column::get` which dispatches through the typed ChunkedArray.
fn series_value_at(s: &polars::prelude::Column, idx: usize) -> SqlValue {
    match s.get(idx) {
        Ok(AnyValue::Null) => SqlValue::Null,
        Ok(AnyValue::Boolean(v)) => SqlValue::Integer(v as i64),
        Ok(AnyValue::Int8(v)) => SqlValue::Integer(v as i64),
        Ok(AnyValue::Int16(v)) => SqlValue::Integer(v as i64),
        Ok(AnyValue::Int32(v)) => SqlValue::Integer(v as i64),
        Ok(AnyValue::Int64(v)) => SqlValue::Integer(v),
        Ok(AnyValue::UInt8(v)) => SqlValue::Integer(v as i64),
        Ok(AnyValue::UInt16(v)) => SqlValue::Integer(v as i64),
        Ok(AnyValue::UInt32(v)) => SqlValue::Integer(v as i64),
        Ok(AnyValue::UInt64(v)) => match i64::try_from(v) {
            Ok(i) => SqlValue::Integer(i),
            Err(_) => SqlValue::Text(v.to_string()),
        },
        Ok(AnyValue::Float32(v)) => SqlValue::Real(v as f64),
        Ok(AnyValue::Float64(v)) => SqlValue::Real(v),
        Ok(AnyValue::Date(v)) => SqlValue::Integer(v as i64),
        Ok(AnyValue::Datetime(v, _, _)) => SqlValue::Integer(v),
        Ok(AnyValue::Duration(v, _)) => SqlValue::Integer(v),
        Ok(AnyValue::Time(v)) => SqlValue::Integer(v),
        Ok(AnyValue::String(v)) => SqlValue::Text(v.to_string()),
        Ok(AnyValue::StringOwned(v)) => SqlValue::Text(v.to_string()),
        Ok(AnyValue::Binary(v)) => SqlValue::Blob(v.to_vec()),
        Ok(AnyValue::BinaryOwned(v)) => SqlValue::Blob(v),
        Ok(v) => SqlValue::Text(v.to_string()),
        Err(_) => SqlValue::Null,
    }
}

/// Read a CSV file and stream its rows directly into a SQLite table.
///
/// Uses the same bulk-load strategy as `arrow_to_sqlite`:
/// single transaction, multi-row batch INSERTs, and aggressive PRAGMAs.
/// Values are type-inferred per cell (INTEGER → REAL → TEXT fallback).
///
/// # Arguments
/// * `csv_file`   - Source CSV file path (must have a header row)
/// * `sqlite_db`  - Destination SQLite file (created if absent)
/// * `table_name` - Target table; defaults to the CSV file's stem
pub async fn csv_to_sqlite(
    csv_file: &str,
    sqlite_db: &str,
    table_name: Option<&str>,
) -> Result<()> {
    let derived;
    let table = match table_name {
        Some(n) => n,
        None => {
            derived = Path::new(csv_file)
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow!("Cannot derive table name from: {csv_file}"))?
                .to_string();
            derived.as_str()
        }
    };
    let table = table.trim().to_string();
    if table.is_empty() {
        return Err(anyhow!("table_name cannot be empty"));
    }

    let csv_file = csv_file.to_string();
    let sqlite_db = sqlite_db.to_string();

    tokio::task::spawn_blocking(move || csv_write_blocking(&csv_file, &sqlite_db, &table)).await??;
    Ok(())
}

/// 64 MiB read buffer — dramatically cuts syscall count on multi-GB files.
const CSV_READ_BUF: usize = 64 * 1024 * 1024;

fn csv_write_blocking(csv_file: &str, sqlite_db: &str, table: &str) -> Result<()> {
    println!("Reading {csv_file}");
    let file = File::open(csv_file)?;
    // Large internal buffer reduces how often the CSV parser refills from the OS.
    let mut rdr = csv::ReaderBuilder::new()
        .buffer_capacity(CSV_READ_BUF)
        .from_reader(file);

    // Read headers as strings (used once for DDL); subsequent iteration uses byte_records().
    let headers = rdr
        .headers()
        .map_err(|e| anyhow!("Failed to read CSV headers: {e}"))?
        .clone();
    let ncols = headers.len();
    let batch_size = (MAX_SQL_PARAMS / ncols.max(1)).clamp(1, 100);

    println!("Writing to '{sqlite_db}' table '{table}' (batch_size={batch_size})");

    let mut conn = Connection::open(sqlite_db)?;
    conn.execute_batch(
        "PRAGMA page_size    = 65536;
         PRAGMA journal_mode = OFF;
         PRAGMA synchronous  = OFF;
         PRAGMA cache_size   = -1048576;
         PRAGMA temp_store   = MEMORY;
         PRAGMA mmap_size    = 30000000000;
         PRAGMA locking_mode = EXCLUSIVE;",
    )?;

    // Column names only — no type declaration lets SQLite store whatever we bind.
    let qt = quote_ident(table);
    let qcols = headers
        .iter()
        .map(quote_ident)
        .collect::<Vec<_>>()
        .join(", ");
    conn.execute_batch(&format!(
        "DROP TABLE IF EXISTS {qt}; CREATE TABLE {qt} ({qcols});"
    ))?;

    let full_batch_sql = make_insert_sql(&qt, &qcols, ncols, batch_size);
    let tx = conn.transaction()?;
    let mut pending: Vec<SqlValue> = Vec::with_capacity(batch_size * ncols);
    let mut total: u64 = 0;

    {
        let mut stmt = tx.prepare(&full_batch_sql)?;

        // byte_records() skips per-field UTF-8 validation — a measurable win on large files.
        for result in rdr.byte_records() {
            let record = result.map_err(|e| anyhow!("CSV read error: {e}"))?;

            for field in record.iter() {
                pending.push(infer_csv_value(field));
            }

            if pending.len() == batch_size * ncols {
                stmt.execute(rusqlite::params_from_iter(pending.iter()))?;
                total += batch_size as u64;
                pending.clear();
            }
        }
    }

    if !pending.is_empty() {
        let rem = pending.len() / ncols;
        tx.execute(
            &make_insert_sql(&qt, &qcols, ncols, rem),
            rusqlite::params_from_iter(pending.iter()),
        )?;
        total += rem as u64;
    }

    tx.commit()?;
    println!("Wrote {total} rows to table '{table}'");
    Ok(())
}

/// Infer the tightest SQLite type that fits a raw CSV byte field.
/// Empty → NULL, parseable integer → INTEGER,
/// parseable float → REAL, valid UTF-8 → TEXT, otherwise → BLOB.
fn infer_csv_value(b: &[u8]) -> SqlValue {
    if b.is_empty() {
        return SqlValue::Null;
    }
    if let Ok(s) = std::str::from_utf8(b) {
        if let Ok(i) = s.parse::<i64>() {
            return SqlValue::Integer(i);
        }
        if let Ok(f) = s.parse::<f64>() {
            return SqlValue::Real(f);
        }
        return SqlValue::Text(s.to_string());
    }
    SqlValue::Blob(b.to_vec())
}

/// Build a multi-row `INSERT INTO t (cols) VALUES (?,?,...), (?,?,...)` statement.
fn make_insert_sql(qt: &str, qcols: &str, ncols: usize, nrows: usize) -> String {
    let row = format!("({})", vec!["?"; ncols].join(","));
    let rows = vec![row.as_str(); nrows].join(",");
    format!("INSERT INTO {qt} ({qcols}) VALUES {rows}")
}

fn quote_ident(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
