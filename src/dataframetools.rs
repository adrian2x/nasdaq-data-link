use ::zip::ZipArchive;
use anyhow::{Result, anyhow};
use limbo::{Builder, Value, params_from_iter};
use polars::io::csv::write::CsvWriter;
use polars::io::ipc::IpcWriter;
use polars::prelude::*;
use std::fs::File;
use std::io::copy;
use std::time::Instant;

/// Annualization factor for daily returns: sqrt(252 trading days)
const ANNUALIZATION_FACTOR: f64 = 15.874507866387544; // 252.0_f64.sqrt()
const SQLITE_OUTPUT_FILENAME: &str = "nasdaq.db";

/// Write a Polars DataFrame to a CSV file
///
/// # Arguments
/// * `df`       - DataFrame to write
/// * `filename` - Output file path (e.g., "output/data.csv")
///
/// # Returns
/// * `Result<()>` - Success or error
pub fn write_df_to_csv(df: &mut DataFrame, filename: &str) -> Result<()> {
    let file = File::create(filename)?;
    CsvWriter::new(file).finish(df)?;
    Ok(())
}

/// Write a Polars DataFrame to an Arrow IPC file
///
/// # Arguments
/// * `df` - DataFrame to write
/// * `filename` - Output file path (e.g., "output/data.arrow")
///
/// # Returns
/// * `Result<()>` - Success or error
pub fn write_df_to_arrow(df: &mut DataFrame, filename: &str) -> Result<()> {
    let file = File::create(filename)?;
    IpcWriter::new(file).finish(df)?;
    Ok(())
}

/// Write a Polars DataFrame to a local SQLite file in the current working directory (via Limbo)
///
/// Creates or opens `nasdaq.db`, replaces the destination table, and inserts all rows.
///
/// # Arguments
/// * `df` - DataFrame to write
/// * `table_name` - Destination table name
///
/// # Returns
/// * `Result<()>` - Success or error
pub async fn write_df_to_sqlite(df: &DataFrame, table_name: &str) -> Result<()> {
    println!("writing table {}", table_name);
    let table_name = table_name.trim();
    if table_name.is_empty() {
        return Err(anyhow!("table_name cannot be empty"));
    }

    let db = Builder::new_local(SQLITE_OUTPUT_FILENAME).build().await?;
    let conn = db.connect()?;

    let quoted_table = quote_sqlite_identifier(table_name);
    let columns = df.get_columns();
    let column_defs = columns
        .iter()
        .map(|column| {
            format!(
                "{} {}",
                quote_sqlite_identifier(column.name().as_str()),
                sqlite_type_for_dtype(column.dtype())
            )
        })
        .collect::<Vec<_>>();

    conn.execute(&format!("DROP TABLE IF EXISTS {}", quoted_table), ())
        .await?;
    conn.execute(
        &format!("CREATE TABLE {} ({})", quoted_table, column_defs.join(", ")),
        (),
    )
    .await?;

    if df.height() == 0 {
        return Ok(());
    }

    let placeholders = (0..columns.len())
        .map(|idx| format!("?{}", idx + 1))
        .collect::<Vec<_>>()
        .join(", ");
    let quoted_columns = columns
        .iter()
        .map(|column| quote_sqlite_identifier(column.name().as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let insert_sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        quoted_table, quoted_columns, placeholders
    );

    for row_idx in 0..df.height() {
        let row = df.get_row(row_idx)?;
        let params = row
            .0
            .into_iter()
            .map(any_value_to_limbo_value)
            .collect::<Vec<_>>();
        conn.execute(&insert_sql, params_from_iter(params)).await?;
    }

    Ok(())
}

fn quote_sqlite_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('\"', "\"\""))
}

fn sqlite_type_for_dtype(dtype: &DataType) -> &'static str {
    match dtype {
        DataType::Boolean
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64 => "INTEGER",
        DataType::Float32 | DataType::Float64 => "REAL",
        DataType::Binary => "BLOB",
        DataType::String => "TEXT",
        _ => "TEXT",
    }
}

fn any_value_to_limbo_value(value: AnyValue<'_>) -> Value {
    match value {
        AnyValue::Null => Value::Null,
        AnyValue::Boolean(v) => Value::Integer(if v { 1 } else { 0 }),
        AnyValue::UInt8(v) => Value::Integer(v as i64),
        AnyValue::UInt16(v) => Value::Integer(v as i64),
        AnyValue::UInt32(v) => Value::Integer(v as i64),
        AnyValue::UInt64(v) => match i64::try_from(v) {
            Ok(integer) => Value::Integer(integer),
            Err(_) => Value::Text(v.to_string()),
        },
        AnyValue::Int8(v) => Value::Integer(v as i64),
        AnyValue::Int16(v) => Value::Integer(v as i64),
        AnyValue::Int32(v) => Value::Integer(v as i64),
        AnyValue::Int64(v) => Value::Integer(v),
        AnyValue::Float32(v) => Value::Real(v as f64),
        AnyValue::Float64(v) => Value::Real(v),
        AnyValue::String(v) => Value::Text(v.to_string()),
        AnyValue::StringOwned(v) => Value::Text(v.to_string()),
        AnyValue::Binary(v) => Value::Blob(v.to_vec()),
        AnyValue::BinaryOwned(v) => Value::Blob(v),
        _ => Value::Text(value.to_string()),
    }
}

/// Extract a zip file to a new file with the same name minus the .zip extension
///
/// # Arguments
/// * `zip_filename` - Path to the zip file (e.g., "output/stocks_eod.csv.zip")
///
/// # Returns
/// * `Result<String>` - Path to the extracted file (e.g., "output/stocks_eod.csv")
pub fn extract_zip_file(zip_filename: &str) -> Result<String> {
    // Remove .zip extension to get output filename
    let output_filename = zip_filename.strip_suffix(".zip").unwrap_or(zip_filename);

    // Extract the zip file
    let zip_file = File::open(zip_filename)?;
    let mut archive = ZipArchive::new(zip_file)?;
    let mut csv_file = archive.by_index(0)?;

    // Write to output file
    let mut output_file = File::create(output_filename)?;
    copy(&mut csv_file, &mut output_file)?;

    Ok(output_filename.to_string())
}

/// Read a CSV zip file and load it into a Polars DataFrame
///
/// Extracts the zip file and then loads it with Polars.
/// Adjusts open, high, low, close prices using the closeadj factor.
///
/// # Arguments
/// * `path` - Path to the zip file (e.g., "output/stocks_eod.csv.zip")
///
/// # Returns
/// * `Result<DataFrame>` - DataFrame containing the adjusted stocks EOD data
pub fn adjust_prices(path: &str) -> Result<DataFrame> {
    println!("adjust_prices...");
    let start = Instant::now();
    // Extract the zip file
    let csv_path = extract_zip_file(path)?;

    // Load the extracted CSV into Polars
    let mut df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(csv_path.into()))?
        .finish()?;

    // Calculate adjustment factor: closeadj / close
    let adjustment_factor = col("closeadj") / col("close");

    // Adjust prices, drop close, and rename closeadj to close
    df = df
        .lazy()
        .with_columns([
            (col("low") * adjustment_factor.clone()).alias("low"),
            (col("high") * adjustment_factor.clone()).alias("high"),
            (col("open") * adjustment_factor).alias("open"),
        ])
        .drop(["close", "closeunadj", "lastupdated"])
        .rename(["closeadj"], ["close"], false)
        .sort(["ticker", "date"], Default::default())
        .collect()?;

    let elapsed = start.elapsed();
    println!("adjust_prices completed in {:.2?}", elapsed);
    Ok(df)
}

/// Read a fundamentals CSV zip file and load it into a Polars DataFrame
///
/// Extracts the zip file, loads it with Polars, renames "calendardate" to "date",
/// and drops the "dimension" and "lastupdated" metadata columns.
///
/// # Arguments
/// * `path` - Path to the zip file (e.g., "output/fundamentals.csv.zip")
///
/// # Returns
/// * `Result<DataFrame>` - DataFrame sorted by ["ticker", "date"]
pub fn adjust_fundamentals(path: &str) -> Result<DataFrame> {
    println!("adjust_fundamentals...");
    let start = Instant::now();

    let csv_path = extract_zip_file(path)?;

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(csv_path.into()))?
        .finish()?
        .lazy()
        .rename(["calendardate"], ["date"], false)
        .drop(["dimension", "lastupdated"])
        .with_columns([
            (f("sharesbas") * f("sharefactor")).alias("shares"),
            (f("dps") / f("fxusd")).alias("dpsusd"),
            (f("eps") / f("fxusd")).alias("epsusd"),
            (f("ncfo") / f("fxusd")).alias("ncfousd"),
            (f("fcf") / f("fxusd")).alias("fcfusd"),
            (f("assets") - f("liabilities")).alias("equity"),
            (f("debt") - f("cashneq")).alias("netdebt"),
        ])
        // MARGINS
        .with_columns([(f("ebit") / f("revenue")).alias("ebitmargin")])
        // FINANCIAL RATIOS
        .with_columns([
            (f("debt") / f("equity")).alias("de"),
            (f("ncfo") / f("opinc")).alias("cfc"),
            (f("ebit") / (f("assets") - f("liabilitiesc"))).alias("roce"),
            (f("ebit") / f("intexp")).alias("icr"),
        ])
        // APPLY ADJUSTMENTS
        .with_columns([
            ((f("netinc") - f("netincdis")) / f("fxusd")).alias("netincadj"), // in USD
        ])
        .with_columns([
            (f("netincadj") / f("shares")).alias("epsadj"), // in USD
            (f("marketcap") / f("netincadj")).alias("pe"),
        ])
        .sort(["ticker", "date"], Default::default())
        .collect()? // materialize sort before order-dependent window functions
        .lazy()
        // All shift().over(["ticker"]) expressions below rely on date-sorted rows
        .with_columns([
            (f("workingcapital") - f("workingcapital").shift(lit(1)))
                .over([col("ticker")])
                .alias("net_workingcapital"),
        ])
        .with_columns([(f("depamor") + f("net_workingcapital")).alias("maintenance_capex")])
        .with_columns([((f("ncfo")
            - f("netincdis")
            - f("maintenance_capex")
            - f("sbcomp"))
            / f("fxusd"))
        .alias("fcfadj")]) // in USD
        .with_columns([
            (f("fcfadj") / f("shares")).alias("fcfpsadj"), // in USD
            (f("marketcap") / f("fcfadj")).alias("pfcf"),
        ])
        // GROWTH METRICS
        .with_columns([
            pct_change("revenue", 4).alias("revenueyoy"),
            pct_change("ebitda", 4).alias("ebitdayoy"),
            pct_change("fcfpsadj", 4).alias("fcfpsyoy"),
        ])
        .collect()?;

    let elapsed = start.elapsed();
    println!("adjust_fundamentals completed in {:.2?}", elapsed);
    Ok(df)
}

/// Prepare DataFrame for technical indicator calculations
///
/// Sorts the DataFrame by ticker and date, then groups by ticker
///
/// # Arguments
/// * `df` - Input DataFrame to process
///
/// # Returns
/// * `Result<DataFrame>` - DataFrame with technical indicators
pub fn technical_indicators(df: DataFrame) -> Result<DataFrame> {
    println!("technical_indicators...");
    let start = Instant::now();

    let grouped = df
        .lazy()
        .sort(["ticker", "date"], Default::default())
        .with_columns([((col("volume") * col("close")).round(2)).alias("volumeusd")]);

    let grouped = grouped
        .with_columns([
            // Price returns
            chg_expr("close", 1).over([col("ticker")]).alias("pct"),
            chg_expr("close", 5).over([col("ticker")]).alias("chg5"),
            chg_expr("close", 10).over([col("ticker")]).alias("chg10"),
            chg_expr("close", 21).over([col("ticker")]).alias("chg1m"),
            chg_expr("close", 3 * 21)
                .over([col("ticker")])
                .alias("chg1q"),
            chg_expr("close", 6 * 21)
                .over([col("ticker")])
                .alias("chg2q"),
            chg_expr("close", 9 * 21)
                .over([col("ticker")])
                .alias("chg3q"),
            chg_expr("close", 12 * 21)
                .over([col("ticker")])
                .alias("chg1y"),
            // Moving averages
            rolling_mean_expr("volumeusd", 50)
                .over([col("ticker")])
                .alias("avgvolumeusd"),
            rolling_mean_expr("close", 10)
                .over([col("ticker")])
                .alias("sma10"),
            rolling_mean_expr("close", 20)
                .over([col("ticker")])
                .alias("sma20"),
            // exponential_moving_average_expr("close", 21)
            //     .over([col("ticker")])
            //     .alias("ema21"),
            rolling_mean_expr("close", 200)
                .over([col("ticker")])
                .alias("sma200"),
            // Rate of Change of price
            // roc_expr("close", 10).over([col("ticker")]).alias("roc10"),
            // roc_expr("close", 20).over([col("ticker")]).alias("roc20"),
            // roc_expr("close", 100).over([col("ticker")]).alias("roc100"),
            // roc_expr("close", 200).over([col("ticker")]).alias("roc200"),
            // Log returns for realized volatility
            log_return_expr("close")
                .over([col("ticker")])
                .alias("log_return"),
        ])
        // Relative strength must be in a separate with_columns since it references chg* columns
        // .with_columns([
        //     (lit(0.4) * col("chg1q") +
        //      lit(0.2) * col("chg2q") +
        //      lit(0.2) * col("chg3q") +
        //      lit(0.2) * col("chg1y")).over([col("ticker")]).alias("rs1y"),
        // ])
        // Realized volatility calculation
        .with_columns([
            rolling_std_expr("log_return", 5, ANNUALIZATION_FACTOR)
                .over([col("ticker")])
                .alias("rv5"),
            rolling_std_expr("log_return", 10, ANNUALIZATION_FACTOR)
                .over([col("ticker")])
                .alias("rv10"),
            rolling_std_expr("log_return", 20, ANNUALIZATION_FACTOR)
                .over([col("ticker")])
                .alias("rv20"),
            rolling_std_expr("log_return", 60, ANNUALIZATION_FACTOR)
                .over([col("ticker")])
                .alias("rv60"),
            rolling_std_expr("log_return", 252, ANNUALIZATION_FACTOR)
                .over([col("ticker")])
                .alias("rv252"),
        ])
        .drop(["log_return"])
        .collect()?;

    let elapsed = start.elapsed();
    println!("technical_indicators completed in {:.2?}", elapsed);

    Ok(grouped)
}

/// Helper function to create a rate of change (ROC) expression
///
/// # Arguments
/// * `column` - Column name to calculate ROC for
/// * `period` - Number of periods to look back
///
/// # Returns
/// * `Expr` - Polars expression for ROC calculation
fn roc_expr(column: &str, period: i64) -> Expr {
    ((col(column) / col(column).shift(lit(period))) - lit(1.0)) * lit(100.0)
}

/// Helper function to create a rolling moving average expression
///
/// # Arguments
/// * `column` - Column name to calculate moving average for
/// * `period` - Window size for the moving average
///
/// # Returns
/// * `Expr` - Polars expression for rolling mean calculation
fn rolling_mean_expr(column: &str, period: usize) -> Expr {
    col(column).rolling_mean(RollingOptionsFixedWindow {
        window_size: period,
        min_periods: 1,
        ..Default::default()
    })
}

/// Helper function to create an exponential moving average (EMA) expression
///
/// # Arguments
/// * `column` - Column name to calculate EMA for
/// * `period` - Span/period for the EMA
///
/// # Returns
/// * `Expr` - Polars expression for EMA calculation
fn exponential_moving_average_expr(column: &str, period: usize) -> Expr {
    col(column).ewm_mean(
        EWMOptions::default()
            .and_span(period)
            .and_adjust(false)
            .and_min_periods(1),
    )
}

/// Helper function to create a log returns expression
///
/// # Arguments
/// * `column` - Column name to calculate log returns for
///
/// # Returns
/// * `Expr` - Polars expression for log returns: ln(price_t / price_t-1)
fn log_return_expr(column: &str) -> Expr {
    (col(column) / col(column).shift(lit(1))).log(std::f64::consts::E)
}

/// Helper function to create a rolling standard deviation expression (annualized)
///
/// # Arguments
/// * `column` - Column name to calculate rolling std for
/// * `period` - Window size for the rolling std
/// * `annualization_factor` - Factor to annualize the volatility (defaults to sqrt(252) for daily)
///
/// # Returns
/// * `Expr` - Polars expression for annualized rolling std calculation
fn rolling_std_expr(column: &str, period: usize, annualization_factor: f64) -> Expr {
    col(column).rolling_std(RollingOptionsFixedWindow {
        window_size: period,
        min_periods: 2,
        ..Default::default()
    }) * lit(annualization_factor)
}

/// Shorthand: reference a column cast to Float64 for safe arithmetic.
fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Helper function to create a percentage change expression
///
/// # Arguments
/// * `column` - Column name to calculate change for
/// * `period` - Number of periods to look back
///
/// # Returns
/// * `Expr` - Polars expression for percentage change: ((price / price_shifted) - 1) * 100, rounded to 2 decimals
fn chg_expr(column: &str, period: i64) -> Expr {
    (((f(column) / f(column).shift(lit(period))) - lit(1.0)) * lit(100.0)).round(2)
}

/// Build a percentage-change expression for a column, scoped to ticker groups.
///
/// The result is aliased as `{column}pct` and is ready to pass directly to
/// `with_columns`. The first row per ticker will be `null` (no prior period).
///
/// # Arguments
/// * `column`  - Source column name
/// * `periods` - Look-back distance (default: 1)
///
/// # Returns
/// * `Expr` - `(current / prev - 1) * 100` rounded to 2 dp, `.over([ticker])`
pub fn pct_change(column: &str, periods: i64) -> Expr {
    let prev = f(column).shift(lit(periods));
    (((f(column) - prev.clone()) / prev.abs()) * lit(100.0))
        .round(2)
        .over([col("ticker")])
}

/// Calculate rate of change (ROC) for multiple periods
///
/// Adds ROC columns for 10, 20, 100, and 200 period lookbacks
///
/// # Arguments
/// * `df` - Input DataFrame (should be sorted by ticker and date)
///
/// # Returns
/// * `Result<DataFrame>` - DataFrame with added roc10, roc20, roc100, roc200 columns
pub fn roc(df: DataFrame) -> Result<DataFrame> {
    let result = df
        .lazy()
        .with_columns([
            roc_expr("close", 10).over([col("ticker")]).alias("roc10"),
            roc_expr("close", 20).over([col("ticker")]).alias("roc20"),
            roc_expr("close", 100).over([col("ticker")]).alias("roc100"),
            roc_expr("close", 200).over([col("ticker")]).alias("roc200"),
        ])
        .collect()?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_moving_averages_adds_expected_values() {
        let df = df![
            "ticker" => ["A", "A", "A", "A"],
            "close" => [1.0_f64, 2.0, 3.0, 4.0]
        ]
        .expect("failed to create test dataframe");

        let result = exponential_moving_averages(df, vec![3])
            .expect("failed to calculate exponential moving averages");

        let ema = result
            .column("ema3")
            .expect("ema3 column should exist")
            .f64()
            .expect("ema3 should be float64");

        let actual: Vec<f64> = ema.into_no_null_iter().collect();
        let expected = [1.0_f64, 1.5, 2.25, 3.125];

        for (value, exp) in actual.iter().zip(expected.iter()) {
            assert!((value - exp).abs() < 1e-10, "expected {exp}, got {value}");
        }
    }
}

/// Calculate rolling moving averages for multiple periods
///
/// Adds moving average columns for specified periods on the close price
///
/// # Arguments
/// * `df` - Input DataFrame (should be sorted by ticker and date)
/// * `periods` - Vector of periods to calculate moving averages for
///
/// # Returns
/// * `Result<DataFrame>` - DataFrame with added MA columns (e.g., ma10, ma20, ma50, ma200)
pub fn moving_averages(df: DataFrame, periods: Vec<usize>) -> Result<DataFrame> {
    let mut cols: Vec<Expr> = Vec::new();

    for period in periods {
        cols.push(
            rolling_mean_expr("close", period)
                .over([col("ticker")])
                .alias(&format!("ma{}", period)),
        );
    }

    let result = df.lazy().with_columns(cols).collect()?;

    Ok(result)
}

/// Calculate exponential moving averages (EMA) for multiple periods
///
/// Adds EMA columns for specified periods on the close price
///
/// # Arguments
/// * `df` - Input DataFrame (should be sorted by ticker and date)
/// * `periods` - Vector of periods to calculate EMAs for
///
/// # Returns
/// * `Result<DataFrame>` - DataFrame with added EMA columns (e.g., ema10, ema20, ema50, ema200)
pub fn exponential_moving_averages(df: DataFrame, periods: Vec<usize>) -> Result<DataFrame> {
    let mut cols: Vec<Expr> = Vec::new();

    for period in periods {
        cols.push(
            exponential_moving_average_expr("close", period)
                .over([col("ticker")])
                .alias(&format!("ema{}", period)),
        );
    }

    let result = df.lazy().with_columns(cols).collect()?;

    Ok(result)
}

/// Calculate Ichimoku Cloud indicator
///
/// Adds the following Ichimoku components:
/// - tenkan_sen: (9-period high + 9-period low) / 2 (Conversion Line)
/// - kijun_sen: (26-period high + 26-period low) / 2 (Base Line)
/// - senkou_span_a: (tenkan_sen + kijun_sen) / 2, shifted 26 periods forward (Leading Span A)
/// - senkou_span_b: (52-period high + 52-period low) / 2, shifted 26 periods forward (Leading Span B)
/// - chikou_span: Close price shifted 26 periods backward (Lagging Span)
///
/// # Arguments
/// * `df` - Input DataFrame (should be sorted by ticker and date)
///
/// # Returns
/// * `Result<DataFrame>` - DataFrame with added Ichimoku indicator columns
pub fn ichimoku(df: DataFrame) -> Result<DataFrame> {
    let result = df
        .lazy()
        .with_columns([
            // Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            ((col("high").rolling_max(RollingOptionsFixedWindow {
                window_size: 9,
                min_periods: 1,
                ..Default::default()
            }) + col("low").rolling_min(RollingOptionsFixedWindow {
                window_size: 9,
                min_periods: 1,
                ..Default::default()
            })) / lit(2.0))
            .over([col("ticker")])
            .alias("tenkan_sen"),
            // Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            ((col("high").rolling_max(RollingOptionsFixedWindow {
                window_size: 26,
                min_periods: 1,
                ..Default::default()
            }) + col("low").rolling_min(RollingOptionsFixedWindow {
                window_size: 26,
                min_periods: 1,
                ..Default::default()
            })) / lit(2.0))
            .over([col("ticker")])
            .alias("kijun_sen"),
        ])
        .with_columns([
            // Senkou Span A: (tenkan_sen + kijun_sen) / 2, shifted 26 periods forward
            ((col("tenkan_sen") + col("kijun_sen")) / lit(2.0))
                .shift(lit(-26))
                .over([col("ticker")])
                .alias("senkou_span_a"),
            // Senkou Span B: (52-period high + 52-period low) / 2, shifted 26 periods forward
            ((col("high").rolling_max(RollingOptionsFixedWindow {
                window_size: 52,
                min_periods: 1,
                ..Default::default()
            }) + col("low").rolling_min(RollingOptionsFixedWindow {
                window_size: 52,
                min_periods: 1,
                ..Default::default()
            })) / lit(2.0))
            .shift(lit(-26))
            .over([col("ticker")])
            .alias("senkou_span_b"),
            // Chikou Span: Close price shifted 26 periods backward
            col("close")
                .shift(lit(26))
                .over([col("ticker")])
                .alias("chikou_span"),
        ])
        .collect()?;

    Ok(result)
}

/// Calculate Average True Range (ATR)
///
/// ATR is a volatility indicator that measures the degree of price volatility.
/// It calculates the True Range (TR) as the greatest of:
/// 1. Current High - Current Low
/// 2. Abs(Current High - Previous Close)
/// 3. Abs(Current Low - Previous Close)
///
/// Then calculates ATR as a moving average of TR over the specified period.
///
/// # Arguments
/// * `df` - Input DataFrame (should be sorted by ticker and date)
/// * `period` - Period for the ATR moving average (default is typically 14)
///
/// # Returns
/// * `Result<DataFrame>` - DataFrame with added true_range and atr columns
pub fn atr(df: DataFrame, period: usize) -> Result<DataFrame> {
    let result = df
        .lazy()
        .with_columns([
            // Calculate the three components of True Range
            // TR1: high - low
            (col("high") - col("low")).alias("tr1"),
            // TR2: abs(high - previous close)
            (col("high") - col("close").shift(lit(1)))
                .abs()
                .over([col("ticker")])
                .alias("tr2"),
            // TR3: abs(low - previous close)
            (col("low") - col("close").shift(lit(1)))
                .abs()
                .over([col("ticker")])
                .alias("tr3"),
        ])
        .with_columns([
            // True Range is the maximum of tr1, tr2, tr3
            when(
                col("tr1")
                    .gt_eq(col("tr2"))
                    .and(col("tr1").gt_eq(col("tr3"))),
            )
            .then(col("tr1"))
            .otherwise(
                when(col("tr2").gt_eq(col("tr3")))
                    .then(col("tr2"))
                    .otherwise(col("tr3")),
            )
            .alias("true_range"),
        ])
        .drop(["tr1", "tr2", "tr3"])
        .with_columns([
            // ATR is the moving average of True Range
            rolling_mean_expr("true_range", period)
                .over([col("ticker")])
                .alias(&format!("atr{}", period)),
        ])
        .collect()?;

    Ok(result)
}
