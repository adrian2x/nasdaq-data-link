use anyhow::{Result, anyhow};
use futures::stream::{FuturesUnordered, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::{DataFrame, DataType};
use std::{
    collections::HashMap,
    env,
    io::{self, Write},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

mod api;
use api::nasdaq_api_get;

mod dataframetools;
mod filetools;
mod fractaltools;
use filetools::{ensure_directory, save_file};

const DOWNLOADS_DIR: &str = "downloads";
const OUTPUT_DIR: &str = "output";
const PATHS_FILE: &str = "paths.txt";
const ENV_FILE: &str = ".env";

// ---------------------------------------------------------------------------
// API key handling
// ---------------------------------------------------------------------------

/// Prompt the user via stdin and trim the response.
fn prompt(message: &str) -> Result<String> {
    print!("{}", message);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

/// Load the NASDAQ API key from .env, or prompt and persist it on first run.
fn load_or_create_api_key() -> Result<String> {
    if Path::new(ENV_FILE).exists() {
        dotenv::dotenv().ok();
        if let Ok(key) = env::var("NASDAQ_API_KEY") {
            if !key.is_empty() {
                return Ok(key);
            }
        }
    }

    println!("\nNo NASDAQ API key found.");
    println!("Get one for free at https://data.nasdaq.com/, then paste it below.");

    let key = loop {
        let entered = prompt("API key: ")?;
        if !entered.is_empty() {
            break entered;
        }
        println!("Key cannot be empty.");
    };

    // Persist on first run so the user only has to do this once.
    let env_content = format!("# NASDAQ API Configuration\nNASDAQ_API_KEY={}\n", key);
    match std::fs::write(ENV_FILE, env_content) {
        Ok(_) => println!("✓ Saved to .env (add it to .gitignore to keep it private)"),
        Err(e) => eprintln!("Warning: could not write .env: {}", e),
    }

    Ok(key)
}

// ---------------------------------------------------------------------------
// paths.txt parsing
// ---------------------------------------------------------------------------

/// A single dataset to download, parsed from a line in paths.txt.
struct PathSpec {
    path: String,
    query: Option<HashMap<String, String>>,
    output: Option<String>,
}

/// Parse a `key=value&key=value` query string into a map. Invalid pairs are
/// skipped with a warning.
fn parse_query(s: &str) -> HashMap<String, String> {
    s.split('&')
        .filter_map(|pair| {
            let pair = pair.trim();
            match pair.split_once('=') {
                Some((k, v)) => Some((k.trim().to_string(), v.trim().to_string())),
                None if pair.is_empty() => None,
                None => {
                    eprintln!("Warning: ignoring invalid query parameter: {}", pair);
                    None
                }
            }
        })
        .collect()
}

/// Parse a single paths.txt line. Format: `PATH [-q QUERY] [-o OUTPUT]`.
/// Flags may appear in any order. Returns `None` for blank lines and comments.
fn parse_path_line(line: &str) -> Option<PathSpec> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    // Tokenize on whitespace; first token is the path, remaining tokens are
    // flag/value pairs. Values are *not* shell-quoted (paths.txt is internal).
    let mut tokens = line.split_whitespace();
    let path = tokens.next()?.to_string();

    let mut query = None;
    let mut output = None;
    while let Some(flag) = tokens.next() {
        let value = tokens.next().unwrap_or_else(|| {
            eprintln!("Warning: flag {} in '{}' has no value", flag, line);
            ""
        });
        match flag {
            "-q" => {
                let parsed = parse_query(value);
                query = (!parsed.is_empty()).then_some(parsed);
            }
            "-o" => {
                if !value.is_empty() {
                    output = Some(value.to_string());
                }
            }
            other => eprintln!("Warning: unknown flag '{}' in '{}'", other, line),
        }
    }

    Some(PathSpec {
        path,
        query,
        output,
    })
}

// ---------------------------------------------------------------------------
// Downloader
// ---------------------------------------------------------------------------

/// Download a single dataset and write the response body to disk.
async fn download_one(api_key: Arc<str>, spec: PathSpec) -> bool {
    let PathSpec {
        path,
        query,
        output,
    } = spec;

    let response = match nasdaq_api_get(&path, &api_key, query).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("✗ {} -> download failed: {}", path, e);
            return false;
        }
    };

    let filepath = match output {
        Some(name) => format!("{}/{}", DOWNLOADS_DIR, name),
        None => {
            let base = path.replace('/', "_").replace(".json", "") + "_data";
            format!("{}/{}.zip", DOWNLOADS_DIR, base)
        }
    };

    if let Err(e) = save_file(&response.body, &filepath) {
        eprintln!("✗ {} -> save failed: {}", path, e);
        return false;
    }
    true
}

/// Read paths.txt, parse it, and download every entry concurrently.
async fn run_downloader(api_key: &str) -> Result<()> {
    let start = Instant::now();
    println!("=== NASDAQ Data Downloader ===");

    let paths_content = std::fs::read_to_string(PATHS_FILE).map_err(|e| {
        anyhow!(
            "Failed to read {}: {}. Create it with one entry per line, e.g.:\n  SHARADAR/SF1.json -q dimension=MRT -o financials.csv",
            PATHS_FILE,
            e
        )
    })?;

    let specs: Vec<PathSpec> = paths_content.lines().filter_map(parse_path_line).collect();
    if specs.is_empty() {
        return Err(anyhow!(
            "{} contained no valid entries (only blank lines or comments).",
            PATHS_FILE
        ));
    }

    let shared_key: Arc<str> = Arc::from(api_key);
    let mut tasks: FuturesUnordered<_> = specs
        .into_iter()
        .map(|spec| {
            let key = Arc::clone(&shared_key);
            tokio::spawn(download_one(key, spec))
        })
        .collect();

    let total = tasks.len();
    println!("Starting {} concurrent downloads...", total);

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("invalid progress bar template")
            .progress_chars("#>-"),
    );
    pb.set_message("downloads");

    let (mut ok, mut fail) = (0_usize, 0_usize);
    while let Some(result) = tasks.next().await {
        match result {
            Ok(true) => ok += 1,
            Ok(false) | Err(_) => fail += 1,
        }
        pb.inc(1);
    }
    pb.finish_with_message("done");

    println!(
        "\nDownloaded {}/{} in {:.2}s ({} failed)",
        ok,
        total,
        start.elapsed().as_secs_f64(),
        fail
    );

    if fail > 0 {
        Err(anyhow!("{} downloads failed", fail))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Writer (sync — no concurrent work, runs on the calling thread)
// ---------------------------------------------------------------------------

/// Run a closure while displaying an animated spinner with `label`. When the
/// closure returns, the spinner is cleared and replaced by a one-line summary
/// (`✓ label (1.23s)` on success, `✗ label` on error).
///
/// The spinner runs on its own thread via `enable_steady_tick`, redrawing at
/// 80ms intervals — negligible CPU cost. Any `println!` output from inside
/// the closure scrolls cleanly above the spinner line.
fn with_spinner<T>(label: &str, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("invalid spinner template"),
    );
    pb.set_message(label.to_string());
    pb.enable_steady_tick(Duration::from_millis(80));

    let start = Instant::now();
    match work() {
        Ok(value) => {
            pb.finish_and_clear();
            println!("✓ {} ({:.2}s)", label, start.elapsed().as_secs_f64());
            Ok(value)
        }
        Err(e) => {
            pb.finish_and_clear();
            println!("✗ {}", label);
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// Writer pipeline
//
// Each `write_*` function owns the full lifecycle for one data domain: load
// the CSV, transform it, write to DuckDB. Functions that produce data used
// downstream return the relevant DataFrame; terminal functions return ().
// `run_writer` is the thin orchestrator that wires them together.
// ---------------------------------------------------------------------------

/// Load → adjust → indicators → write stocks (daily and weekly).
/// Returns the daily indicators frame because `write_companies` needs it for
/// the latest-snapshot join.
fn write_stocks() -> Result<DataFrame> {
    use dataframetools::*;

    // Pin the 6 numeric columns at parse time to skip inference for them on
    // a 45M-row file. `date` is left as String here and cast to Date inside
    // `adjust_prices` so downstream temporal ops (resample) work.
    let raw = with_spinner("loading raw prices", || {
        load_csv_zip(
            &format!("{}/stocks_eod.csv.zip", DOWNLOADS_DIR),
            Some(&[
                ("open", DataType::Float64),
                ("high", DataType::Float64),
                ("low", DataType::Float64),
                ("close", DataType::Float64),
                ("closeadj", DataType::Float64),
                ("volume", DataType::Float64),
            ]),
        )
    })?;
    let adjusted = with_spinner("adjusting prices", || adjust_prices(raw))?;

    // Daily pipeline: Hurst at 500-bar (~2y) window, then daily indicators.
    let daily_hurst_cfg = fractaltools::HurstConfig::default(); // window=500, vol_window=20
    let daily_with_hurst = with_spinner("computing daily hurst", || {
        fractaltools::with_hurst(adjusted.clone(), daily_hurst_cfg).map_err(anyhow::Error::from)
    })?;
    let stocks_daily = with_spinner("computing daily indicators", || {
        technical_indicators_daily(daily_with_hurst)
    })?;
    with_spinner("writing stocks_daily to duckdb", || {
        df_to_duckdb(&stocks_daily, "stocks_daily")
    })?;

    // Weekly pipeline: resample on adjusted daily bars (correct order —
    // aggregating unadjusted bars then adjusting after would produce
    // incoherent intrabar ranges), Hurst at 100-bar (~2y of weeks) window,
    // then weekly indicators.
    let weekly_bars = with_spinner("resampling to weekly", || resample(adjusted, "1w"))?;
    let weekly_hurst_cfg = fractaltools::HurstConfig {
        window: 100,   // ~2 years of weekly bars
        vol_window: 4, // ~1 month
        ..Default::default()
    };
    let weekly_with_hurst = with_spinner("computing weekly hurst", || {
        fractaltools::with_hurst(weekly_bars, weekly_hurst_cfg).map_err(anyhow::Error::from)
    })?;
    let stocks_weekly = with_spinner("computing weekly indicators", || {
        technical_indicators_weekly(weekly_with_hurst)
    })?;
    with_spinner("writing stocks_weekly to duckdb", || {
        df_to_duckdb(&stocks_weekly, "stocks_weekly")
    })?;

    Ok(stocks_daily)
}

/// Load → adjust → write fundamentals. Returns the adjusted frame because
/// `write_companies` needs it for the latest-snapshot join.
fn write_financials() -> Result<DataFrame> {
    use dataframetools::*;

    let raw = with_spinner("loading raw fundamentals", || {
        load_csv_zip(&format!("{}/financials_ttm.csv.zip", DOWNLOADS_DIR), None)
    })?;
    let financials_ttm = with_spinner("adjusting fundamentals", || adjust_fundamentals(raw))?;
    with_spinner("writing financials_ttm to duckdb", || {
        df_to_duckdb(&financials_ttm, "financials_ttm")
    })?;
    Ok(financials_ttm)
}

/// Load → filter → join → write the company snapshot. Joins metadata with
/// the latest fundamentals and latest daily prices per ticker; computes
/// `rs1y` percentile rank, `marketcap`, and `ev` in the process.
fn write_companies(financials_ttm: DataFrame, stocks_daily: DataFrame) -> Result<()> {
    use dataframetools::*;

    let raw = with_spinner("loading raw companies metadata", || {
        load_csv_zip(&format!("{}/companies.csv.zip", DOWNLOADS_DIR), None)
    })?;
    let meta = with_spinner("filtering companies metadata", || {
        filter_companies_meta(raw)
    })?;
    let companies = with_spinner("joining company snapshot", || {
        join_company_financials(financials_ttm, meta, stocks_daily)
    })?;
    with_spinner("writing companies to duckdb", || {
        df_to_duckdb(&companies, "companies")
    })?;
    Ok(())
}

/// Load → transform → write insider transactions. Self-contained, no
/// downstream consumers. `formtype` starts with numeric codes ("4", "5")
/// and then hits strings like "RESTATED - 4" hundreds of rows in, so
/// default inference picks Int64 and crashes mid-file. Pin to String at
/// load time.
fn write_insiders() -> Result<()> {
    use dataframetools::*;

    let raw = with_spinner("loading raw insider transactions", || {
        load_csv_zip(
            &format!("{}/insiders.csv.zip", DOWNLOADS_DIR),
            Some(&[("formtype", DataType::String)]),
        )
    })?;
    let insiders = with_spinner("transforming insider transactions", || update_insiders(raw))?;
    with_spinner("writing insiders to duckdb", || {
        df_to_duckdb(&insiders, "insiders")
    })?;
    Ok(())
}

fn run_writer() -> Result<()> {
    let stocks_daily = write_stocks()?;
    let financials_ttm = write_financials()?;
    write_companies(financials_ttm, stocks_daily)?;
    write_insiders()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    ensure_directory(DOWNLOADS_DIR)?;
    ensure_directory(OUTPUT_DIR)?;

    // --sync skips the downloader and just rebuilds DuckDB from cached files.
    if env::args().skip(1).any(|arg| arg == "--sync") {
        return run_writer();
    }

    let api_key = load_or_create_api_key()?;
    run_downloader(&api_key).await?;
    run_writer()
}
