//! Downloads configured NASDAQ bulk datasets concurrently.
use anyhow::{Result, anyhow};
use futures::stream::{self, FuturesUnordered, StreamExt};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use tokio::{fs, io::AsyncWriteExt};

use crate::{
    api::{get_logodev_api, nasdaq_api_get},
    config::{DOWNLOADS_DIR, PathSpec},
    filetools::extract_zip_file,
    ui::new_progress_bar,
};

async fn download_one(api_key: Arc<str>, spec: PathSpec) -> bool {
    let PathSpec {
        path,
        query,
        output,
    } = spec;

    let mut stream = match nasdaq_api_get(&path, &api_key, query.as_ref()).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("✗ {} -> download failed: {}", path, e);
            return false;
        }
    };

    let filepath = match output {
        Some(name) => PathBuf::from(DOWNLOADS_DIR).join(name),
        None => {
            let base = path
                .strip_suffix(".json")
                .unwrap_or(&path)
                .replace('/', "_")
                + "_data.zip";
            PathBuf::from(DOWNLOADS_DIR).join(base)
        }
    };

    if let Some(parent) = filepath.parent()
        && let Err(e) = fs::create_dir_all(parent).await
    {
        eprintln!("✗ {} -> mkdir failed: {}", path, e);
        return false;
    }

    let mut file = match fs::File::create(&filepath).await {
        Ok(f) => f,
        Err(e) => {
            eprintln!("✗ {} -> save failed: {}", path, e);
            return false;
        }
    };

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                eprintln!("✗ {} -> stream read failed: {}", path, e);
                return false;
            }
        };
        if let Err(e) = file.write_all(&chunk).await {
            eprintln!("✗ {} -> write failed: {}", path, e);
            return false;
        }
    }

    if let Err(e) = file.flush().await {
        eprintln!("✗ {} -> flush failed: {}", path, e);
        return false;
    }

    let is_zip = filepath
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("zip"))
        .unwrap_or(false);
    if is_zip {
        let extracted = tokio::task::spawn_blocking(move || extract_zip_file(filepath)).await;
        match extracted {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                eprintln!("✗ {} -> extract failed: {}", path, e);
                return false;
            }
            Err(e) => {
                eprintln!("✗ {} -> extract task failed: {}", path, e);
                return false;
            }
        }
    }
    true
}

/// Downloads all configured datasets and extracts ZIP payloads.
///
/// # Failure
/// Returns an error if any download task fails.
pub async fn run_downloader(api_key: &str, specs: Vec<PathSpec>) -> Result<()> {
    let start = Instant::now();
    println!("=== NASDAQ Data Downloader ===");

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
    let pb = new_progress_bar(total as u64, "downloads");

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

/// Reads `downloads/companies.csv` and downloads logos for each ticker concurrently.
///
/// # Failure
/// Returns an error when `companies.csv` is missing/invalid.
pub async fn run_logo_downloader() -> Result<()> {
    let file = File::open("downloads/companies.csv")?;
    let mut lines = BufReader::new(file).lines();

    let header = lines
        .next()
        .transpose()?
        .ok_or_else(|| anyhow!("downloads/companies.csv is empty"))?;
    let ticker_index = header
        .split(',')
        .position(|column| column.trim() == "ticker")
        .ok_or_else(|| anyhow!("'ticker' column not found in downloads/companies.csv"))?;

    stream::iter(lines)
        .for_each_concurrent(Some(64), move |line_result| async move {
            let line = match line_result {
                Ok(line) => line,
                Err(e) => {
                    eprintln!("Failed to read CSV line: {e}");
                    return;
                }
            };

            let Some(ticker) = line
                .split(',')
                .nth(ticker_index)
                .map(str::trim)
                .filter(|ticker| !ticker.is_empty())
            else {
                return;
            };

            print!("https://img.logo.dev/ticker/{}", ticker);
            if let Err(e) = get_logodev_api(ticker, Some(100), Some("webp")).await {
                eprintln!("Failed to fetch logo for {ticker}: {e}");
            }
        })
        .await;

    Ok(())
}
