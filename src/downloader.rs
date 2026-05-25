use anyhow::{Result, anyhow};
use futures::stream::{FuturesUnordered, StreamExt};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tokio::{fs, io::AsyncWriteExt};

use crate::{
    api::nasdaq_api_get,
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
            let base = path.replace('/', "_").replace(".json", "") + "_data.zip";
            PathBuf::from(DOWNLOADS_DIR).join(base)
        }
    };

    if let Some(parent) = filepath.parent() {
        if let Err(e) = fs::create_dir_all(parent).await {
            eprintln!("✗ {} -> mkdir failed: {}", path, e);
            return false;
        }
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
        let zip_path = filepath.clone();
        let extracted = tokio::task::spawn_blocking(move || extract_zip_file(&zip_path)).await;
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
