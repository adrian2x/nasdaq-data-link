//! Runs the NASDAQ dataset download and output-table build pipeline.
mod api;
mod config;
mod downloader;
mod filetools;
mod fractaltools;
mod indicators;
mod pipeline;
mod sqltools;
mod ui;
mod writer;

use anyhow::Result;
use filetools::ensure_directory;

#[tokio::main]
async fn main() -> Result<()> {
    ensure_directory(config::DOWNLOADS_DIR)?;
    ensure_directory(config::OUTPUT_DIR)?;
    let api_key = config::load_or_create_api_key()?;
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|arg| arg == "--sync") {
        return writer::run_writer();
    }

    if args.iter().any(|arg| arg == "--logos") {
        return downloader::run_logo_downloader().await;
    }

    let specs = config::load_path_specs()?;
    downloader::run_downloader(&api_key, specs).await?;
    writer::run_writer()
}
