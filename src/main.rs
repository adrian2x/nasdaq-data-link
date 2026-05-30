//! Runs the NASDAQ dataset download and output-table build pipeline.
mod api;
mod config;
mod downloader;
mod filetools;
mod fractaltools;
pub mod indicators;
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
    let mut sync = false;
    let mut logos = false;
    let mut export_arrow = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--sync" => sync = true,
            "--logos" => logos = true,
            "--export-arrow" => export_arrow = true,
            _ => {}
        }
    }

    if sync {
        return writer::run_writer(export_arrow);
    }

    if logos {
        return downloader::run_logo_downloader().await;
    }

    let specs = config::load_path_specs()?;
    downloader::run_downloader(&api_key, specs).await?;
    writer::run_writer(export_arrow)
}
