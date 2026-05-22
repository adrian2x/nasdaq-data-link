mod api;
mod config;
mod dataframetools;
mod downloader;
mod filetools;
mod fractaltools;
mod sqltools;
mod ui;
mod writer;

use anyhow::Result;
use filetools::ensure_directory;

#[tokio::main]
async fn main() -> Result<()> {
    let v: String =
        duckdb::Connection::open_in_memory()?.query_row("SELECT version()", [], |r| r.get(0))?;
    println!("DuckDB engine: {v}");
    ensure_directory(config::DOWNLOADS_DIR)?;
    ensure_directory(config::OUTPUT_DIR)?;

    if std::env::args().skip(1).any(|arg| arg == "--sync") {
        return writer::run_writer();
    }

    let api_key = config::load_or_create_api_key()?;
    let specs = config::load_path_specs()?;
    downloader::run_downloader(&api_key, specs).await?;
    writer::run_writer()
}
