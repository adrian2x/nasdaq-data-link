use anyhow::{Result, anyhow};
use futures::stream::{FuturesUnordered, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::HashMap,
    env,
    io::{self, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

mod api;
use api::nasdaq_api_get;

mod dataframetools;
mod filetools;
mod fractaltools;
use filetools::{ensure_directory, save_file};

const DOWNLOADS_DIR: &str = "downloads";
const OUTPUT_DIR: &str = "output";

/// Get user input from stdin with better error handling
fn get_user_input(prompt: &str) -> Result<String> {
    print!("{}", prompt);
    io::stdout()
        .flush()
        .map_err(|e| anyhow!("Failed to flush stdout: {}", e))?;

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .map_err(|e| anyhow!("Failed to read user input: {}", e))?;

    Ok(input.trim().to_string())
}

/// Parse query string in format "key=value&key=value" into HashMap
fn parse_query_string(query_str: &str) -> HashMap<String, String> {
    if query_str.is_empty() {
        return HashMap::new();
    }

    query_str
        .split('&')
        .filter_map(|pair| {
            let pair = pair.trim();
            match pair.split_once('=') {
                Some((key, value)) => Some((key.trim().to_string(), value.trim().to_string())),
                None => {
                    eprintln!("Warning: Ignoring invalid query parameter: {}", pair);
                    None
                }
            }
        })
        .collect()
}

/// Load or create .env file with NASDAQ API key
///
/// # Returns
/// * `Result<String>` - The NASDAQ API key
fn load_or_create_api_key() -> Result<String> {
    let env_file = ".env";

    // Try to load existing .env file
    if Path::new(env_file).exists() {
        println!("Loading API key from .env file...");
        dotenv::dotenv().ok();

        if let Ok(api_key) = env::var("NASDAQ_API_KEY") {
            if !api_key.is_empty() {
                println!("✓ API key loaded from .env file");
                return Ok(api_key);
            }
        }

        println!("Warning: .env file exists but NASDAQ_API_KEY is missing or empty");
    } else {
        println!("No .env file found");
    }

    // Prompt user for API key
    println!("\nNASDAQ API key is required to download data.");
    println!("You can get a free API key from: https://data.nasdaq.com/");

    let api_key = loop {
        match get_user_input("\nEnter your NASDAQ API key: ") {
            Ok(key) if !key.is_empty() => break key,
            Ok(_) => println!("API key cannot be empty. Please try again."),
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                return Err(e);
            }
        }
    };

    // Ask if user wants to save the API key
    println!("\nWould you like to save this API key to a .env file for future use? (y/n)");
    match get_user_input("Save API key: ") {
        Ok(answer) if answer.to_lowercase().starts_with('y') => {
            let env_content = format!("# NASDAQ API Configuration\nNASDAQ_API_KEY={}\n", api_key);
            match std::fs::write(env_file, env_content) {
                Ok(_) => {
                    println!("✓ API key saved to .env file");
                    println!("Note: Add .env to your .gitignore file to keep your API key secure!");
                }
                Err(e) => {
                    eprintln!("✗ Failed to save .env file: {}", e);
                    println!("You'll need to enter your API key each time you run the program.");
                }
            }
        }
        Ok(_) => {
            println!("API key not saved. You'll need to enter it each time you run the program.")
        }
        Err(e) => eprintln!("Error reading input: {}", e),
    }

    Ok(api_key)
}

/// Parse a path line from paths.txt, extracting path, query parameters, and output filename
///
/// # Arguments
/// * `line` - A line from paths.txt (e.g., "SHARADAR/SF1.json -q dimension=MRT -o companies.csv")
///
/// # Returns
/// * `Option<(String, Option<HashMap<String, String>>, Option<String>)>` - Path, optional query params, and optional output filename
fn parse_path_line(
    line: &str,
) -> Option<(String, Option<HashMap<String, String>>, Option<String>)> {
    let line = line.trim();

    // Skip empty lines and comments
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    let mut path = line.to_string();
    let mut query_params: Option<HashMap<String, String>> = None;
    let mut output_filename: Option<String> = None;

    // Parse -q flag for query parameters
    if let Some(q_pos) = line.find(" -q ") {
        path = line[..q_pos].trim().to_string();
        let remaining = &line[q_pos + 4..]; // Skip " -q "

        // Check if there's also an -o flag after -q
        if let Some(o_pos) = remaining.find(" -o ") {
            let query_str = remaining[..o_pos].trim();
            let output_str = remaining[o_pos + 4..].trim(); // Skip " -o "

            if !query_str.is_empty() {
                let parsed_params = parse_query_string(query_str);
                query_params = if parsed_params.is_empty() {
                    None
                } else {
                    Some(parsed_params)
                };
            }

            if !output_str.is_empty() {
                output_filename = Some(output_str.to_string());
            }
        } else {
            // Only -q flag, no -o flag
            let query_str = remaining.trim();
            if !query_str.is_empty() {
                let parsed_params = parse_query_string(query_str);
                query_params = if parsed_params.is_empty() {
                    None
                } else {
                    Some(parsed_params)
                };
            }
        }
    } else if let Some(o_pos) = line.find(" -o ") {
        // Only -o flag, no -q flag
        path = line[..o_pos].trim().to_string();
        let output_str = line[o_pos + 4..].trim(); // Skip " -o "

        if !output_str.is_empty() {
            output_filename = Some(output_str.to_string());
        }
    }

    Some((path, query_params, output_filename))
}

/// Process a single path download and persist the response body.
///
/// Returns `true` on success and `false` when download or save fails.
async fn process_single_download(
    api_key: &str,
    path: String,
    query_params: Option<HashMap<String, String>>,
    custom_output_filename: Option<String>,
) -> bool {
    let response = match nasdaq_api_get(&path, api_key, query_params).await {
        Ok(response) => response,
        Err(e) => {
            eprintln!("✗ {} -> Download failed: {}", path, e);
            return false;
        }
    };

    // Use custom filename if provided, otherwise use default naming.
    let filepath = if let Some(custom_filename) = custom_output_filename {
        format!("{}/{}", DOWNLOADS_DIR, custom_filename)
    } else {
        let base_filename = path.replace('/', "_").replace(".json", "") + "_data";
        format!("{}/{}.zip", DOWNLOADS_DIR, base_filename)
    };

    if let Err(e) = save_file(&response.body, &filepath) {
        eprintln!("✗ {} -> Save failed: {}", path, e);
        return false;
    }

    true
}

/// Process all paths from paths.txt file concurrently
///
/// # Arguments
/// * `api_key` - The NASDAQ API key
///
/// # Returns
/// * `Result<()>` - Success or error
async fn process_all_paths(api_key: &str) -> Result<()> {
    let batch_start_time = Instant::now();

    println!("\n=== Processing All Paths from paths.txt (Concurrent Mode) ===");

    // Read the paths.txt file
    let paths_content = match std::fs::read_to_string("paths.txt") {
        Ok(content) => content,
        Err(e) => {
            return Err(anyhow!(
                "Failed to read paths.txt file: {}. Make sure the file exists.",
                e
            ));
        }
    };

    let shared_api_key: Arc<str> = Arc::from(api_key);
    let mut pending_tasks = FuturesUnordered::new();

    // Collect all valid paths and create concurrent tasks.
    for line in paths_content.lines() {
        if let Some((path, query_params, output_filename)) = parse_path_line(line) {
            let api_key = Arc::clone(&shared_api_key);
            pending_tasks.push(tokio::spawn(async move {
                process_single_download(api_key.as_ref(), path, query_params, output_filename).await
            }));
        }
    }

    let total_tasks = pending_tasks.len();
    println!("Starting {} concurrent downloads...", total_tasks);

    // Create progress bar for NASDAQ downloads
    let nasdaq_pb = ProgressBar::new(total_tasks as u64);
    nasdaq_pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );
    nasdaq_pb.set_message("NASDAQ datasets");

    let mut success_count = 0;
    let mut failed_count = 0;
    // Process task results as they complete (avoids storing all results in memory).
    while let Some(result) = pending_tasks.next().await {
        match result {
            Ok(true) => success_count += 1,
            Ok(false) | Err(_) => failed_count += 1,
        }
        nasdaq_pb.inc(1);
    }
    nasdaq_pb.finish_with_message("NASDAQ downloads completed!");

    let batch_duration = batch_start_time.elapsed();
    let total_processed = success_count + failed_count;

    println!("\n=== Concurrent Processing Complete ===");
    println!("Total processed: {} paths", total_processed);
    println!("Successful: {} downloads", success_count);
    println!("Failed: {} downloads", failed_count);
    println!("Total time: {:.2} seconds", batch_duration.as_secs_f64());

    Ok(())
}

async fn downloader() -> Result<()> {
    println!("=== NASDAQ Data Downloader ===");
    println!("This tool downloads data from NASDAQ's API.\n");
    // Ensure required directories exist
    ensure_directory(DOWNLOADS_DIR)?;
    ensure_directory(OUTPUT_DIR)?;

    // Load API key from .env file or prompt user
    let api_key = load_or_create_api_key()?;

    // Check if paths.txt exists and choose mode accordingly
    if Path::new("paths.txt").exists() {
        println!("Found paths.txt file. Processing all paths in batch mode...");
        process_all_paths(&api_key).await?;
    } else {
        println!("No paths.txt file found. Using interactive mode...");
        process_single_path(&api_key).await?;
    }

    Ok(())
}

async fn writer() -> Result<()> {
    let stock_prices =
        dataframetools::adjust_prices(&format!("{}/stocks_eod.csv.zip", DOWNLOADS_DIR))?;
    let stock_prices = dataframetools::technical_indicators(stock_prices)?;
    dataframetools::write_df_to_duckdb(stock_prices, "stock_prices").await?;
    let financials_ttm =
        dataframetools::adjust_fundamentals(&format!("{}/financials_ttm.csv.zip", DOWNLOADS_DIR))?;
    let financials_ttm =
        dataframetools::write_df_to_duckdb(financials_ttm, "financials_ttm").await?;
    let companies_meta =
        dataframetools::load_companies_meta(&format!("{}/companies.csv.zip", DOWNLOADS_DIR))?;
    let companies = dataframetools::join_company_financials(financials_ttm, companies_meta)?;
    dataframetools::write_df_to_duckdb(companies, "companies").await?;
    let insiders = dataframetools::update_insiders(&format!("{}/insiders.csv.zip", DOWNLOADS_DIR))?;
    dataframetools::write_df_to_duckdb(insiders, "insiders").await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    if env::args().skip(1).any(|arg| arg == "--sync") {
        return writer().await;
    }

    downloader().await?;
    writer().await
}

/// Process a single path interactively
async fn process_single_path(api_key: &str) -> Result<()> {
    // Get dataset path from user
    let path = loop {
        println!("\nEnter the dataset path (e.g., 'WIKI/PRICES', 'ZACKS/FC', 'EOD/AAPL'):");
        match get_user_input("Path: ") {
            Ok(p) if !p.is_empty() => break p,
            Ok(_) => println!("Path cannot be empty. Please try again."),
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                return Err(e);
            }
        }
    };

    // Get query parameters from user
    println!("\nEnter additional query parameters (optional):");
    println!("Format: key=value&key=value (e.g., 'ticker=AAPL&date.gte=2023-01-01')");
    let query_input = match get_user_input("Query params (or press Enter to skip): ") {
        Ok(input) => input,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            return Err(e);
        }
    };

    let query_params = if query_input.is_empty() {
        None
    } else {
        let parsed = parse_query_string(&query_input);
        if parsed.is_empty() {
            println!("No valid query parameters found.");
            None
        } else {
            println!("Parsed query parameters: {:?}", parsed);
            Some(parsed)
        }
    };

    println!("\n=== Making NASDAQ API Request ===");
    println!("Path: {}", path);
    if let Some(ref params) = query_params {
        println!("Query params: {:?}", params);
    }
    println!("");

    let request_start_time = Instant::now();

    // Make the NASDAQ API request
    match nasdaq_api_get(&path, api_key, query_params).await {
        Ok(response) => {
            let download_time = request_start_time.elapsed();
            println!("✓ Success!");
            println!("Status Code: {}", response.status_code);
            println!("Data size: {} bytes", response.body.len());
            println!("Download time: {:.2} seconds", download_time.as_secs_f64());

            // Automatically save the zip file
            let base_filename = format!("{}_data", path.replace('/', "_"));

            let zip_filepath = format!("{}/{}.zip", DOWNLOADS_DIR, base_filename);
            match save_file(&response.body, &zip_filepath) {
                Ok(zip_filename) => {
                    let total_time = request_start_time.elapsed();
                    println!("✓ Zip file saved to: {}", zip_filename);
                    println!(
                        "Total processing time: {:.2} seconds",
                        total_time.as_secs_f64()
                    );
                }
                Err(e) => {
                    eprintln!("✗ Failed to save zip file: {}", e);
                }
            }
        }
        Err(e) => {
            let failed_time = request_start_time.elapsed();
            eprintln!(
                "✗ NASDAQ API request failed (took {:.2}s): {}",
                failed_time.as_secs_f64(),
                e
            );
            println!("\nTroubleshooting tips:");
            println!("1. Check your API key is valid");
            println!("2. Verify the dataset path exists");
            println!("3. Check your query parameters are correct");
            println!("4. Ensure you have sufficient API quota");
        }
    }

    Ok(())
}
