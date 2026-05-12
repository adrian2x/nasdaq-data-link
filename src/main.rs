use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{self, Write},
    path::Path,
    time::Instant,
};

mod api;
use api::{get_logodev_api, nasdaq_api_get};

mod dataframetools;
mod sqltools;

const OUTPUT_DIR: &str = "output";

/// Save data to the specified file path
///
/// # Arguments
/// * `data` - The binary data to write to file
/// * `filepath` - The complete file path including filename and extension
///
/// # Returns
/// * `Result<String>` - The full path to the saved file
///
/// # Examples
/// ```
/// save_file(&zip_data, "output/data.zip")?;           // Creates output/data.zip
/// save_file(&json_data, "output/configs/app.json")?;  // Creates output/configs/app.json
/// save_file(&csv_data, "exports/2023/prices.csv")?;   // Creates exports/2023/prices.csv
/// ```
fn save_file(data: &[u8], filepath: &str) -> Result<String> {
    let path = Path::new(filepath);

    // Create parent directories if they don't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("Failed to create directories for '{}': {}", filepath, e))?;
    }

    // Write the file
    std::fs::write(path, data)
        .map_err(|e| anyhow!("Failed to write file '{}': {}", filepath, e))?;

    Ok(filepath.to_string())
}

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

/// Ensure output directory exists, create if it doesn't
fn ensure_output_dir() -> Result<()> {
    let output_path = Path::new(OUTPUT_DIR);

    if !output_path.exists() {
        println!("Creating output directory: {}", OUTPUT_DIR);
        std::fs::create_dir(output_path)
            .map_err(|e| anyhow!("Failed to create output directory '{}': {}", OUTPUT_DIR, e))?;
    }

    Ok(())
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

#[derive(Debug)]
struct DownloadResult {
    message: String,
    duration: f64,
    success: bool,
}

/// Process a single path with timing - optimized for performance
async fn process_single_download(
    api_key: &str,
    path: String,
    query_params: Option<HashMap<String, String>>,
    custom_output_filename: Option<String>,
) -> Result<DownloadResult> {
    let start_time = Instant::now();

    let result = nasdaq_api_get(&path, api_key, query_params).await;

    match result {
        Ok(response) => {
            // Use custom filename if provided, otherwise use default naming
            let filepath = if let Some(custom_filename) = custom_output_filename {
                format!("{}/{}", OUTPUT_DIR, custom_filename)
            } else {
                let base_filename = path.replace('/', "_").replace(".json", "") + "_data";
                format!("{}/{}.zip", OUTPUT_DIR, base_filename)
            };

            match save_file(&response.body, &filepath) {
                Ok(saved_filename) => {
                    let duration = start_time.elapsed().as_secs_f64();
                    Ok(DownloadResult {
                        message: format!(
                            "✓ {} -> {} ({} bytes, {:.2}s)",
                            path,
                            saved_filename,
                            response.body.len(),
                            duration
                        ),
                        duration,
                        success: true,
                    })
                }
                Err(e) => {
                    let duration = start_time.elapsed().as_secs_f64();
                    eprintln!("✗ {} -> Save failed: {}", path, e);
                    Ok(DownloadResult {
                        message: format!("✗ {} -> Save failed: {} ({:.2}s)", path, e, duration),
                        duration,
                        success: false,
                    })
                }
            }
        }
        Err(e) => {
            let duration = start_time.elapsed().as_secs_f64();
            eprintln!("✗ {} -> Download failed: {}", path, e);
            Ok(DownloadResult {
                message: format!("✗ {} -> Download failed: {} ({:.2}s)", path, e, duration),
                duration,
                success: false,
            })
        }
    }
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

    let lines: Vec<&str> = paths_content.lines().collect();
    let mut tasks = Vec::new();

    // Collect all valid paths and create concurrent tasks
    for line in lines {
        if let Some((path, query_params, output_filename)) = parse_path_line(line) {
            let api_key = api_key.to_string();
            tasks.push(tokio::spawn(async move {
                process_single_download(&api_key, path, query_params, output_filename).await
            }));
        }
    }

    println!("Starting {} concurrent downloads...", tasks.len());

    // Create progress bar for NASDAQ downloads
    let nasdaq_pb = ProgressBar::new(tasks.len() as u64);
    nasdaq_pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );
    nasdaq_pb.set_message("NASDAQ datasets");

    // Process tasks and update progress as they complete
    let mut completed_results = Vec::new();
    let mut pending_tasks = tasks;

    while !pending_tasks.is_empty() {
        // Use select to wait for the first task to complete
        let (result, _index, remaining) = futures::future::select_all(pending_tasks).await;
        completed_results.push(result);
        pending_tasks = remaining;

        // Update progress
        nasdaq_pb.inc(1);
    }

    let results = completed_results;
    nasdaq_pb.finish_with_message("NASDAQ downloads completed!");

    let mut success_count = 0;
    let mut failed_count = 0;

    // Process results silently
    for result in results {
        match result {
            Ok(Ok(download_result)) => {
                if download_result.success {
                    success_count += 1;
                } else {
                    failed_count += 1;
                }
            }
            Ok(Err(_)) | Err(_) => {
                failed_count += 1;
            }
        }
    }

    let batch_duration = batch_start_time.elapsed();
    let total_processed = success_count + failed_count;

    println!("\n=== Concurrent Processing Complete ===");
    println!("Total processed: {} paths", total_processed);
    println!("Successful: {} downloads", success_count);
    println!("Failed: {} downloads", failed_count);
    println!("Total time: {:.2} seconds", batch_duration.as_secs_f64());

    Ok(())
}

/// Process companies.csv file and download logos for all tickers
///
/// This function:
/// 1. Looks for companies.csv or companies.csv.zip in the output directory
/// 2. Unzips if needed and reads the CSV into a Polars DataFrame
/// 3. Extracts all ticker symbols from the "ticker" column
/// 4. Downloads logos for all tickers using the Logo.dev API
///
/// # Returns
/// * `Result<()>` - Success or error
async fn process_companies_and_download_logos() -> Result<()> {
    println!("\n=== Processing Companies and Downloading Logos ===");

    // Check for companies.csv.zip first, then companies.csv
    let csv_path = if Path::new("output/companies.csv.zip").exists() {
        println!("Found companies.csv.zip, extracting...");

        // Extract the zip file
        let file = File::open("output/companies.csv.zip")
            .map_err(|e| anyhow!("Failed to open companies.csv.zip: {}", e))?;

        let mut archive = ::zip::ZipArchive::new(file)
            .map_err(|e| anyhow!("Failed to read zip archive: {}", e))?;

        // Look for CSV file in the archive
        let mut csv_found = false;
        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| anyhow!("Failed to access file {} in zip: {}", i, e))?;

            if file.name().ends_with(".csv") {
                println!("Extracting {} from zip file...", file.name());

                let mut contents = Vec::new();
                std::io::copy(&mut file, &mut contents)
                    .map_err(|e| anyhow!("Failed to read CSV from zip: {}", e))?;

                // Write extracted CSV to output directory
                std::fs::write("output/companies.csv", contents)
                    .map_err(|e| anyhow!("Failed to save extracted CSV: {}", e))?;

                csv_found = true;
                break;
            }
        }

        if !csv_found {
            return Err(anyhow!("No CSV file found in companies.csv.zip"));
        }

        "output/companies.csv"
    } else if Path::new("output/companies.csv").exists() {
        println!("Found companies.csv file");
        "output/companies.csv"
    } else {
        return Err(anyhow!(
            "Neither companies.csv nor companies.csv.zip found in output directory"
        ));
    };

    // Read CSV file and extract tickers
    println!("Reading CSV file and extracting tickers...");
    let csv_content =
        std::fs::read_to_string(csv_path).map_err(|e| anyhow!("Failed to read CSV file: {}", e))?;

    let mut tickers = Vec::new();
    let mut ticker_column_index = None;

    for (line_idx, line) in csv_content.lines().enumerate() {
        let fields: Vec<&str> = line
            .split(',')
            .map(|s| s.trim().trim_matches('"'))
            .collect();

        if line_idx == 0 {
            // Find ticker column index
            for (col_idx, field) in fields.iter().enumerate() {
                if field.to_lowercase() == "ticker" {
                    ticker_column_index = Some(col_idx);
                    break;
                }
            }

            if ticker_column_index.is_none() {
                return Err(anyhow!(
                    "No 'ticker' column found in CSV. Available columns: {:?}",
                    fields
                ));
            }

            println!(
                "Found ticker column at index {}",
                ticker_column_index.unwrap()
            );
        } else if let Some(col_idx) = ticker_column_index {
            // Extract ticker from this row
            if let Some(ticker) = fields.get(col_idx) {
                if !ticker.is_empty() {
                    tickers.push(ticker.to_string());
                }
            }
        }
    }

    println!("Found {} tickers to process", tickers.len());

    if tickers.is_empty() {
        println!("No tickers found, skipping logo downloads");
        return Ok(());
    }

    // Download all logos concurrently
    println!("\nDownloading {} logos concurrently...", tickers.len());

    let mut tasks = Vec::new();

    for ticker in tickers {
        tasks.push(tokio::spawn(async move {
            match get_logodev_api(&ticker, None, None).await {
                Ok(_) => Ok(()),
                Err(e) => Err(e),
            }
        }));
    }

    // Create progress bar for logo downloads
    let logo_pb = ProgressBar::new(tasks.len() as u64);
    logo_pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.green/blue} {pos}/{len} {msg}")
            .expect("Failed to set progress bar template")
            .progress_chars("#>-"),
    );
    logo_pb.set_message("company logos");

    // Process tasks and update progress as they complete
    let mut completed_results = Vec::new();
    let mut pending_tasks = tasks;

    while !pending_tasks.is_empty() {
        // Use select to wait for the first task to complete
        let (result, _index, remaining) = futures::future::select_all(pending_tasks).await;
        completed_results.push(result);
        pending_tasks = remaining;

        // Update progress
        logo_pb.inc(1);
    }

    let results = completed_results;
    logo_pb.finish_with_message("Logo downloads completed!");

    let mut success_count = 0;
    let mut failed_count = 0;

    for result in results {
        match result {
            Ok(Ok(())) => success_count += 1,
            Ok(Err(_)) | Err(_) => failed_count += 1,
        }
    }

    println!("\n=== Logo Download Complete ===");
    println!("Successful: {} logos", success_count);
    println!("Failed: {} logos", failed_count);
    println!("Total processed: {} tickers", success_count + failed_count);

    Ok(())
}

async fn downloader() -> Result<()> {
    let start_time = Instant::now();

    println!("=== NASDAQ Data Downloader ===");
    println!("This tool downloads data from NASDAQ's API.\n");

    // Ensure output directory exists
    ensure_output_dir()?;

    // Load API key from .env file or prompt user
    let api_key = load_or_create_api_key()?;

    // Check if paths.txt exists and choose mode accordingly
    if Path::new("paths.txt").exists() {
        println!("Found paths.txt file. Processing all paths in batch mode...");
        process_all_paths(&api_key).await?;

        // After all downloads complete, process companies CSV and download logos
        // if let Err(e) = process_companies_and_download_logos().await {
        //     eprintln!("Warning: Failed to process companies and download logos: {}", e);
        //     eprintln!("This is not critical - the main data downloads were successful.");
        // }
    } else {
        println!("No paths.txt file found. Using interactive mode...");
        process_single_path(&api_key).await?;
    }

    // Report total execution time
    // let duration = start_time.elapsed();
    // println!("\n=== Execution Complete ===");
    // println!("Total execution time: {:.2} seconds", duration.as_secs_f64());

    Ok(())
}

async fn write_stock_prices() -> Result<()> {
    let prices = dataframetools::adjust_prices("output/stocks_eod.csv.zip")?;
    sqltools::df_to_sqlite(prices, "nasdaq.db", "stock_prices").await?;
    Ok(())
}

async fn write_fundamentals() -> Result<()> {
    let prices = dataframetools::adjust_fundamentals("output/financials_ttm.csv.zip")?;
    sqltools::df_to_sqlite(prices, "nasdaq.db", "financials_ttm").await?;
    Ok(())
}

async fn writer() -> Result<()> {
    write_stock_prices().await?;
    write_fundamentals().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let sync_only = env::args().skip(1).any(|arg| arg == "--sync");
    if sync_only {
        return writer().await;
    }

    downloader().await
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

            let zip_filepath = format!("{}/{}.zip", OUTPUT_DIR, base_filename);
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
