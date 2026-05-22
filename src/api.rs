//! # NASDAQ API Client Module
//!
//! This module provides HTTP client functionality specifically designed for interacting
//! with the NASDAQ Data API. It includes robust retry logic, connection pooling, and
//! optimized performance for downloading large datasets.
//!
//! ## Features
//!
//! - Connection pooling and reuse for improved performance
//! - Exponential backoff retry logic for resilient API calls
//! - Automatic bulk download handling for NASDAQ datasets
//! - Memory-efficient response handling
//!
//! ## Example
//!
//! ```rust
//! use std::collections::HashMap;
//! use nasdaq_api::nasdaq_api_get;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut query = HashMap::new();
//!     query.insert("ticker".to_string(), "AAPL".to_string());
//!     
//!     let response = nasdaq_api_get(
//!         "WIKI/PRICES",
//!         "your-api-key",
//!         Some(query)
//!     ).await?;
//!     
//!     println!("Downloaded {} bytes", response.body.len());
//!     Ok(())
//! }
//! ```

use anyhow::{Result, anyhow};
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

/// HTTP response containing the response body and status information.
///
/// This struct represents a complete HTTP response with the raw body data
/// and status code. It's optimized for memory efficiency by only storing
/// the essential response data.
///
/// # Examples
///
/// ```rust
/// # use nasdaq_api::HttpResponse;
/// # fn main() -> anyhow::Result<()> {
/// let response = HttpResponse {
///     body: b"Hello, world!".to_vec(),
///     status_code: 200,
/// };
///
/// // Convert to text if it's UTF-8 encoded
/// let text = response.text()?;
/// println!("Response text: {}", text);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct HttpResponse {
    /// The raw response body as bytes.
    ///
    /// This contains the complete response body data, which could be text,
    /// JSON, binary data, or compressed content depending on the endpoint.
    pub body: Vec<u8>,

    /// The HTTP status code returned by the server.
    ///
    /// Standard HTTP status codes such as:
    /// - `200` - OK (successful request)
    /// - `400` - Bad Request
    /// - `401` - Unauthorized (invalid API key)
    /// - `404` - Not Found (invalid endpoint)
    /// - `429` - Rate Limited
    /// - `500` - Internal Server Error
    pub status_code: u16,
}

impl HttpResponse {
    /// Converts the response body to a UTF-8 encoded string.
    ///
    /// This method attempts to interpret the raw response bytes as a UTF-8
    /// encoded string. This is useful for text-based responses such as JSON,
    /// XML, HTML, or plain text.
    ///
    /// # Returns
    ///
    /// - `Ok(String)` - The response body as a UTF-8 string
    /// - `Err(anyhow::Error)` - If the bytes are not valid UTF-8
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nasdaq_api::HttpResponse;
    /// # fn main() -> anyhow::Result<()> {
    /// let response = HttpResponse {
    ///     body: b"{\"message\": \"Hello\"}".to_vec(),
    ///     status_code: 200,
    /// };
    ///
    /// let json_text = response.text()?;
    /// println!("JSON response: {}", json_text);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the response body contains invalid UTF-8 sequences.
    /// This commonly occurs with binary data such as images, compressed files,
    /// or other non-text content.
    pub fn into_text(self) -> Result<String> {
        String::from_utf8(self.body)
            .map_err(|e| anyhow!("Failed to convert bytes to UTF-8 string: {}", e))
    }
}

/// Makes an HTTP GET request with robust retry logic and exponential backoff.
///
/// This function provides a reliable HTTP client with built-in retry capabilities,
/// connection pooling, and configurable timeouts. It's designed to handle transient
/// network failures and API rate limiting gracefully.
///
/// The function uses a static HTTP client with connection pooling to improve
/// performance for multiple requests. Retries use exponential backoff (1s, 2s, 4s, 8s...)
/// to avoid overwhelming servers during outages.
///
/// # Parameters
///
/// * `url` - The complete URL to send the GET request to. Must be a valid HTTP/HTTPS URL.
/// * `headers` - Optional HTTP headers to include with the request. Common headers include:
///   - `"Authorization"` for API keys or bearer tokens
///   - `"User-Agent"` for client identification
///   - `"Accept"` for content type negotiation
/// * `query` - Optional query parameters to append to the URL. These will be properly
///   URL-encoded and appended as `?key1=value1&key2=value2`.
/// * `retries` - Maximum number of retry attempts. Must be at least 1. Each retry uses
///   exponential backoff with delays of 1s, 2s, 4s, 8s, etc.
/// * `timeout_secs` - Request timeout in seconds for each individual attempt. Does not
///   include retry delays. Recommended values: 30-120 seconds.
///
/// # Returns
///
/// Returns `Result<HttpResponse>` containing:
/// - `Ok(HttpResponse)` - Successful response with status code 2xx
/// - `Err(anyhow::Error)` - Request failed after all retries, invalid URL, network error,
///   or timeout
///
/// # Examples
///
/// ## Basic GET request
/// ```rust
/// # use std::collections::HashMap;
/// # use nasdaq_api::http_get;
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let response = http_get(
///     "https://api.example.com/data",
///     None,           // No custom headers
///     None,           // No query parameters
///     3,              // 3 retry attempts
///     30              // 30 second timeout
/// ).await?;
///
/// println!("Status: {}", response.status_code);
/// println!("Body size: {} bytes", response.body.len());
/// # Ok(())
/// # }
/// ```
///
/// ## Request with headers and query parameters
/// ```rust
/// # use std::collections::HashMap;
/// # use nasdaq_api::http_get;
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let mut headers = HashMap::new();
/// headers.insert("Authorization".to_string(), "Bearer token123".to_string());
/// headers.insert("User-Agent".to_string(), "MyApp/1.0".to_string());
///
/// let mut query = HashMap::new();
/// query.insert("limit".to_string(), "100".to_string());
/// query.insert("offset".to_string(), "0".to_string());
///
/// let response = http_get(
///     "https://api.example.com/data",
///     Some(&headers),
///     Some(&query),
///     5,              // 5 retry attempts
///     60              // 60 second timeout
/// ).await?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// This function returns an error in the following cases:
/// - **Network errors**: DNS resolution failures, connection timeouts, network unreachable
/// - **HTTP errors**: 4xx and 5xx status codes that persist after all retries
/// - **Timeout errors**: Individual requests exceed the specified timeout
/// - **Invalid URLs**: Malformed URLs or unsupported protocols
/// - **Response errors**: Failed to read response body or corrupted data
///
/// # Performance Considerations
///
/// - Uses a static HTTP client with connection pooling (up to 10 idle connections per host)
/// - Connections are reused for 90 seconds to improve performance
/// - Exponential backoff prevents overwhelming servers during outages
/// - Consider using reasonable timeout values (30-120s) based on expected response times
///
/// # Thread Safety
///
/// This function is fully async and can be called concurrently from multiple tasks.
/// The underlying HTTP client is thread-safe and optimized for concurrent use.
pub async fn http_get(
    url: &str,
    headers: Option<&HashMap<String, String>>,
    query: Option<&HashMap<String, String>>,
    retries: u32,
    timeout_secs: u64,
) -> Result<HttpResponse> {
    // Reuse client for connection pooling
    static CLIENT: std::sync::OnceLock<Client> = std::sync::OnceLock::new();
    let client = CLIENT.get_or_init(|| {
        Client::builder()
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("Failed to create HTTP client")
    });

    for attempt in 1..=retries {
        let mut request_builder = client.get(url).timeout(Duration::from_secs(timeout_secs));

        // Add headers if provided
        if let Some(headers_map) = headers {
            for (key, value) in headers_map {
                request_builder = request_builder.header(key, value);
            }
        }

        // Add query parameters if provided
        if let Some(query_map) = query {
            request_builder = request_builder.query(query_map);
        }

        // Make the request
        let result = request_builder.send().await;

        match result {
            Ok(response) if response.status().is_success() => {
                let status_code = response.status().as_u16();

                match response.bytes().await {
                    Ok(body_bytes) => {
                        return Ok(HttpResponse {
                            body: body_bytes.to_vec(),
                            status_code,
                        });
                    }
                    Err(e) if attempt == retries => {
                        return Err(anyhow!("Failed to read response body: {}", e));
                    }
                    _ => {} // Continue to retry
                }
            }
            Ok(response) if attempt == retries => {
                return Err(anyhow!(
                    "HTTP request failed with status {} after {} attempts",
                    response.status(),
                    retries
                ));
            }
            Err(e) if attempt == retries => {
                return Err(anyhow!("Request failed after {} attempts: {}", retries, e));
            }
            _ => {} // Continue to retry
        }

        // Simple exponential backoff
        if attempt < retries {
            let delay_secs = 2_u64.pow(attempt - 1); // 1s, 2s, 4s, 8s...
            sleep(Duration::from_secs(delay_secs)).await;
        }
    }

    unreachable!("Loop should have returned or errored")
}

/// Base URL for NASDAQ Data API v3 datatables endpoint.
///
/// All dataset requests are made to this base URL with the dataset path appended.
const NASDAQ_BASE_URL: &str = "https://data.nasdaq.com/api/v3/datatables";

/// Maximum number of retry attempts for NASDAQ API requests.
///
/// This includes retries for both the initial bulk download request and
/// the subsequent zip file download.
const MAX_RETRIES: u32 = 3;

/// Timeout in seconds for the initial NASDAQ API request to get download URL.
///
/// This should be sufficient for the API to process the request and return
/// the bulk download information.
const INITIAL_TIMEOUT: u64 = 60;

/// Timeout in seconds for downloading the actual zip file.
///
/// Zip files can be large (hundreds of MB), so this timeout is longer
/// to accommodate slower connections and large datasets.
const DOWNLOAD_TIMEOUT: u64 = 600;

/// Makes a request to the NASDAQ Data API and downloads the resulting dataset as a zip file.
///
/// This function implements the complete NASDAQ bulk download workflow:
/// 1. Makes an initial API request to the datatables endpoint with export options
/// 2. Extracts the bulk download URL from the JSON response
/// 3. Downloads the zip file from the provided URL
/// 4. Returns the complete zip file data
///
/// The function handles the NASDAQ-specific API flow where datasets are not returned
/// directly, but instead require a two-step process to get a download link and then
/// fetch the actual data.
///
/// # Parameters
///
/// * `path` - The dataset path to append to the NASDAQ API base URL. This should be
///   in the format `"PROVIDER/DATASET"` such as:
///   - `"WIKI/PRICES"` for Wikipedia stock prices
///   - `"SHARADAR/SF1"` for Sharadar fundamentals
///   - `"ZACKS/FC"` for Zacks fundamentals
///   - `"EOD/AAPL"` for end-of-day data for specific ticker
///
/// * `api_key` - Your NASDAQ Data API key. This is required for all requests.
///   You can obtain a free API key by registering at <https://data.nasdaq.com/>
///   The API key will be automatically added to the request as a query parameter.
///
/// * `query` - Optional additional query parameters to filter or customize the dataset.
///   Common parameters include:
///   - `"ticker"` - Filter by specific stock ticker symbol
///   - `"date.gte"` - Filter for dates greater than or equal to specified date
///   - `"date.lte"` - Filter for dates less than or equal to specified date
///   - `"dimension"` - For fundamentals data, specify reporting dimension
///   
///   Example: `{"ticker": "AAPL", "date.gte": "2023-01-01"}`
///
/// # Returns
///
/// Returns `Result<HttpResponse>` containing:
/// - `Ok(HttpResponse)` - The zip file data with:
///   - `body`: Complete zip file as bytes, ready to be written to disk or extracted
///   - `status_code`: HTTP status (should be 200 for successful downloads)
/// - `Err(anyhow::Error)` - Request failed with detailed error information:
///   - Invalid API key (401 Unauthorized)
///   - Dataset not found (404 Not Found)
///   - Rate limit exceeded (429 Too Many Requests)
///   - Network or timeout errors
///   - Invalid dataset path or query parameters
///
/// # Examples
///
/// ## Download complete dataset
/// ```rust
/// # use std::collections::HashMap;
/// # use nasdaq_api::nasdaq_api_get;
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let response = nasdaq_api_get(
///     "WIKI/PRICES",
///     "your-api-key-here",
///     None
/// ).await?;
///
/// // Save to file
/// std::fs::write("wiki_prices.zip", &response.body)?;
/// println!("Downloaded {} bytes", response.body.len());
/// # Ok(())
/// # }
/// ```
///
/// ## Download filtered dataset
/// ```rust
/// # use std::collections::HashMap;
/// # use nasdaq_api::nasdaq_api_get;
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let mut query = HashMap::new();
/// query.insert("ticker".to_string(), "AAPL".to_string());
/// query.insert("date.gte".to_string(), "2023-01-01".to_string());
/// query.insert("date.lte".to_string(), "2023-12-31".to_string());
///
/// let response = nasdaq_api_get(
///     "EOD/AAPL",
///     "your-api-key-here",
///     Some(query)
/// ).await?;
///
/// println!("Downloaded AAPL data: {} bytes", response.body.len());
/// # Ok(())
/// # }
/// ```
///
/// # Error Handling
///
/// This function implements robust retry logic with exponential backoff:
/// - Up to 5 retry attempts for both API and download requests
/// - Exponential backoff delays: 1s, 2s, 4s, 8s, 16s
/// - Separate timeouts for API requests (60s) and downloads (120s)
/// - Automatic handling of transient network failures
///
/// # API Rate Limits
///
/// NASDAQ APIs have rate limits that vary by subscription tier:
/// - Free tier: 50 requests per day
/// - Paid tiers: Higher limits based on plan
///
/// The function will return a 429 error if you exceed your rate limit.
/// Consider implementing additional delay between calls if needed.
///
/// # Data Format
///
/// The returned zip file typically contains CSV data with:
/// - Header row with column names
/// - UTF-8 encoded text data
/// - Standard CSV format (comma-separated)
/// - Dates in YYYY-MM-DD format
///
/// You can extract and process the CSV data using standard zip and CSV libraries.
pub async fn nasdaq_api_get(
    path: &str,
    api_key: &str,
    query: Option<&HashMap<String, String>>,
) -> Result<HttpResponse> {
    let base_url = format!("{}/{}", NASDAQ_BASE_URL, path);

    // Build query parameters efficiently
    let mut query_params = HashMap::with_capacity(2 + query.as_ref().map_or(0, |q| q.len()));
    query_params.insert("api_key".to_string(), api_key.to_string());
    query_params.insert("qopts.export".to_string(), "true".to_string());

    if let Some(extra_params) = query {
        query_params.extend(extra_params.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    // Retry logic for getting download URL
    for attempt in 1..=MAX_RETRIES {
        let response = http_get(&base_url, None, Some(&query_params), 1, INITIAL_TIMEOUT).await;

        let response = match response {
            Ok(r) => r,
            Err(e) if attempt == MAX_RETRIES => {
                return Err(anyhow!(
                    "HTTP request failed after {} attempts: {}",
                    MAX_RETRIES,
                    e
                ));
            }
            Err(_) => {
                // Exponential backoff: 1s, 2s, 4s, 8s, 16s
                let delay_secs = 2_u64.pow(attempt - 1);
                sleep(Duration::from_secs(delay_secs)).await;
                continue;
            }
        };

        let text = match response.into_text() {
            Ok(t) => t,
            Err(e) if attempt == MAX_RETRIES => {
                return Err(anyhow!(
                    "Failed to read response text after {} attempts: {}",
                    MAX_RETRIES,
                    e
                ));
            }
            Err(_) => {
                // Exponential backoff: 1s, 2s, 4s, 8s, 16s
                let delay_secs = 2_u64.pow(attempt - 1);
                sleep(Duration::from_secs(delay_secs)).await;
                continue;
            }
        };

        let json: Value = match serde_json::from_str(&text) {
            Ok(j) => j,
            Err(e) if attempt == MAX_RETRIES => {
                return Err(anyhow!(
                    "Failed to parse JSON after {} attempts: {}",
                    MAX_RETRIES,
                    e
                ));
            }
            Err(_) => {
                // Exponential backoff: 1s, 2s, 4s, 8s, 16s
                let delay_secs = 2_u64.pow(attempt - 1);
                sleep(Duration::from_secs(delay_secs)).await;
                continue;
            }
        };

        // Extract download URL using more concise pattern
        let download_url = json
            .pointer("/datatable_bulk_download/file/link")
            .and_then(Value::as_str)
            .filter(|url| !url.is_empty());

        let download_url = match download_url {
            Some(url) => url,
            None if attempt == MAX_RETRIES => {
                return Err(anyhow!(
                    "No valid bulk download link found after {} attempts",
                    MAX_RETRIES
                ));
            }
            None => {
                // Exponential backoff: 1s, 2s, 4s, 8s, 16s
                let delay_secs = 2_u64.pow(attempt - 1);
                sleep(Duration::from_secs(delay_secs)).await;
                continue;
            }
        };

        // Download the zip file with separate timeout
        match http_get(download_url, None, None, 3, DOWNLOAD_TIMEOUT).await {
            Ok(zip_resp) => return Ok(zip_resp),
            Err(e) if attempt == MAX_RETRIES => {
                return Err(anyhow!(
                    "Failed to download zip file after {} attempts: {}",
                    MAX_RETRIES,
                    e
                ));
            }
            Err(_) => {
                // Exponential backoff: 1s, 2s, 4s, 8s, 16s
                let delay_secs = 2_u64.pow(attempt - 1);
                sleep(Duration::from_secs(delay_secs)).await;
            }
        }
    }

    unreachable!("Loop should have returned or errored")
}

/// Get company logo from Logo.dev API and save it to output/logos folder
///
/// This function fetches a company logo from Logo.dev API using the provided stock symbol
/// and saves it to the local filesystem in the output/logos directory.
///
/// # Parameters
///
/// * `symbol` - Stock ticker symbol (e.g., "AAPL", "GOOGL", "MSFT")
/// * `size` - Optional logo size in pixels (default: 100). Common sizes: 32, 64, 100, 128, 256
/// * `format` - Optional image format (default: "png"). Supported formats: "png", "webp"
///
/// # Returns
///
/// Returns `Result<String>` containing:
/// - `Ok(String)` - Path to the saved logo file
/// - `Err(anyhow::Error)` - Request failed, invalid format, or file save error
///
/// # Examples
///
/// ```rust
/// # use nasdaq_api::get_logodev_api;
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// // Get Apple logo with defaults (PNG, 100px)
/// let logo_path = get_logodev_api("AAPL", None, None).await?;
/// println!("Apple logo saved to: {}", logo_path);
///
/// // Get Google logo with custom size
/// let logo_path = get_logodev_api("GOOGL", Some(128), None).await?;
///
/// // Get Microsoft logo in WebP format
/// let webp_path = get_logodev_api("MSFT", Some(256), Some("webp")).await?;
/// println!("Microsoft logo saved to: {}", webp_path);
/// # Ok(())
/// # }
/// ```
///
/// # Environment Variables
///
/// Requires `LOGO_KEY` environment variable to be set with your Logo.dev API token.
/// You can get an API key from <https://logo.dev/>
///
/// # File Organization
///
/// Logos are saved to:
/// - Directory: `output/logos/`
/// - Filename format: `{symbol}.{format}`
/// - Examples: `AAPL.png`, `GOOGL.webp`
pub async fn get_logodev_api(
    symbol: &str,
    size: Option<u32>,
    format: Option<&str>,
) -> Result<String> {
    use std::env;
    use std::path::Path;

    // Apply default values
    let size = size.unwrap_or(100);
    let format = format.unwrap_or("png");

    // Validate format
    if format != "png" && format != "webp" {
        return Err(anyhow!(
            "Invalid format '{}'. Only 'png' and 'webp' are supported.",
            format
        ));
    }

    // Get Logo.dev API key from environment
    let logo_key = env::var("LOGO_KEY").map_err(|_| {
        anyhow!("LOGO_KEY environment variable not found. Please set your Logo.dev API token.")
    })?;

    // Construct the Logo.dev API URL
    let url = format!(
        "https://img.logo.dev/ticker/{}?token={}&size={}&fallback=monogram&retina=true&format={}",
        symbol.to_uppercase(),
        logo_key,
        size,
        format
    );

    // Make the HTTP request
    let response = http_get(&url, None, None, 3, 30)
        .await
        .map_err(|e| anyhow!("Failed to fetch logo for {}: {}", symbol, e))?;

    // Ensure output/logos directory exists
    let logos_dir = "output/logos";
    let logos_path = Path::new(logos_dir);
    if !logos_path.exists() {
        std::fs::create_dir_all(logos_path)
            .map_err(|e| anyhow!("Failed to create logos directory '{}': {}", logos_dir, e))?;
    }

    // Create filename and full path
    let filename = format!("{}.{}", symbol.to_uppercase(), format);
    let filepath = format!("{}/{}", logos_dir, filename);

    // Save the logo file
    std::fs::write(&filepath, &response.body)
        .map_err(|e| anyhow!("Failed to save logo to '{}': {}", filepath, e))?;

    Ok(filepath)
}
