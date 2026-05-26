//! Provides HTTP and NASDAQ API clients for the downloader pipeline.
use anyhow::{Result, anyhow};
use futures::Stream;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug)]
pub struct HttpResponse {
    pub body: Vec<u8>,

    pub status_code: u16,
}

impl HttpResponse {
    /// Decodes the response body as UTF-8 text.
    ///
    /// # Failure
    /// Returns an error if the body is not valid UTF-8.
    pub fn into_text(self) -> Result<String> {
        String::from_utf8(self.body)
            .map_err(|e| anyhow!("Failed to convert bytes to UTF-8 string: {}", e))
    }
}

/// Performs a GET request with optional headers and query parameters.
///
/// # Failure
/// Returns an error if all retry attempts fail or the response body cannot be read.
pub async fn http_get(
    url: &str,
    headers: Option<&HashMap<String, String>>,
    query: Option<&HashMap<String, String>>,
    retries: u32,
    timeout_secs: u64,
) -> Result<HttpResponse> {
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

        if let Some(headers_map) = headers {
            for (key, value) in headers_map {
                request_builder = request_builder.header(key, value);
            }
        }

        if let Some(query_map) = query {
            request_builder = request_builder.query(query_map);
        }

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

        if attempt < retries {
            let delay_secs = 2_u64.pow(attempt - 1); // 1s, 2s, 4s, 8s...
            sleep(Duration::from_secs(delay_secs)).await;
        }
    }

    unreachable!("Loop should have returned or errored")
}

const NASDAQ_BASE_URL: &str = "https://data.nasdaq.com/api/v3/datatables";

const MAX_RETRIES: u32 = 3;

const INITIAL_TIMEOUT: u64 = 60;

const DOWNLOAD_TIMEOUT: u64 = 600;
async fn http_get_response(
    url: &str,
    headers: Option<&HashMap<String, String>>,
    query: Option<&HashMap<String, String>>,
    retries: u32,
    timeout_secs: u64,
) -> Result<reqwest::Response> {
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

        if let Some(headers_map) = headers {
            for (key, value) in headers_map {
                request_builder = request_builder.header(key, value);
            }
        }

        if let Some(query_map) = query {
            request_builder = request_builder.query(query_map);
        }

        match request_builder.send().await {
            Ok(response) if response.status().is_success() => return Ok(response),
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
            _ => {}
        }

        if attempt < retries {
            let delay_secs = 2_u64.pow(attempt - 1);
            sleep(Duration::from_secs(delay_secs)).await;
        }
    }

    unreachable!("Loop should have returned or errored")
}

/// Resolves a NASDAQ datatable export URL and returns a ZIP byte stream.
///
/// # Failure
/// Returns an error if export metadata retrieval or ZIP download fails after retries.
pub async fn nasdaq_api_get(
    path: &str,
    api_key: &str,
    query: Option<&HashMap<String, String>>,
) -> Result<impl Stream<Item = reqwest::Result<bytes::Bytes>>> {
    let base_url = format!("{}/{}", NASDAQ_BASE_URL, path);

    let mut query_params = HashMap::with_capacity(2 + query.as_ref().map_or(0, |q| q.len()));
    query_params.insert("api_key".to_string(), api_key.to_string());
    query_params.insert("qopts.export".to_string(), "true".to_string());

    if let Some(extra_params) = query {
        query_params.extend(extra_params.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

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
                let delay_secs = 2_u64.pow(attempt - 1);
                sleep(Duration::from_secs(delay_secs)).await;
                continue;
            }
        };

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
                let delay_secs = 2_u64.pow(attempt - 1);
                sleep(Duration::from_secs(delay_secs)).await;
                continue;
            }
        };

        match http_get_response(download_url, None, None, 3, DOWNLOAD_TIMEOUT).await {
            Ok(zip_resp) => return Ok(zip_resp.bytes_stream()),
            Err(e) if attempt == MAX_RETRIES => {
                return Err(anyhow!(
                    "Failed to download zip file after {} attempts: {}",
                    MAX_RETRIES,
                    e
                ));
            }
            Err(_) => {
                let delay_secs = 2_u64.pow(attempt - 1);
                sleep(Duration::from_secs(delay_secs)).await;
            }
        }
    }

    unreachable!("Loop should have returned or errored")
}

/// Fetches and saves a Logo.dev image for the given ticker symbol.
///
/// # Failure
/// Returns an error if the format is invalid, credentials are missing, or network or file I/O fails.
pub async fn get_logodev_api(
    symbol: &str,
    size: Option<u32>,
    format: Option<&str>,
) -> Result<String> {
    use std::env;
    use std::path::Path;

    let size = size.unwrap_or(100);
    let format = format.unwrap_or("png");

    if format != "png" && format != "webp" {
        return Err(anyhow!(
            "Invalid format '{}'. Only 'png' and 'webp' are supported.",
            format
        ));
    }

    let logo_key = env::var("LOGO_KEY").map_err(|_| {
        anyhow!("LOGO_KEY environment variable not found. Please set your Logo.dev API token.")
    })?;

    let url = format!(
        "https://img.logo.dev/ticker/{}?token={}&size={}&fallback=monogram&retina=true&format={}",
        symbol.to_uppercase(),
        logo_key,
        size,
        format
    );

    let response = http_get(&url, None, None, 3, 30)
        .await
        .map_err(|e| anyhow!("Failed to fetch logo for {}: {}", symbol, e))?;

    let logos_dir = "output/logos";
    let logos_path = Path::new(logos_dir);
    if !logos_path.exists() {
        std::fs::create_dir_all(logos_path)
            .map_err(|e| anyhow!("Failed to create logos directory '{}': {}", logos_dir, e))?;
    }

    let filename = format!("{}.{}", symbol.to_uppercase(), format);
    let filepath = format!("{}/{}", logos_dir, filename);

    std::fs::write(&filepath, &response.body)
        .map_err(|e| anyhow!("Failed to save logo to '{}': {}", filepath, e))?;

    Ok(filepath)
}
