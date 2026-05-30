//! Loads runtime configuration for API credentials and dataset path specs.
use anyhow::{Result, anyhow};
use std::{
    collections::HashMap,
    env,
    io::{self, Write},
};

pub const DOWNLOADS_DIR: &str = "downloads";
pub const OUTPUT_DIR: &str = "output";
pub const DUCKDB_FILENAME: &str = "nasdaq.duckdb";

const PATHS_FILE: &str = "paths.txt";
const ENV_FILE: &str = ".env";

#[derive(Debug)]
pub struct PathSpec {
    pub path: String,
    pub query: Option<HashMap<String, String>>,
    pub output: Option<String>,
}

fn prompt(message: &str) -> Result<String> {
    print!("{message}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

/// Loads the NASDAQ API key from `.env`, prompting and persisting it when missing.
///
/// # Failure
/// Returns an error if reading from stdin or flushing stdout fails.
pub fn load_or_create_api_key() -> Result<String> {
    dotenv::dotenv().ok();
    if let Ok(key) = env::var("NASDAQ_API_KEY")
        && !key.is_empty()
    {
        return Ok(key);
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

    let env_content = format!("# NASDAQ API Configuration\nNASDAQ_API_KEY={}\n", key);
    match std::fs::write(ENV_FILE, env_content) {
        Ok(_) => println!("✓ Saved to .env (add it to .gitignore to keep it private)"),
        Err(e) => eprintln!("Warning: could not write .env: {}", e),
    }

    Ok(key)
}

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

fn parse_path_line(line: &str) -> Option<PathSpec> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    let mut tokens = line.split_whitespace();
    let path = tokens.next()?.to_string();

    let mut query = None;
    let mut output = None;
    while let Some(flag) = tokens.next() {
        let Some(value) = tokens.next() else {
            eprintln!("Warning: flag {flag} in '{line}' has no value");
            break;
        };
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

/// Parses `paths.txt` into download path specifications.
///
/// # Failure
/// Returns an error if `paths.txt` cannot be read or contains no valid entries.
pub fn load_path_specs() -> Result<Vec<PathSpec>> {
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

    Ok(specs)
}
