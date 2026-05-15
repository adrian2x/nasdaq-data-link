use anyhow::{Result, anyhow};
use std::path::Path;

/// Save data to the specified file path.
///
/// # Arguments
/// * `data` - The binary data to write to file
/// * `filepath` - The complete file path including filename and extension
///
/// # Returns
/// * `Result<String>` - The full path to the saved file
pub fn save_file(data: &[u8], filepath: &str) -> Result<String> {
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

/// Ensure a directory exists, creating it when missing.
pub fn ensure_directory(dir: &str) -> Result<()> {
    let dir_path = Path::new(dir);

    if !dir_path.exists() {
        println!("Creating directory: {}", dir);
        std::fs::create_dir_all(dir_path)
            .map_err(|e| anyhow!("Failed to create directory '{}': {}", dir, e))?;
    }

    Ok(())
}
