use anyhow::{Result, anyhow};
use std::path::Path;

/// Save data to the specified file path.
///
/// # Arguments
/// * `data` - The binary data to write to file
/// * `filepath` - The complete file path including filename and extension
///
/// # Returns
/// * `Result<()>` - success/failure
pub fn save_file(data: &[u8], filepath: impl AsRef<Path>) -> Result<()> {
    let filepath = filepath.as_ref();
    let path = filepath;

    // Create parent directories if they don't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("Failed to create directories for '{}': {}", filepath.display(), e))?;
    }

    // Write the file
    std::fs::write(path, data)
        .map_err(|e| anyhow!("Failed to write file '{}': {}", filepath.display(), e))?;
    Ok(())
}

/// Ensure a directory exists, creating it when missing.
pub fn ensure_directory(dir: impl AsRef<Path>) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)
        .map_err(|e| anyhow!("Failed to create directory '{}': {}", dir.display(), e))?;
    Ok(())
}
