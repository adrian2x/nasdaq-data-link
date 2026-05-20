use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

pub fn new_progress_bar(total: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("invalid progress bar template")
            .progress_chars("=>-"),
    );
    pb.set_message(message.to_string());
    pb
}

pub fn with_spinner<T>(label: &str, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("invalid spinner template"),
    );
    pb.set_message(label.to_string());
    pb.enable_steady_tick(Duration::from_millis(80));

    let start = Instant::now();
    match work() {
        Ok(value) => {
            pb.finish_and_clear();
            println!("✓ {} ({:.2}s)", label, start.elapsed().as_secs_f64());
            Ok(value)
        }
        Err(e) => {
            pb.finish_and_clear();
            println!("✗ {}", label);
            Err(e)
        }
    }
}
