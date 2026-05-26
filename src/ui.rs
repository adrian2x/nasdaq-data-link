//! Provides terminal UI helpers for progress bars, spinners, and timed status output.
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;
use std::time::{Duration, Instant};

/// Builds a determinate progress bar for a counted loop, or a no-op hidden
/// bar when stdout is not a terminal (piped, redirected, CI) — a redrawing
/// bar would otherwise emit escape-code garbage into a log. When hidden, a
/// plain `→ message (N items)` start line is printed instead.
///
/// The returned bar is advanced with `.inc(...)` and finished with
/// `.finish_with_message(...)` as usual; on a hidden bar those are no-ops,
/// so loop code needs no TTY-specific branching.
pub fn new_progress_bar(total: u64, message: &str) -> ProgressBar {
    if !std::io::stdout().is_terminal() {
        println!("→ {message} ({total} items)");
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("invalid progress bar template")
            .progress_chars("=>-"),
    );
    pb.set_message(message.to_owned());
    pb
}

/// Builds a spinner `ProgressBar`, or a no-op hidden one when stdout is not a
/// terminal. When hidden, a plain `→ label` start line is printed instead of
/// an animated spinner.
fn new_spinner(label: &str) -> ProgressBar {
    if !std::io::stdout().is_terminal() {
        println!("→ {label}");
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("invalid spinner template"),
    );
    pb.set_message(label.to_owned());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

/// Runs `work` while showing a spinner labeled `label`. On completion the
/// spinner is cleared and a plain `✓ label` / `✗ label` line is printed —
/// no timing. Prefer the `spinner!` macro at call sites.
///
/// # Failure
/// Returns any error produced by `work`.
pub fn spinner_run<T, E>(label: &str, work: impl FnOnce() -> Result<T, E>) -> Result<T>
where
    E: Into<anyhow::Error>,
{
    let pb = new_spinner(label);
    match work() {
        Ok(value) => {
            pb.finish_and_clear();
            println!("✓ {label}");
            Ok(value)
        }
        Err(e) => {
            pb.finish_and_clear();
            println!("✗ {label}");
            Err(e.into())
        }
    }
}

/// Runs `work` like `spinner_run`, and reports elapsed time on success:
/// `✓ label (1.23s)`. Prefer the `timed!` macro at call sites.
///
/// # Failure
/// Returns any error produced by `work`.
pub fn timed_run<T, E>(label: &str, work: impl FnOnce() -> Result<T, E>) -> Result<T>
where
    E: Into<anyhow::Error>,
{
    let pb = new_spinner(label);
    let start = Instant::now();
    match work() {
        Ok(value) => {
            pb.finish_and_clear();
            println!("✓ {label} ({:.2}s)", start.elapsed().as_secs_f64());
            Ok(value)
        }
        Err(e) => {
            pb.finish_and_clear();
            println!("✗ {label}");
            Err(e.into())
        }
    }
}

/// Runs an expression behind a labeled spinner without timing.
///
/// `spinner!("loading", load_data())` expands to
/// `spinner_run("loading", || load_data())`.
macro_rules! spinner {
    ($label:expr, $body:expr) => {
        $crate::ui::spinner_run($label, || $body)
    };
}

/// Runs an expression behind a labeled spinner and reports elapsed time.
///
/// `timed!("indexing", build_index())` expands to
/// `timed_run("indexing", || build_index())`.
macro_rules! timed {
    ($label:expr, $body:expr) => {
        $crate::ui::timed_run($label, || $body)
    };
}

pub(crate) use {spinner, timed};
