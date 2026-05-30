//! Provides terminal UI helpers for progress bars, spinners, and timed status output.
use anyhow::Result;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;
use std::time::{Duration, Instant};

const BAR_CHARS: &str = "█▉▊▋▌▍▎▏  ";

fn stdout_is_terminal() -> bool {
    std::io::stdout().is_terminal()
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.cyan} {prefix:.bold.cyan} [{bar:32.cyan/blue}] \
         {pos:>3.green}/{len:<3.green} {percent:>3}% {wide_msg:.dim}",
    )
    .expect("invalid progress bar template")
    .progress_chars(BAR_CHARS)
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.cyan} {wide_msg}").expect("invalid spinner template")
}

fn print_success(label: &str, elapsed: Option<Duration>) {
    if stdout_is_terminal() {
        match elapsed {
            Some(elapsed) => println!(
                "{} {} {}",
                style("✓").green().bold(),
                style(label).bold(),
                style(format!("({:.2}s)", elapsed.as_secs_f64())).dim()
            ),
            None => println!("{} {}", style("✓").green().bold(), style(label).bold()),
        }
        return;
    }

    match elapsed {
        Some(elapsed) => println!("✓ {label} ({:.2}s)", elapsed.as_secs_f64()),
        None => println!("✓ {label}"),
    }
}

fn print_failure(label: &str) {
    if stdout_is_terminal() {
        println!("{} {}", style("✗").red().bold(), style(label).bold());
    } else {
        println!("✗ {label}");
    }
}

/// Builds a determinate progress bar for a counted loop, or a no-op hidden
/// bar when stdout is not a terminal (piped, redirected, CI) — a redrawing
/// bar would otherwise emit escape-code garbage into a log. When hidden, a
/// plain `→ message (N items)` start line is printed instead.
///
/// The returned bar is advanced with `.inc(...)` and finished with
/// `.finish_with_message(...)` as usual; on a hidden bar those are no-ops,
/// so loop code needs no TTY-specific branching.
pub fn new_progress_bar(total: u64, message: &str) -> ProgressBar {
    if !stdout_is_terminal() {
        println!("→ {message} ({total} items)");
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(total);
    pb.set_style(bar_style());
    pb.set_prefix(message.to_owned());
    pb
}

/// Builds a spinner `ProgressBar`, or a no-op hidden one when stdout is not a
/// terminal. When hidden, a plain `→ label` start line is printed instead of
/// an animated spinner.
fn new_spinner(label: &str) -> ProgressBar {
    if !stdout_is_terminal() {
        println!("→ {label}");
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(spinner_style());
    pb.set_message(label.to_owned());
    pb.enable_steady_tick(Duration::from_millis(90));
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
            print_success(label, None);
            Ok(value)
        }
        Err(e) => {
            pb.finish_and_clear();
            print_failure(label);
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
            print_success(label, Some(start.elapsed()));
            Ok(value)
        }
        Err(e) => {
            pb.finish_and_clear();
            print_failure(label);
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
