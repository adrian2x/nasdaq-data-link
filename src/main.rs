//! Runs the NASDAQ dataset download and output-table build pipeline.
mod api;
mod config;
mod downloader;
mod filetools;
pub mod indicators;
mod pipeline;
mod sqltools;
mod ui;

use anyhow::{Result, bail};
use filetools::ensure_directory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Default,
    Export,
    Help,
    Logos,
    Sync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cli {
    command: Command,
    weekly: bool,
    hurst: bool,
}

fn usage() -> &'static str {
    "Usage: nasdaq [COMMAND] [OPTIONS]\n\nCommands:\n  export  Write Arrow files from existing DuckDB output\n  logos   Download company logos from downloads/companies.csv\n  sync    Run the writer pipeline without Arrow exports\n\nOptions:\n  --weekly  Also write weekly stock prices\n  --hurst   Add Hurst exponent columns to stock prices\n\nWith no command, downloads datasets and runs the writer pipeline with Arrow exports.\n"
}

fn parse_command(args: impl Iterator<Item = String>) -> Result<Cli> {
    let mut cli = Cli {
        command: Command::Default,
        weekly: false,
        hurst: false,
    };
    let mut command_seen = false;

    for arg in args {
        match arg.as_str() {
            "--weekly" => cli.weekly = true,
            "--hurst" => cli.hurst = true,
            "export" | "help" | "--help" | "-h" | "logos" | "sync" => {
                if command_seen {
                    bail!("unexpected argument '{arg}'\n\n{}", usage());
                }
                cli.command = match arg.as_str() {
                    "export" => Command::Export,
                    "help" | "--help" | "-h" => Command::Help,
                    "logos" => Command::Logos,
                    "sync" => Command::Sync,
                    _ => unreachable!(),
                };
                command_seen = true;
            }
            _ if arg.starts_with('-') => bail!("unknown option '{arg}'\n\n{}", usage()),
            _ => bail!("unknown command '{arg}'\n\n{}", usage()),
        }
    }

    if cli.weekly && !matches!(cli.command, Command::Default | Command::Sync) {
        bail!(
            "--weekly can only be used with the default pipeline or 'sync'\n\n{}",
            usage()
        );
    }
    if cli.hurst && !matches!(cli.command, Command::Default | Command::Sync) {
        bail!(
            "--hurst can only be used with the default pipeline or 'sync'\n\n{}",
            usage()
        );
    }

    Ok(cli)
}

#[tokio::main]
async fn main() -> Result<()> {
    match parse_command(std::env::args().skip(1))? {
        Cli {
            command: Command::Default,
            weekly,
            hurst,
        } => {
            ensure_directory(config::DOWNLOADS_DIR)?;
            ensure_directory(config::OUTPUT_DIR)?;

            let api_key = config::load_or_create_api_key()?;
            let specs = config::load_path_specs()?;
            downloader::run_downloader(&api_key, specs).await?;
            pipeline::run_writer(true, weekly, hurst)?;
        }
        Cli {
            command: Command::Export,
            ..
        } => filetools::write_arrow_files()?,
        Cli {
            command: Command::Help,
            ..
        } => print!("{}", usage()),
        Cli {
            command: Command::Logos,
            ..
        } => {
            dotenv::dotenv().ok();
            ensure_directory(config::OUTPUT_DIR)?;
            downloader::run_logo_downloader().await?;
        }
        Cli {
            command: Command::Sync,
            weekly,
            hurst,
        } => {
            ensure_directory(config::DOWNLOADS_DIR)?;
            ensure_directory(config::OUTPUT_DIR)?;
            pipeline::run_writer(false, weekly, hurst)?;
        }
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli> {
        parse_command(args.iter().map(|arg| (*arg).to_string()))
    }

    #[test]
    fn parses_subcommands() -> Result<()> {
        assert_eq!(parse(&[])?.command, Command::Default);
        assert_eq!(parse(&["export"])?.command, Command::Export);
        assert_eq!(parse(&["logos"])?.command, Command::Logos);
        assert_eq!(parse(&["sync"])?.command, Command::Sync);
        Ok(())
    }

    #[test]
    fn parses_weekly_option() -> Result<()> {
        assert!(!parse(&[])?.weekly);
        assert_eq!(
            parse(&["--weekly"])?,
            Cli {
                command: Command::Default,
                weekly: true,
                hurst: false,
            }
        );
        assert_eq!(
            parse(&["sync", "--weekly"])?,
            Cli {
                command: Command::Sync,
                weekly: true,
                hurst: false,
            }
        );
        assert_eq!(
            parse(&["--weekly", "sync"])?,
            Cli {
                command: Command::Sync,
                weekly: true,
                hurst: false,
            }
        );
        Ok(())
    }

    #[test]
    fn parses_hurst_option() -> Result<()> {
        assert!(!parse(&[])?.hurst);
        assert_eq!(
            parse(&["--hurst"])?,
            Cli {
                command: Command::Default,
                weekly: false,
                hurst: true,
            }
        );
        assert_eq!(
            parse(&["sync", "--hurst"])?,
            Cli {
                command: Command::Sync,
                weekly: false,
                hurst: true,
            }
        );
        assert_eq!(
            parse(&["--weekly", "--hurst", "sync"])?,
            Cli {
                command: Command::Sync,
                weekly: true,
                hurst: true,
            }
        );
        Ok(())
    }

    #[test]
    fn rejects_weekly_for_non_writer_commands() {
        assert!(parse(&["export", "--weekly"]).is_err());
        assert!(parse(&["logos", "--weekly"]).is_err());
    }

    #[test]
    fn rejects_hurst_for_non_writer_commands() {
        assert!(parse(&["export", "--hurst"]).is_err());
        assert!(parse(&["logos", "--hurst"]).is_err());
    }
}
