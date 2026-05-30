//! Dataset-preparation pipelines.
//!
//! Each submodule prepares one raw dataset into its analysis form. The
//! submodules are an implementation detail; pipeline entrypoints are
//! re-exported here as the module's public surface.
mod companies;
mod fundamentals;
mod insiders;
mod prices;
mod technical_analysis;

pub use companies::build_company_snapshot;
pub use fundamentals::adjust_fundamentals;
pub use insiders::update_insiders;
pub use prices::{load_prices_adjusted, resample_ohlcv};
pub use technical_analysis::technical_indicators_daily;
