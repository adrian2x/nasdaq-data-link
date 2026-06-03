//! Dataset-preparation pipelines.
//!
//! Each submodule prepares one raw dataset into its analysis form. The
//! submodules are an implementation detail; pipeline entrypoints are
//! re-exported here as the module's public surface.
mod companies;
mod fundamentals_quarter;
mod fundamentals_ttm;
mod insiders;
mod prices;
mod rankings;
mod technical_analysis;
mod writer;

pub use companies::build_company_snapshot;
pub use fundamentals_quarter::{
    adjust_fundamentals_quarter, compute_quarterly_fundamental_metrics,
};
pub use fundamentals_ttm::{adjust_fundamentals, compute_fundamental_metrics};
pub use insiders::update_insiders;
pub use prices::{load_prices_adjusted, resample_ohlcv};
pub use technical_analysis::technical_indicators_daily;
pub use writer::run_writer;
