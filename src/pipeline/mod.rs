//! Dataset-preparation pipelines.
//!
//! Each submodule prepares one raw dataset into its analysis form. The
//! submodules are an implementation detail; the three preparation functions
//! are re-exported here as the module's public surface.

mod fundamentals;
mod insiders;
mod prices;

pub use fundamentals::adjust_fundamentals;
pub use insiders::update_insiders;
pub use prices::adjust_prices;
