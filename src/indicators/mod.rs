//! Daily technical-indicator building blocks.
//!
//! Each indicator lives in its own fully self-contained submodule — no shared
//! helper layer, small primitives re-declared per module. This module declares
//! and re-exports the indicator primitives consumed by pipeline assembly code.

mod adr;
mod adx;
mod atr;
mod bollinger;
mod ema;
mod ewma_volatility;
mod fractaltools;
mod fs_score;
mod highlows;
mod macd;
mod rank;
mod rate_of_change;
mod realized_volatility;
mod rsi;
mod sma;
mod yang_zhang;

pub use self::adr::adr;
pub use self::adx::adx;
pub use self::atr::atr;
pub use self::bollinger::bollinger;
pub use self::ema::{ema, ema_expr};
pub use self::ewma_volatility::ewma_vol;
pub use self::fractaltools::{HurstConfig, compute_hurst, with_hurst};
pub use self::fs_score::fs_score;
pub use self::highlows::highlows;
pub use self::macd::macd;
pub use self::rank::percentile;
pub use self::rate_of_change::rate_of_change;
pub use self::realized_volatility::realized_volatility;
pub use self::rsi::rsi;
pub use self::sma::{sma, sma_expr};
pub use self::yang_zhang::yang_zhang;
