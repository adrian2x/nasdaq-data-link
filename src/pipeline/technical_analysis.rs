//! Daily technical-indicator computation pipeline.

use polars::prelude::*;

use crate::indicators::{
    adr, atr, bollinger, highlows, macd, percentile, rate_of_change, rsi, sma, sma_expr,
};

/// Computes technical indicator columns for each individual ticker.
///
/// Ranking concepts added here:
/// - `rsrank`: Relative Strength. This is IBD/CAN SLIM's core price-momentum
///   idea: rank a stock's price performance against all other stocks.
/// - `volconfirmrank`: volume-confirmed momentum. This rewards price strength
///   when traded dollar volume is expanding versus its medium-term baseline.
/// - `adrank`: Accumulation/Distribution. IBD describes A/D as a 13-week
///   price-and-volume measure of institutional buying or selling pressure. The
///   exact IBD formula is proprietary, so this code uses a transparent proxy:
///   price direction, close location within the daily range, and relative
///   volume, smoothed over roughly one quarter.
///
/// References:
/// - IBD-style A/D descriptions mention closing price, daily range, volume,
///   and a 13-week window:
///   https://deepvue.com/knowledge-base/proprietary-ratings/
/// - Public IBD Composite descriptions include RS and A/D as components:
///   https://finance.yahoo.com/news/composite-rating-helps-spot-highest-220400010.html
pub fn technical_indicators_daily(lf: LazyFrame) -> LazyFrame {
    let lf = lf.sort(["ticker", "date"], Default::default());

    // Average volume of 3 months is often cited as liquidity filter
    // Institutions buy shares in "blocks" of 100k units or more
    let lf = lf.with_columns([sma_expr("volume", 60)
        .over([col("ticker")])
        .alias("avgvolume60")]);

    let lf = lf.with_column(
        (col("close").cast(DataType::Float64) * col("volume").cast(DataType::Float64))
            .alias("__dollarvolume"),
    );
    let lf = lf.with_columns([
        sma_expr("__dollarvolume", 20)
            .over([col("ticker")])
            .alias("avgdollarvolume20"),
        sma_expr("__dollarvolume", 126)
            .over([col("ticker")])
            .alias("avgdollarvolume126"),
    ]);

    // Computes the price range (high/low) equivalent to approx 1m, 3m, and 1y
    let lf = highlows(lf, &[20, 55, 252]).with_columns([
        // Adds a signal (true/false) value if the stock made new 52-week highs or lows
        col("close")
            .gt(col("max252").shift(lit(1)))
            .alias("high252"),
        col("close").lt(col("min252").shift(lit(1))).alias("low252"),
    ]);

    // Volatility measures
    let lf = adr(lf, 20);
    let lf = atr(lf, 20);
    let lf = bollinger(lf, 20, 2.0);

    // Rate of change calculates price momentum over multiple horizons:
    // 1 day, 5 days, about 1 month, 1 quarter, 2 quarters, 3 quarters,
    // and 1 year. These raw columns are time-series returns per ticker.
    let lf = rate_of_change(lf, "close", &[1, 5, 21, 3 * 21, 6 * 21, 9 * 21, 12 * 21]);

    // Relative Strength raw score:
    // `0.4 * 3M return + 0.2 * 6M return + 0.2 * 9M return + 0.2 * 12M return`.
    //
    // The shorter 3-month leg gets the largest weight so the score reacts to
    // newer leadership, while 6/9/12-month returns keep it anchored to the
    // intermediate-horizon momentum window studied in the academic literature.
    let lf = lf.with_columns([(lit(0.4) * col("pct63")
        + lit(0.2) * col("pct126")
        + lit(0.2) * col("pct189")
        + lit(0.2) * col("pct252"))
    .alias("rs1y")
    .cast(DataType::Float32)]);

    // Convert the raw RS score into a daily cross-sectional percentile.
    // `rs1y` and `rsrank` are intentionally equal aliases: `rs1y` preserves the
    // existing API, while `rsrank` reads naturally inside the composite rank.
    let lf = percentile(lf, "rs1y", "date", true, "rs1y");
    let lf = lf.with_column(col("rs1y").alias("rsrank"));

    let lf = lf.with_column(
        when(
            col("avgdollarvolume126")
                .cast(DataType::Float64)
                .gt(lit(0.0)),
        )
        .then(
            col("avgdollarvolume20").cast(DataType::Float64)
                / col("avgdollarvolume126").cast(DataType::Float64)
                - lit(1.0),
        )
        .otherwise(lit(NULL))
        .cast(DataType::Float32)
        .alias("volumeroc"),
    );
    let lf = percentile(lf, "volumeroc", "date", true, "volumerocrank");
    let lf = lf.with_column(
        (((col("rsrank").cast(DataType::Float64) - lit(50.0)) / lit(50.0))
            * (col("volumerocrank").cast(DataType::Float64) / lit(100.0)))
        .alias("__volconfirm"),
    );
    let lf = percentile(lf, "__volconfirm", "date", true, "volconfirmrank");

    // Materialize previous close before building A/D. Polars does not allow a
    // window expression nested inside another windowed aggregation, so the
    // lagged close must be a temporary column rather than an inline expression.
    let lf = lf.with_column(
        col("close")
            .cast(DataType::Float64)
            .shift(lit(1))
            .over([col("ticker")])
            .alias("__prev_close"),
    );

    let close = col("close").cast(DataType::Float64);
    let high = col("high").cast(DataType::Float64);
    let low = col("low").cast(DataType::Float64);
    let volume = col("volume").cast(DataType::Float64);
    let avg_volume = col("avgvolume60").cast(DataType::Float64);
    let prev_close = col("__prev_close");
    let range = high.clone() - low.clone();
    // A/D intraday component:
    // `(2 * close - high - low) / (high - low)`.
    //
    // This is the close-location value. It is near +1 when the stock closes
    // near the high of the day, near -1 when it closes near the low, and 0
    // near the middle. Range-less days are neutral.
    let intraday = when(range.clone().gt(lit(0.0)))
        .then((lit(2.0) * close.clone() - high - low) / range)
        .otherwise(lit(0.0));

    // A/D interday component:
    // sign of today's close minus yesterday's close. It captures whether the
    // market marked the stock up or down, independent of the intraday location.
    let interday = when(close.clone().gt(prev_close.clone()))
        .then(lit(1.0))
        .otherwise(
            when(close.clone().lt(prev_close))
                .then(lit(-1.0))
                .otherwise(lit(0.0)),
        );
    // Relative volume = today's volume / 60-day average volume.
    // A price move on heavy volume carries more information than the same move
    // on quiet volume, matching the CAN SLIM "institutional demand" intuition.
    let volume_weight = when(avg_volume.clone().gt(lit(0.0)))
        .then(volume / avg_volume)
        .otherwise(lit(NULL));

    // A/D raw score:
    // `((0.7 * interday) + (0.3 * intraday)) * relative_volume`.
    //
    // Direction gets more weight than close-location because gaps and full-day
    // repricing matter. Close-location adds nuance for whether buyers or
    // sellers controlled the session. The exponentially weighted mean uses
    // `alpha = 2 / 66`, the standard EMA span formula for a 65-trading-day
    // quarter, with 20 observations required before a score is emitted.
    let lf = lf.with_column(
        ((lit(0.7) * interday + lit(0.3) * intraday) * volume_weight)
            .ewm_mean(EWMOptions {
                alpha: 2.0 / 66.0,
                min_periods: 20,
                ..Default::default()
            })
            .over([col("ticker")])
            .alias("__adscore"),
    );

    // `adrank` is the cross-sectional percentile of the smoothed A/D score:
    // high values mean stronger recent accumulation than peers; low values
    // mean distribution or weak sponsorship.
    let lf = percentile(lf, "__adscore", "date", true, "adrank").drop([
        "__dollarvolume",
        "__volconfirm",
        "__prev_close",
        "__adscore",
    ]);

    // Trend / oscillator indicators.
    let lf = sma(lf, "close", &[10, 20, 50, 100, 150, 200]);
    let lf = macd(lf, 12, 26, 9);
    rsi(lf, 14)
}
