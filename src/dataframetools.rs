use polars::prelude::*;

fn rolling_opts(window: usize) -> RollingOptionsFixedWindow {
    RollingOptionsFixedWindow {
        window_size: window,
        min_periods: window,
        ..Default::default()
    }
}

fn f(name: &str) -> Expr {
    col(name).cast(DataType::Float64)
}

/// Partition `expr` per ticker and alias it. Wraps the
/// `.over([col("ticker")]).alias(name)` boilerplate that every per-ticker
/// indicator column repeats.
fn per_ticker(expr: Expr, name: &str) -> Expr {
    expr.over([col("ticker")]).alias(name)
}

/// Like `per_ticker`, but also casts the result to Float32. Used for the
/// bounded-range indicator columns (percent changes, realized vol, scores)
/// where Float32 precision is sufficient and halves the stored width.
fn per_ticker_f32(expr: Expr, name: &str) -> Expr {
    expr.over([col("ticker")])
        .alias(name)
        .cast(DataType::Float32)
}

/// Per-ticker SMA column: `sma("close", 50, "sma50")`.
fn sma(source: &str, period: usize, name: &str) -> Expr {
    per_ticker(sma_expr(source, period), name)
}

/// Per-ticker EMA column: `ema("close", 12, "ema12")`.
fn ema(source: &str, span: usize, name: &str) -> Expr {
    per_ticker(ema_expr(source, span), name)
}

/// Per-ticker percent-change column, cast to Float32:
/// `pct("close", 5, "pct1w")`.
fn pct(source: &str, period: i64, name: &str) -> Expr {
    per_ticker_f32(chg_expr(f(source), period), name)
}

/// Per-ticker realized-volatility column, cast to Float32:
/// `rv("log_ret", 21, 252, "rv21")`.
fn rv(log_ret_col: &str, window: usize, trading_periods: usize, name: &str) -> Expr {
    per_ticker_f32(rv_expr(log_ret_col, window, trading_periods), name)
}

/// Add SMA columns, one per period: `sma5`, `sma10`, ...
fn with_sma_columns(lf: LazyFrame, source: &str, periods: &[usize]) -> LazyFrame {
    let cols: Vec<Expr> = periods
        .iter()
        .map(|&p| sma(source, p, &format!("sma{p}")))
        .collect();
    lf.with_columns(cols)
}

/// Add EMA columns, one per span: `ema8`, `ema12`, ...
fn with_ema_columns(lf: LazyFrame, source: &str, spans: &[usize]) -> LazyFrame {
    let cols: Vec<Expr> = spans
        .iter()
        .map(|&s| ema(source, s, &format!("ema{s}")))
        .collect();
    lf.with_columns(cols)
}

/// Add percent-change columns, one per period: `pct1`, `pct5`, ...
fn with_pct_columns(lf: LazyFrame, source: &str, periods: &[i64]) -> LazyFrame {
    let cols: Vec<Expr> = periods
        .iter()
        .map(|&p| pct(source, p, &format!("pct{p}")))
        .collect();
    lf.with_columns(cols)
}

/// Add realized-volatility columns, one per window: `rv21`, `rv63`, ...
/// Computes log-returns internally from `source`; the caller doesn't need
/// to materialize a log-return column.
fn with_rv_columns(
    lf: LazyFrame,
    source: &str,
    trading_periods: usize,
    windows: &[usize],
) -> LazyFrame {
    let lf = lf.with_columns([per_ticker(
        (f(source) / f(source).shift(lit(1))).log(std::f64::consts::E),
        "_log_ret",
    )]);
    let cols: Vec<Expr> = windows
        .iter()
        .map(|&w| rv("_log_ret", w, trading_periods, &format!("rv{w}")))
        .collect();
    lf.with_columns(cols).drop(["_log_ret"])
}

/// Add rolling min/max (Donchian channel) columns, two per window.
/// Upper band uses intraday highs, lower band uses intraday lows — the
/// standard Donchian construction. The channel covers the *prior* N bars
/// (current bar excluded) so a same-bar high doesn't trivially break its
/// own channel.
fn with_minmax_columns(lf: LazyFrame, windows: &[usize]) -> LazyFrame {
    let cols: Vec<Expr> = windows
        .iter()
        .flat_map(|&w| {
            let opts = RollingOptionsFixedWindow {
                window_size: w,
                min_periods: w,
                ..Default::default()
            };
            [
                per_ticker(
                    f("low").rolling_min(opts.clone()).shift(lit(1)),
                    &format!("min{w}"),
                ),
                per_ticker(
                    f("high").rolling_max(opts).shift(lit(1)),
                    &format!("max{w}"),
                ),
            ]
        })
        .collect();
    lf.with_columns(cols)
}

/// Simple moving average expression.
pub fn sma_expr(source: &str, period: usize) -> Expr {
    f(source).rolling_mean(rolling_opts(period))
}

/// Exponential moving average expression using span-based alpha.
pub fn ema_expr(source: &str, span: usize) -> Expr {
    f(source).ewm_mean(EWMOptions {
        alpha: 2.0 / (span as f64 + 1.0),
        min_periods: span,
        ..Default::default()
    })
}

/// Percent change over `period` rows.
pub fn chg_expr(source: Expr, period: i64) -> Expr {
    ((source.clone() / source.shift(lit(period))) - lit(1.0)) * lit(100.0)
}

/// CAGR assuming quarterly cadence (`periods * 4` row shift).
pub fn cagr_expr(source: Expr, periods: i64) -> Expr {
    ((source.clone() / source.shift(lit(periods * 4))).pow(lit(1.0 / periods as f64)) - lit(1.0))
        * lit(100.0)
}

/// Annualized realized volatility from log returns.
pub fn rv_expr(log_ret_col: &str, window: usize, trading_periods: usize) -> Expr {
    let log_ret = col(log_ret_col);
    let mean = log_ret.clone().rolling_mean(rolling_opts(window));
    let mean_sq = (log_ret.clone() * log_ret).rolling_mean(rolling_opts(window));
    let pop_std = (mean_sq - mean.clone() * mean).abs().pow(lit(0.5));
    pop_std * lit((trading_periods as f64).sqrt()) * lit(100.0)
}

fn wilder_smooth(source: Expr, period: usize) -> Expr {
    source.ewm_mean(EWMOptions {
        alpha: 1.0 / period as f64,
        min_periods: period,
        ..Default::default()
    })
}

pub fn macd_line_expr(fast: usize, slow: usize) -> Expr {
    ema_expr("close", fast) - ema_expr("close", slow)
}

pub fn bbtop_expr(period: usize, multiplier: f64) -> Expr {
    let sma = f("close").rolling_mean(rolling_opts(period));
    let stdev = f("close").rolling_std(rolling_opts(period));
    sma + lit(multiplier) * stdev
}

pub fn bbbot_expr(period: usize, multiplier: f64) -> Expr {
    let sma = f("close").rolling_mean(rolling_opts(period));
    let stdev = f("close").rolling_std(rolling_opts(period));
    sma - lit(multiplier) * stdev
}

pub fn true_range_expr() -> Expr {
    let prev_close = f("close").shift(lit(1));
    let hl = f("high") - f("low");
    let hc = (f("high") - prev_close.clone()).abs();
    let lc = (f("low") - prev_close).abs();
    let max_hl_hc = when(hl.clone().gt_eq(hc.clone())).then(hl).otherwise(hc);
    when(max_hl_hc.clone().gt_eq(lc.clone()))
        .then(max_hl_hc)
        .otherwise(lc)
}

/// Add MACD columns.
pub fn with_macd(lf: LazyFrame, fast: usize, slow: usize, signal: usize) -> LazyFrame {
    let is_default = fast == 12 && slow == 26 && signal == 9;
    let (ema_fast_col, ema_slow_col, macd_col, signal_col) = if is_default {
        (
            "ema12".to_string(),
            "ema26".to_string(),
            "macd".to_string(),
            "macdsignal".to_string(),
        )
    } else {
        (
            format!("ema{fast}"),
            format!("ema{slow}"),
            format!("macd_{fast}_{slow}"),
            format!("macdsignal_{fast}_{slow}_{signal}"),
        )
    };

    // Signal is computed in a second wave because it depends on `macd_col`.
    lf.with_columns([
        ema_expr("close", fast)
            .over([col("ticker")])
            .alias(ema_fast_col.as_str()),
        ema_expr("close", slow)
            .over([col("ticker")])
            .alias(ema_slow_col.as_str()),
        macd_line_expr(fast, slow)
            .over([col("ticker")])
            .alias(macd_col.as_str()),
    ])
    .with_columns([ema_expr(macd_col.as_str(), signal)
        .over([col("ticker")])
        .alias(signal_col.as_str())])
}

/// Add Bollinger band columns.
pub fn with_bollinger(lf: LazyFrame, period: usize, multiplier: f64) -> LazyFrame {
    let is_default = period == 20 && (multiplier - 2.0).abs() < 1e-9;
    let (top_col, bot_col) = if is_default {
        ("bbtop".to_string(), "bbbot".to_string())
    } else {
        let mult_str = if (multiplier - multiplier.round()).abs() < 1e-9 {
            format!("{}", multiplier as i64)
        } else {
            format!("{multiplier}")
        };
        (
            format!("bbtop_{period}_{mult_str}"),
            format!("bbbot_{period}_{mult_str}"),
        )
    };

    lf.with_columns([
        bbbot_expr(period, multiplier)
            .over([col("ticker")])
            .alias(bot_col.as_str()),
        bbtop_expr(period, multiplier)
            .over([col("ticker")])
            .alias(top_col.as_str()),
    ])
}

/// Add RSI column.
pub fn with_rsi(lf: LazyFrame, period: usize) -> LazyFrame {
    let rsi_col = format!("rsi{period}");

    lf.with_columns([(f("close") - f("close").shift(lit(1)))
        .over([col("ticker")])
        .alias("_delta")])
        .with_columns([
            when(col("_delta").gt(lit(0.0)))
                .then(col("_delta"))
                .otherwise(lit(0.0))
                .alias("_gain"),
            when(col("_delta").lt(lit(0.0)))
                .then(-col("_delta"))
                .otherwise(lit(0.0))
                .alias("_loss"),
        ])
        .with_columns([
            wilder_smooth(col("_gain"), period)
                .over([col("ticker")])
                .alias("_avg_gain"),
            wilder_smooth(col("_loss"), period)
                .over([col("ticker")])
                .alias("_avg_loss"),
        ])
        .with_columns([(lit(100.0)
            - lit(100.0) / (lit(1.0) + col("_avg_gain") / col("_avg_loss")))
        .alias(rsi_col.as_str())])
        .drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"])
}

/// Add ATR column.
pub fn with_atr(lf: LazyFrame, period: usize) -> LazyFrame {
    let atr_col = format!("atr{period}");

    lf.with_columns([f("close")
        .shift(lit(1))
        .over([col("ticker")])
        .alias("_prev_close")])
        .with_columns([{
            let hl = f("high") - f("low");
            let hc = (f("high") - col("_prev_close")).abs();
            let lc = (f("low") - col("_prev_close")).abs();
            let max_hl_hc = when(hl.clone().gt_eq(hc.clone())).then(hl).otherwise(hc);
            when(max_hl_hc.clone().gt_eq(lc.clone()))
                .then(max_hl_hc)
                .otherwise(lc)
                .alias("_tr")
        }])
        .with_columns([wilder_smooth(col("_tr"), period)
            .over([col("ticker")])
            .alias(atr_col.as_str())])
        .drop(["_prev_close", "_tr"])
}

/// Add ADX column.
pub fn with_adx(lf: LazyFrame, period: usize) -> LazyFrame {
    let adx_col = format!("adx{period}");

    lf.with_columns([
        (f("high") - f("high").shift(lit(1)))
            .over([col("ticker")])
            .alias("_up_move"),
        (f("low").shift(lit(1)) - f("low"))
            .over([col("ticker")])
            .alias("_down_move"),
        f("close")
            .shift(lit(1))
            .over([col("ticker")])
            .alias("_prev_close"),
    ])
    .with_columns([
        {
            let hl = f("high") - f("low");
            let hc = (f("high") - col("_prev_close")).abs();
            let lc = (f("low") - col("_prev_close")).abs();
            let max_hl_hc = when(hl.clone().gt_eq(hc.clone())).then(hl).otherwise(hc);
            when(max_hl_hc.clone().gt_eq(lc.clone()))
                .then(max_hl_hc)
                .otherwise(lc)
                .alias("_tr")
        },
        when(
            col("_up_move")
                .gt(col("_down_move"))
                .and(col("_up_move").gt(lit(0.0))),
        )
        .then(col("_up_move"))
        .otherwise(lit(0.0))
        .alias("_plus_dm"),
        when(
            col("_down_move")
                .gt(col("_up_move"))
                .and(col("_down_move").gt(lit(0.0))),
        )
        .then(col("_down_move"))
        .otherwise(lit(0.0))
        .alias("_minus_dm"),
    ])
    .with_columns([
        wilder_smooth(col("_tr"), period)
            .over([col("ticker")])
            .alias("_smooth_tr"),
        wilder_smooth(col("_plus_dm"), period)
            .over([col("ticker")])
            .alias("_smooth_plus_dm"),
        wilder_smooth(col("_minus_dm"), period)
            .over([col("ticker")])
            .alias("_smooth_minus_dm"),
    ])
    .with_columns([
        (lit(100.0) * col("_smooth_plus_dm") / col("_smooth_tr")).alias("_plus_di"),
        (lit(100.0) * col("_smooth_minus_dm") / col("_smooth_tr")).alias("_minus_di"),
    ])
    .with_columns([wilder_smooth(
        lit(100.0) * (col("_plus_di") - col("_minus_di")).abs()
            / (col("_plus_di") + col("_minus_di")),
        period,
    )
    .over([col("ticker")])
    .alias(adx_col.as_str())])
    .drop([
        "_up_move",
        "_down_move",
        "_prev_close",
        "_tr",
        "_plus_dm",
        "_minus_dm",
        "_smooth_tr",
        "_smooth_plus_dm",
        "_smooth_minus_dm",
        "_plus_di",
        "_minus_di",
    ])
}

/// Adjust raw OHLC values using `closeadj / close`.
pub fn adjust_prices(lf: LazyFrame) -> LazyFrame {
    // Keep only adjusted OHLCV columns used downstream.
    let adjustment_factor =
        col("closeadj").cast(DataType::Float64) / col("close").cast(DataType::Float64);

    lf.select([
        col("ticker"),
        col("date").cast(DataType::Date),
        (col("open").cast(DataType::Float64) * adjustment_factor.clone()).alias("open"),
        (col("high").cast(DataType::Float64) * adjustment_factor.clone()).alias("high"),
        (col("low").cast(DataType::Float64) * adjustment_factor).alias("low"),
        col("closeadj").cast(DataType::Float64).alias("close"),
        col("volume").cast(DataType::Float64),
    ])
}

/// Resample OHLCV bars at `interval` per ticker.
pub fn resample(lf: LazyFrame, interval: &str) -> LazyFrame {
    // Aggregate OHLCV per ticker on a dynamic time window.
    lf.group_by_dynamic(
        col("date"),
        [col("ticker")],
        DynamicGroupOptions {
            every: Duration::parse(interval),
            period: Duration::parse(interval),
            offset: Duration::parse("0d"),
            label: Label::Left,
            closed_window: ClosedWindow::Left,
            include_boundaries: false,
            start_by: StartBy::WindowBound,
            ..Default::default()
        },
    )
    .agg([
        col("open").first().alias("open"),
        col("high").max().alias("high"),
        col("low").min().alias("low"),
        col("close").last().alias("close"),
        col("volume").sum().alias("volume"),
    ])
    .sort(["ticker", "date"], Default::default())
}

pub fn with_rank(
    lf: LazyFrame,
    source: &str,
    date_col: &str,
    ascending: bool,
    alias: &str,
) -> LazyFrame {
    let opts = RankOptions {
        descending: !ascending,
        ..Default::default()
    };

    // n = count of non-null `source` values within the date group.
    let n = col(source).count().over([col(date_col)]);
    let rank = col(source).rank(opts, None).over([col(date_col)]);

    lf.with_columns([(((rank - lit(1.0)) / (n - lit(1.0))) * lit(100.0))
        .alias(alias)
        .cast(DataType::Float32)])
}

/// Transform raw fundamentals into analysis columns.
pub fn adjust_fundamentals(lf: LazyFrame) -> LazyFrame {
    lf.with_columns([col("calendardate").cast(DataType::Date)])
        .drop(["dimension", "lastupdated"])
        .with_columns([
            (f("sharesbas") * f("sharefactor")).alias("shares"),
            (f("dps") / f("fxusd")).alias("dpsusd"),
            (f("eps") / f("fxusd")).alias("epsusd"),
            (f("ncfo") / f("fxusd")).alias("ncfousd"),
            (f("fcf") / f("fxusd")).alias("fcfusd"),
            (f("assets") - f("liabilities")).alias("equity"),
            ((f("debt") - f("cashneq")) / f("fxusd")).alias("netdebtusd"),
            (lit(100.0) * f("roe")).alias("roe"),
            (lit(100.0) * f("roic")).alias("roic"),
            (lit(100.0) * f("roa")).alias("roa"),
            (f("ncfo") / f("opinc")).alias("cfc"),
            (f("ebit") / f("intexp")).alias("icr"),
            (lit(100.0) * f("ebit") / (f("assets") - f("liabilitiesc"))).alias("roce"),
            ((f("netinc") - f("netincdis")) / f("fxusd")).alias("netincadj"),
            ((lit(-1.0) * f("ncfcommon") / f("fxusd")) / f("marketcap") * lit(100.0))
                .alias("bbyield"),
            (lit(100.0) * f("divyield")).alias("divyield"),
            ((lit(-1.0) * f("ncfdiv") + lit(-1.0) * f("ncfcommon")) / f("fxusd"))
                .alias("shreturnusd"),
            (lit(100.0) * f("gp") / f("revenue")).alias("grossmargin"),
            (lit(100.0) * f("ebitda") / f("revenue")).alias("ebitdamargin"),
            (lit(100.0) * f("ebit") / f("revenue")).alias("ebitmargin"),
        ])
        .with_columns([
            (f("debt") / f("equity")).alias("de"),
            (f("netincadj") / f("shares")).alias("epsadj"),
            (lit(100.0) * f("netincadj") / f("revenueusd")).alias("netmargin"),
            (lit(100.0) * f("shreturnusd") / f("marketcap")).alias("shyield"),
        ])
        .sort(["ticker", "calendardate"], Default::default())
        // `shift`-based metrics require sorted time order.
        .with_columns({
            let fcfadj = (f("ncfo")
                - f("netincdis")
                - f("depamor")
                - (f("workingcapital") - f("workingcapital").shift(lit(1))).over([col("ticker")])
                - f("sbcomp"))
                / f("fxusd");
            [
                fcfadj.clone().alias("fcfadj"),
                (fcfadj / f("shares")).alias("fcfpsadj"),
            ]
        })
        .with_columns([
            chg_expr(f("revenue") / f("shares"), 4)
                .over([col("ticker")])
                .alias("revenueyoy"),
            cagr_expr(f("revenue") / f("shares"), 2)
                .over([col("ticker")])
                .alias("revenuecagr"),
            chg_expr(f("ebitdausd") / f("shares"), 4)
                .over([col("ticker")])
                .alias("ebitdayoy"),
            cagr_expr(f("ebitdausd") / f("shares"), 2)
                .over([col("ticker")])
                .alias("ebitdacagr"),
            chg_expr(f("epsadj"), 4)
                .over([col("ticker")])
                .alias("epsyoy"),
            cagr_expr(f("epsadj"), 2)
                .over([col("ticker")])
                .alias("epscagr"),
            chg_expr(f("fcfpsadj"), 4)
                .over([col("ticker")])
                .alias("fcfpsyoy"),
            cagr_expr(f("fcfpsadj"), 2)
                .over([col("ticker")])
                .alias("fcfpscagr"),
            chg_expr(f("revenue") / f("shares"), 4 * 3)
                .over([col("ticker")])
                .alias("revenue3y"),
            chg_expr(f("ebitdausd") / f("shares"), 4 * 3)
                .over([col("ticker")])
                .alias("ebitda3y"),
            chg_expr(f("epsadj"), 4 * 3)
                .over([col("ticker")])
                .alias("eps3y"),
            chg_expr(f("fcfpsadj"), 4 * 3)
                .over([col("ticker")])
                .alias("fcfps3y"),
        ])
}

/// Aggregate insider transactions over the last ~6 months.
pub fn update_insiders(lf: LazyFrame) -> LazyFrame {
    let six_months_ago = chrono::Utc::now().date_naive() - chrono::Duration::weeks(26);

    lf.with_columns([
        col("formtype").cast(DataType::String),
        col("transactiondate").cast(DataType::Date).alias("date"),
        col("transactionshares")
            .cast(DataType::Float64)
            .abs()
            .alias("_transactionshares_abs"),
    ])
    .filter(
        col("date")
            .gt_eq(lit(six_months_ago))
            .and(f("transactionvalue").neq(lit(0.0))),
    )
    .group_by([
        col("ticker"),
        col("date"),
        col("issuername"),
        col("ownername"),
        col("transactioncode"),
        col("securityadcode"),
        col("securitytitle"),
        col("officertitle"),
        col("isofficer"),
        col("isdirector"),
        col("istenpercentowner"),
    ])
    .agg([
        col("transactionvalue").sum().alias("transactionvalue"),
        col("_transactionshares_abs")
            .sum()
            .alias("transactionshares"),
        col("transactionpricepershare")
            .mean()
            .alias("transactionpricepershare"),
    ])
    .sort(
        ["date", "transactionvalue"],
        SortMultipleOptions::default().with_order_descending_multi([true, true]),
    )
    .with_columns([when(col("isofficer").eq(lit("Y")))
        .then(col("officertitle").fill_null(lit("")))
        .when(col("isdirector").eq(lit("Y")))
        .then(lit("Director"))
        .when(col("istenpercentowner").eq(lit("Y")))
        .then(lit("10% Owner"))
        .otherwise(col("officertitle").fill_null(lit("")))
        .alias("officertitle")])
}

/// Compute daily technical indicator columns.
pub fn technical_indicators_daily(lf: LazyFrame) -> LazyFrame {
    // Average traded volume (liquidity filter)
    let lf = lf
        .sort(["ticker", "date"], Default::default())
        .with_columns([sma("volume", 50, "avgvolume3m")]);

    // 52-Week High and Low price range
    let lf = with_minmax_columns(lf, &[252]);
    // Donchian channel ranges (commonly 1m and 1q highs and lows)
    let lf = with_minmax_columns(lf, &[20, 55]);

    // Time-series momentum (rate of change of price, in percent).
    // Lookbacks ~1w, 1m, 3m, 6m, 9m, 12m at 21 trading days per month.
    let lf = with_pct_columns(lf, "close", &[5, 21, 3 * 21, 6 * 21, 9 * 21, 12 * 21]);

    // Composite Relative Strength (cross-sectional momentum),
    // a weighted blend of the 3m/6m/9m/12m lookbacks.
    let lf = lf.with_columns([(lit(0.4) * col("pct63")
        + lit(0.2) * col("pct126")
        + lit(0.2) * col("pct189")
        + lit(0.2) * col("pct252"))
    .alias("rs1y")
    .cast(DataType::Float32)]);
    // Replace the scalar value of Relative Strength with the cross-sectional
    // ranking from 0-100 were 100 is best.
    let lf = with_rank(lf, "rs1y", "date", true, "rs1y");

    // Moving Averages used for trend filters
    let lf = with_sma_columns(lf, "close", &[10, 20, 50, 100, 150, 200]);
    // MACD often used to measure trend and momentum
    let lf = with_macd(lf, 12, 26, 9);
    // ADX used to measure trend strength
    let lf = with_adx(lf, 14);

    // Realized volatility (annualized, in percent)
    let lf = with_rv_columns(lf, "close", 252, &[21, 63, 252]);

    // Bollinger Band indicator and ATR for volatility breakouts, mean reversion
    let lf = with_bollinger(lf, 20, 2.0);
    let lf = with_atr(lf, 14);
    // RSI indicator used for mean reversion
    with_rsi(lf, 14)
}
