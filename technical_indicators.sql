-- =====================================================================
-- stocks_indicators_1d / stocks_indicators_1w
--
-- Views of non-recursive technical-indicator building blocks, computed
-- straight from `stocks_daily` / `stocks_weekly`. These are the indicators
-- that translate cleanly to SQL window functions; recursive ones (RSI,
-- ATR, ADX — Wilder smoothing) are deliberately NOT here. Volatility is
-- measured by ADR (Average Daily Range %), which is non-recursive.
--
-- WINDOW SEMANTICS: a fixed-window aggregate is wrapped in a COUNT() guard
-- so it returns NULL until the window is full — matching Polars rolling
-- with min_periods == window. Rolling extrema use min_periods = 2.
--
-- Column naming: ROC indicators are exposed as `pct*` (percentage change).
--
-- Window sizes mirror the Rust pipeline's technical_indicators_daily /
-- technical_indicators_weekly so the views agree with the Rust output.
--
-- Named windows are prefixed per view (d_* daily, w_* weekly): running
-- both CREATE VIEW statements in one batch, a shared WINDOW name collides
-- in DuckDB's parser, so each view uses its own distinct names.
-- =====================================================================


-- #####################################################################
-- DAILY  —  over stocks_daily
--   SMA 5/10/20/200 · pct (1d)/5/20/63/126/189/252 · RV 10/20/60/252
--   avgvolume20/50-bar · min1y/max1y = 250-bar · RV annualized x252
-- #####################################################################
CREATE OR REPLACE VIEW stocks_indicators_1d AS
WITH base AS (
    SELECT
        ticker,
        date,
        open, high, low, close, volume,
        ln(close / lag(close) OVER d_seq)               AS log_ret
    FROM stocks_daily
    WINDOW d_seq AS (PARTITION BY ticker ORDER BY date)
)
SELECT
    ticker,
    date,
    open, high, low, close, volume,

    -- ---- Percentage change (ROC -> pct), row-based shifts ----------
    (close / lag(close,   1) OVER d_seq - 1) * 100       AS pct1d,
    (close / lag(close,   5) OVER d_seq - 1) * 100       AS pct1w,
    (close / lag(close,  20) OVER d_seq - 1) * 100       AS pct1m,
    (close / lag(close,  63) OVER d_seq - 1) * 100       AS pct1q,
    (close / lag(close, 126) OVER d_seq - 1) * 100       AS pct2q,
    (close / lag(close, 189) OVER d_seq - 1) * 100       AS pct3q,
    (close / lag(close, 252) OVER d_seq - 1) * 100       AS pct1y,

    -- ---- Simple moving averages (NULL until window full) -----------
    CASE WHEN count(close) OVER d_5   = 5   THEN avg(close) OVER d_5   END AS sma5,
    CASE WHEN count(close) OVER d_10  = 10  THEN avg(close) OVER d_10  END AS sma10,
    CASE WHEN count(close) OVER d_20  = 20  THEN avg(close) OVER d_20  END AS sma20,
    CASE WHEN count(close) OVER d_50  = 50  THEN avg(close) OVER d_50  END AS sma50,
    CASE WHEN count(close) OVER d_200 = 200 THEN avg(close) OVER d_200 END AS sma200,

    -- ---- Average volume: 1 month (~20 bars) and 3 months (~50 bars) -
    CASE WHEN count(volume) OVER d_20 = 20 THEN avg(volume) OVER d_20 END AS avgvolume1m,
    CASE WHEN count(volume) OVER d_50 = 50 THEN avg(volume) OVER d_50 END AS avgvolume3m,

    -- ---- ADR: Average Daily Range %, 20-bar (Qullamaggie's formula).
    -- 100 * (avg(high/low) over 20 bars - 1). Uses the plain high/low
    -- ratio (intraday range only, no gaps) and a plain 20-bar SMA - the
    -- canonical momentum/tradeability filter.
    CASE WHEN count(close) OVER d_20 = 20 THEN
        (avg(high / low) OVER d_20 - 1) * 100
    END                                                  AS adr,

    -- ---- Bollinger Bands (20-bar, k=2, population stdev) -----------
    -- Matches the pipeline's bbtop_expr/bbbot_expr (Polars rolling_std
    -- is population std, so stddev_pop is the correct match).
    CASE WHEN count(close) OVER d_20 = 20 THEN
        avg(close) OVER d_20 + 2 * stddev_pop(close) OVER d_20
    END                                                  AS bbtop,
    CASE WHEN count(close) OVER d_20 = 20 THEN
        avg(close) OVER d_20 - 2 * stddev_pop(close) OVER d_20
    END                                                  AS bbbot,

    -- ---- Rolling 1-year extrema (250 bars, min_periods = 2) --------
    CASE WHEN count(close) OVER d_250 >= 2 THEN max(close) OVER d_250 END AS max1y,
    CASE WHEN count(close) OVER d_250 >= 2 THEN min(close) OVER d_250 END AS min1y,

    -- ---- Realized volatility (annualized %, 252 trading periods) ----
    CASE WHEN count(log_ret) OVER d_10  = 10  THEN
        sqrt(abs(avg(log_ret*log_ret) OVER d_10
               - pow(avg(log_ret) OVER d_10,  2))) * sqrt(252) * 100 END AS rv10,
    CASE WHEN count(log_ret) OVER d_20  = 20  THEN
        sqrt(abs(avg(log_ret*log_ret) OVER d_20
               - pow(avg(log_ret) OVER d_20,  2))) * sqrt(252) * 100 END AS rv20,
    CASE WHEN count(log_ret) OVER d_60  = 60  THEN
        sqrt(abs(avg(log_ret*log_ret) OVER d_60
               - pow(avg(log_ret) OVER d_60,  2))) * sqrt(252) * 100 END AS rv60,
    CASE WHEN count(log_ret) OVER d_252 = 252 THEN
        sqrt(abs(avg(log_ret*log_ret) OVER d_252
               - pow(avg(log_ret) OVER d_252, 2))) * sqrt(252) * 100 END AS rv252,

    -- ---- rs1y: weighted composite of pct1q/2q/3q/1y -----------------
    0.4 * (close / lag(close,  63) OVER d_seq - 1) * 100
  + 0.2 * (close / lag(close, 126) OVER d_seq - 1) * 100
  + 0.2 * (close / lag(close, 189) OVER d_seq - 1) * 100
  + 0.2 * (close / lag(close, 252) OVER d_seq - 1) * 100  AS rs1y

FROM base
WINDOW
    d_seq AS (PARTITION BY ticker ORDER BY date),
    d_5   AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4   PRECEDING AND CURRENT ROW),
    d_10  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9   PRECEDING AND CURRENT ROW),
    d_20  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19  PRECEDING AND CURRENT ROW),
    d_50  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49  PRECEDING AND CURRENT ROW),
    d_60  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 59  PRECEDING AND CURRENT ROW),
    d_200 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW),
    d_250 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW),
    d_252 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
ORDER BY ticker, date;


-- #####################################################################
-- WEEKLY  —  over stocks_weekly
--   Mirrors technical_indicators_weekly: SMA 10/30/40/200 weeks
--   (Weinstein-style), pct (1w)/4/13/26/39/52, RV 4/13/52 weeks
--   (annualized x52), avgvolume13 = 13-bar, min1y/max1y = 52-bar.
-- #####################################################################
CREATE OR REPLACE VIEW stocks_indicators_1w AS
WITH base AS (
    SELECT
        ticker,
        date,
        open, high, low, close, volume,
        ln(close / lag(close) OVER w_seq)               AS log_ret
    FROM stocks_weekly
    WINDOW w_seq AS (PARTITION BY ticker ORDER BY date)
)
SELECT
    ticker,
    date,
    open, high, low, close, volume,

    -- ---- Percentage change (ROC -> pct), weekly cadence ------------
    (close / lag(close,  1) OVER w_seq - 1) * 100        AS pct1w,
    (close / lag(close,  4) OVER w_seq - 1) * 100        AS pct1m,
    (close / lag(close, 13) OVER w_seq - 1) * 100        AS pct1q,
    (close / lag(close, 26) OVER w_seq - 1) * 100        AS pct2q,
    (close / lag(close, 39) OVER w_seq - 1) * 100        AS pct3q,
    (close / lag(close, 52) OVER w_seq - 1) * 100        AS pct1y,

    -- ---- Simple moving averages (Weinstein 10/30/40/200 weeks) -----
    CASE WHEN count(close) OVER w_10  = 10  THEN avg(close) OVER w_10  END AS sma10,
    CASE WHEN count(close) OVER w_30  = 30  THEN avg(close) OVER w_30  END AS sma30,
    CASE WHEN count(close) OVER w_40  = 40  THEN avg(close) OVER w_40  END AS sma40,
    CASE WHEN count(close) OVER w_200 = 200 THEN avg(close) OVER w_200 END AS sma200,

    -- ---- Average volume (~3 months ≈ 13 weekly bars) ---------------
    CASE WHEN count(volume) OVER w_13 = 13 THEN avg(volume) OVER w_13 END AS avgvolume3m,

    -- ---- ADR: Average (weekly) Range %, 20-bar. Weekly parallel of
    -- Qullamaggie's daily ADR(20): 100 * (avg(high/low) over 20 bars
    -- - 1). Note Qullamaggie's ADR is canonically a daily-chart tool;
    -- this 20-week form is the same formula on weekly bars.
    CASE WHEN count(close) OVER w_20 = 20 THEN
        (avg(high / low) OVER w_20 - 1) * 100
    END                                                  AS adr,

    -- ---- Bollinger Bands (20-week, k=2, population stdev) ----------
    CASE WHEN count(close) OVER w_20 = 20 THEN
        avg(close) OVER w_20 + 2 * stddev_pop(close) OVER w_20
    END                                                  AS bbtop,
    CASE WHEN count(close) OVER w_20 = 20 THEN
        avg(close) OVER w_20 - 2 * stddev_pop(close) OVER w_20
    END                                                  AS bbbot,

    -- ---- Rolling 1-year extrema (52 bars, min_periods = 2) ---------
    CASE WHEN count(close) OVER w_52 >= 2 THEN max(close) OVER w_52 END AS max1y,
    CASE WHEN count(close) OVER w_52 >= 2 THEN min(close) OVER w_52 END AS min1y,

    -- ---- Realized volatility (annualized %, 52 weeks/year) ---------
    CASE WHEN count(log_ret) OVER w_4  = 4  THEN
        sqrt(abs(avg(log_ret*log_ret) OVER w_4
               - pow(avg(log_ret) OVER w_4,  2))) * sqrt(52) * 100 END AS rv4,
    CASE WHEN count(log_ret) OVER w_13 = 13 THEN
        sqrt(abs(avg(log_ret*log_ret) OVER w_13
               - pow(avg(log_ret) OVER w_13, 2))) * sqrt(52) * 100 END AS rv13,
    CASE WHEN count(log_ret) OVER w_52 = 52 THEN
        sqrt(abs(avg(log_ret*log_ret) OVER w_52
               - pow(avg(log_ret) OVER w_52, 2))) * sqrt(52) * 100 END AS rv52,

    -- ---- rs1y: weighted composite of pct1q/2q/3q/1y -----------------
    0.4 * (close / lag(close, 13) OVER w_seq - 1) * 100
  + 0.2 * (close / lag(close, 26) OVER w_seq - 1) * 100
  + 0.2 * (close / lag(close, 39) OVER w_seq - 1) * 100
  + 0.2 * (close / lag(close, 52) OVER w_seq - 1) * 100   AS rs1y

FROM base
WINDOW
    w_seq AS (PARTITION BY ticker ORDER BY date),
    w_4   AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 3   PRECEDING AND CURRENT ROW),
    w_10  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9   PRECEDING AND CURRENT ROW),
    w_13  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 12  PRECEDING AND CURRENT ROW),
    w_20  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19  PRECEDING AND CURRENT ROW),
    w_30  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29  PRECEDING AND CURRENT ROW),
    w_40  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 39  PRECEDING AND CURRENT ROW),
    w_52  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 51  PRECEDING AND CURRENT ROW),
    w_200 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW)
ORDER BY ticker, date;