-- =====================================================================
-- build_companies.sql
--
-- Builds the `companies` snapshot: active companies x latest fundamentals
-- row x latest daily price row.
--
-- Run by write_companies() (in writer.rs) via sqltools::execute_sql_file.
-- This script is self-contained for the companies metadata: it reads the
-- raw parquet directly, so no companies_meta_raw table is created or
-- dropped by the Rust side.
--
-- DEPENDS ON (must exist before this runs):
--   downloads/companies.parquet  — raw metadata parquet, written by
--                                  write_companies() (phase-1 conversion)
--   financials_ttm               — table, written by write_financials()
--   stocks_daily                 — table, written by write_stocks()
-- =====================================================================
DROP TABLE IF EXISTS companies;
CREATE TABLE companies AS
WITH active_meta AS (
    -- Raw companies metadata is read straight from the phase-1 parquet;
    -- no transform, so it is never materialized as its own table.
    SELECT
        ticker, name, exchange, category, currency, location,
        sector, industry, companysite, secfilings
    FROM read_parquet('downloads/companies.parquet')
    WHERE isdelisted = 'N'
),
ranked_financials AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY calendardate DESC) AS rn
    FROM financials_ttm
),
latest_financials AS (
    SELECT * EXCLUDE (rn) FROM ranked_financials WHERE rn = 1
),
ranked_prices AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
    FROM stocks_daily
),
-- Latest price row per ticker, but ONLY for tickers still actively traded.
-- "Actively traded" = the ticker's most recent price row falls on the
-- single most recent trading date across the whole stocks_daily table.
-- A company that stopped trading a while ago has a stale latest row (an
-- older date), so it fails this filter and — via the INNER JOIN below —
-- drops out of the snapshot entirely. This keeps `companies` a snapshot
-- of whatever is currently trading, at its latest price.
latest_prices AS (
    SELECT * EXCLUDE (rn)
    FROM ranked_prices
    WHERE rn = 1
      AND date = (SELECT max(date) FROM stocks_daily)
)
SELECT
    m.*,
    -- marketcap and ev both exist as raw columns in financials_ttm; both
    -- are recomputed against the latest close price. ev recomputes the
    -- marketcap term inline because the REPLACE alias is not visible to a
    -- sibling expression in the same SELECT.
    f.* EXCLUDE (ticker) REPLACE (
        f.shares * p.close                AS marketcap,
        f.shares * p.close + f.netdebtusd AS ev
    ),
    p.* EXCLUDE (ticker)
FROM active_meta m
INNER JOIN latest_financials f ON f.ticker = m.ticker
INNER JOIN latest_prices    p ON p.ticker = m.ticker;

-- NOTE: INNER JOIN on both sides — an active ticker missing either a
-- fundamentals row or a daily price row is dropped from the snapshot.