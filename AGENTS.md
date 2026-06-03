This is a Rust cli project. 

# Working agreements
- Optimize code for both runtime and memory usage

Pipeline:
- Download datasets from NASADQ api
- write_stocks
  - Load daily stock prices
  - Adjust closing prices
  - Apply technical indicators
  - Write the combined prices and indicators to db
- write_stocks_weekly
  - Like write_stocks but resamples to weekly prices
- write_financials
  - Load financials TTM data
  - Adjust currency to USD, shares outstanding, and percent formatting
  - Computes fundamental metrics
  - Applies rankings
  - Writes the combined reported financials and metrics to db
- write_companies
  - Loads company data
  - Joins with latest reported financials and price data
  - Writes one row for each company latest data
- write_insiders
  - Loads insider trades reports
  - Adjust for recent transactions
  - Formats titles and codes
  - Writes to db
