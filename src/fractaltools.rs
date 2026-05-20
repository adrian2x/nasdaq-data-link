// src/fractaltools.rs
//
// Rolling Hurst exponent estimation via DFA1 on a volatility- and
// volume-weighted composite signal, computed per ticker on a Polars DataFrame.
//
// Targets Polars 0.46.
//
// Design notes:
//   1. DFA1 inner loop is O(n_boxes) per scale instead of O(window), using
//      prefix sums of {P, P*P, j*P} over the integrated profile. This is
//      the dominant runtime speedup.
//   2. Per-ticker scratch buffers (profile + 3 prefix-sum arrays of length
//      W+1) are reused across all window positions for that ticker.
//   3. Signal construction (Garman-Klass + rolling-volume + composite) is
//      fused into a single forward pass with ring buffers of length
//      vol_window. No length-L intermediates; only the final `signal` vec.
//   4. OHLCV is NOT materialized into per-ticker copies — `build_composite_indexed`
//      reads directly from the global arrays via sorted row indices.
//   5. Ticker grouping replaces df.clone() + partition_by with a single
//      HashMap pass over the ticker column. No DataFrame clone, no per-
//      partition column materialization.
//   6. Parallelism is per-ticker via rayon. With thousands of tickers,
//      this saturates any reasonable core count without nested
//      oversubscription.
//   7. cfg.reverse_pass toggles the second (tail-anchored) DFA box pass.
//      Disable for ~2x DFA speedup at modest fidelity cost.

use polars::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct HurstConfig {
    pub window: usize,
    pub vol_window: usize,
    pub min_scale: usize,
    pub max_scale_frac: f64,
    pub n_scales: usize,
    /// If true, use both forward- and reverse-anchored DFA boxes (more
    /// statistically robust, ~2x slower). If false, forward boxes only.
    pub reverse_pass: bool,
}

impl Default for HurstConfig {
    fn default() -> Self {
        Self {
            window: 500,
            vol_window: 20,
            min_scale: 8,
            max_scale_frac: 0.25,
            n_scales: 8,
            reverse_pass: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute rolling Hurst exponents per ticker.
///
/// Required columns and dtypes:
///   - `ticker`: String (or any dtype castable to String)
///   - `date`:   Date, Datetime, integer, or ISO-format String
///   - `open`, `high`, `low`, `close`, `volume`: Float64
///
/// The Float64 requirement on numeric columns is a hard contract — callers
/// must cast upstream if their schema uses other dtypes. This keeps the
/// boundary clean and forces type-conversion decisions (precision tradeoffs,
/// integer-vs-float semantics) to live in the caller, which has the context
/// to make them.
pub fn compute_hurst(df: &DataFrame, cfg: HurstConfig) -> PolarsResult<Series> {
    // println!("compute_hurst...");
    let start = Instant::now();
    let n_rows = df.height();
    if n_rows == 0 {
        return Ok(Series::new("hurst".into(), Vec::<f64>::new()));
    }

    let open = f64_to_vec(df, "open")?;
    let high = f64_to_vec(df, "high")?;
    let low = f64_to_vec(df, "low")?;
    let close = f64_to_vec(df, "close")?;
    let volume = f64_to_vec(df, "volume")?;
    let dates = date_col_to_i64(df.column("date")?)?;

    // Group rows by ticker without cloning the DataFrame. We extract a
    // small (n_rows × u32) ticker-id vector, then bucket row indices by
    // ticker-id in a single pass. Peak overhead: ~8 bytes/row (one u32
    // for ids during extraction, one u32 per row in the bucket vectors)
    // vs. the previous approach which materialized a full DataFrame clone
    // and per-partition column copies.
    let group_indices: Vec<Vec<u32>> = group_row_indices_by_ticker(df)?;

    // Precompute the deduplicated, filtered scale set once.
    let scales = build_scales(&cfg);

    // Process tickers in parallel. We do NOT nest a second par_iter inside;
    // tickers are the parallel unit.
    let hurst_per_group: Vec<Vec<(u32, f64)>> = group_indices
        .par_iter()
        .map(|indices| {
            process_ticker(
                indices, &dates, &open, &high, &low, &close, &volume, &cfg, &scales,
            )
        })
        .collect();

    // Scatter results back to original row order.
    let mut hurst_out = vec![f64::NAN; n_rows];
    for group in hurst_per_group {
        for (orig, val) in group {
            hurst_out[orig as usize] = val;
        }
    }

    // println!("compute_hurst completed in {:.2?}", start.elapsed());
    Ok(Series::new("hurst".into(), hurst_out))
}

pub fn with_hurst(mut df: DataFrame, cfg: HurstConfig) -> PolarsResult<DataFrame> {
    let hurst = compute_hurst(&df, cfg)?;
    if hurst.len() != df.height() {
        return Err(PolarsError::ShapeMismatch(
            format!(
                "hurst length {} does not match dataframe height {}",
                hurst.len(),
                df.height()
            )
            .into(),
        ));
    }
    df.with_column(hurst.into_column())?;
    Ok(df)
}

// ---------------------------------------------------------------------------
// Ticker grouping (low-memory replacement for df.clone + partition_by)
// ---------------------------------------------------------------------------

/// Group row indices by ticker without cloning the DataFrame.
///
/// Returns `Vec<Vec<u32>>` where each inner vec contains the original row
/// indices for one ticker. Order of tickers is arbitrary; order of indices
/// within each ticker matches the original row order (sorting by date is
/// done downstream in `process_ticker`).
///
/// Fast path: ticker column is already String. Fallback: cast to String,
/// which handles Categorical, Enum, or any other string-castable dtype
/// without requiring the polars dtype-categorical feature flag.
fn group_row_indices_by_ticker(df: &DataFrame) -> PolarsResult<Vec<Vec<u32>>> {
    use std::collections::HashMap;
    let col = df.column("ticker")?;
    let s = col.as_materialized_series();

    let mut buckets: HashMap<String, Vec<u32>> = HashMap::new();

    // Try the fast path first: column already String.
    if matches!(s.dtype(), DataType::String) {
        let ca = s.str()?;
        for (i, opt) in ca.iter().enumerate() {
            if let Some(t) = opt {
                buckets
                    .entry(t.to_string())
                    .or_insert_with(|| Vec::with_capacity(64))
                    .push(i as u32);
            }
        }
    } else {
        // Fallback: cast whatever it is to String. Works for Categorical,
        // Enum, and any other dtype that supports a String cast. Returns
        // a clear error if the cast itself fails.
        let casted = s.cast(&DataType::String).map_err(|e| {
            PolarsError::ComputeError(
                format!(
                    "ticker column has dtype {:?} which cannot be cast to String: {}",
                    s.dtype(),
                    e
                )
                .into(),
            )
        })?;
        let ca = casted.str()?;
        for (i, opt) in ca.iter().enumerate() {
            if let Some(t) = opt {
                buckets
                    .entry(t.to_string())
                    .or_insert_with(|| Vec::with_capacity(64))
                    .push(i as u32);
            }
        }
    }

    Ok(buckets.into_values().collect())
}

// ---------------------------------------------------------------------------
// Per-ticker worker
// ---------------------------------------------------------------------------

fn process_ticker(
    indices: &[u32],
    dates: &[i64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    cfg: &HurstConfig,
    scales: &[usize],
) -> Vec<(u32, f64)> {
    let mut sorted_idx: Vec<u32> = indices.to_vec();

    // Fast path: if the input is already sorted by date (which it is when
    // called from the standard pipeline that pre-sorts the DataFrame by
    // ["ticker", "date"]), skip the per-ticker sort entirely. The check is
    // a single O(n) scan; the avoided sort is O(n log n). For thousands of
    // tickers × ~14k bars each, this is a meaningful win.
    let already_sorted = sorted_idx
        .windows(2)
        .all(|w| dates[w[0] as usize] <= dates[w[1] as usize]);
    if !already_sorted {
        sorted_idx.sort_unstable_by_key(|&i| dates[i as usize]);
    }

    let len = sorted_idx.len();
    if len < cfg.window {
        return sorted_idx.into_iter().map(|i| (i, f64::NAN)).collect();
    }

    // Build composite signal directly from global arrays via sorted_idx,
    // without materializing per-ticker OHLCV copies. The only length-L
    // allocation here is `signal` itself.
    let signal = build_composite_indexed(&sorted_idx, open, high, low, close, volume, cfg);
    let hurst = rolling_hurst(&signal, cfg, scales);

    sorted_idx.into_iter().zip(hurst).collect()
}

// ---------------------------------------------------------------------------
// Column extraction helpers
// ---------------------------------------------------------------------------

/// Materialize a Float64 column into a Vec<f64>, replacing nulls with NaN.
/// Fast path: no nulls -> single extend_from_slice per chunk.
///
/// Caller contract: the column must be Float64. Upstream is responsible for
/// any dtype conversion. This function returns an error if the column is not
/// Float64 rather than silently coercing — the caller has more context to
/// decide whether casting (and potential precision loss) is appropriate.
fn f64_to_vec(df: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    let ca = df.column(name)?.f64()?;
    let mut out = Vec::with_capacity(ca.len());
    for chunk in ca.downcast_iter() {
        match chunk.validity() {
            None => out.extend_from_slice(chunk.values()),
            Some(validity) => {
                // Word-level scan of the validity bitmap.
                let values = chunk.values();
                let len = values.len();
                let mut i = 0;
                // Bulk-process 64 bits at a time.
                while i + 64 <= len {
                    // Check if any of the next 64 bits are unset.
                    let mut all_valid = true;
                    for j in 0..64 {
                        if !validity.get_bit(i + j) {
                            all_valid = false;
                            break;
                        }
                    }
                    if all_valid {
                        out.extend_from_slice(&values[i..i + 64]);
                    } else {
                        for j in 0..64 {
                            out.push(if validity.get_bit(i + j) {
                                values[i + j]
                            } else {
                                f64::NAN
                            });
                        }
                    }
                    i += 64;
                }
                while i < len {
                    out.push(if validity.get_bit(i) {
                        values[i]
                    } else {
                        f64::NAN
                    });
                    i += 1;
                }
            }
        }
    }
    Ok(out)
}

fn date_col_to_i64(col: &Column) -> PolarsResult<Vec<i64>> {
    let s = col.as_materialized_series();
    let casted = match s.dtype() {
        DataType::Date => s.cast(&DataType::Int32)?.cast(&DataType::Int64)?,
        DataType::Datetime(_, _) => s.cast(&DataType::Int64)?,
        DataType::Int64 => s.clone(),
        DataType::Int32 => s.cast(&DataType::Int64)?,
        DataType::UInt32 => s.cast(&DataType::Int64)?,
        DataType::UInt64 => s.cast(&DataType::Int64)?,
        DataType::String => {
            let parsed = s.str()?.as_date(Some("%Y-%m-%d"), false)?.into_series();
            parsed.cast(&DataType::Int32)?.cast(&DataType::Int64)?
        }
        other => {
            return Err(PolarsError::ComputeError(
                format!("unsupported date dtype: {:?}", other).into(),
            ));
        }
    };
    let ca = casted.i64()?;
    let mut out = Vec::with_capacity(ca.len());
    for chunk in ca.downcast_iter() {
        match chunk.validity() {
            None => out.extend_from_slice(chunk.values()),
            Some(validity) => {
                for (i, &v) in chunk.values().iter().enumerate() {
                    out.push(if validity.get_bit(i) { v } else { i64::MIN });
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Signal construction (fused, indexed, low-memory)
// ---------------------------------------------------------------------------

/// Build the composite signal directly from sorted row indices into the
/// global OHLCV arrays. Fuses:
///   - Garman-Klass per-bar variance + rolling mean
///   - Rolling mean of volume
///   - Final composite z[i] = sign(r) * |r| * sqrt(v/vbar) / sigma
/// into a single forward pass over the ticker.
///
/// Memory: one length-L output vec `z`, plus two length-vol_window ring
/// buffers (typically ~160 bytes total). The previous version allocated
/// 5 length-L OHLCV gathers + length-L sigma + length-L v_mean intermediates.
fn build_composite_indexed(
    sorted_idx: &[u32],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    cfg: &HurstConfig,
) -> Vec<f64> {
    let n = sorted_idx.len();
    let vw = cfg.vol_window;
    let ln2 = 2.0_f64.ln();

    // Ring buffers for per-bar GK and for volume — both of length vol_window.
    let mut gk_ring = vec![f64::NAN; vw];
    let mut vol_ring = vec![f64::NAN; vw];
    let mut gk_sum = 0.0_f64;
    let mut gk_count = 0usize;
    let mut vol_sum = 0.0_f64;
    let mut vol_count = 0usize;

    let mut z = vec![f64::NAN; n];

    // Track previous close (in date order) for log-return computation.
    let mut prev_close: f64 = f64::NAN;

    for i in 0..n {
        let orig = sorted_idx[i] as usize;
        let o = open[orig];
        let h = high[orig];
        let l = low[orig];
        let c = close[orig];
        let v = volume[orig];

        // Per-bar Garman-Klass variance.
        let gk = if o > 0.0 && l > 0.0 && h > 0.0 && c > 0.0 {
            let hl = (h / l).ln();
            let co = (c / o).ln();
            0.5 * hl * hl - (2.0 * ln2 - 1.0) * co * co
        } else {
            f64::NAN
        };

        // Evict values that have fallen out of the rolling window.
        if i >= vw {
            let slot = i % vw;
            let old_gk = gk_ring[slot];
            if old_gk.is_finite() {
                gk_sum -= old_gk;
                gk_count -= 1;
            }
            let old_v = vol_ring[slot];
            if old_v.is_finite() && old_v > 0.0 {
                vol_sum -= old_v;
                vol_count -= 1;
            }
        }
        let slot = i % vw;
        gk_ring[slot] = gk;
        if gk.is_finite() {
            gk_sum += gk;
            gk_count += 1;
        }
        vol_ring[slot] = v;
        if v.is_finite() && v > 0.0 {
            vol_sum += v;
            vol_count += 1;
        }

        // Compute composite z[i] once rolling windows are warm.
        if i + 1 >= vw && i >= 1 && gk_count > 0 && vol_count > 0 {
            let sigma = (gk_sum / gk_count as f64).max(1e-12).sqrt();
            let vbar = vol_sum / vol_count as f64;
            if sigma.is_finite()
                && sigma > 0.0
                && vbar > 0.0
                && v > 0.0
                && prev_close > 0.0
                && c > 0.0
            {
                let r = (c / prev_close).ln();
                let vw_factor = (v / vbar).sqrt();
                z[i] = r.signum() * r.abs() * vw_factor / sigma;
            }
        }

        prev_close = c;
    }

    z
}

// ---------------------------------------------------------------------------
// DFA core
// ---------------------------------------------------------------------------

fn build_scales(cfg: &HurstConfig) -> Vec<usize> {
    let max_scale = ((cfg.window as f64) * cfg.max_scale_frac) as usize;
    let lo = (cfg.min_scale as f64).ln();
    let hi = (max_scale as f64).ln();
    let denom = (cfg.n_scales - 1).max(1) as f64;
    let mut scales: Vec<usize> = (0..cfg.n_scales)
        .map(|i| {
            let t = i as f64 / denom;
            (lo + t * (hi - lo)).exp() as usize
        })
        .collect();
    scales.sort_unstable();
    scales.dedup();
    scales.retain(|&s| s >= 4 && s <= cfg.window / 2);
    scales
}

/// Scratch buffers reused across all window positions for one ticker.
struct DfaScratch {
    profile: Vec<f64>, // integrated profile P[0..w]
    ps: Vec<f64>,      // prefix sums of P:  ps[k] = sum_{j<k} P[j]
    pss: Vec<f64>,     // prefix sums of P*P
    pks: Vec<f64>,     // prefix sums of (j * P[j]) using global index j
    log_s: Vec<f64>,
    log_f: Vec<f64>,
}

impl DfaScratch {
    fn new(w: usize, max_scales: usize) -> Self {
        Self {
            profile: vec![0.0; w],
            ps: vec![0.0; w + 1],
            pss: vec![0.0; w + 1],
            pks: vec![0.0; w + 1],
            log_s: Vec::with_capacity(max_scales),
            log_f: Vec::with_capacity(max_scales),
        }
    }
}

/// DFA1 with prefix-sum acceleration.
///
/// Within each box [a, a+s) the points are (k, P[a+k]) for k = 0..s.
/// The OLS residual sum of squares with linear detrending equals:
///     SS = Syy - (Sxy - Sx*Sy/s)^2 / (Sxx - Sx^2/s)
/// where Sx, Sxx depend only on s (deterministic), and:
///     Sy   = sum_{k<s} P[a+k]
///     Syy  = sum_{k<s} P[a+k]^2
///     Sxy  = sum_{k<s} k * P[a+k]
///
/// Sy and Syy come directly from prefix sums of P and P*P. Sxy is derived from
/// prefix sums of (j * P[j]) and (P[j]) using:
///     Sxy = sum_{j=a..a+s} (j-a) * P[j]
///         = (sum_{j=a..a+s} j*P[j]) - a * (sum_{j=a..a+s} P[j])
fn dfa1_from_profile(
    profile: &[f64],
    ps: &[f64],
    pss: &[f64],
    pks: &[f64],
    scales: &[usize],
    reverse_pass: bool,
    log_s: &mut Vec<f64>,
    log_f: &mut Vec<f64>,
) -> f64 {
    let n = profile.len();
    log_s.clear();
    log_f.clear();

    for &s in scales {
        if s < 4 || s > n / 2 {
            continue;
        }
        let n_boxes = n / s;
        if n_boxes < 2 {
            continue;
        }

        // Deterministic x-statistics (k = 0..s-1).
        let sf = s as f64;
        let sx = sf * (sf - 1.0) * 0.5;
        let sxx = (sf - 1.0) * sf * (2.0 * sf - 1.0) / 6.0;
        let denom_x = sxx - sx * sx / sf;
        if denom_x <= 0.0 {
            continue;
        }

        let mut ss_total = 0.0_f64;
        let mut count_total = 0usize;

        // Forward boxes.
        for b in 0..n_boxes {
            let a = b * s;
            ss_total += box_ss(a, s, ps, pss, pks, sx, denom_x, sf);
            count_total += s;
        }
        // Reverse boxes (anchored at tail) — optional.
        if reverse_pass {
            for b in 0..n_boxes {
                let a = n - (b + 1) * s;
                ss_total += box_ss(a, s, ps, pss, pks, sx, denom_x, sf);
                count_total += s;
            }
        }

        if count_total == 0 {
            continue;
        }
        let f = (ss_total / count_total as f64).sqrt();
        if f > 0.0 {
            log_s.push((s as f64).ln());
            log_f.push(f.ln());
        }
    }

    if log_s.len() < 3 {
        return f64::NAN;
    }
    let m = log_s.len() as f64;
    let sx: f64 = log_s.iter().sum();
    let sy: f64 = log_f.iter().sum();
    let sxy: f64 = log_s.iter().zip(log_f.iter()).map(|(x, y)| x * y).sum();
    let sxx: f64 = log_s.iter().map(|x| x * x).sum();
    let denom = m * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return f64::NAN;
    }
    (m * sxy - sx * sy) / denom
}

#[inline(always)]
fn box_ss(
    a: usize,
    s: usize,
    ps: &[f64],
    pss: &[f64],
    pks: &[f64],
    sx: f64,
    denom_x: f64,
    sf: f64,
) -> f64 {
    // Range [a, a+s) using prefix-sum convention ps[i] = sum_{j<i} P[j].
    let end = a + s;
    let sy = ps[end] - ps[a];
    let syy = pss[end] - pss[a];
    let sky = pks[end] - pks[a]; // sum_{j in box} j * P[j], j is global index
    let sxy = sky - (a as f64) * sy; // shift x = j - a

    // OLS residual SS.
    let num = sxy - sx * sy / sf;
    let ss = syy - sy * sy / sf - num * num / denom_x;
    if ss > 0.0 { ss } else { 0.0 }
}

/// Rolling Hurst over one ticker's composite signal.
fn rolling_hurst(signal: &[f64], cfg: &HurstConfig, scales: &[usize]) -> Vec<f64> {
    let n = signal.len();
    let w = cfg.window;
    let mut out = vec![f64::NAN; n];
    if n < w {
        return out;
    }

    let mut scratch = DfaScratch::new(w, scales.len());

    for i in (w - 1)..n {
        let start = i + 1 - w;
        let win = &signal[start..=i];

        // Combined NaN check + mean computation in a single pass over the
        // window. Early-exit on first non-finite. No O(L) auxiliary state.
        let mut sum = 0.0_f64;
        let mut has_bad = false;
        for &x in win {
            if !x.is_finite() {
                has_bad = true;
                break;
            }
            sum += x;
        }
        if has_bad {
            continue;
        }
        let mean = sum / w as f64;

        // Integrated profile and its three prefix sums, in a single pass.
        scratch.ps[0] = 0.0;
        scratch.pss[0] = 0.0;
        scratch.pks[0] = 0.0;
        let mut acc = 0.0_f64;
        for k in 0..w {
            acc += win[k] - mean;
            scratch.profile[k] = acc;
            let kf = k as f64;
            scratch.ps[k + 1] = scratch.ps[k] + acc;
            scratch.pss[k + 1] = scratch.pss[k] + acc * acc;
            scratch.pks[k + 1] = scratch.pks[k] + kf * acc;
        }

        out[i] = dfa1_from_profile(
            &scratch.profile,
            &scratch.ps,
            &scratch.pss,
            &scratch.pks,
            scales,
            cfg.reverse_pass,
            &mut scratch.log_s,
            &mut scratch.log_f,
        );
    }

    out
}
