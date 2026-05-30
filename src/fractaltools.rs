//! Computes Hurst-exponent series for per-ticker OHLCV data.
use polars::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct HurstConfig {
    pub window: usize,
    pub vol_window: usize,
    pub min_scale: usize,
    pub max_scale_frac: f64,
    pub n_scales: usize,
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

/// Computes a Hurst series from OHLCV data, returning one value per input row.
///
/// # Failure
/// Returns an error if required columns are missing or cannot be converted to expected dtypes.
pub fn compute_hurst(df: &DataFrame, cfg: HurstConfig) -> PolarsResult<Series> {
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

    let group_indices = group_row_indices_by_ticker(df)?;

    let scales = build_scales(&cfg);
    let inputs = HurstInputs {
        dates: &dates,
        open: &open,
        high: &high,
        low: &low,
        close: &close,
        volume: &volume,
    };

    let hurst_per_group: Vec<Vec<(u32, f64)>> = group_indices
        .into_par_iter()
        .map(|indices| process_ticker(indices, &inputs, &cfg, &scales))
        .collect();

    let mut hurst_out = vec![f64::NAN; n_rows];
    for group in hurst_per_group {
        for (orig, val) in group {
            hurst_out[orig as usize] = val;
        }
    }

    Ok(Series::new("hurst".into(), hurst_out))
}

/// Computes Hurst exponent values and appends them as a `hurst` column.
///
/// # Failure
/// Returns an error if Hurst computation fails or the produced series length is invalid.
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

fn group_row_indices_by_ticker(df: &DataFrame) -> PolarsResult<Vec<Vec<u32>>> {
    use std::collections::HashMap;
    let col = df.column("ticker")?;
    let s = col.as_materialized_series();

    let mut buckets: HashMap<String, Vec<u32>> = HashMap::new();

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

struct HurstInputs<'a> {
    dates: &'a [i64],
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
}

fn process_ticker(
    mut sorted_idx: Vec<u32>,
    inputs: &HurstInputs<'_>,
    cfg: &HurstConfig,
    scales: &[usize],
) -> Vec<(u32, f64)> {
    let already_sorted = sorted_idx
        .windows(2)
        .all(|w| inputs.dates[w[0] as usize] <= inputs.dates[w[1] as usize]);
    if !already_sorted {
        sorted_idx.sort_unstable_by_key(|&i| inputs.dates[i as usize]);
    }

    let len = sorted_idx.len();
    if len < cfg.window {
        return sorted_idx.into_iter().map(|i| (i, f64::NAN)).collect();
    }

    let signal = build_composite_indexed(&sorted_idx, inputs, cfg);
    let hurst = rolling_hurst(&signal, cfg, scales);

    sorted_idx.into_iter().zip(hurst).collect()
}

fn f64_to_vec(df: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    let ca = df.column(name)?.f64()?;
    let mut out = Vec::with_capacity(ca.len());
    for chunk in ca.downcast_iter() {
        match chunk.validity() {
            None => out.extend_from_slice(chunk.values()),
            Some(validity) => {
                let values = chunk.values();
                let len = values.len();
                let mut i = 0;
                while i + 64 <= len {
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

fn build_composite_indexed(
    sorted_idx: &[u32],
    inputs: &HurstInputs<'_>,
    cfg: &HurstConfig,
) -> Vec<f64> {
    let n = sorted_idx.len();
    let vw = cfg.vol_window;
    let ln2 = 2.0_f64.ln();

    let mut gk_ring = vec![f64::NAN; vw];
    let mut vol_ring = vec![f64::NAN; vw];
    let mut gk_sum = 0.0_f64;
    let mut gk_count = 0usize;
    let mut vol_sum = 0.0_f64;
    let mut vol_count = 0usize;

    let mut z = vec![f64::NAN; n];

    let mut prev_close: f64 = f64::NAN;

    for (i, &idx) in sorted_idx.iter().enumerate() {
        let orig = idx as usize;
        let o = inputs.open[orig];
        let h = inputs.high[orig];
        let l = inputs.low[orig];
        let c = inputs.close[orig];
        let v = inputs.volume[orig];

        let gk = if o > 0.0 && l > 0.0 && h > 0.0 && c > 0.0 {
            let hl = (h / l).ln();
            let co = (c / o).ln();
            0.5 * hl * hl - (2.0 * ln2 - 1.0) * co * co
        } else {
            f64::NAN
        };

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

fn dfa1_from_profile(scratch: &mut DfaScratch, scales: &[usize], reverse_pass: bool) -> f64 {
    let DfaScratch {
        profile,
        ps,
        pss,
        pks,
        log_s,
        log_f,
    } = scratch;
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

        let sf = s as f64;
        let sx = sf * (sf - 1.0) * 0.5;
        let sxx = (sf - 1.0) * sf * (2.0 * sf - 1.0) / 6.0;
        let denom_x = sxx - sx * sx / sf;
        if denom_x <= 0.0 {
            continue;
        }
        let box_stats = BoxStats {
            ps,
            pss,
            pks,
            sx,
            denom_x,
            sf,
        };

        let mut ss_total = 0.0_f64;
        let mut count_total = 0usize;

        for b in 0..n_boxes {
            let a = b * s;
            ss_total += box_ss(a, s, &box_stats);
            count_total += s;
        }
        if reverse_pass {
            for b in 0..n_boxes {
                let a = n - (b + 1) * s;
                ss_total += box_ss(a, s, &box_stats);
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

struct BoxStats<'a> {
    ps: &'a [f64],
    pss: &'a [f64],
    pks: &'a [f64],
    sx: f64,
    denom_x: f64,
    sf: f64,
}

#[inline(always)]
fn box_ss(a: usize, s: usize, stats: &BoxStats<'_>) -> f64 {
    let end = a + s;
    let sy = stats.ps[end] - stats.ps[a];
    let syy = stats.pss[end] - stats.pss[a];
    let sky = stats.pks[end] - stats.pks[a]; // sum_{j in box} j * P[j], j is global index
    let sxy = sky - (a as f64) * sy; // shift x = j - a

    let num = sxy - stats.sx * sy / stats.sf;
    let ss = syy - sy * sy / stats.sf - num * num / stats.denom_x;
    if ss > 0.0 { ss } else { 0.0 }
}

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

        scratch.ps[0] = 0.0;
        scratch.pss[0] = 0.0;
        scratch.pks[0] = 0.0;
        let mut acc = 0.0_f64;
        for (k, &x) in win.iter().enumerate() {
            acc += x - mean;
            scratch.profile[k] = acc;
            let kf = k as f64;
            scratch.ps[k + 1] = scratch.ps[k] + acc;
            scratch.pss[k + 1] = scratch.pss[k] + acc * acc;
            scratch.pks[k + 1] = scratch.pks[k] + kf * acc;
        }

        out[i] = dfa1_from_profile(&mut scratch, scales, cfg.reverse_pass);
    }

    out
}
