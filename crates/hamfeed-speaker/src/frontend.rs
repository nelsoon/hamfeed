//! 80-bin log-mel filterbank frontend with utterance-level CMVN.
//!
//! 16 kHz input, 25 ms symmetric Hamming window, 10 ms shift, 512-point
//! DFT power spectrum (naive O(n²); 512 pts × ~100 frames/s is cheap),
//! 80 HTK-mel triangular filters over 0–8000 Hz, natural log with floor
//! 1e-10, pre-emphasis 0.97, no dither. CMVN (mean AND variance) is the
//! default and only mode: mean-only CMN collapses cross-speaker
//! separation (spike: 0.88) while CMVN passes the gate (0.33).

/// Sample rate the frontend expects (Hz).
pub const SAMPLE_RATE: u32 = 16_000;
/// Window length in samples (25 ms @ 16 kHz).
pub const FRAME_LEN: usize = 400;
/// Frame shift in samples (10 ms @ 16 kHz).
pub const FRAME_SHIFT: usize = 160;
/// Number of mel bins.
pub const NUM_MEL: usize = 80;
/// Mel filterbank edge frequencies (Hz).
pub const MEL_LOW_HZ: f32 = 0.0;
pub const MEL_HIGH_HZ: f32 = 8000.0;
/// Floor applied before the natural log.
pub const LOG_FLOOR: f32 = 1e-10;
/// Floor for the CMVN standard deviation: bins deader than this are
/// constant across the utterance and are set to zero instead of
/// normalized (normalizing would amplify rounding noise).
pub const CMVN_STD_FLOOR: f32 = 1e-10;
/// DFT size (Kaldi/WeSpeaker training convention for 25 ms frames:
/// 400 samples zero-padded with 112 zeros).
pub const NFFT: usize = 512;
/// Number of unique magnitude bins for real input (0..=NFFT/2).
pub const N_BINS: usize = NFFT / 2 + 1;
/// Pre-emphasis coefficient.
pub const PREEMPHASIS: f32 = 0.97;

/// Number of frames for `pcm_len` samples: `0` when shorter than one
/// window, else `1 + floor((pcm_len - 400) / 160)`.
pub fn num_frames(pcm_len: usize) -> usize {
    if pcm_len < FRAME_LEN {
        0
    } else {
        1 + (pcm_len - FRAME_LEN) / FRAME_SHIFT
    }
}

fn hz_to_mel_htk(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Triangular HTK-mel filterbank: `[mel_bin][dft_bin]` (bins 0..=256
/// cover 0–8000 Hz at 31.25 Hz resolution).
fn mel_filterbank() -> Vec<Vec<f32>> {
    let mel_low = hz_to_mel_htk(MEL_LOW_HZ);
    let mel_high = hz_to_mel_htk(MEL_HIGH_HZ);
    let points: Vec<f32> = (0..NUM_MEL + 2)
        .map(|i| {
            let mel = mel_low + (mel_high - mel_low) * i as f32 / (NUM_MEL + 1) as f32;
            mel_to_hz_htk(mel) * NFFT as f32 / SAMPLE_RATE as f32
        })
        .collect();
    (0..NUM_MEL)
        .map(|m| {
            (0..N_BINS)
                .map(|k| {
                    let k = k as f32;
                    let (lo, c, hi) = (points[m], points[m + 1], points[m + 2]);
                    if k < lo || k > hi {
                        0.0
                    } else if k <= c {
                        (k - lo) / (c - lo)
                    } else {
                        (hi - k) / (hi - c)
                    }
                })
                .collect()
        })
        .collect()
}

/// Symmetric Hamming window of length 400.
fn hamming_window() -> Vec<f32> {
    (0..FRAME_LEN)
        .map(|n| {
            0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / (FRAME_LEN - 1) as f32).cos()
        })
        .collect()
}

/// Power spectrum (bins 0..=256) of a windowed 400-sample frame via
/// naive 512-point DFT.
///
/// Twiddle factors and accumulation are f64: with f32 math the twiddle
/// angle error leaks ~1e-9 power into empty bins — above the 1e-10 log
/// floor — so near-empty bins would carry DFT rounding noise instead of
/// sitting at the floor.
fn power_spectrum(frame: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; N_BINS];
    for (k, bin) in out.iter_mut().enumerate() {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (n, &x) in frame.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * n as f64 / NFFT as f64;
            re += x as f64 * angle.cos();
            im -= x as f64 * angle.sin();
        }
        *bin = (re * re + im * im) as f32;
    }
    out
}

/// 80-bin log-mel filterbank with utterance-level CMVN.
///
/// Returns one `[f32; 80]` row per frame; empty vec when input is shorter
/// than one 400-sample window.
pub fn fbank80(pcm: &[i16]) -> Vec<[f32; NUM_MEL]> {
    let n_frames = num_frames(pcm.len());
    if n_frames == 0 {
        return Vec::new();
    }
    // i16 -> float in [-1, 1) + pre-emphasis 0.97.
    let mut sig = Vec::with_capacity(pcm.len());
    let mut prev = 0.0f32;
    for &s in pcm {
        let x = s as f32 / 32768.0;
        sig.push(x - PREEMPHASIS * prev);
        prev = x;
    }
    let window = hamming_window();
    let filters = mel_filterbank();
    let mut feats: Vec<[f32; NUM_MEL]> = Vec::with_capacity(n_frames);
    let mut frame = vec![0.0f32; FRAME_LEN];
    for t in 0..n_frames {
        let start = t * FRAME_SHIFT;
        for n in 0..FRAME_LEN {
            frame[n] = sig[start + n] * window[n];
        }
        let power = power_spectrum(&frame);
        let mut row = [0.0f32; NUM_MEL];
        for (m, filt) in filters.iter().enumerate() {
            let e: f32 = filt.iter().zip(power.iter()).map(|(w, p)| w * p).sum();
            row[m] = e.max(LOG_FLOOR).ln();
        }
        feats.push(row);
    }
    // Utterance-level CMVN. Mean/variance accumulate in f64 so the
    // constant-bin detector below is not fooled by f32 summation error.
    let n_frames_f = feats.len() as f64;
    for m in 0..NUM_MEL {
        let mean: f64 = feats.iter().map(|r| r[m] as f64).sum::<f64>() / n_frames_f;
        let var: f64 = feats
            .iter()
            .map(|r| {
                let d = r[m] as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n_frames_f;
        let std = var.sqrt();
        if std < CMVN_STD_FLOOR as f64 {
            for row in feats.iter_mut() {
                row[m] = 0.0;
            }
        } else {
            let (mean, std) = (mean as f32, std as f32);
            for row in feats.iter_mut() {
                row[m] = (row[m] - mean) / std;
            }
        }
    }
    feats
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean per-frame cosine distance between two feature matrices.
    fn mean_cosine_distance(a: &[[f32; 80]], b: &[[f32; 80]]) -> f32 {
        let n = a.len().min(b.len());
        assert!(n > 0);
        let mut sum = 0.0f32;
        for i in 0..n {
            let (x, y) = (&a[i], &b[i]);
            let dot: f32 = x.iter().zip(y.iter()).map(|(p, q)| p * q).sum();
            let nx: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let ny: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
            sum += 1.0 - dot / (nx * ny).max(1e-12);
        }
        sum / n as f32
    }

    /// Spike recipe: harmonic tones a fifth apart + harmonics, 1.5 s.
    fn synth(f0: f64) -> Vec<i16> {
        (0..24000)
            .map(|i| {
                let t = i as f64 / 16000.0;
                let s = (2.0 * std::f64::consts::PI * f0 * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 2.0 * f0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 3.0 * f0 * t).sin();
                (s * 6000.0) as i16
            })
            .collect()
    }

    #[test]
    fn frames_count() {
        assert_eq!(num_frames(16000), 98); // 1 + floor((16000-400)/160)
        assert_eq!(num_frames(800), 3);
        assert!(fbank80(&[]).is_empty());
        assert!(fbank80(&vec![0i16; 399]).is_empty());
        assert_eq!(fbank80(&vec![0i16; 400]).len(), 1);
    }

    #[test]
    fn frontend_deterministic() {
        let pcm = synth(110.0);
        assert_eq!(fbank80(&pcm), fbank80(&pcm));
    }

    #[test]
    fn cmvn_separates_synth_tones() {
        // CMVN must preserve inter-speaker differences (guards against a
        // degenerate frontend: all-zero/NaN output, or normalization that
        // erases every difference).
        let a = fbank80(&synth(110.0));
        let b = fbank80(&synth(190.0));
        assert_eq!(a.len(), b.len());
        let dist = mean_cosine_distance(&a, &b);
        assert!(
            dist > 0.1,
            "CMVN must preserve pitch/timbre differences, got {dist}"
        );
    }

    #[test]
    fn silence_maps_near_floor_after_cmvn() {
        let fb = fbank80(&vec![0i16; 16000]);
        assert_eq!(fb.len(), 98);
        assert!(
            fb.iter().flatten().all(|v| v.abs() < 1e-3),
            "CMVN silences flat input"
        );
    }

    #[test]
    fn tone_maps_to_expected_mel_region() {
        // 440 Hz is HTK-mel bin ~15.7 and 512-pt DFT bin ~14.1: the
        // windowed DFT + filterbank must peak there (checked pre-CMVN —
        // CMVN equalizes perfectly stationary signals by design).
        let window = hamming_window();
        let filters = mel_filterbank();
        let frame: Vec<f32> = (0..FRAME_LEN)
            .map(|n| {
                (2.0 * std::f64::consts::PI * 440.0 * n as f64 / 16000.0).cos() as f32 * window[n]
            })
            .collect();
        let power = power_spectrum(&frame);
        let dft_peak = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(dft_peak.abs_diff(14) <= 1, "DFT peak at {dft_peak}");
        let logmel: Vec<f32> = filters
            .iter()
            .map(|f| {
                f.iter()
                    .zip(power.iter())
                    .map(|(w, p)| w * p)
                    .sum::<f32>()
                    .max(LOG_FLOOR)
                    .ln()
            })
            .collect();
        let mpeak = logmel
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(mpeak.abs_diff(15) <= 2, "mel peak at {mpeak}");
    }
}
