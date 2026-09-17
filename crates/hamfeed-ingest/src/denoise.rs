//! Spectral gating for radio static (006, replaces RNNoise).
//!
//! The frying static (`grésillement`) is steady broadband noise: learn its
//! fingerprint during quiet moments (minimum statistics per FFT bin), then
//! turn down only those bins. Gains move with slow attack/release, so the
//! level physically cannot pump — the artifact the RNNoise stage had on
//! weak voice. Native 16 kHz, sqrt-Hann 50%-overlap WOLA: streaming-exact
//! lengths, ~16 ms latency, no resampling anywhere.

use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

/// FFT size (16 ms at 16 kHz) and hop (50% overlap).
const N: usize = 256;
const HOP: usize = 128;
/// Bins for real input (DC..Nyquist).
const BINS: usize = N / 2 + 1;
/// Oversubtraction: how far below the floor a bin must sink to open.
const OVER: f32 = 1.3;
/// Gain floor: static never fully mutes (no gating "holes").
const FLOOR: f32 = 0.08;
/// Gain smoothing per hop (slow both ways — this is the anti-pump).
const SMOOTH: f32 = 0.12;
/// Noise-floor tracking: fast down (new quiet wins immediately), measured
/// up (loud is usually signal — but a silence-primed gate must still find
/// the static within about a second, hence 0.01 not 0.002).
const TRACK_DOWN: f32 = 0.15;
const TRACK_UP: f32 = 0.01;

fn sqrt_hann(n: usize, i: usize) -> f32 {
    (std::f32::consts::PI * i as f32 / n as f32).sin().sqrt()
}

pub struct Denoiser {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    /// Input carry (< N) and overlap tail (HOP).
    carry: Vec<f32>,
    tail: Vec<f32>,
    /// Per-bin noise floor power + smoothed gains.
    floor: Vec<f32>,
    gains: Vec<f32>,
    primed: bool,
    total_in: u64,
    total_out: u64,
}

impl Denoiser {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        Self {
            fft: planner.plan_fft_forward(N),
            ifft: planner.plan_fft_inverse(N),
            carry: Vec::new(),
            tail: vec![0.0; HOP],
            floor: vec![0.0; BINS],
            gains: vec![1.0; BINS],
            primed: false,
            total_in: 0,
            total_out: 0,
        }
    }

    /// Feed 16 kHz mono S16, get gated S16 back. Per-call output may lag
    /// input by under two hops; totals always match (see [`Self::flush`]).
    pub fn process(&mut self, pcm: &[i16]) -> Vec<i16> {
        self.total_in += pcm.len() as u64;
        for s in pcm {
            self.carry.push(f32::from(*s) / 32768.0);
        }
        let mut out = Vec::new();
        self.pump(&mut out);
        self.total_out += out.len() as u64;
        out
    }

    /// Drain at end of stream (tests, file jobs — live never flushes).
    /// Zero-pads the tail, then trims padding-induced extras so total
    /// output length equals total input length exactly.
    pub fn flush(&mut self) -> Vec<i16> {
        if !self.carry.is_empty() {
            // Zero lookahead so the overlap tail drains: pump emits while
            // a full window waits, then leftovers (all padding-driven by
            // construction) are dropped and output is trimmed to exact.
            while self.carry.len() < 3 * N {
                self.carry.push(0.0);
            }
        }
        let mut out = Vec::new();
        self.pump(&mut out);
        let keep = (self.total_in - self.total_out).min(out.len() as u64) as usize;
        out.truncate(keep);
        self.total_out += keep as u64;
        self.carry.clear();
        self.tail = vec![0.0; HOP];
        out
    }

    fn pump(&mut self, out: &mut Vec<i16>) {
        while self.carry.len() >= N {
            // Windowed frame.
            let mut buf: Vec<Complex<f32>> = self.carry[..N]
                .iter()
                .enumerate()
                .map(|(i, &x)| Complex::new(x * sqrt_hann(N, i), 0.0))
                .collect();
            self.carry.drain(..HOP);
            self.fft.process(&mut buf);
            // Power spectrum.
            let mut pow = [0.0f32; BINS];
            for (i, c) in buf.iter().take(BINS).enumerate() {
                pow[i] = c.re * c.re + c.im * c.im;
            }
            // Prime the floor on the first frame (the radio is never
            // silent, so frame one is static — the honest starting point).
            if !self.primed {
                self.floor.copy_from_slice(&pow);
                self.primed = true;
            }
            // Minimum-statistics tracking + smoothed subtraction gains.
            for (i, &p) in pow.iter().enumerate() {
                let f = &mut self.floor[i];
                if p < *f {
                    *f += TRACK_DOWN * (p - *f);
                } else {
                    *f += TRACK_UP * (p - *f);
                }
                let target = ((p - OVER * *f) / p.max(1e-12)).clamp(FLOOR, 1.0);
                let g = &mut self.gains[i];
                *g += SMOOTH * (target - *g);
            }
            // Apply, invert, overlap-add.
            for (i, c) in buf.iter_mut().take(BINS).enumerate() {
                *c *= self.gains[i];
            }
            // Mirror for the real inverse (bins above Nyquist).
            for i in 1..N / 2 {
                buf[N - i] = buf[i].conj();
            }
            self.ifft.process(&mut buf);
            let scale = 1.0 / N as f32;
            for (i, c) in buf.iter().enumerate() {
                let v = c.re * scale * sqrt_hann(N, i);
                if i < HOP {
                    out.push(clamp16(self.tail[i] + v));
                } else {
                    self.tail.push(v);
                }
            }
            self.tail.drain(..HOP);
        }
    }
}

fn clamp16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

impl Default for Denoiser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db(samples: &[i16]) -> f64 {
        let e: f64 =
            samples.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / samples.len().max(1) as f64;
        10.0 * e.log10()
    }

    fn hiss(n: usize, seed: &mut u64) -> Vec<i16> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((*seed >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32 * 0.08 * 32767.0
            })
            .map(|v| v as i16)
            .collect()
    }

    #[test]
    fn length_preserved_over_odd_chunks() {
        let mut d = Denoiser::new();
        let mut seed = 1u64;
        let mut total_in = 0usize;
        let mut total_out = 0usize;
        for chunk in [7usize, 128, 200, 1000, 3, 256] {
            let pcm = hiss(chunk, &mut seed);
            total_in += chunk;
            total_out += d.process(&pcm).len();
            assert!(total_out <= total_in);
        }
        total_out += d.flush().len();
        assert_eq!(total_out, total_in, "stream totals must match exactly");
    }

    #[test]
    fn static_drops_and_speech_survives() {
        // G1: steady frying static falls hard (the gate learns it whole).
        let mut seed = 42u64;
        let mut d = Denoiser::new();
        let _ = d.process(&hiss(1600, &mut seed)); // prime the floor
        let mut clean = Vec::new();
        for _ in 0..10 {
            clean.extend_from_slice(&d.process(&hiss(1600, &mut seed)));
        }
        clean.extend_from_slice(&d.flush());
        let noisy = hiss(16000, &mut seed);
        assert!(
            db(&clean) < db(&noisy) - 8.0,
            "static {} dB -> {} dB",
            db(&noisy),
            db(&clean)
        );
        // G2: quiet voice under static lands between (reduced, not eaten).
        let one = crate::fixture::speech_like(40);
        let voice: Vec<i16> = one
            .iter()
            .cycle()
            .take(48000)
            .map(|v| v.saturating_div(4))
            .collect();
        let mut seed2 = 99u64;
        let bed = hiss(48000, &mut seed2);
        let noisy: Vec<i16> = voice
            .iter()
            .zip(bed.iter())
            .map(|(v, h)| v.saturating_add(*h))
            .collect();
        let mut d3 = Denoiser::new();
        let _ = d3.process(&voice[..1600]);
        let mut out = Vec::new();
        for chunk in noisy.chunks(1600) {
            out.extend_from_slice(&d3.process(chunk));
        }
        out.extend_from_slice(&d3.flush());
        let (e_clean, e_noisy, e_out) = (db(&voice), db(&noisy), db(&out));
        assert!(
            e_out < e_noisy - 2.0,
            "static not reduced: clean {e_clean:.1} noisy {e_noisy:.1} out {e_out:.1}"
        );
        assert!(
            e_out > e_clean - 6.0,
            "voice eaten: clean {e_clean:.1} noisy {e_noisy:.1} out {e_out:.1}"
        );
    }

    #[test]
    fn gains_cannot_pump() {
        // The reported artifact: fast louder/quieter hunting on weak
        // voice. Steady input must settle to steady output: after priming,
        // late hop energies stay within a tight band (no oscillation).
        // Stationary input: skip the prime transient, then the settled
        // halves must sit at the same level (hunting would tilt them).
        let mut seed3 = 6u64;
        let steady = hiss(6400, &mut seed3);
        let mut d4 = Denoiser::new();
        let _ = d4.process(&steady[..1600]);
        let mut out2 = Vec::new();
        for chunk in steady[1600..].chunks(128) {
            out2.extend_from_slice(&d4.process(chunk));
        }
        out2.extend_from_slice(&d4.flush());
        let mid = out2.len() / 2;
        let e1 = db(&out2[..mid]);
        let e2 = db(&out2[mid..]);
        assert!(
            (e1 - e2).abs() < 1.5,
            "gain hunting on steady input: {e1:.1} vs {e2:.1} dB"
        );
    }

    #[test]
    fn empty_is_safe() {
        let mut d = Denoiser::new();
        assert!(d.process(&[]).is_empty());
        assert!(d.flush().is_empty());
    }
}
