//! Split-band de-esser for sibilance (007, ear loop).
//!
//! The remaining "shhh" is quiet voice itself (gap spectrum matches
//! speech at −7 dB), so nothing may touch it broadband. This stage
//! only turns down the 4.5 kHz-and-up band, and only while that band
//! dominates the signal (sibilant) — vowels pass bit-identical.
//! First in the chain (needs full-band sibilance, ahead of the LP).

use crate::SAMPLE_RATE;

/// Sibilance band split.
const HF_FC: f32 = 4500.0;
const Q: f32 = std::f32::consts::FRAC_1_SQRT_2;
/// Fires only when HF dominates (vowel harmonics sit far below this —
/// real voice carries only ~10–30 % of its energy above 4.5 kHz — and
/// clears an absolute floor (quiet hash never triggers).
const RATIO_THRESH: f32 = 0.5;
const ABS_FLOOR: f32 = 0.0056; // −45 dB
/// Deepest HF cut (−6 dB); threshold crossing is continuous by
/// construction (target → 1 as the ratio → threshold).
const MIN_GAIN: f32 = 0.5;
/// Envelope followers: snappy HF attack, stable broadband reference.
const HF_ATTACK: f32 = 0.06;
const HF_RELEASE: f32 = 0.001;
const FULL_ATTACK: f32 = 0.01;
const FULL_RELEASE: f32 = 0.0005;
/// Gain slew: dives in milliseconds, releases over ~30 ms (no flutter).
const GAIN_ATTACK: f32 = 0.05;
const GAIN_RELEASE: f32 = 0.002;

/// Matched low-pass: the HF path is derived as `x - lp(x)`, so the
/// split reconstructs EXACTLY at unity gain (no crossover ripple to
/// color vowels when the stage is open, which is almost always).
struct Lowpass {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Lowpass {
    fn new(freq: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * freq / SAMPLE_RATE as f32;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2.0 * Q);
        let b = (1.0 - c) / 2.0;
        let (a0, a1, a2) = (1.0 + alpha, -2.0 * c, 1.0 - alpha);
        Self {
            b0: b / a0,
            b1: (1.0 - c) / a0,
            b2: b / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

pub struct Deesser {
    lp: Lowpass,
    hf_env: f32,
    full_env: f32,
    gain: f32,
}

impl Deesser {
    pub fn new() -> Self {
        Self {
            lp: Lowpass::new(HF_FC),
            hf_env: 0.0,
            full_env: 0.0,
            gain: 1.0,
        }
    }

    /// Tame sibilance in place (float mono, ±1.0). Sample-exact: no
    /// delay, no tail — nothing to flush. Only ever turns the HF band
    /// down; lows pass untouched always.
    pub fn process(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            let l = self.lp.sample(*x);
            let h = *x - l;
            self.hf_env += (if h.abs() > self.hf_env {
                HF_ATTACK
            } else {
                HF_RELEASE
            }) * (h.abs() - self.hf_env);
            self.full_env += (if x.abs() > self.full_env {
                FULL_ATTACK
            } else {
                FULL_RELEASE
            }) * (x.abs() - self.full_env);
            let mut target = 1.0;
            if self.hf_env > ABS_FLOOR && self.hf_env > RATIO_THRESH * (self.full_env + 1e-9) {
                target = (RATIO_THRESH * self.full_env / (self.hf_env + 1e-9)).clamp(MIN_GAIN, 1.0);
            }
            let rate = if target < self.gain {
                GAIN_ATTACK
            } else {
                GAIN_RELEASE
            };
            self.gain += rate * (target - self.gain);
            *x = l + h * self.gain;
        }
    }
}

impl Default for Deesser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_f32(pcm: &[i16]) -> Vec<f32> {
        pcm.iter().map(|s| f32::from(*s) / 32768.0).collect()
    }

    fn hf_energy(buf: &[f32]) -> f64 {
        // Energy above ~4.5 kHz via first-difference (rises with HF).
        buf.windows(2)
            .map(|w| (w[1] - w[0]) as f64)
            .map(|d| d * d)
            .sum::<f64>()
            / buf.len().max(1) as f64
    }

    fn mean_abs(v: &[f32]) -> f32 {
        v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32
    }

    #[test]
    fn sibilance_ducked_vowels_untouched() {
        // Broadband HF burst (sibilant-like, phase-averaged): HF energy
        // drops clearly. Pure tones at the crossover are NOT used —
        // their phase makes any split read arbitrary.
        let freqs = [4600.0f32, 5300.0, 6100.0, 6900.0, 7500.0];
        let tones: Vec<Vec<i16>> = freqs
            .iter()
            .map(|f| crate::fixture::tone_ms(*f, 1000, 2500))
            .collect();
        let sib: Vec<i16> = (0..tones[0].len())
            .map(|i| tones.iter().map(|t| t[i] as i32).sum::<i32>() / freqs.len() as i32)
            .map(|v| v.clamp(-32768, 32767) as i16)
            .collect();
        let mut buf = to_f32(&sib);
        Deesser::new().process(&mut buf);
        let drop = 10.0 * (hf_energy(&buf) / hf_energy(&to_f32(&sib)).max(1e-12)).log10();
        assert!(
            (3.0..=7.0).contains(&-drop),
            "sibilance trim out of band: {drop:.1} dB"
        );
        // Pure vowel (no HF content at all): bit-near-identical.
        let vowel = to_f32(&crate::fixture::tone_ms(440.0, 1000, 9000));
        let mut vout = vowel.clone();
        Deesser::new().process(&mut vout);
        let kept = 20.0 * (mean_abs(&vout[160..]) / mean_abs(&vowel[160..]).max(1e-9)).log10();
        assert!(kept.abs() < 0.5, "vowel touched: {kept:.1} dB");
    }

    #[test]
    fn quiet_hf_alongside_voice_passes() {
        // Vowel carrying modest HF (ratio under threshold): untouched —
        // selectivity, not just level.
        let a = crate::fixture::tone_ms(440.0, 1000, 9000);
        let b = crate::fixture::tone_ms(6000.0, 1000, 1500);
        let mix: Vec<i16> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x.saturating_add(*y))
            .collect();
        let mut buf = to_f32(&mix);
        Deesser::new().process(&mut buf);
        let kept =
            20.0 * (mean_abs(&buf[160..]) / mean_abs(&to_f32(&mix)[160..]).max(1e-9)).log10();
        assert!(kept.abs() < 1.0, "mixed voice ducked: {kept:.1} dB");
    }

    #[test]
    fn empty_is_safe() {
        let mut d = Deesser::new();
        let mut buf = Vec::new();
        d.process(&mut buf);
        assert!(buf.is_empty());
    }
}
