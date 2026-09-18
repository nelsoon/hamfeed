//! Voice clarity chain for FM speech (007, sub-stage of 006).
//!
//! Narrow FM voice lives in ~300–3400 Hz. This stage keeps only that:
//! a dual high-pass (~250 Hz corner) kills CTCSS/hum/rumble, a dual
//! low-pass (~3.6 kHz corner) cuts hiss above the voice, then a
//! downward expander rides silence down 20 dB — whole-signal, so
//! speech passes bit-identical and nothing can sound carved or
//! robotic. A split-band de-esser rounds sibilance first (HF-only
//! gain, vowels untouched). A slow AGC rides speech level (simplex
//! fading) without ever hunting, and a soft limiter caps peaks before
//! Opus/live/STT. De-esser, expander sit before AGC so silence stays
//! silent. Native 16 kHz, streaming, no new deps.

use std::f32::consts::PI;

use crate::{Deesser, Expander, SAMPLE_RATE};

/// -3 dB near 250 Hz voice edge; -24 dB at 100 Hz hum.
/// -3 dB near 3.6 kHz voice edge; -12 dB at 6 kHz hiss.
const HP_FC: f32 = 200.0;
const LP_FC: f32 = 4500.0;
const Q: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// AGC: ~0.5 s RMS follower, speech target, ±12 dB ride, per-sample
/// smoothing (the follower is the slow part — gain itself never jumps).
const TARGET_RMS: f32 = 0.1;
const MAX_GAIN: f32 = 4.0;
const MIN_GAIN: f32 = 0.25;
const GAIN_SMOOTH: f32 = 0.005;
/// Post-expander block RMS below this freezes adaptation: quiet stays
/// quiet. Midpoint between the quietest test voice (0.06) and the
/// loudest fixture hiss (0.026) — the real weak-voice line lives with
/// the VAD, this only stops the AGC from riding pure beds up.
const SILENCE_RMS: f32 = 0.035;
/// Soft-limiter knee; output asymptotically approaches full scale.
const LIM_T: f32 = 0.89;

fn env_coef() -> f32 {
    1.0 - (-1.0 / (SAMPLE_RATE as f32 * 0.5)).exp()
}

/// 2nd-order RBJ biquad, transposed direct-form II.
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn highpass(freq: f32) -> Self {
        let w0 = 2.0 * PI * freq / SAMPLE_RATE as f32;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2.0 * Q);
        let b = (1.0 + c) / 2.0;
        Self::norm(b, -(1.0 + c), b, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    fn lowpass(freq: f32) -> Self {
        let w0 = 2.0 * PI * freq / SAMPLE_RATE as f32;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2.0 * Q);
        let b = (1.0 - c) / 2.0;
        Self::norm(b, 1.0 - c, b, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    fn norm(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
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

fn to16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

fn limit(x: f32) -> f32 {
    let a = x.abs();
    if a <= LIM_T {
        x
    } else {
        x.signum() * (LIM_T + (1.0 - LIM_T) * (1.0 - (-(a - LIM_T) / (1.0 - LIM_T)).exp()))
    }
}

pub struct VoiceClarity {
    de: Deesser,
    hp1: Biquad,
    hp2: Biquad,
    lp1: Biquad,
    lp2: Biquad,
    exp: Expander,
    env: f32,
    gain: f32,
    total_in: u64,
    total_out: u64,
}

impl VoiceClarity {
    pub fn new() -> Self {
        Self {
            de: Deesser::new(),
            hp1: Biquad::highpass(HP_FC),
            hp2: Biquad::highpass(HP_FC),
            lp1: Biquad::lowpass(LP_FC),
            lp2: Biquad::lowpass(LP_FC),
            exp: Expander::new(),
            env: TARGET_RMS * TARGET_RMS,
            gain: 1.0,
            total_in: 0,
            total_out: 0,
        }
    }

    /// Feed 16 kHz mono S16, get enhanced S16 back. Every stage is
    /// sample-exact, so output length always equals input length;
    /// [`Self::flush`] is a no-op kept for API stability.
    pub fn process(&mut self, pcm: &[i16]) -> Vec<i16> {
        self.total_in += pcm.len() as u64;
        let fwd = self.forward(pcm);
        let out = self.finish(&fwd);
        self.total_out += out.len() as u64;
        out
    }

    /// No-op: no stage holds a tail (live never flushes; file jobs may).
    pub fn flush(&mut self) -> Vec<i16> {
        debug_assert_eq!(self.total_in, self.total_out);
        Vec::new()
    }

    /// De-esser (full-band sibilance, sample-exact) + band filters +
    /// downward expander. No per-bin gate anywhere on this chain (ear
    /// loop: even the gentle preset warbles audibly) — sibilance is
    /// rounded, silence is ridden, neither is carved.
    fn forward(&mut self, pcm: &[i16]) -> Vec<i16> {
        let mut f: Vec<f32> = pcm.iter().map(|s| f32::from(*s) / 32768.0).collect();
        self.de.process(&mut f);
        for v in f.iter_mut() {
            *v = self
                .lp2
                .sample(self.lp1.sample(self.hp2.sample(self.hp1.sample(*v))));
        }
        self.exp.process(&mut f);
        f.iter().map(|v| to16(*v)).collect()
    }

    /// Slow AGC (frozen on silence) + soft limiter, sample-exact.
    fn finish(&mut self, gated: &[i16]) -> Vec<i16> {
        if gated.is_empty() {
            return Vec::new();
        }
        let rms = (gated
            .iter()
            .map(|s| f32::from(*s) / 32768.0)
            .map(|x| x * x)
            .sum::<f32>()
            / gated.len() as f32)
            .sqrt();
        let adapt = rms >= SILENCE_RMS;
        let mut out = Vec::with_capacity(gated.len());
        for s in gated {
            let x = f32::from(*s) / 32768.0;
            if adapt {
                self.env += env_coef() * (x * x - self.env);
                let target = (TARGET_RMS / (self.env + 1e-12).sqrt()).clamp(MIN_GAIN, MAX_GAIN);
                self.gain += GAIN_SMOOTH * (target - self.gain);
            }
            out.push(to16(limit(x * self.gain)));
        }
        out
    }
}

impl Default for VoiceClarity {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[i16]) -> f64 {
        (samples.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / samples.len().max(1) as f64)
            .sqrt()
    }

    fn att_db(before: &[i16], after: &[i16]) -> f64 {
        20.0 * (rms(after) / rms(before).max(1e-9)).log10()
    }

    /// Steady-state response of one biquad stage (skips the transient).
    fn stage_db(mut b: Biquad, freq: f32) -> f64 {
        let tone = crate::fixture::tone_ms(freq, 3000, 20_000);
        let mut ys = Vec::with_capacity(tone.len());
        for s in &tone {
            ys.push(to16(b.sample(f32::from(*s) / 32768.0)));
        }
        att_db(&tone[16000..], &ys[16000..])
    }

    /// Run the full chain over `pcm` in live-like chunks; prime on hiss
    /// first (frame one on a real channel is static, never speech).
    fn chain(pcm: &[i16]) -> (Vec<i16>, VoiceClarity) {
        let mut v = VoiceClarity::new();
        let prime = crate::fixture::noise_floor_ms(200, 2000);
        let _ = v.process(&prime);
        let mut out = Vec::new();
        for chunk in pcm.chunks(1600) {
            out.extend_from_slice(&v.process(chunk));
        }
        out.extend_from_slice(&v.flush());
        (out, v)
    }

    #[test]
    fn hp_stage_kills_lows_keeps_voice() {
        assert!(stage_db(Biquad::highpass(HP_FC), 100.0) < -10.0);
        assert!(stage_db(Biquad::highpass(HP_FC), 1000.0).abs() < 1.0);
    }

    #[test]
    fn lp_stage_kills_highs_keeps_voice() {
        assert!(stage_db(Biquad::lowpass(LP_FC), 6000.0) < -5.0);
        assert!(stage_db(Biquad::lowpass(LP_FC), 1000.0).abs() < 1.0);
    }

    #[test]
    fn chain_kills_ctcss_band() {
        // G1: hum / low CTCSS representative drops ≥20 dB end to end.
        let tone = crate::fixture::tone_ms(100.0, 2000, 20_000);
        let (out, _) = chain(&tone);
        assert!(att_db(&tone, &out) < -20.0, "low rumble survives");
    }

    #[test]
    fn chain_kills_hiss_band() {
        // G2: frying-static highs drop ≥10 dB end to end.
        let tone = crate::fixture::tone_ms(6000.0, 2000, 8000);
        let (out, _) = chain(&tone);
        assert!(att_db(&tone, &out) < -10.0, "hiss band survives");
    }

    #[test]
    fn chain_levels_voice_and_freezes_on_quiet() {
        // G3: 10 dB-apart voice evens out; the hiss bed after loud
        // voice is not ridden up (gain frozen on quiet). Bursts with
        // gaps (like speech) so the gate floor keeps re-learning down —
        // a continuous stepped tone is not speech and would pin it.
        fn bursts(amp: i16) -> Vec<i16> {
            let mut v = Vec::new();
            for _ in 0..13 {
                v.extend(crate::fixture::tone_ms(440.0, 150, amp));
                v.extend(crate::fixture::silence_ms(80));
            }
            v
        }
        let loud = bursts(9000);
        let quiet = bursts(2845);
        let bed = crate::fixture::noise_floor_ms(2000, 1500);
        let mut v = VoiceClarity::new();
        let prime = crate::fixture::noise_floor_ms(200, 2000);
        let _ = v.process(&prime);
        let mut out_loud = Vec::new();
        for chunk in loud.chunks(1600) {
            out_loud.extend_from_slice(&v.process(chunk));
        }
        let mut out_quiet = Vec::new();
        for chunk in quiet.chunks(1600) {
            out_quiet.extend_from_slice(&v.process(chunk));
        }
        let mut out_bed = Vec::new();
        for chunk in bed.chunks(1600) {
            out_bed.extend_from_slice(&v.process(chunk));
        }
        out_bed.extend_from_slice(&v.flush());
        let tail = |o: &Vec<i16>| rms(&o[o.len() - 16000..]);
        let (rl, rq, rb) = (tail(&out_loud), tail(&out_quiet), tail(&out_bed));
        let db = |e: f64| 20.0 * e.log10();
        assert!(
            (db(rl) - db(rq)).abs() < 3.0,
            "levels not evened: loud {rl:.0} quiet {rq:.0}"
        );
        assert!(
            (db(rl) - db(0.1 * 32768.0)).abs() < 3.0,
            "not riding to target: loud {rl:.0}"
        );
        assert!(
            db(rb) < db(rl) - 10.0,
            "quiet lifted after speech: bed {rb:.0} vs voice {rl:.0}"
        );
    }

    #[test]
    fn hot_peak_never_clips() {
        // G4: a hot transient (fresh high gain) still cannot clip.
        let hot = crate::fixture::tone_ms(440.0, 500, 30_000);
        let (out, _) = chain(&hot);
        let peak = out.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak < i16::MAX, "clipped at {peak}");
    }

    #[test]
    fn length_exact_and_empty_safe() {
        let mut v = VoiceClarity::new();
        let tone = crate::fixture::tone_ms(440.0, 111, 9000);
        let mut n = 0usize;
        for chunk in tone.chunks(7) {
            n += v.process(chunk).len();
            assert!(n <= tone.len());
        }
        n += v.flush().len();
        assert_eq!(n, tone.len());
        assert!(v.process(&[]).is_empty());
        assert!(v.flush().is_empty());
    }
}
