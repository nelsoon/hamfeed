//! Downward expander for radio silence (007 evolution).
//!
//! The per-bin gate could hush silence deeply or sound natural, never
//! both (ear loop: deep carves the voice robotic, gentle leaks). The
//! expander sidesteps the dilemma: it rides the WHOLE signal level, so
//! open speech passes bit-identical and nothing can sound carved.
//!
//! Opening is two-speed: an instantaneous punch-through for loud onsets
//! (crisp first syllables, ~6 ms) plus a slow sustained confirmation
//! for weak voice. Closing needs a slow mean-level fall (tails fade,
//! stop-gaps bridge). Thresholds are calibrated on the WX channel, not
//! fixtures: post-band silence peaks med -32 dB, speech peaks med -9,
//! so the line sits between with hysteresis. Brief loud transients
//! (squelch crashes) punch through like every gate — they are loud.

use crate::SAMPLE_RATE;

/// Slow mean-level follower (both directions): stationary beds settle
/// at their mean, brief crashes barely move it.
const ENV_TC_S: f32 = 0.1;
/// Instantaneous punch-through: definitely voice (or a crash), open now.
const INST_OPEN_DB: f32 = -18.0;
/// Sustained confirmation: weak voice holding above this opens.
const SLOW_OPEN_DB: f32 = -30.0;
/// Close line with hysteresis: the bed never reaches up here.
const CLOSE_DB: f32 = -32.0;
/// Gain slew: fast open (~3 ms, no zipper: ≤2 %/sample), slow fade.
const GAIN_ATTACK: f32 = 0.02;
const GAIN_RELEASE: f32 = 0.0004;
/// Closed gain (-20 dB): silence goes dark, never mutes (no holes).
const MIN_GAIN: f32 = 0.1;

fn env_coef() -> f32 {
    1.0 - (-1.0 / (SAMPLE_RATE as f32 * ENV_TC_S)).exp()
}

pub struct Expander {
    env: f32,
    gain: f32,
    open: bool,
}

impl Expander {
    pub fn new() -> Self {
        Self {
            env: 0.0,
            gain: MIN_GAIN,
            open: false,
        }
    }

    /// Apply the expanding gain in place (16 kHz float mono, ±1.0).
    /// Sample-exact: no delay, no tail — nothing to flush. Gain never
    /// exceeds 1.0: the stage only ever turns things down.
    pub fn process(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            let a = x.abs();
            self.env += env_coef() * (a - self.env);
            let inst_db = 20.0 * (a + 1e-9).log10();
            let env_db = 20.0 * (self.env + 1e-9).log10();
            if inst_db > INST_OPEN_DB || env_db > SLOW_OPEN_DB {
                self.open = true;
            } else if env_db < CLOSE_DB {
                self.open = false;
            }
            let target = if self.open { 1.0 } else { MIN_GAIN };
            let rate = if target > self.gain {
                GAIN_ATTACK
            } else {
                GAIN_RELEASE
            };
            self.gain += rate * (target - self.gain);
            *x *= self.gain;
        }
    }
}

impl Default for Expander {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    fn mean_abs(v: &[f32]) -> f32 {
        v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32
    }

    fn to_f32(pcm: &[i16]) -> Vec<f32> {
        pcm.iter().map(|s| f32::from(*s) / 32768.0).collect()
    }

    fn att_db(before: &[f32], after: &[f32]) -> f32 {
        20.0 * (mean_abs(after) / mean_abs(before).max(1e-9)).log10()
    }

    #[test]
    fn silence_dives_deep() {
        // Quiet hiss bed (mean -42 dB, under the close line): ≥12 dB down.
        let bed = to_f32(&fixture::noise_floor_ms(2000, 300));
        let mut e = Expander::new();
        let mut out = bed.clone();
        e.process(&mut out);
        assert!(att_db(&bed, &out) < -12.0, "silence not expanded");
    }

    #[test]
    fn never_amplifies() {
        // Structural: gain ≤ 1 always, so a loud bed can pass but never
        // grow — the worst case is unity, never a lift.
        let bed = to_f32(&fixture::noise_floor_ms(2000, 1500));
        let mut e = Expander::new();
        let mut out = bed.clone();
        e.process(&mut out);
        assert!(att_db(&bed, &out) <= 0.5, "expander lifted the bed");
    }

    #[test]
    fn speech_passes_whole() {
        // Voice-band tone at speech level: open, bit-near-identical past
        // the millisecond attack transient.
        let tone = to_f32(&fixture::tone_ms(1000.0, 1000, 9000));
        let mut e = Expander::new();
        let mut out = tone.clone();
        e.process(&mut out);
        assert!(
            att_db(&tone[160..], &out[160..]).abs() < 1.0,
            "speech touched"
        );
    }

    #[test]
    fn onset_not_chopped() {
        // Burst out of silence: instantaneous punch opens in milliseconds —
        // the first 10 ms already carries most of the energy.
        let mut sig = vec![0i16; 16000];
        sig.extend_from_slice(&fixture::tone_ms(1000.0, 500, 9000));
        let buf = to_f32(&sig);
        let mut e = Expander::new();
        let mut out = buf.clone();
        e.process(&mut out);
        let first = mean_abs(&out[16000..16160]);
        let steady = mean_abs(&out[20000..]);
        assert!(
            20.0 * (first / steady).log10() > -6.0,
            "onset chopped: first {first:.5} vs steady {steady:.5}"
        );
    }

    #[test]
    fn tail_fades_smoothly_then_closes() {
        // Burst then a hiss tail (radio is never digitally silent): no
        // 20 ms window may drop hard (no gating edge), yet the late bed
        // sits dark next to speech.
        let mut sig = fixture::tone_ms(1000.0, 500, 9000);
        sig.extend_from_slice(&fixture::noise_floor_ms(800, 300));
        let buf = to_f32(&sig);
        let mut e = Expander::new();
        let mut out = buf.clone();
        e.process(&mut out);
        let start = 8000; // first tail sample
        let mut prev = mean_abs(&out[start..start + 320]);
        for k in 1..6 {
            let w = mean_abs(&out[start + k * 320..start + (k + 1) * 320]);
            let drop = 20.0 * (w / prev.max(1e-9)).log10();
            assert!(drop > -12.0, "hard gate edge at window {k}: {drop:.1} dB");
            prev = w;
        }
        let late = mean_abs(&out[out.len() - 320..]);
        let loud = mean_abs(&out[..8000]);
        assert!(
            20.0 * (late / loud).log10() < -15.0,
            "tail never closed: {late:.5} vs speech {loud:.5}"
        );
    }

    #[test]
    fn empty_is_safe() {
        let mut e = Expander::new();
        let mut buf = Vec::new();
        e.process(&mut buf);
        assert!(buf.is_empty());
    }
}
