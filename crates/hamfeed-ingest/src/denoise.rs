//! RNNoise suppression for 16 kHz ingest (006).
//!
//! RNNoise eats 10 ms frames at 48 kHz, so this module upsamples 16→48
//! (Hermite interpolation, exact 3×), denoises in 480-sample frames, low-
//! passes (2nd-order Butterworth @8 kHz) and decimates back to 16 kHz.
//! Everything is streaming-stateful across [`Denoiser::process`] calls:
//! total output length always equals total input length (no drift, only a
//! fixed ~30 ms pipeline latency). Partial trailing samples wait in the
//! carry buffers for the next call.

use nnnoiseless::DenoiseState;

/// RNNoise frame: 10 ms at 48 kHz.
const FRAME48: usize = 480;
/// 16 kHz chunk feeding one 48 kHz frame.
const CHUNK16: usize = 160;

/// 2nd-order Butterworth lowpass @8 kHz, fs=48 kHz (RBJ cookbook).
/// Keeps decimation honest: folds nothing above 8 kHz back into voice.
const LP_B: [f32; 3] = [0.20657, 0.41314, 0.20657];
const LP_A: [f32; 3] = [1.0, -0.36953, 0.19582];

fn clamp16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// Hermite (C1-smooth) interpolation between x0 (t=0) and x1 (t=1) with
/// neighbor slopes from x_1 and x2. t in thirds gives the two inserted
/// samples for exact 3× upsampling.
fn hermite(x_1: f32, x0: f32, x1: f32, x2: f32, t: f32) -> f32 {
    let m0 = 0.5 * (x1 - x_1);
    let m1 = 0.5 * (x2 - x0);
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * x0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * x1
        + (t3 - t2) * m1
}

pub struct Denoiser {
    dn: Box<DenoiseState<'static>>,
    /// Last 16 kHz sample (Hermite left neighbor across calls).
    prev16: f32,
    /// 16 kHz carry (< CHUNK16) and 48 kHz carry (< FRAME48).
    carry16: Vec<f32>,
    carry48: Vec<f32>,
    /// Butterworth IIR state (x_1, x_2, y_1, y_2).
    lp: [f32; 4],
    /// Stream totals: flush trims padding-induced extras so output
    /// length always equals input length.
    total_in: u64,
    total_out: u64,
}

impl Denoiser {
    pub fn new() -> Self {
        Self {
            dn: DenoiseState::new(),
            prev16: 0.0,
            carry16: Vec::new(),
            carry48: Vec::new(),
            lp: [0.0; 4],
            total_in: 0,
            total_out: 0,
        }
    }

    /// Feed 16 kHz mono S16, get denoised S16 back. Per-call output may
    /// lag input by under two frames; totals always match (see [`Self::flush`]).
    pub fn process(&mut self, pcm: &[i16]) -> Vec<i16> {
        self.total_in += pcm.len() as u64;
        for s in pcm {
            self.carry16.push(f32::from(*s) / 32768.0);
        }
        let mut out = Vec::new();
        self.pump(&mut out);
        self.total_out += out.len() as u64;
        out
    }

    /// Drain the carries at end of stream (tests, file jobs — live never
    /// flushes). Zero-pads the tail, then trims padding-induced extras so
    /// total output length equals total input length exactly.
    pub fn flush(&mut self) -> Vec<i16> {
        while !self.carry16.len().is_multiple_of(CHUNK16) {
            self.carry16.push(0.0);
        }
        while !self.carry48.len().is_multiple_of(FRAME48) {
            self.carry48.push(0.0);
        }
        let mut out = Vec::new();
        self.pump(&mut out);
        let keep = (self.total_in - self.total_out).min(out.len() as u64) as usize;
        out.truncate(keep);
        self.total_out += keep as u64;
        self.carry16.clear();
        self.carry48.clear();
        out
    }

    /// Upsample/denoise/decimate whatever full chunks wait in the carries.
    fn pump(&mut self, out: &mut Vec<i16>) {
        while self.carry16.len() >= CHUNK16 {
            // Upsample one 160-sample chunk to exactly 480. Chunk edges
            // use flat slopes when the neighbor hasn't arrived yet (one
            // sample per chunk, inaudible); the streaming left neighbor
            // stays exact via prev16.
            let mut up = Vec::with_capacity(FRAME48);
            let c = &self.carry16;
            for i in 0..CHUNK16 {
                let x_1 = if i == 0 { self.prev16 } else { c[i - 1] };
                let x0 = c[i];
                let x1 = c.get(i + 1).copied().unwrap_or(x0);
                let x2 = c.get(i + 2).copied().unwrap_or(x1);
                up.push(x0);
                up.push(hermite(x_1, x0, x1, x2, 1.0 / 3.0));
                up.push(hermite(x_1, x0, x1, x2, 2.0 / 3.0));
            }
            self.prev16 = c[CHUNK16 - 1];
            self.carry16.drain(..CHUNK16);
            self.carry48.extend_from_slice(&up);
            // Denoise full 480-sample frames.
            while self.carry48.len() >= FRAME48 {
                let mut clean = [0.0f32; FRAME48];
                self.dn.process_frame(&mut clean, &self.carry48[..FRAME48]);
                self.carry48.drain(..FRAME48);
                // Lowpass + decimate ×3 back to 160 samples.
                for (i, &x) in clean.iter().enumerate() {
                    let y = LP_B[0] * x + LP_B[1] * self.lp[0] + LP_B[2] * self.lp[1]
                        - LP_A[1] * self.lp[2]
                        - LP_A[2] * self.lp[3];
                    self.lp[1] = self.lp[0];
                    self.lp[0] = x;
                    self.lp[3] = self.lp[2];
                    self.lp[2] = y;
                    if i % 3 == 0 {
                        out.push(clamp16(y));
                    }
                }
            }
        }
    }
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
        // Deterministic white noise at FM-hiss level.
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
        // Streaming totals match exactly, whatever the splits; per-call
        // output never runs ahead of input (bounded lag, no drift).
        let mut d = Denoiser::new();
        let mut seed = 1u64;
        let mut total_in = 0usize;
        let mut total_out = 0usize;
        for chunk in [7usize, 160, 161, 1000, 3, 480] {
            let pcm = hiss(chunk, &mut seed);
            total_in += chunk;
            total_out += d.process(&pcm).len();
            assert!(total_out <= total_in);
        }
        total_out += d.flush().len();
        assert_eq!(total_out, total_in, "stream totals must match exactly");
    }

    #[test]
    fn hiss_drops_and_speech_survives() {
        // G1: pure hiss falls (RNNoise is VAD-gated and conservative on
        // noise-only input — direct 48 kHz measurement says ~6 dB, so the
        // bar sits there, not at an imagined 10).
        let mut seed = 42u64;
        let mut d = Denoiser::new();
        let _ = d.process(&hiss(1600, &mut seed)); // prime the filters
        let mut clean = Vec::new();
        for _ in 0..10 {
            clean.extend_from_slice(&d.process(&hiss(1600, &mut seed)));
        }
        let noisy = hiss(16000, &mut seed);
        assert!(
            db(&clean) < db(&noisy) - 5.0,
            "hiss {} dB -> {} dB",
            db(&noisy),
            db(&clean)
        );
        // G1b (the use case): quiet voice under full hiss. Output must
        // land between: hiss audibly reduced, voice not eaten.
        let one = crate::fixture::speech_like(40);
        let voice: Vec<i16> = one
            .iter()
            .cycle()
            .take(9600)
            .map(|v| v.saturating_div(4))
            .collect();
        let mut seed2 = 99u64;
        let bed = hiss(9600, &mut seed2);
        let noisy: Vec<i16> = voice
            .iter()
            .zip(bed.iter())
            .map(|(v, h)| v.saturating_add(*h))
            .collect();
        let mut d3 = Denoiser::new();
        let _ = d3.process(&voice[..1600]); // prime on speech
        let mut out = Vec::new();
        for chunk in noisy.chunks(1600) {
            out.extend_from_slice(&d3.process(chunk));
        }
        out.extend_from_slice(&d3.flush());
        let (e_clean, e_noisy, e_out) = (db(&voice), db(&noisy), db(&out));
        assert!(
            e_out < e_noisy - 2.0,
            "hiss not reduced: clean {e_clean:.1} noisy {e_noisy:.1} out {e_out:.1}"
        );
        assert!(
            e_out > e_clean - 6.0,
            "voice eaten: clean {e_clean:.1} noisy {e_noisy:.1} out {e_out:.1}"
        );
        // G2: speech-like fixture stays within 3 dB (no eaten voice).
        let mut d2 = Denoiser::new();
        let one = crate::fixture::speech_like(40);
        let voice: Vec<i16> = one.iter().cycle().take(6400).copied().collect();
        let _ = d2.process(&voice[..1600]);
        let mut kept = Vec::new();
        for chunk in voice.chunks(1600) {
            kept.extend_from_slice(&d2.process(chunk));
        }
        kept.extend_from_slice(&d2.flush());
        assert!(
            (db(&kept) - db(&voice)).abs() < 3.0,
            "voice {} dB -> {} dB",
            db(&voice),
            db(&kept)
        );
    }

    #[test]
    fn empty_is_safe() {
        let mut d = Denoiser::new();
        assert!(d.process(&[]).is_empty());
    }
}
