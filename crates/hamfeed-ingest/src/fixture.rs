//! Synthetic PCM helpers (test-only shapes, checked in per T5).
//!
//! Produces deterministic 16 kHz mono S16 streams: silence, tones, noise
//! floor, and syllable-like bursts. Used by ingest tests now and by the
//! simulated end-to-loop (T12) later.

use crate::SAMPLE_RATE;

/// `secs` seconds of digital silence.
pub fn silence_secs(secs: u64) -> Vec<i16> {
    vec![0; secs as usize * SAMPLE_RATE as usize]
}

/// `ms` milliseconds of digital silence.
pub fn silence_ms(ms: u64) -> Vec<i16> {
    vec![0; ms as usize * SAMPLE_RATE as usize / 1000]
}

/// Two simultaneous steady tones (dual-tone repeater beeps).
pub fn dual_tone_ms(f1_hz: f32, f2_hz: f32, ms: u64, amplitude: i16) -> Vec<i16> {
    let n = ms as usize * SAMPLE_RATE as usize / 1000;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        let v = (2.0 * std::f32::consts::PI * f1_hz * t).sin()
            + (2.0 * std::f32::consts::PI * f2_hz * t).sin();
        out.push((v * amplitude as f32 / 2.0) as i16);
    }
    out
}

/// Steady tone (loud, clearly above VAD threshold).
pub fn tone_ms(freq_hz: f32, ms: u64, amplitude: i16) -> Vec<i16> {
    let n = ms as usize * SAMPLE_RATE as usize / 1000;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        let v = (2.0 * std::f32::consts::PI * freq_hz * t).sin();
        out.push((v * amplitude as f32) as i16);
    }
    out
}

/// Low-level hiss standing in for a simplex noise floor (below threshold).
pub fn noise_floor_ms(ms: u64, amplitude: i16) -> Vec<i16> {
    // Deterministic xorshift so fixtures are stable across runs.
    let n = ms as usize * SAMPLE_RATE as usize / 1000;
    let mut out = Vec::with_capacity(n);
    let mut x: u32 = 0x1234_5678;
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let v = (x >> 9) as i16;
        out.push(v % (amplitude + 1));
    }
    out
}

/// Speech-like: `syllables` 150 ms tone bursts separated by 80 ms gaps.
pub fn speech_like(syllables: usize) -> Vec<i16> {
    let mut out = Vec::new();
    for s in 0..syllables {
        let freq = 180.0 + (s as f32 % 4.0) * 60.0;
        out.extend(tone_ms(freq, 150, 9_000));
        out.extend(silence_ms(80));
    }
    out
}

/// Split `pcm` into chunks of `len` samples (last chunk may be short).
pub fn chunks(pcm: &[i16], len: usize) -> Vec<Vec<i16>> {
    pcm.chunks(len).map(|c| c.to_vec()).collect()
}
