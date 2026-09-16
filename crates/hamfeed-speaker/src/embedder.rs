//! ONNX speaker-embedding session wrapper (T7, `voice` feature only).
//!
//! [`Embedder::open`] loads the wespeaker model pinned by the 2026-09-13
//! diarization spike. A missing or unloadable model file is an `Err` — the
//! caller treats that as voice-disabled and carries on without voiceprints.
//! There is no "disabled" [`Embedder`] value: absence of an `Embedder` ==
//! disabled, so [`Embedder::is_enabled`] always returns `true` (it exists
//! for call-site readability).

use std::sync::Mutex;

use ort::session::Session;

/// Pinned download URL for the embedding model (T12 helper uses it).
pub const MODEL_URL: &str = "https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/6a61a1833ff2583aabeba044f5c8221f00b67ceb/onnx/model.onnx";
/// SHA-256 of the model file at the pinned revision (T12 verifies it).
pub const MODEL_SHA256: &str = "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb";
/// Expected file size in bytes (26_535_549 — the packaging is smaller than
/// the 90–110 MB brief estimate, but I/O and behavior check out).
pub const MODEL_BYTES: u64 = 26_535_549;

/// Speaker-embedding extractor: 16 kHz PCM in, L2-normalized 256-d vector out.
pub struct Embedder {
    session: Mutex<Session>,
    /// Embedding dimension ([`crate::EXPECTED_DIM`]).
    pub dim: usize,
}

impl Embedder {
    /// Load `path` as an ONNX session. `Err` when the file is missing or
    /// unloadable — the caller disables voice in that case.
    ///
    /// Adaptation note: `SessionBuilder::commit_from_file` needs ort's
    /// `std` feature, which this crate deliberately omits (pure-Rust TLS
    /// via `tls-rustls`, no OpenSSL dev headers); the bytes are read with
    /// `std::fs` and committed via `commit_from_memory` instead.
    pub fn open(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let session = Session::builder()?.commit_from_memory(&bytes)?;
        Ok(Self {
            session: Mutex::new(session),
            dim: crate::EXPECTED_DIM,
        })
    }

    /// Embed one utterance: frontend fbank → `[1, frames, 80]` tensor →
    /// L2-normalized embedding of length [`Embedder::dim`].
    pub fn embed(&self, pcm: &[i16]) -> anyhow::Result<Vec<f32>> {
        let fb = crate::frontend::fbank80(pcm);
        if fb.is_empty() {
            anyhow::bail!("embed needs at least one 400-sample frame");
        }
        let frames = fb.len();
        let flat: Vec<f32> = fb.iter().flatten().copied().collect();
        let input = ort::value::Tensor::from_array(([1usize, frames, 80usize], flat))?;
        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("session lock: {e}"))?;
        let out = session.run(ort::inputs![crate::INPUT_NAME => input])?;
        let (_shape, data) = out[crate::OUTPUT_NAME].try_extract_tensor::<f32>()?;
        if data.len() != self.dim {
            anyhow::bail!("expected {}-d embedding, got {}", self.dim, data.len());
        }
        let n = data.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        Ok(data.iter().map(|x| x / n).collect())
    }

    /// Always `true`: absence of an `Embedder` is what "disabled" means.
    pub fn is_enabled(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_disables_without_panic() {
        assert!(Embedder::open("/nonexistent/wespeaker.onnx").is_err());
    }

    /// Spike recipe: harmonic tones a fifth apart + harmonics, 3 s each.
    fn synth(f0: f64) -> Vec<i16> {
        (0..48000)
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
    fn embed_synth_separates() {
        // Real-model gate: runs only when HAMFEED_TEST_VOICE_MODEL points
        // at a wespeaker file. Unset → pass with a note (CI has no model);
        // set-but-missing → LOUD failure, never a silent skip.
        let path = match std::env::var("HAMFEED_TEST_VOICE_MODEL") {
            Ok(p) => p,
            Err(_) => {
                println!("skipping: set HAMFEED_TEST_VOICE_MODEL to run the real-model gate");
                return;
            }
        };
        let emb = Embedder::open(&path)
            .unwrap_or_else(|e| panic!("HAMFEED_TEST_VOICE_MODEL={path} unreadable: {e:?}"));
        let a = emb.embed(&synth(110.0)).unwrap();
        let b = emb.embed(&synth(190.0)).unwrap();
        assert_eq!(a.len(), crate::EXPECTED_DIM);
        let cross = crate::cosine(&a, &b);
        // Spike measured 0.33; 0.7 leaves wide margin while still
        // falsifying a collapsed pipeline (mean-only CMN gave 0.88).
        assert!(cross < 0.7, "synth speakers must separate, got {cross}");
        // Determinism: same input embeds identically.
        let a2 = emb.embed(&synth(110.0)).unwrap();
        assert!((crate::cosine(&a, &a2) - 1.0).abs() < 1e-5);
    }
}
