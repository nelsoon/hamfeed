//! hamfeed-speaker: voiceprints for sender linking (003 T6).
//!
//! Pure-Rust voice math, no model, no I/O: mel frontend, cosine geometry,
//! and greedy per-window clustering. The ONNX embedder lands in T7 behind
//! the `voice` cargo feature; everything here compiles on default builds.
//!
//! Conventions (spike-verified on synthetic speakers: same 1.0, cross
//! 0.33): 16 kHz input, 25 ms Hamming window, 10 ms shift, 512-pt DFT,
//! 80 HTK-mel bins, pre-emphasis 0.97, utterance-level CMVN (mean AND
//! variance — mean-only CMN collapses separation and is NOT used).

pub mod cluster;
pub mod frontend;

#[cfg(feature = "voice")]
pub mod embedder;

/// ONNX input name (wespeaker resnet34-LM packaging).
pub const INPUT_NAME: &str = "input_features";
/// ONNX output name (utterance-level 256-d vector, no temporal axis).
pub const OUTPUT_NAME: &str = "last_hidden_state";
/// Embedding dimension.
pub const EXPECTED_DIM: usize = 256;

/// Cosine similarity (unit vectors: plain dot product).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        let a = vec![1.0f32, 0.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
        assert!((cosine(&a, &[0.0, 1.0])).abs() < 1e-6);
    }
}
