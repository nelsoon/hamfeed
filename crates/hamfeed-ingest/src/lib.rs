//! hamfeed-ingest: VAD segmentation + split (T5).
//!
//! `Segmenter` consumes 16 kHz mono S16 and emits `Segment`s with PCM
//! attached. Sample-clock driven (no wall clock): timestamps derive from the
//! sample count, which keeps tests deterministic. No files, no DB here —
//! encode/queue/spill land in T6.

pub mod denoise;
#[cfg(feature = "opus")]
pub mod encode;
pub mod fixture;
pub mod queue;
pub mod segmenter;
pub mod spill;

pub use denoise::Denoiser;
#[cfg(feature = "opus")]
pub use encode::{
    clip_path, decode_ogg_to_pcm, encode_pcm_to_ogg, write_clip_atomic, ClipMeta, LiveEncoder,
};
pub use queue::{IngestQueue, QueueItem};
pub use segmenter::{Segment, Segmenter, SegmenterConfig};
pub use spill::{SpillDir, SpillRow};

/// Internal rate, shared with `hamfeed-source`.
pub const SAMPLE_RATE: u32 = 16_000;
