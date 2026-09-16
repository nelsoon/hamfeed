//! hamfeed-stt: local whisper transcription (T8).
//!
//! `Transcriber` wraps whisper-rs: stored Opus clip → decode PCM in-RAM →
//! language detect (probabilities kept) → constrained transcribe in the
//! detected language, never translated. Input stays local; nothing here
//! touches the network. Missing model fails loudly at open (S11), never as
//! a silent empty transcript.

use std::path::Path;

use anyhow::Context;
use whisper_rs::{
    get_lang_str, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

/// Transcription result: source language kept as-transcribed (ADR-4).
#[derive(Debug, Clone, PartialEq)]
pub struct SttOut {
    pub transcript: String,
    pub lang: String,
    pub lang_conf: f64,
    pub stt_conf: f64,
}

/// Why a clip could not be transcribed (S5/S11 surface).
#[derive(Debug, Clone, PartialEq)]
pub enum SttErr {
    /// Not decodable audio, no speech, or empty transcript.
    Undecodable(String),
    /// Detected language outside the configured whitelist.
    UnsupportedLang {
        lang: String,
        whitelist: Vec<String>,
    },
    /// No model file: carries provisioning instructions (S11).
    ModelMissing(String),
    /// whisper backend failure.
    Backend(String),
}

impl std::fmt::Display for SttErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SttErr::Undecodable(m) => write!(f, "undecodable clip: {m}"),
            SttErr::UnsupportedLang { lang, whitelist } => write!(
                f,
                "detected language '{lang}' outside whitelist [{}]",
                whitelist.join(",")
            ),
            SttErr::ModelMissing(hint) => write!(f, "{hint}"),
            SttErr::Backend(m) => write!(f, "stt backend: {m}"),
        }
    }
}

/// Local transcriber bound to one model file + language whitelist.
pub struct Transcriber {
    ctx: WhisperContext,
    whitelist: Vec<String>,
    threads: usize,
}

impl Transcriber {
    /// Open `model_path` (ggml whisper model). A missing file returns
    /// [`SttErr::ModelMissing`] with download + drop-in instructions —
    /// the pipeline turns this into a loud startup refusal (S11).
    pub fn open(model_path: &Path, whitelist: &[String]) -> Result<Self, SttErr> {
        if !model_path.exists() {
            return Err(SttErr::ModelMissing(hamfeed_config::missing_model_hint(
                &model_path.to_string_lossy(),
            )));
        }
        let path = model_path.to_string_lossy().into_owned();
        let ctx = WhisperContext::new_with_params(&path, WhisperContextParameters::default())
            .map_err(|e| SttErr::Backend(format!("cannot load {path}: {e:?}")))?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 4))
            .unwrap_or(1);
        Ok(Self {
            ctx,
            whitelist: whitelist.to_vec(),
            threads,
        })
    }

    /// Transcribe one stored Opus clip.
    pub fn transcribe(&self, clip: &Path) -> Result<SttOut, SttErr> {
        let bytes = std::fs::read(clip)
            .with_context(|| format!("cannot read {}", clip.display()))
            .map_err(|e| SttErr::Undecodable(format!("{e:?}")))?;
        let pcm = hamfeed_ingest::decode_ogg_to_pcm(&bytes)
            .map_err(|e| SttErr::Undecodable(format!("{e:?}")))?;
        if pcm.is_empty() {
            return Err(SttErr::Undecodable("no audio frames".into()));
        }
        let pcm_f32: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| SttErr::Backend(format!("state: {e:?}")))?;

        // Language detect first: probabilities are the lang confidence.
        // (The `_with_state` detector needs mel features up front.)
        state
            .pcm_to_mel(&pcm_f32, self.threads)
            .map_err(|e| SttErr::Backend(format!("mel: {e:?}")))?;
        let (lang_id, probs) = state
            .lang_detect(0, self.threads)
            .map_err(|e| SttErr::Backend(format!("lang detect: {e:?}")))?;
        let lang = get_lang_str(lang_id).unwrap_or("unknown").to_string();
        let lang_conf = probs.get(lang_id as usize).copied().unwrap_or(0.0) as f64;
        if !self.whitelist.iter().any(|w| w == &lang) {
            return Err(SttErr::UnsupportedLang {
                lang,
                whitelist: self.whitelist.clone(),
            });
        }

        // Constrained decode in the detected language; never translate.
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads as std::ffi::c_int);
        params.set_language(Some(&lang));
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        state
            .full(params, &pcm_f32)
            .map_err(|e| SttErr::Backend(format!("decode: {e:?}")))?;

        let n_seg = state
            .full_n_segments()
            .map_err(|e| SttErr::Backend(format!("segments: {e:?}")))?;
        if n_seg == 0 {
            return Err(SttErr::Undecodable("no speech segments".into()));
        }
        let mut transcript = String::new();
        let mut prob_sum = 0f64;
        let mut prob_n = 0u64;
        for i in 0..n_seg {
            let text = state
                .full_get_segment_text(i)
                .map_err(|e| SttErr::Backend(format!("text: {e:?}")))?;
            transcript.push_str(text.trim());
            transcript.push(' ');
            let n_tok = state
                .full_n_tokens(i)
                .map_err(|e| SttErr::Backend(format!("tokens: {e:?}")))?;
            for t in 0..n_tok {
                if let Ok(p) = state.full_get_token_prob(i, t) {
                    prob_sum += p as f64;
                    prob_n += 1;
                }
            }
        }
        let transcript = transcript.trim().to_string();
        if transcript.is_empty() {
            return Err(SttErr::Undecodable("empty transcript".into()));
        }
        let stt_conf = if prob_n > 0 {
            prob_sum / prob_n as f64
        } else {
            0.0
        };
        Ok(SttOut {
            transcript,
            lang,
            lang_conf,
            stt_conf,
        })
    }
}

/// Fixture model lookup for tests: `$HAMFEED_TEST_MODEL`, else the workspace
/// `models/ggml-tiny.bin` (fetched by `scripts/download-model.sh`, cached in
/// CI). Panics LOUDLY when absent — a silent skip would hide a dead STT.
#[cfg(test)]
pub(crate) fn test_transcriber() -> Transcriber {
    let from_env = std::env::var("HAMFEED_TEST_MODEL").ok();
    let fallback =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ggml-tiny.bin");
    let path = from_env.map(std::path::PathBuf::from).unwrap_or(fallback);
    if !path.exists() {
        panic!(
            "STT fixture test needs a whisper model at {}.\n\
             Fetch it: sh scripts/download-model.sh tiny\n\
             (from the workspace root; CI caches it). Refusing to skip.",
            path.display()
        );
    }
    Transcriber::open(&path, &["fr".to_string(), "en".to_string()]).expect("test model must load")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn en_fixture() {
        let t = test_transcriber();
        let out = t.transcribe(&fixture("en.ogg")).expect("en transcribes");
        assert_eq!(out.lang, "en", "lang must be en, got {}", out.lang);
        assert!(!out.transcript.is_empty(), "transcript must be non-empty");
        assert!(
            out.transcript.is_ascii(),
            "en must stay untranslated: {}",
            out.transcript
        );
    }

    #[test]
    fn fr_fixture() {
        let t = test_transcriber();
        let out = t.transcribe(&fixture("fr.ogg")).expect("fr transcribes");
        assert_eq!(out.lang, "fr", "lang must be fr, got {}", out.lang);
        assert!(!out.transcript.is_empty(), "transcript must be non-empty");
    }

    #[test]
    fn missing_model_is_loud() {
        let err =
            match Transcriber::open(Path::new("/nonexistent/ggml-tiny.bin"), &["fr".to_string()]) {
                Ok(_) => panic!("must refuse a missing model"),
                Err(e) => e,
            };
        match err {
            SttErr::ModelMissing(hint) => {
                assert!(hint.contains("scripts/download-model.sh"));
            }
            other => panic!("want ModelMissing, got {other:?}"),
        }
    }

    #[test]
    fn corrupt_clip_is_undecodable() {
        let t = test_transcriber();
        let dir = std::env::temp_dir().join(format!("hamfeed-stt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("bad.ogg");
        std::fs::write(&bad, b"not audio at all").unwrap();
        let err = t.transcribe(&bad).unwrap_err();
        assert!(matches!(err, SttErr::Undecodable(_)), "got {err:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
