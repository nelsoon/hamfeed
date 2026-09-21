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

/// Default whisper initial prompt: bilingual (FR/EN) amateur-repeater
/// context. Steers the decoder toward callsigns, NATO words, and Q codes
/// instead of same-sounding everyday words. Overridable via
/// `[stt] initial_prompt`; pass `None` to [`Transcriber::open`] for this.
pub const DEFAULT_INITIAL_PROMPT: &str = "VE2DEM, Victor Echo Two Delta \
    Echo Mike, bonsoir, je vous reçois cinq neuf, QTH Montréal, 73, à la \
    prochaine, over. Yeah, good evening, thanks for the call, QSB tonight, \
    seventy-three. Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India \
    Juliett Kilo Lima Mike November Oscar Papa Quebec Romeo Sierra Tango \
    Uniform Victor Whiskey X-ray Yankee Zulu, un deux trois quatre cinq \
    six sept huit neuf, ici VE2LHA, Lima Hotel Alpha.";

/// Local transcriber bound to one model file + language whitelist.
pub struct Transcriber {
    ctx: WhisperContext,
    whitelist: Vec<String>,
    threads: usize,
    prompt: String,
    lang_min_conf: f64,
    lang_fallback: Option<String>,
}

/// Decoder thread budget: an explicit cap wins; otherwise all cores
/// minus one (never 0, never more than 4 — whisper scales poorly
/// past that on small boxes). Taking every core starves capture and
/// the live relay, which is worse than a slower transcript.
pub fn resolve_threads(available: usize, explicit: Option<usize>) -> usize {
    if let Some(t) = explicit {
        return t.clamp(1, 4);
    }
    available.saturating_sub(1).clamp(1, 4)
}

/// Decode language: the detection stands at or above `min_conf`;
/// below it a whitelisted `fallback` wins over a low-confidence
/// guess (short/noisy clips misdetect most). No fallback configured
/// (or not whitelisted) → detection stands regardless.
pub fn decode_lang(
    detected: &str,
    conf: f64,
    whitelist: &[String],
    min_conf: f64,
    fallback: Option<&str>,
) -> String {
    if conf < min_conf {
        if let Some(fb) = fallback {
            if whitelist.iter().any(|w| w == fb) && fb != detected {
                return fb.to_string();
            }
        }
    }
    detected.to_string()
}

impl Transcriber {
    /// Open `model_path` (ggml whisper model). A missing file returns
    /// [`SttErr::ModelMissing`] with download + drop-in instructions —
    /// the pipeline turns this into a loud startup refusal (S11).
    /// `prompt` overrides [`DEFAULT_INITIAL_PROMPT`] (`None` keeps it).
    /// `threads` caps decoder threads (`None` → all cores minus one).
    pub fn open(
        model_path: &Path,
        whitelist: &[String],
        prompt: Option<&str>,
        threads: Option<usize>,
    ) -> Result<Self, SttErr> {
        if !model_path.exists() {
            return Err(SttErr::ModelMissing(hamfeed_config::missing_model_hint(
                &model_path.to_string_lossy(),
            )));
        }
        let path = model_path.to_string_lossy().into_owned();
        let ctx = WhisperContext::new_with_params(&path, WhisperContextParameters::default())
            .map_err(|e| SttErr::Backend(format!("cannot load {path}: {e:?}")))?;
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = resolve_threads(available, threads);
        Ok(Self {
            ctx,
            whitelist: whitelist.to_vec(),
            threads,
            prompt: prompt.unwrap_or(DEFAULT_INITIAL_PROMPT).to_string(),
            lang_min_conf: 0.0,
            lang_fallback: None,
        })
    }

    /// Set the language-confidence fallback after open (config-wired by
    /// the pipeline; tests use it directly).
    pub fn set_lang_fallback(&mut self, min_conf: f64, fallback: Option<String>) {
        self.lang_min_conf = min_conf;
        self.lang_fallback = fallback;
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

        // Constrained decode in the detection language (or the
        // configured fallback when detection is weak); never translate.
        // The initial prompt biases vocabulary toward repeater traffic.
        let lang = decode_lang(
            &lang,
            lang_conf,
            &self.whitelist,
            self.lang_min_conf,
            self.lang_fallback.as_deref(),
        );
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_initial_prompt(&self.prompt);
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
            // Lossy read: whisper occasionally emits bytes that are not
            // valid UTF-8 (more often with larger models). One bad segment
            // must not fail the whole clip — scrub to U+FFFD instead. This
            // fixed a real `text: InvalidUtf8` failure on a 120 s FR clip.
            let text = state
                .full_get_segment_text_lossy(i)
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

/// NATO phonetic alphabet + digit words, both common spellings where
/// whisper varies (`alpha`/`alfa`, `juliet`/`juliett`, `x-ray`/`xray`).
fn phonetic_letter(word: &str) -> Option<char> {
    Some(match word {
        "alfa" | "alpha" => 'A',
        "bravo" => 'B',
        "charlie" => 'C',
        "delta" => 'D',
        "echo" => 'E',
        "foxtrot" => 'F',
        "golf" => 'G',
        "hotel" => 'H',
        "india" => 'I',
        "juliet" | "juliett" => 'J',
        "kilo" => 'K',
        "lima" => 'L',
        "mike" => 'M',
        "november" => 'N',
        "oscar" => 'O',
        "papa" => 'P',
        "quebec" => 'Q',
        "romeo" => 'R',
        "sierra" => 'S',
        "tango" => 'T',
        "uniform" => 'U',
        "victor" => 'V',
        "whiskey" => 'W',
        "xray" | "x-ray" => 'X',
        "yankee" => 'Y',
        "zulu" => 'Z',
        "zero" | "oh" => '0',
        "one" => '1',
        "two" => '2',
        "three" | "tree" => '3',
        "four" | "fower" => '4',
        "five" | "fife" => '5',
        "six" => '6',
        "seven" => '7',
        "eight" | "ait" => '8',
        "nine" | "niner" => '9',
        _ => return None,
    })
}

/// One token's letter, if it is a phonetic word, a digit word, or a bare
/// digit (`2` in "victor echo 2" joins the run as `2`). Anything else —
/// normal words, ham shortcuts (`73`, `QTH`), callsign fragments already
/// written as letters — yields `None` and ends the run.
fn token_letter(word: &str) -> Option<char> {
    let lower = word.to_lowercase();
    let t = lower.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
    if let Some(c) = phonetic_letter(t) {
        return Some(c);
    }
    let t = t.trim_matches(|c: char| !c.is_alphanumeric());
    if t.len() == 1 {
        if let Some(c) = t.chars().next() {
            if c.is_ascii_digit() {
                return Some(c);
            }
        }
    }
    None
}

/// Collapse spoken phonetics into letter groups: "alpha lima lima oscar"
/// becomes "ALLO"; callsign-style runs with digits collapse the same way.
/// Only runs of two or more tokens collapse — a lone "echo" or "mike" in
/// tokens collapse — a lone "echo" or "mike" in normal speech stays
/// untouched, as do ham shortcuts. Separators inside a run may be spaces
/// or hyphens ("x-ray yankee" and "alpha-lima" both work).
pub fn normalize_phonetics(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let mut letters: Vec<char> = Vec::new();
        let mut k = i;
        while k < words.len() {
            // Whole word first, so "x-ray" matches as a unit; hyphenated
            // pairs ("alpha-lima") fall back to per-part matching.
            if let Some(c) = token_letter(words[k]) {
                letters.push(c);
                k += 1;
                continue;
            }
            let before = letters.len();
            let mut matched_all = true;
            let mut any = false;
            for chunk in words[k].split('-') {
                if chunk.is_empty() {
                    continue;
                }
                any = true;
                match token_letter(chunk) {
                    Some(c) => letters.push(c),
                    None => {
                        matched_all = false;
                        break;
                    }
                }
            }
            if !matched_all || !any {
                letters.truncate(before);
                break;
            }
            k += 1;
        }
        if letters.len() >= 2 {
            out.push(letters.into_iter().collect());
            i = k;
        } else {
            out.push(words[i].to_string());
            i += 1;
        }
    }
    out.join(" ")
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
    Transcriber::open(&path, &["fr".to_string(), "en".to_string()], None, None)
        .expect("test model must load")
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
    fn custom_prompt_transcribes() {
        // A custom initial prompt must flow into the decoder without
        // breaking it: same fixture, still English, still non-empty.
        let from_env = std::env::var("HAMFEED_TEST_MODEL").ok();
        let fallback =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ggml-tiny.bin");
        let path = from_env.map(std::path::PathBuf::from).unwrap_or(fallback);
        let t = Transcriber::open(&path, &["en".to_string()], Some("VE2ABC net, over"), None)
            .expect("prompted model must load");
        let out = t.transcribe(&fixture("en.ogg")).expect("en transcribes");
        assert_eq!(out.lang, "en", "lang must be en, got {}", out.lang);
        assert!(!out.transcript.is_empty(), "transcript must be non-empty");
    }

    #[test]
    fn phonetics_collapse() {
        assert_eq!(normalize_phonetics("alpha lima lima oscar"), "ALLO");
        assert_eq!(
            normalize_phonetics("Whiskey One Alpha Whiskey calling"),
            "W1AW calling"
        );
        assert_eq!(normalize_phonetics("tango hotel five"), "TH5");
        assert_eq!(
            normalize_phonetics("contact alpha lima on simplex"),
            "contact AL on simplex"
        );
        assert_eq!(normalize_phonetics("x-ray yankee"), "XY");
        assert_eq!(normalize_phonetics("alpha-lima"), "AL");
        assert_eq!(normalize_phonetics("lima oscar, over"), "LO over");
    }

    #[test]
    fn phonetics_leave_normal_speech() {
        // Singletons are ordinary words until they form a run.
        assert_eq!(normalize_phonetics("say echo again"), "say echo again");
        assert_eq!(normalize_phonetics("thanks mike"), "thanks mike");
        // Ham shortcuts pass through untouched.
        assert_eq!(normalize_phonetics("73 and 88"), "73 and 88");
        assert_eq!(normalize_phonetics("QTH is here"), "QTH is here");
        assert_eq!(
            normalize_phonetics("hello radio world"),
            "hello radio world"
        );
    }

    #[test]
    fn threads_leave_a_core_free() {
        // Explicit cap wins (clamped); otherwise all cores minus one,
        // never 0 — a saturated box starves capture + live relay.
        assert_eq!(resolve_threads(4, Some(2)), 2);
        assert_eq!(resolve_threads(4, Some(99)), 4);
        assert_eq!(resolve_threads(4, Some(0)), 1);
        assert_eq!(resolve_threads(4, None), 3);
        assert_eq!(resolve_threads(16, None), 4);
        assert_eq!(resolve_threads(1, None), 1);
        assert_eq!(resolve_threads(2, None), 1);
    }

    #[test]
    fn weak_detection_falls_back() {
        let wl = vec!["fr".to_string(), "en".to_string()];
        // Confident detection stands, even for the non-fallback language.
        assert_eq!(decode_lang("en", 0.97, &wl, 0.85, Some("fr")), "en");
        // Weak detection yields to a whitelisted fallback…
        assert_eq!(decode_lang("en", 0.71, &wl, 0.85, Some("fr")), "fr");
        // …but never to a language outside the whitelist…
        assert_eq!(decode_lang("en", 0.71, &wl, 0.85, Some("es")), "en");
        // …and 0.0 floor (default) keeps pure detection.
        assert_eq!(decode_lang("en", 0.01, &wl, 0.0, Some("fr")), "en");
        // Fallback equal to detection is a no-op.
        assert_eq!(decode_lang("fr", 0.5, &wl, 0.85, Some("fr")), "fr");
    }

    #[test]
    fn prompt_carries_nato_table() {
        // The decoder can only reach for words the prompt offers:
        // full NATO alphabet + French digits + a spelled callsign.
        for w in [
            "Lima", "Juliett", "X-ray", "Yankee", "Zulu", "sept", "VE2LHA",
        ] {
            assert!(DEFAULT_INITIAL_PROMPT.contains(w), "prompt must offer {w}");
        }
    }

    #[test]
    fn missing_model_is_loud() {
        let err = match Transcriber::open(
            Path::new("/nonexistent/ggml-tiny.bin"),
            &["fr".to_string()],
            None,
            None,
        ) {
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
