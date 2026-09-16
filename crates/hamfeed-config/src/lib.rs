//! hamfeed-config: TOML load + validate (T3).
//!
//! Owns the `hamfeed.toml` schema. Invalid TOML or out-of-range values fail
//! with actionable messages; missing-model provisioning text lives here so
//! the pipeline startup check (T9) and the hint test share one wording.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Top-level `hamfeed.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub audio: Audio,
    pub vad: Vad,
    pub segment: Segment,
    pub stt: Stt,
    pub storage: Storage,
    pub station: Station,
    /// Sender carry window (Slice 2). Old configs omit it.
    #[serde(default)]
    pub identity: Identity,
    /// Local callbook database (Slice 2). Old configs omit it.
    #[serde(default)]
    pub callbook: CallbookCfg,
    /// Notification cue lists (Slice 2). Old configs omit it.
    #[serde(default)]
    pub notify: Notify,
    /// Voiceprint settings (Slice 2). Old configs omit it.
    #[serde(default)]
    pub voiceprint: Voiceprint,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Audio {
    pub device: String,
    pub sample_rate: u32,
}

fn default_beep_split() -> bool {
    true
}

fn default_beep_min_ms() -> u64 {
    150
}

#[derive(Debug, Clone, Deserialize)]
pub struct Vad {
    pub engine: String,
    pub hang_ms: u64,
    #[serde(default)]
    pub profiles: VadProfiles,
    /// Cut segments on repeater courtesy beeps (B4). Old configs omit it.
    #[serde(default = "default_beep_split")]
    pub beep_split: bool,
    /// Sustained-tone persistence before a beep cuts (B4, ms).
    #[serde(default = "default_beep_min_ms")]
    pub beep_min_ms: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct VadProfiles {
    #[serde(default)]
    pub repeater: Option<VadProfile>,
    #[serde(default)]
    pub simplex: Option<VadProfile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VadProfile {
    pub hang_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Segment {
    pub max_s: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Stt {
    pub model_path: String,
    pub lang_whitelist: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Storage {
    pub dir: String,
    pub db_path: String,
    pub retention_days: u64,
    #[serde(default)]
    pub delete_audio_on_drop: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Station {
    pub freq_label: String,
    #[serde(default)]
    pub my_callsign: Option<String>,
}

fn default_link_window_min() -> u64 {
    30
}

/// Sender carry window: follow-ups within this long of the last heard
/// self-ID inherit its sender, marked inferred (R2).
#[derive(Debug, Clone, Deserialize)]
pub struct Identity {
    #[serde(default = "default_link_window_min")]
    pub link_window_min: u64,
}

impl Default for Identity {
    fn default() -> Self {
        Self {
            link_window_min: default_link_window_min(),
        }
    }
}

/// Local CA/US callbook database path (R3). Empty `db_path` derives
/// `<storage.dir>/callbook.db`; a missing file degrades to nameless
/// badges, never an error.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CallbookCfg {
    #[serde(default)]
    pub db_path: String,
}

impl CallbookCfg {
    pub fn resolved_db_path(&self, storage_dir: &str) -> String {
        if self.db_path.trim().is_empty() {
            format!("{}/callbook.db", storage_dir.trim_end_matches('/'))
        } else {
            self.db_path.clone()
        }
    }
}

fn default_emergency_cues() -> Vec<String> {
    [
        "mayday",
        "may-day",
        "urgence",
        "emergency",
        "sos",
        "détresse",
        "detresse",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Notification cue lists (R5–R6). `my_callsign` lives under `[station]`;
/// emergency cues match case-insensitively against transcripts.
#[derive(Debug, Clone, Deserialize)]
pub struct Notify {
    #[serde(default = "default_emergency_cues")]
    pub emergency_cues: Vec<String>,
}

impl Default for Notify {
    fn default() -> Self {
        Self {
            emergency_cues: default_emergency_cues(),
        }
    }
}

fn default_min_embed_s() -> f32 {
    1.5
}

fn default_voice_threshold() -> f32 {
    0.55
}

fn default_suggest_min_conf() -> f32 {
    0.5
}

fn default_voice_retention_days() -> u64 {
    7
}

/// Voiceprint settings (R4). Empty `model_path` (or an unreadable file)
/// disables voice with a loud log line — never a startup refusal.
#[derive(Debug, Clone, Deserialize)]
pub struct Voiceprint {
    #[serde(default)]
    pub model_path: String,
    #[serde(default = "default_min_embed_s")]
    pub min_embed_s: f32,
    #[serde(default = "default_voice_threshold")]
    pub threshold: f32,
    #[serde(default = "default_suggest_min_conf")]
    pub suggest_min_conf: f32,
    #[serde(default = "default_voice_retention_days")]
    pub retention_days: u64,
}

impl Default for Voiceprint {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            min_embed_s: default_min_embed_s(),
            threshold: default_voice_threshold(),
            suggest_min_conf: default_suggest_min_conf(),
            retention_days: default_voice_retention_days(),
        }
    }
}

/// Supported STT languages for Slice 1 (ADR-4: local whisper, FR/EN only).
pub const SUPPORTED_LANGS: &[&str] = &["fr", "en"];

fn non_empty(what: &str, v: &str) -> Result<()> {
    if v.trim().is_empty() {
        anyhow::bail!("{what} must not be empty")
    }
    Ok(())
}

fn hang_in_range(what: &str, hang_ms: u64) -> Result<()> {
    if !(100..=5000).contains(&hang_ms) {
        anyhow::bail!("{what} = {hang_ms}ms out of range (want 100..=5000ms)");
    }
    Ok(())
}

impl Config {
    fn validate(&self) -> Result<()> {
        non_empty("[audio] device", &self.audio.device)?;
        if !(8000..=48000).contains(&self.audio.sample_rate) {
            anyhow::bail!(
                "[audio] sample_rate = {} out of range (want 8000..=48000)",
                self.audio.sample_rate
            );
        }
        if self.vad.engine.trim().is_empty() {
            anyhow::bail!("[vad] engine must not be empty (want \"energy\")");
        }
        if self.vad.engine != "energy" {
            anyhow::bail!(
                "[vad] engine = \"{}\" unsupported (want \"energy\")",
                self.vad.engine
            );
        }
        hang_in_range("[vad] hang_ms", self.vad.hang_ms)?;
        if !(50..=1000).contains(&self.vad.beep_min_ms) {
            anyhow::bail!(
                "[vad] beep_min_ms = {} out of range (want 50..=1000)",
                self.vad.beep_min_ms
            );
        }
        if let Some(p) = &self.vad.profiles.repeater {
            hang_in_range("[vad.profiles.repeater] hang_ms", p.hang_ms)?;
        }
        if let Some(p) = &self.vad.profiles.simplex {
            hang_in_range("[vad.profiles.simplex] hang_ms", p.hang_ms)?;
        }
        if !(5..=600).contains(&self.segment.max_s) {
            anyhow::bail!(
                "[segment] max_s = {} out of range (want 5..=600)",
                self.segment.max_s
            );
        }
        non_empty("[stt] model_path", &self.stt.model_path)?;
        if self.stt.lang_whitelist.is_empty() {
            anyhow::bail!("[stt] lang_whitelist must list at least one language");
        }
        for lang in &self.stt.lang_whitelist {
            if !SUPPORTED_LANGS.contains(&lang.as_str()) {
                anyhow::bail!(
                    "[stt] lang_whitelist has unsupported \"{lang}\" (want subset of fr,en)"
                );
            }
        }
        non_empty("[storage] dir", &self.storage.dir)?;
        non_empty("[storage] db_path", &self.storage.db_path)?;
        if !(1..=3650).contains(&self.storage.retention_days) {
            anyhow::bail!(
                "[storage] retention_days = {} out of range (want 1..=3650)",
                self.storage.retention_days
            );
        }
        non_empty("[station] freq_label", &self.station.freq_label)?;
        if !(1..=240).contains(&self.identity.link_window_min) {
            anyhow::bail!(
                "[identity] link_window_min = {} out of range (want 1..=240)",
                self.identity.link_window_min
            );
        }
        if !(0.0..=1.0).contains(&self.voiceprint.threshold) {
            anyhow::bail!(
                "[voiceprint] threshold = {} out of range (want 0.0..=1.0)",
                self.voiceprint.threshold
            );
        }
        if !(0.0..=1.0).contains(&self.voiceprint.suggest_min_conf) {
            anyhow::bail!(
                "[voiceprint] suggest_min_conf = {} out of range (want 0.0..=1.0)",
                self.voiceprint.suggest_min_conf
            );
        }
        if !(0.1..=30.0).contains(&self.voiceprint.min_embed_s) {
            anyhow::bail!(
                "[voiceprint] min_embed_s = {} out of range (want 0.1..=30.0)",
                self.voiceprint.min_embed_s
            );
        }
        if !(1..=3650).contains(&self.voiceprint.retention_days) {
            anyhow::bail!(
                "[voiceprint] retention_days = {} out of range (want 1..=3650)",
                self.voiceprint.retention_days
            );
        }
        Ok(())
    }

    /// Effective hang time for a named VAD profile (`repeater`/`simplex`),
    /// falling back to the top-level `hang_ms` for unknown names.
    pub fn hang_ms(&self, profile: &str) -> u64 {
        match profile {
            "repeater" => self
                .vad
                .profiles
                .repeater
                .as_ref()
                .map(|p| p.hang_ms)
                .unwrap_or(self.vad.hang_ms),
            "simplex" => self
                .vad
                .profiles
                .simplex
                .as_ref()
                .map(|p| p.hang_ms)
                .unwrap_or(self.vad.hang_ms),
            _ => self.vad.hang_ms,
        }
    }
}

/// Load and validate `hamfeed.toml` at `path`.
pub fn load(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read config {}", path.display()))?;
    parse(&text)
}

/// Parse and validate config from a TOML string (tests + inline overrides).
pub fn parse(text: &str) -> Result<Config> {
    let cfg: Config = toml::from_str(text).context("invalid TOML (see hamfeed.toml.example)")?;
    cfg.validate()
        .context("config validation failed (see hamfeed.toml.example)")?;
    Ok(cfg)
}

/// Provisioning instructions shown when the STT model file is missing (S11).
/// Names both the download helper and the manual drop-in directory.
pub fn missing_model_hint(model_path: &str) -> String {
    format!(
        "whisper model not found at {model_path}.\n\
         Fetch it with the download helper:\n  \
           sh scripts/download-model.sh tiny\n\
         or drop a compatible ggml model in manually:\n  \
           mkdir -p models && cp <your-ggml-model> {model_path}\n\
         (tiny/base/small all work; CI caches the tiny model)."
    )
}

/// Provisioning instructions shown when the voice model is missing (003
/// R9). Voice is optional: this text is logged loudly, never a refusal.
pub fn missing_voice_model_hint(model_path: &str) -> String {
    format!(
        "voiceprint model not found at {model_path} — voiceprints off.\n\
         Fetch it with the download helper:\n  \
           sh scripts/download-voice-model.sh models\n\
         or drop the file in manually:\n  \
           mkdir -p models && cp <wespeaker.onnx> {model_path}\n\
         (pinned wespeaker revision; SHA-256 verified by the helper)."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = include_str!("../../../hamfeed.toml.example");

    #[test]
    fn loads_example() {
        let cfg = parse(EXAMPLE).expect("example must parse");
        assert_eq!(cfg.audio.sample_rate, 16000);
        assert_eq!(cfg.hang_ms("repeater"), 800);
        assert_eq!(cfg.hang_ms("simplex"), 1200);
        assert_eq!(cfg.segment.max_s, 120);
        assert_eq!(cfg.stt.lang_whitelist, vec!["fr", "en"]);
        assert_eq!(cfg.storage.retention_days, 90);
    }

    #[test]
    fn rejects_bad_toml() {
        let err = parse("not [valid toml ==").unwrap_err();
        assert!(format!("{err:?}").contains("invalid TOML"));
    }

    #[test]
    fn validates_ranges() {
        for (label, patch) in [
            ("sample_rate", "sample_rate = 96000"),
            ("hang", "hang_ms = 50"),
            ("max_s", "max_s = 3"),
            ("retention", "retention_days = 99999"),
            ("whitelist", "lang_whitelist = [\"de\"]"),
        ] {
            let mut text = EXAMPLE.to_string();
            // Replace only the first occurrence inside the right section is
            // overkill; these keys are unique except hang_ms — patch all.
            if label == "hang" {
                text = text.replace("hang_ms = 800", patch);
                text = text.replace("hang_ms = 1200", patch);
            } else {
                text = text.replacen(
                    match label {
                        "sample_rate" => "sample_rate = 16000",
                        "max_s" => "max_s = 120",
                        "retention" => "retention_days = 90",
                        "whitelist" => "lang_whitelist = [\"fr\", \"en\"]",
                        _ => unreachable!(),
                    },
                    patch,
                    1,
                );
            }
            let err = parse(&text).unwrap_err();
            assert!(
                format!("{err:?}").contains("validation failed"),
                "{label}: {err:?}"
            );
        }
    }

    #[test]
    fn voice_sections_default() {
        let cfg = parse(EXAMPLE).expect("example must parse");
        assert_eq!(cfg.identity.link_window_min, 30);
        assert!(cfg.notify.emergency_cues.contains(&"mayday".to_string()));
        assert!(cfg.notify.emergency_cues.contains(&"urgence".to_string()));
        assert_eq!(cfg.voiceprint.threshold, 0.55);
        assert_eq!(cfg.voiceprint.min_embed_s, 1.5);
        assert_eq!(cfg.voiceprint.retention_days, 7);
        // Empty db_path derives from the storage dir.
        assert_eq!(
            cfg.callbook.resolved_db_path(&cfg.storage.dir),
            format!("{}/callbook.db", cfg.storage.dir)
        );
    }

    #[test]
    fn old_config_without_slice2_sections_parses() {
        let text = r#"
[audio]
device = "default"
sample_rate = 16000
[vad]
engine = "energy"
hang_ms = 400
[segment]
max_s = 120
[stt]
model_path = "models/ggml-tiny.bin"
lang_whitelist = ["fr", "en"]
[storage]
dir = "./data/audio"
db_path = "./data/hamfeed.db"
retention_days = 90
[station]
freq_label = "TEST"
"#;
        let cfg = parse(text).expect("pre-slice-2 config must still parse");
        assert_eq!(cfg.identity.link_window_min, 30);
        assert!(cfg.voiceprint.model_path.is_empty());
    }

    #[test]
    fn rejects_bad_threshold() {
        let text = EXAMPLE.replace("threshold = 0.55", "threshold = 1.5");
        let err = parse(&text).unwrap_err();
        assert!(format!("{err:?}").contains("threshold"));
    }

    #[test]
    fn rejects_zero_window() {
        let text = EXAMPLE.replace("link_window_min = 30", "link_window_min = 0");
        let err = parse(&text).unwrap_err();
        assert!(format!("{err:?}").contains("link_window_min"));
    }

    #[test]
    fn missing_voice_model_hint_text() {
        let hint = missing_voice_model_hint("models/wespeaker.onnx");
        assert!(hint.contains("scripts/download-voice-model.sh"));
        assert!(hint.contains("models/wespeaker.onnx"));
        assert!(hint.contains("voiceprints off"));
    }

    #[test]
    fn missing_model_hint_text() {
        let hint = missing_model_hint("models/whisper-small.bin");
        assert!(hint.contains("scripts/download-model.sh"));
        assert!(hint.contains("models"));
        assert!(hint.contains("models/whisper-small.bin"));
    }
}
