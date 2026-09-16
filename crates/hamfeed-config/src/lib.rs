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
}

#[derive(Debug, Clone, Deserialize)]
pub struct Audio {
    pub device: String,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Vad {
    pub engine: String,
    pub hang_ms: u64,
    #[serde(default)]
    pub profiles: VadProfiles,
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
    fn missing_model_hint_text() {
        let hint = missing_model_hint("models/whisper-small.bin");
        assert!(hint.contains("scripts/download-model.sh"));
        assert!(hint.contains("models"));
        assert!(hint.contains("models/whisper-small.bin"));
    }
}
