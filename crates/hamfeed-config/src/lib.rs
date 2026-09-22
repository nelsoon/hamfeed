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
    /// Input selector (old configs omit it → `mic`, unchanged behavior).
    #[serde(default)]
    pub source: Source,
    /// B210 station parameters (old configs omit them → measured
    /// defaults, empty channel list).
    #[serde(default)]
    pub sdr: Sdr,
    #[serde(default)]
    pub ingest: Ingest,
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

/// Audio input selector. `mic` is the cpal path (unchanged default);
/// `sdr` spawns the UHD capture shim and feeds gated 16 kHz frames.
#[derive(Debug, Clone, Deserialize)]
pub struct Source {
    #[serde(default = "default_source_kind")]
    pub kind: String,
}

fn default_source_kind() -> String {
    "mic".into()
}

impl Default for Source {
    fn default() -> Self {
        Self {
            kind: default_source_kind(),
        }
    }
}

impl Default for Sdr {
    fn default() -> Self {
        Self {
            python: default_sdr_python(),
            script: default_sdr_script(),
            gain: default_sdr_gain(),
            rate_hz: default_sdr_rate(),
            bandwidth_hz: default_sdr_bandwidth(),
            squelch_db: default_sdr_squelch(),
            hang_s: default_sdr_hang(),
            antenna: default_sdr_antenna(),
            active: String::new(),
            channels: Vec::new(),
        }
    }
}

/// One listenable channel: a named frequency. Bandwidth and gain are
/// station constants (measured), not per-channel knobs — the mode
/// alone selects the demod path (`nbfm` today; `am`/`ssb` later).
#[derive(Debug, Clone, Deserialize)]
pub struct SdrChannel {
    pub name: String,
    pub freq_hz: f64,
    #[serde(default = "default_sdr_mode")]
    pub mode: String,
}

fn default_sdr_mode() -> String {
    "nbfm".into()
}

/// B210 station parameters, measured on 10.0.2.102 — do not re-tune
/// blindly. Gain 20 (usable 10-30; past 30 the AD9361 table handoffs
/// pull adjacent junk); 250 ksps with analog BW capped at 200 kHz
/// (the chip's own anti-intermod filter against LTE/DVB-T); squelch
/// gate 14.5 dB with ~1.5 s hang (DC-removed PSD plus a
/// concentration test; the empty-channel median sits ~11 dB).
/// TX is never touched — the shim parks it at 0.
#[derive(Debug, Clone, Deserialize)]
pub struct Sdr {
    #[serde(default = "default_sdr_python")]
    pub python: String,
    #[serde(default = "default_sdr_script")]
    pub script: String,
    #[serde(default = "default_sdr_gain")]
    pub gain: f64,
    #[serde(default = "default_sdr_rate")]
    pub rate_hz: f64,
    #[serde(default = "default_sdr_bandwidth")]
    pub bandwidth_hz: f64,
    #[serde(default = "default_sdr_squelch")]
    pub squelch_db: f64,
    #[serde(default = "default_sdr_hang")]
    pub hang_s: f64,
    #[serde(default = "default_sdr_antenna")]
    pub antenna: String,
    /// Active channel name; empty selects the first channel.
    #[serde(default)]
    pub active: String,
    #[serde(default)]
    pub channels: Vec<SdrChannel>,
}

fn default_sdr_python() -> String {
    "python3".into()
}
fn default_sdr_script() -> String {
    "sdr/sdr_rx.py".into()
}
fn default_sdr_gain() -> f64 {
    20.0
}
fn default_sdr_rate() -> f64 {
    250_000.0
}
fn default_sdr_bandwidth() -> f64 {
    200_000.0
}
fn default_sdr_squelch() -> f64 {
    14.5
}
fn default_sdr_hang() -> f64 {
    1.5
}
fn default_sdr_antenna() -> String {
    "RX2".into()
}

impl Sdr {
    /// Active channel: the named one, else the first. `None` when no
    /// channels are configured at all.
    pub fn active_channel(&self) -> Option<&SdrChannel> {
        if self.channels.is_empty() {
            return None;
        }
        if !self.active.is_empty() {
            if let Some(c) = self.channels.iter().find(|c| c.name == self.active) {
                return Some(c);
            }
        }
        self.channels.first()
    }

    /// Look up a channel by name (UI switching + validation).
    pub fn channel(&self, name: &str) -> Option<&SdrChannel> {
        self.channels.iter().find(|c| c.name == name)
    }
}

fn default_beep_split() -> bool {
    true
}

fn default_beep_min_ms() -> u64 {
    100
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

/// Enhancement ahead of the segmenter (006/007). Old configs omit both
/// flags (stay off); prod enables them explicitly after the ear test.
/// `voice_clarity` subsumes `denoise`: its chain already contains the
/// spectral gate, so setting it alone is the full treatment.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Ingest {
    #[serde(default)]
    pub denoise: bool,
    #[serde(default)]
    pub voice_clarity: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Stt {
    pub model_path: String,
    pub lang_whitelist: Vec<String>,
    /// Whisper initial prompt override. Omitted → built-in bilingual
    /// repeater context (`hamfeed_stt::DEFAULT_INITIAL_PROMPT`).
    #[serde(default)]
    pub initial_prompt: Option<String>,
    /// Decoder thread cap. Omitted → all cores minus one (never 0):
    /// a saturated box starves capture + the live relay, which is
    /// worse than a slower transcript.
    #[serde(default)]
    pub threads: Option<usize>,
    /// Language-detection confidence floor. When the top detection
    /// scores below this AND `lang_fallback` names a whitelisted
    /// language, the decode runs in the fallback instead of a
    /// low-confidence guess (short/noisy clips misdetect most).
    /// 0.0 (default) keeps pure detection.
    #[serde(default)]
    pub lang_min_conf: f64,
    /// See `lang_min_conf`. Omitted → no fallback.
    #[serde(default)]
    pub lang_fallback: Option<String>,
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

/// Voiceprint settings (R4). `enabled` is the master switch (default
/// off): embeddings are derived only when the operator opts in AND a
/// readable model is configured. An unreadable file still disables voice
/// with a loud log line — never a startup refusal.
#[derive(Debug, Clone, Deserialize)]
pub struct Voiceprint {
    #[serde(default)]
    pub enabled: bool,
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
            enabled: false,
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
    /// Station parameters are measured on 10.0.2.102 — ranges below
    /// encode what was verified, not guesses. A value outside fails
    /// loudly rather than capturing garbage.
    fn validate_sdr(&self) -> Result<()> {
        let s = &self.sdr;
        if s.channels.is_empty() {
            anyhow::bail!("[sdr] kind is \"sdr\" but no [[sdr.channels]] are configured");
        }
        let mut seen = std::collections::HashSet::new();
        for c in &s.channels {
            if c.name.trim().is_empty() {
                anyhow::bail!("[sdr] channel with empty name");
            }
            if !seen.insert(c.name.clone()) {
                anyhow::bail!("[sdr] duplicate channel name {:?}", c.name);
            }
            if !(1e6..=6e9).contains(&c.freq_hz) {
                anyhow::bail!(
                    "[sdr] channel {:?} freq {} out of range (want 1 MHz..=6 GHz)",
                    c.name,
                    c.freq_hz
                );
            }
            if c.mode != "nbfm" {
                anyhow::bail!(
                    "[sdr] channel {:?} mode = {:?} unsupported (want \"nbfm\"; am/ssb later)",
                    c.name,
                    c.mode
                );
            }
        }
        if !s.active.is_empty() && s.channel(&s.active).is_none() {
            anyhow::bail!("[sdr] active = {:?} names no configured channel", s.active);
        }
        if !(0.0..=40.0).contains(&s.gain) {
            anyhow::bail!(
                "[sdr] gain = {} out of range (want 0..=40; measured usable 10-30)",
                s.gain
            );
        }
        if !(50_000.0..=4_000_000.0).contains(&s.rate_hz) {
            anyhow::bail!("[sdr] rate_hz = {} out of range (want 50k..=4M)", s.rate_hz);
        }
        if !(1_000.0..=200_000.0).contains(&s.bandwidth_hz) {
            anyhow::bail!(
                "[sdr] bandwidth_hz = {} out of range (want 1k..=200k: wider re-admits LTE/DVB-T)",
                s.bandwidth_hz
            );
        }
        if s.bandwidth_hz > s.rate_hz {
            anyhow::bail!(
                "[sdr] bandwidth_hz = {} exceeds rate_hz = {}",
                s.bandwidth_hz,
                s.rate_hz
            );
        }
        if !(0.0..=40.0).contains(&s.squelch_db) {
            anyhow::bail!(
                "[sdr] squelch_db = {} out of range (want 0..=40; calibrated gate >= 14, empty-channel median ~11)",
                s.squelch_db
            );
        }
        if !(0.2..=5.0).contains(&s.hang_s) {
            anyhow::bail!("[sdr] hang_s = {} out of range (want 0.2..=5)", s.hang_s);
        }
        non_empty("[sdr] python", &s.python)?;
        non_empty("[sdr] script", &s.script)?;
        non_empty("[sdr] antenna", &s.antenna)?;
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        non_empty("[audio] device", &self.audio.device)?;
        if !(8000..=48000).contains(&self.audio.sample_rate) {
            anyhow::bail!(
                "[audio] sample_rate = {} out of range (want 8000..=48000)",
                self.audio.sample_rate
            );
        }
        if self.source.kind != "mic" && self.source.kind != "sdr" {
            anyhow::bail!(
                "[source] kind = \"{}\" unsupported (want \"mic\" or \"sdr\")",
                self.source.kind
            );
        }
        if self.source.kind == "sdr" {
            self.validate_sdr()?;
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
        assert_eq!(cfg.stt.initial_prompt, None);
        // New [stt] knobs default to old behavior: pure detection,
        // automatic thread budget (old configs parse unchanged).
        assert_eq!(cfg.stt.threads, None);
        assert_eq!(cfg.stt.lang_min_conf, 0.0);
        assert_eq!(cfg.stt.lang_fallback, None);
        assert_eq!(cfg.storage.retention_days, 90);
    }

    #[test]
    fn source_defaults_to_mic() {
        // Old configs (and the example) omit [source]/[sdr]: capture
        // behavior is unchanged, with measured SDR defaults waiting.
        let cfg = parse(EXAMPLE).expect("example must parse");
        assert_eq!(cfg.source.kind, "mic");
        assert!(cfg.sdr.channels.is_empty());
        assert_eq!(cfg.sdr.gain, 20.0);
        assert_eq!(cfg.sdr.rate_hz, 250_000.0);
        assert_eq!(cfg.sdr.squelch_db, 14.5);
        assert!(cfg.sdr.active_channel().is_none());
    }

    const SDR_OK: &str = r#"
[sdr]
active = "marine"
[[sdr.channels]]
name = "2m VE2"
freq_hz = 145110000.0
[[sdr.channels]]
name = "marine"
freq_hz = 161750000.0
mode = "nbfm"
"#;

    /// Example (mic default) plus the station channels, flipped to sdr.
    fn sdr_example() -> String {
        EXAMPLE.replace("kind = \"mic\"", "kind = \"sdr\"") + SDR_OK
    }

    #[test]
    fn sdr_channels_parse_and_resolve() {
        let text = sdr_example();
        let cfg = parse(&text).expect("sdr config must parse");
        assert_eq!(cfg.source.kind, "sdr");
        assert_eq!(cfg.sdr.channels.len(), 2);
        // Named active wins; unnamed falls back to the first channel.
        assert_eq!(cfg.sdr.active_channel().unwrap().name, "marine");
        let mut anon = cfg.sdr.clone();
        anon.active.clear();
        assert_eq!(anon.active_channel().unwrap().name, "2m VE2");
        assert_eq!(anon.channel("marine").unwrap().freq_hz, 161750000.0);
        assert!(anon.channel("nope").is_none());
    }

    #[test]
    fn sdr_validation_rejects_garbage() {
        // kind=sdr with no channels.
        let bad = EXAMPLE.replace("kind = \"mic\"", "kind = \"sdr\"");
        assert!(parse(&bad).is_err());
        // Unknown kind.
        let bad = EXAMPLE.replace("kind = \"mic\"", "kind = \"fm\"");
        assert!(parse(&bad).is_err());
        // Overrides splice into the [sdr] header (a second [sdr] table
        // after the channels would be invalid TOML, not a validation
        // failure — test the validator, not the parser).
        let with =
            |key: &str| sdr_example().replace("[sdr]\nactive", &format!("[sdr]\n{key}\nactive"));
        // Gain past the measured-usable ceiling.
        assert!(parse(&with("gain = 45.0")).is_err());
        // Wide-open analog bandwidth re-admits the local giants.
        assert!(parse(&with("bandwidth_hz = 1000000.0")).is_err());
        // Active naming nothing (duplicate keys would fail in the
        // parser instead — swap the value, don't add a second key).
        let bad = sdr_example().replace("active = \"marine\"", "active = \"nope\"");
        assert!(parse(&bad).is_err());
        // Non-nbfm mode has no demod path yet.
        let bad = sdr_example().replace("marine\"\nfreq_hz", "marine\"\nmode = \"ssb\"\nfreq_hz");
        assert!(parse(&bad).is_err());
    }

    #[test]
    fn stt_knobs_parse() {
        let text = EXAMPLE.replace(
            "lang_whitelist = [\"fr\", \"en\"]",
            "lang_whitelist = [\"fr\", \"en\"]\nthreads = 2\nlang_min_conf = 0.85\nlang_fallback = \"fr\"",
        );
        let stt = parse(&text).expect("stt knobs must parse").stt;
        assert_eq!(stt.threads, Some(2));
        assert_eq!(stt.lang_min_conf, 0.85);
        assert_eq!(stt.lang_fallback.as_deref(), Some("fr"));
    }

    #[test]
    fn ingest_voice_clarity_flag() {
        // 007: absent section stays off (old configs unchanged); explicit
        // true parses through, independently of denoise.
        assert!(
            !parse(EXAMPLE)
                .expect("example must parse")
                .ingest
                .voice_clarity
        );
        let text = EXAMPLE.replace("[vad]", "[ingest]\nvoice_clarity = true\n\n[vad]");
        let cfg = parse(&text).expect("ingest section must parse").ingest;
        assert!(cfg.voice_clarity);
        assert!(!cfg.denoise);
    }

    #[test]
    fn ingest_denoise_flag() {
        // 006: absent section stays off (old configs unchanged); explicit
        // true parses through.
        assert!(!parse(EXAMPLE).expect("example must parse").ingest.denoise);
        let text = EXAMPLE.replace("[vad]", "[ingest]\ndenoise = true\n\n[vad]");
        assert!(
            parse(&text)
                .expect("ingest section must parse")
                .ingest
                .denoise
        );
    }

    #[test]
    fn initial_prompt_override() {
        let text = EXAMPLE.replace(
            "lang_whitelist = [\"fr\", \"en\"]",
            "lang_whitelist = [\"fr\", \"en\"]\ninitial_prompt = \"VE2ABC net\"",
        );
        let cfg = parse(&text).expect("override must parse");
        assert_eq!(cfg.stt.initial_prompt.as_deref(), Some("VE2ABC net"));
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
        // Voice is opt-in: the example ships disabled.
        assert!(!cfg.voiceprint.enabled);
        let on = parse(&EXAMPLE.replace("enabled = false", "enabled = true"))
            .expect("enabled = true must parse");
        assert!(on.voiceprint.enabled);
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
"#;
        let cfg = parse(text).expect("pre-slice-2 config must still parse");
        assert_eq!(cfg.identity.link_window_min, 30);
        assert!(cfg.voiceprint.model_path.is_empty());
        // Old configs omit the switch: voice stays off (serde default).
        assert!(!cfg.voiceprint.enabled);
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
