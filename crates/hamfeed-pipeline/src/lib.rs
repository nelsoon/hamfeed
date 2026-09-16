//! hamfeed-pipeline: ingest → stt → store orchestration (T9).
//!
//! Owns the queue drain, status mapping (`ok|failed|system|dropped`), triage
//! transitions, system gap markers, and the loud missing-model startup check
//! (S11). Group contract: web may call the `enrich`/triage helpers here, but
//! pipeline never depends on web.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use hamfeed_callbook::Callbook;
use hamfeed_config::Config;
use hamfeed_ingest::{IngestQueue, QueueItem, Segment, Segmenter, SegmenterConfig, SpillDir};
use hamfeed_source::AudioSource;
use hamfeed_speaker::cluster::Clusterer;
#[cfg(feature = "voice")]
use hamfeed_speaker::embedder::Embedder;
use hamfeed_store::{Message, NewMessage, SearchQuery, Store, TriageAction, SHORT_MS};
use hamfeed_stt::{SttErr, Transcriber};

/// `stt_conf` at or above this marks a row `ok`, below marks `low`.
pub const CONF_OK: f64 = 0.6;

/// Result of sender enrichment for one message (Slice 2, 003).
/// `sender_name` lands in T5 (callbook), `alert` in T9 (notify),
/// `speaker_key` in T8 (voice).
#[derive(Debug, Clone, PartialEq)]
pub struct Enrichment {
    pub sender_callsign: Option<String>,
    pub sender_name: Option<String>,
    pub sender_source: String,
    pub alert: i32,
    pub speaker_key: Option<String>,
}

impl Enrichment {
    fn none() -> Self {
        Self {
            sender_callsign: None,
            sender_name: None,
            sender_source: "none".into(),
            alert: 0,
            speaker_key: None,
        }
    }
}

/// Voice-link state (R4, T8): per-group clusterer window plus the keyed
/// segments of the current group for alias linking. Without the `voice`
/// feature no embedder exists and no keys are ever minted — the alias and
/// suggestion paths then idle on empty state.
pub struct VoiceState {
    clusterer: Clusterer,
    group_keys: Vec<(String, u64)>,
    last_group_id: String,
    #[cfg(feature = "voice")]
    embedder: Option<Embedder>,
}

impl VoiceState {
    fn new(threshold: f32, #[cfg(feature = "voice")] embedder: Option<Embedder>) -> Self {
        Self {
            clusterer: Clusterer::new(threshold),
            group_keys: Vec::new(),
            last_group_id: String::new(),
            #[cfg(feature = "voice")]
            embedder,
        }
    }

    /// Roll the clusterer window on `group_id` change (one window = one
    /// transmission group; N allocation stays store-global, never reused).
    fn roll_window(&mut self, group_id: &str) {
        if self.last_group_id != group_id {
            self.clusterer.reset_window();
            self.group_keys.clear();
            self.last_group_id = group_id.to_string();
        }
    }

    /// Per-segment speaker gate: long enough AND not failed → embed →
    /// assign → `Unknown-N|label|day` key; else `None` (short blips,
    /// failed rows, and voice-off builds stay keyless).
    #[allow(clippy::too_many_arguments)]
    fn key_for(
        &mut self,
        store: &Store,
        pcm: &[i16],
        duration_ms: u64,
        status: &str,
        freq_label: &str,
        day: &str,
        min_embed_s: f32,
        seg_id: &str,
    ) -> Option<String> {
        // Float seconds (no float→int truncation at the boundary).
        if (duration_ms as f32) < min_embed_s * 1000.0 {
            return None;
        }
        if status == "failed" {
            return None;
        }
        #[cfg(feature = "voice")]
        {
            let emb = self.embedder.as_ref()?;
            match emb.embed(pcm) {
                Ok(vec) => {
                    let mut alloc_failed = false;
                    let mut alloc = || match store.alloc_speaker_n(freq_label, day) {
                        Ok(n) => n,
                        Err(err) => {
                            eprintln!("pipeline: alloc_speaker_n failed: {err}");
                            alloc_failed = true;
                            0
                        }
                    };
                    let (n, _is_new) = self.clusterer.assign(&vec, &mut alloc);
                    if !alloc_failed && n != 0 {
                        return Some(format!(
                            "{}|{}|{}",
                            Clusterer::label(n),
                            freq_label.replace('|', "-"),
                            day
                        ));
                    }
                    None
                }
                Err(err) => {
                    eprintln!("pipeline: embed failed for {seg_id}: {err}");
                    None
                }
            }
        }
        #[cfg(not(feature = "voice"))]
        {
            let _ = (store, pcm, freq_label, day, seg_id);
            None
        }
    }
}

/// Epoch-day number for a millisecond timestamp (voice-key scope).
pub fn day_of(ts_ms: u64) -> String {
    (ts_ms / 86_400_000).to_string()
}

/// Link every keyed segment in the current group to a heard callsign.
///
/// Confidence decays linearly from 1.0 at dt=0 to 0.5 at the window edge.
/// Writes alias rows only — never `sender_*` fields.
pub fn link_alias(
    store: &Store,
    group_keys: &[(String, u64)],
    heard_ts_ms: u64,
    callsign: &str,
    window_ms: u64,
) {
    if window_ms == 0 {
        return;
    }
    let now_ms = now_ms();
    for (key, ts) in group_keys {
        let dt = heard_ts_ms.abs_diff(*ts) as f64;
        if dt <= window_ms as f64 {
            let conf = 0.5 + 0.5 * (1.0 - dt / window_ms as f64);
            let _ = store.set_alias(key, callsign, conf as f32, now_ms);
        }
    }
}

/// In-memory heard-sender carry (R2): the last self-ID heard plus its
/// timestamp. A restart clears it by design (carry is best-effort).
#[derive(Debug, Default)]
pub struct CarryState {
    last_callsign: Option<String>,
    last_ts_ms: u64,
    window_ms: u64,
}

impl CarryState {
    pub fn new(window_ms: u64) -> Self {
        Self {
            last_callsign: None,
            last_ts_ms: 0,
            window_ms,
        }
    }

    /// Record a heard self-ID (or clear nothing when `callsign` is `None`).
    pub fn observe(&mut self, callsign: Option<String>, ts_ms: u64) {
        if let Some(cs) = callsign {
            self.last_callsign = Some(cs);
            self.last_ts_ms = ts_ms;
        }
    }

    /// The carried sender when the last heard self-ID is still in-window.
    pub fn carried(&self, ts_ms: u64) -> Option<&str> {
        let cs = self.last_callsign.as_deref()?;
        if ts_ms.saturating_sub(self.last_ts_ms) <= self.window_ms {
            Some(cs)
        } else {
            None
        }
    }
}

/// Alert bit: the message addresses my configured callsign (R5).
pub const ALERT_ME: i32 = 1;
/// Alert bit: the message matches an emergency cue (R6).
pub const ALERT_EMERGENCY: i32 = 2;
/// Alert bit: the message matches a disaster-profile cue (Slice 3).
/// Set alone, never alongside ALERT_EMERGENCY (ADR-8): Normal vs
/// disaster classifications stay distinguishable and G5 is exact.
pub const ALERT_DISASTER: i32 = 4;

/// True when the transcript addresses my callsign — plain or spelled
/// (both resolve through the same extractor, so `VE2ABC` and `victor
/// echo two ...` match alike).
pub fn matches_me(transcript: &str, lang: &str, my_callsign: &str) -> bool {
    let mine = hamfeed_callsign::normalize(my_callsign);
    if mine.is_empty() {
        return false;
    }
    hamfeed_callsign::extract(transcript, lang)
        .iter()
        .any(|h| h.normalized == mine)
}

/// True when the transcript contains any emergency cue (accent-folded,
/// case-insensitive substring — cues are ordinary words, not tokens).
pub fn matches_emergency(transcript: &str, cues: &[String]) -> bool {
    let folded = hamfeed_callsign::fold(transcript);
    cues.iter().any(|c| {
        let cue = hamfeed_callsign::fold(c);
        !cue.trim().is_empty() && folded.contains(&cue)
    })
}

/// Enrich one transcript: heard self-ID wins; otherwise an in-window
/// carried sender attaches (marked `carried`); otherwise sender-less.
/// `carry` is the in-window callsign, if any — resolved by the caller via
/// [`CarryState::carried`] so this stays pure and unit-testable.
/// `lookup` maps a normalized callsign to an operator name (callbook);
/// a degraded callbook simply returns `None` and badges show the
/// callsign alone. `my_callsign`/`emergency_cues` raise the alert bits
/// on any message with a transcript (R5–R6).
pub fn enrich(
    transcript: &str,
    lang: &str,
    carry: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
    my_callsign: Option<&str>,
    emergency_cues: &[String],
) -> Enrichment {
    enrich_profiled(
        transcript,
        lang,
        carry,
        lookup,
        my_callsign,
        emergency_cues,
        &[],
    )
}

/// Profile-aware enrichment (Slice 3): `disaster_cues` is the active
/// profile's extra set (empty under Normal). A disaster hit sets ONLY
/// ALERT_DISASTER (ADR-8), so baseline callers see bit-identical output.
pub fn enrich_profiled(
    transcript: &str,
    lang: &str,
    carry: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
    my_callsign: Option<&str>,
    emergency_cues: &[String],
    disaster_cues: &[String],
) -> Enrichment {
    let mut alert = 0;
    if let Some(mine) = my_callsign {
        if matches_me(transcript, lang, mine) {
            alert |= ALERT_ME;
        }
    }
    if matches_emergency(transcript, emergency_cues) {
        alert |= ALERT_EMERGENCY;
    }
    if matches_emergency(transcript, disaster_cues) {
        alert |= ALERT_DISASTER;
    }
    if let Some(hit) = hamfeed_callsign::extract(transcript, lang).first() {
        let cs = hit.normalized.clone();
        return Enrichment {
            sender_callsign: Some(cs.clone()),
            sender_name: lookup(&cs),
            sender_source: "heard".into(),
            alert,
            speaker_key: None,
        };
    }
    if let Some(cs) = carry {
        return Enrichment {
            sender_callsign: Some(cs.to_string()),
            sender_name: lookup(cs),
            sender_source: "carried".into(),
            alert,
            speaker_key: None,
        };
    }
    let mut e = Enrichment::none();
    e.alert = alert;
    e
}

/// Language recorded when detection was skipped (failed/system rows).
pub const LANG_UNKNOWN: &str = "und";

/// Live monitor tap (Slice 3, R7): the `run_source` loop forks every
/// source PCM chunk here; `/api/live` readers follow from the tail.
/// Bounded (30 s at 16 kHz); overflow drops the oldest. `seq` counts every
/// sample ever pushed, so readers detect overwrite and skip ahead instead
/// of replaying stale audio.
pub const LIVE_TAP_CAP: usize = 16_000 * 30;

pub struct LiveTap {
    inner: Mutex<LiveTapInner>,
}

struct LiveTapInner {
    buf: VecDeque<i16>,
    seq: u64,
}

impl LiveTap {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(LiveTapInner {
                buf: VecDeque::new(),
                seq: 0,
            }),
        }
    }

    pub fn push(&self, pcm: &[i16]) {
        let mut g = self.inner.lock().expect("live tap mutex");
        for &s in pcm {
            if g.buf.len() == LIVE_TAP_CAP {
                g.buf.pop_front();
            }
            g.buf.push_back(s);
        }
        g.seq += pcm.len() as u64;
    }

    /// Samples pushed since `since`, plus the current sequence. When `since`
    /// predates the retained window the reader jumps to its oldest sample
    /// (live, not archive — T3 skip-ahead lives here, not in web).
    pub fn read_since(&self, since: u64) -> (u64, Vec<i16>) {
        let g = self.inner.lock().expect("live tap mutex");
        let oldest = g.seq - g.buf.len() as u64;
        let skip = since.saturating_sub(oldest) as usize;
        (g.seq, g.buf.iter().skip(skip).copied().collect())
    }

    pub fn current_seq(&self) -> u64 {
        self.inner.lock().expect("live tap mutex").seq
    }
}

impl Default for LiveTap {
    fn default() -> Self {
        Self::new()
    }
}

/// Live pipeline: config + store + transcriber + queue + spill.
pub struct Pipeline {
    cfg: Config,
    store: Store,
    transcriber: Transcriber,
    queue: IngestQueue,
    spill: SpillDir,
    storage_dir: PathBuf,
    drains: u64,
    carry: CarryState,
    callbook: Callbook,
    voice: VoiceState,
    tap: Arc<LiveTap>,
}

impl Pipeline {
    /// Open everything. A missing STT model fails LOUD with provisioning
    /// instructions (S11) — the binary exits non-zero, never loops silently.
    pub fn open(config_path: &Path) -> Result<Self> {
        let cfg = hamfeed_config::load(config_path)?;
        Self::open_with(cfg)
    }

    pub fn open_with(cfg: Config) -> Result<Self> {
        let transcriber = Transcriber::open(
            Path::new(&cfg.stt.model_path),
            &cfg.stt.lang_whitelist,
            cfg.stt.initial_prompt.as_deref(),
        )
        .map_err(|e| match e {
            SttErr::ModelMissing(hint) => anyhow::anyhow!("{hint}"),
            other => anyhow::anyhow!("stt backend: {other}"),
        })?;
        let store = Store::open(Path::new(&cfg.storage.db_path))?;
        let storage_dir = PathBuf::from(&cfg.storage.dir);
        std::fs::create_dir_all(&storage_dir)
            .with_context(|| format!("cannot create {}", storage_dir.display()))?;
        let spill = SpillDir::open(&storage_dir.join("spill"))?;
        let window_ms = cfg.identity.link_window_min * 60_000;
        let callbook = hamfeed_callbook::open(&cfg.callbook.resolved_db_path(&cfg.storage.dir));
        #[cfg(feature = "voice")]
        let embedder = match cfg.voiceprint.model_path.trim() {
            "" => {
                eprintln!(
                    "voice: no model configured ([voiceprint] model_path empty) — voiceprints off"
                );
                None
            }
            path => match Embedder::open(path) {
                Ok(e) => Some(e),
                Err(err) => {
                    eprintln!("voice: cannot load {path} ({err:?})");
                    eprintln!(
                        "{}",
                        hamfeed_config::missing_voice_model_hint(&cfg.voiceprint.model_path)
                    );
                    None
                }
            },
        };
        Ok(Self {
            cfg: cfg.clone(),
            store,
            transcriber,
            queue: IngestQueue::new(100),
            drains: 0,
            spill,
            storage_dir,
            carry: CarryState::new(window_ms),
            callbook,
            voice: VoiceState::new(
                cfg.voiceprint.threshold,
                #[cfg(feature = "voice")]
                embedder,
            ),
            tap: Arc::new(LiveTap::new()),
        })
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Live monitor tap (Slice 3): web `/api/live` readers share this.
    pub fn live_tap(&self) -> Arc<LiveTap> {
        Arc::clone(&self.tap)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn storage_dir(&self) -> &Path {
        &self.storage_dir
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Boot recovery: replay spilled rows into the queue, oldest first, and
    /// leave a system gap marker when anything was recovered (S9). Also
    /// purges expired voice aliases (R4 library bound).
    pub fn startup_recovery(&mut self) -> Result<usize> {
        let purged = self
            .store
            .purge_aliases(self.cfg.voiceprint.retention_days, now_ms())?;
        if purged > 0 {
            eprintln!("voice: purged {purged} expired voice alias(es)");
        }
        let items = self.spill.replay_items()?;
        let n = items.len();
        for item in items {
            self.queue.push(item, &self.spill)?;
        }
        if n > 0 {
            self.record_gap(&format!(
                "recovered {n} spilled clip(s) after restart; replayed oldest-first"
            ))?;
        }
        Ok(n)
    }

    /// Persist one closed segment's audio, then queue its metadata.
    /// The queue item exists only after the file has landed (S6).
    pub fn enqueue_segment(&mut self, seg: &Segment) -> Result<()> {
        let meta = hamfeed_ingest::write_clip_atomic(
            &self.storage_dir,
            &seg.id,
            seg.ts_start_ms,
            seg.duration_ms,
            &seg.pcm,
        )?;
        self.queue.push(
            QueueItem {
                id: meta.id,
                audio_path: meta.path.to_string_lossy().into_owned(),
                duration_ms: meta.duration_ms,
                ts_start_ms: meta.ts_start_ms,
                group_id: seg.group_id.clone(),
                seq: seg.seq,
            },
            &self.spill,
        )?;
        Ok(())
    }

    /// Drain the queue: transcribe each clip, store the row. Returns the
    /// number of clips processed. Spilled-then-replayed items are removed
    /// from the spill dir once stored.
    pub fn drain(&mut self) -> Result<usize> {
        self.drains += 1;
        let mut n = 0;
        while let Some(item) = self.queue.pop() {
            self.process_item(&item)?;
            let _ = self.spill.remove(&item.id);
            n += 1;
        }
        Ok(n)
    }

    /// How many times the queue was drained (live loop drains per segment).
    pub fn drain_count(&self) -> u64 {
        self.drains
    }

    /// Past-window voice suggestion (S7): a stored alias for this key at
    /// or above the suggestion confidence becomes a `suggested` sender,
    /// awaiting confirm/correct. Never auto-links — the operator decides.
    fn suggest_from_voice(&self, speaker_key: &str) -> Option<Enrichment> {
        let (callsign, conf) = self.store.get_alias(speaker_key).ok()??;
        if conf < self.cfg.voiceprint.suggest_min_conf {
            return None;
        }
        Some(Enrichment {
            sender_callsign: Some(callsign.clone()),
            sender_name: self
                .callbook
                .lookup(&callsign)
                .ok()
                .flatten()
                .map(|c| c.name),
            sender_source: "suggested".into(),
            alert: 0,
            speaker_key: Some(speaker_key.to_string()),
        })
    }

    /// Operator confirm/correct (S8): attach the verdict to the message
    /// and teach the voice library (`set_alias` at 1.0) when the message
    /// has a speaker key. Corrections are the same call with a different
    /// callsign — the library reassigns.
    pub fn confirm_sender(&self, id: &str, callsign: &str) -> Result<()> {
        let cs = hamfeed_callsign::normalize(callsign);
        if !hamfeed_callsign::is_valid(&cs) {
            anyhow::bail!("confirm_sender: {callsign} is not a callsign");
        }
        let name = self.callbook.lookup(&cs).ok().flatten().map(|c| c.name);
        self.store
            .set_sender(id, Some(&cs), name.as_deref(), "confirmed")?;
        let msg: Message = self
            .store
            .get(id)?
            .with_context(|| format!("confirm_sender: no such message {id}"))?;
        if let Some(key) = msg.speaker_key {
            self.store.overwrite_alias(&key, &cs, 1.0, now_ms())?;
        }
        Ok(())
    }

    fn process_item(&mut self, item: &QueueItem) -> Result<()> {
        let msg = match self.transcriber.transcribe(Path::new(&item.audio_path)) {
            Ok(out) => {
                // Spoken phonetics collapse to letter groups ("alpha lima
                // lima oscar" -> "ALLO") so callsigns read and search as
                // written; ham shortcuts ("73", "QTH") pass through.
                let transcript = hamfeed_stt::normalize_phonetics(&out.transcript);
                // Voice key first: the alias link below needs it.
                self.voice.roll_window(&item.group_id);
                let speaker_key = self.voice.key_for(
                    &self.store,
                    &decode_for_voice(&item.audio_path),
                    item.duration_ms,
                    "ok",
                    &self.cfg.station.freq_label,
                    &day_of(item.ts_start_ms),
                    self.cfg.voiceprint.min_embed_s,
                    &item.id,
                );
                if let Some(k) = &speaker_key {
                    self.voice.group_keys.push((k.clone(), item.ts_start_ms));
                }
                // Active profile cues per segment: always fresh across the
                // web/pipeline handles with no cache to invalidate. A failed
                // read degrades to Normal (empty cues), never louder.
                let disaster_cues = self.store.active_cues().unwrap_or_default();
                let mut e = enrich_profiled(
                    &transcript,
                    &out.lang,
                    self.carry.carried(item.ts_start_ms),
                    &|cs| self.callbook.lookup(cs).ok().flatten().map(|c| c.name),
                    self.cfg.station.my_callsign.as_deref(),
                    &self.cfg.notify.emergency_cues,
                    &disaster_cues,
                );
                e.speaker_key = speaker_key.clone();
                // A heard self-ID refreshes the carry window (R2) and
                // teaches the voice library (link_alias writes alias rows
                // only — never sender_*).
                if e.sender_source == "heard" {
                    if let Some(cs) = &e.sender_callsign {
                        self.carry.observe(Some(cs.clone()), item.ts_start_ms);
                        link_alias(
                            &self.store,
                            &self.voice.group_keys,
                            item.ts_start_ms,
                            cs,
                            self.cfg.identity.link_window_min * 60_000,
                        );
                    }
                }
                // Carry expired and nothing heard: ask the voice library.
                if e.sender_source == "none" {
                    if let Some(k) = &speaker_key {
                        if let Some(sugg) = self.suggest_from_voice(k) {
                            e = sugg;
                        }
                    }
                }
                NewMessage {
                    id: item.id.clone(),
                    ts_start_ms: item.ts_start_ms,
                    ts_end_ms: item.ts_start_ms + item.duration_ms,
                    freq_label: self.cfg.station.freq_label.clone(),
                    lang: out.lang,
                    lang_conf: out.lang_conf,
                    transcript,
                    stt_conf: out.stt_conf,
                    conf_flag: if out.stt_conf >= CONF_OK { "ok" } else { "low" }.into(),
                    status: "ok".into(),
                    fail_reason: None,
                    audio_path: Some(item.audio_path.clone()),
                    duration_ms: Some(item.duration_ms),
                    size_bytes: file_size(&item.audio_path),
                    short_flag: item.duration_ms < SHORT_MS,
                    group_id: item.group_id.clone(),
                    seq: item.seq,
                    sender_callsign: e.sender_callsign,
                    sender_name: e.sender_name,
                    sender_source: e.sender_source,
                    alert: e.alert,
                    speaker_key: e.speaker_key,
                    // The pipeline never invents corrections (004).
                    corrected_text: None,
                }
            }
            // Failed rows skip enrichment: sender-less, alert 0 (S11).
            Err(e) => NewMessage {
                id: item.id.clone(),
                ts_start_ms: item.ts_start_ms,
                ts_end_ms: item.ts_start_ms + item.duration_ms,
                freq_label: self.cfg.station.freq_label.clone(),
                lang: LANG_UNKNOWN.into(),
                lang_conf: 0.0,
                transcript: String::new(),
                stt_conf: 0.0,
                conf_flag: "low".into(),
                status: "failed".into(),
                fail_reason: Some(e.to_string()),
                audio_path: Some(item.audio_path.clone()),
                duration_ms: Some(item.duration_ms),
                size_bytes: file_size(&item.audio_path),
                short_flag: item.duration_ms < SHORT_MS,
                group_id: item.group_id.clone(),
                seq: item.seq,
                sender_callsign: None,
                sender_name: None,
                sender_source: "none".into(),
                alert: 0,
                speaker_key: None,
                corrected_text: None,
            },
        };
        // Upsert: a retried clip already has its row (S5 retry path).
        self.store.upsert(&msg)?;
        Ok(())
    }

    /// Operator triage (S5). `Drop` carries its own `delete_audio` flag;
    /// callers pass `delete_audio_on_drop` from config when the operator did
    /// not choose. `Retry` re-queues the clip for another pass.
    ///
    /// Web helper: [`drop_with_config`] resolves the config default.
    pub fn set_triage(&mut self, id: &str, action: TriageAction) -> Result<()> {
        match &action {
            TriageAction::Drop { .. } => {
                self.store.set_triage(id, &action)?;
                // A retry may still be queued or draining: the row stays
                // dropped (upsert skips dropped rows) and the queue entry
                // goes away so it is never re-processed.
                self.queue.remove(id);
            }
            TriageAction::Retry => {
                self.store.set_triage(id, &TriageAction::Retry)?;
                let msg: Message = self
                    .store
                    .get(id)?
                    .with_context(|| format!("retry: no such message {id}"))?;
                let audio = msg
                    .audio_path
                    .with_context(|| format!("retry: {id} has no audio to re-transcribe"))?;
                if !Path::new(&audio).exists() {
                    anyhow::bail!("retry: audio missing for {id}");
                }
                self.queue.push(
                    QueueItem {
                        id: msg.id,
                        audio_path: audio,
                        duration_ms: msg.duration_ms.unwrap_or(0),
                        ts_start_ms: msg.ts_start_ms,
                        group_id: msg.group_id,
                        seq: msg.seq,
                    },
                    &self.spill,
                )?;
            }
            _ => {
                self.store.set_triage(id, &action)?;
            }
        }
        Ok(())
    }

    /// Record a system gap marker (S10/S9): notice text, no audio.
    pub fn record_gap(&self, notice: &str) -> Result<()> {
        let now = now_ms();
        self.store.insert(&NewMessage {
            id: uuid::Uuid::new_v4().to_string(),
            ts_start_ms: now,
            ts_end_ms: now,
            freq_label: self.cfg.station.freq_label.clone(),
            lang: LANG_UNKNOWN.into(),
            lang_conf: 0.0,
            transcript: format!("system notice: {notice}"),
            stt_conf: 0.0,
            conf_flag: "low".into(),
            status: "system".into(),
            fail_reason: None,
            audio_path: None,
            duration_ms: None,
            size_bytes: None,
            short_flag: false,
            group_id: uuid::Uuid::new_v4().to_string(),
            seq: 0,
            sender_callsign: None,
            sender_name: None,
            sender_source: "none".into(),
            alert: 0,
            speaker_key: None,
            corrected_text: None,
        })?;
        Ok(())
    }

    /// Newest-first feed slice for the web layer (thin read-through).
    pub fn latest(&self, limit: usize) -> Result<Vec<Message>> {
        Ok(self
            .store
            .search(&SearchQuery {
                limit,
                ..Default::default()
            })?
            .messages)
    }

    /// Feed one audio source through segment → clip → queue → store.
    /// Headless Slice-1 loop without web (T9 Done clause).
    pub fn run_source(
        &mut self,
        source: &mut dyn AudioSource,
        hang_ms: u64,
        max_s: u64,
        beep_split: bool,
        beep_min_ms: u64,
    ) -> Result<usize> {
        let mut seg = Segmenter::new(
            SegmenterConfig {
                hang_ms,
                max_segment_ms: max_s * 1000,
                beep_split,
                beep_min_ms,
                ..Default::default()
            },
            now_ms(),
        );
        let mut total = 0;
        let stream = source.stream();
        for frame in stream {
            self.tap.push(&frame.samples);
            for s in seg.push(&frame.samples) {
                self.enqueue_segment(&s)?;
                // Drain per segment: a live mic never ends its stream, so a
                // drain-only-at-end would buffer forever and store nothing.
                // STT latency only delays the feed; capture keeps buffering
                // frames in order behind it.
                total += self.drain()?;
            }
        }
        for s in seg.flush() {
            self.enqueue_segment(&s)?;
        }
        total += self.drain()?;
        Ok(total)
    }
}

/// `Drop` with the config default for `delete_audio` (web triage buttons).
pub fn drop_with_config(cfg: &Config, delete_audio: Option<bool>) -> TriageAction {
    TriageAction::Drop {
        delete_audio: delete_audio.unwrap_or(cfg.storage.delete_audio_on_drop),
    }
}

fn file_size(path: &str) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// Decode a stored clip for voice embedding. Failure yields empty PCM,
/// which the speaker gate refuses (keyless, never an error).
fn decode_for_voice(path: &str) -> Vec<i16> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    hamfeed_ingest::decode_ogg_to_pcm(&bytes).unwrap_or_default()
}

pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hamfeed_ingest::fixture;
    use hamfeed_source::FakeSource;

    fn test_config(dir: &Path, model: &Path) -> Config {
        let text = format!(
            r#"
[audio]
device = "default"
sample_rate = 16000
[vad]
engine = "energy"
hang_ms = 400
[vad.profiles.repeater]
hang_ms = 400
[vad.profiles.simplex]
hang_ms = 600
[segment]
max_s = 120
[stt]
model_path = "{}"
lang_whitelist = ["fr", "en"]
[storage]
dir = "{}"
db_path = "{}"
retention_days = 90
delete_audio_on_drop = false
[station]
freq_label = "TEST"
"#,
            model.display(),
            dir.join("audio").display(),
            dir.join("test.db").display()
        );
        hamfeed_config::parse(&text).expect("test config parses")
    }

    fn test_model() -> PathBuf {
        if let Ok(p) = std::env::var("HAMFEED_TEST_MODEL") {
            return PathBuf::from(p);
        }
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ggml-tiny.bin")
    }

    fn test_dir(name: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let p =
            std::env::temp_dir().join(format!("hamfeed-pipe-{name}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn failed_item(dir: &Path, id: &str) -> QueueItem {
        // Corrupt bytes persisted as a clip stand in for an undecodable
        // transmission; the file exists, decoding fails (S5).
        let path = dir.join(format!("{id}.ogg"));
        std::fs::write(&path, b"corrupt clip bytes").unwrap();
        QueueItem {
            id: id.into(),
            audio_path: path.to_string_lossy().into_owned(),
            duration_ms: 1000,
            ts_start_ms: 1000,
            group_id: "g".into(),
            seq: 0,
        }
    }

    #[test]
    fn undecodable_clip_triage() {
        let dir = test_dir("triage");
        let mut pipe = Pipeline::open_with(test_config(&dir, &test_model())).unwrap();
        pipe.startup_recovery().unwrap();

        // Corrupt clip → failed row with error text.
        let item = failed_item(&dir, "bad1");
        pipe.process_item(&item).unwrap();
        let row = pipe.store.get("bad1").unwrap().expect("row stored");
        assert_eq!(row.status, "failed");
        assert!(row.fail_reason.is_some(), "error must show");
        assert!(row.audio_path.is_some(), "clip must be kept");

        // All four triage transitions work from failed.
        pipe.set_triage(
            "bad1",
            TriageAction::Flag {
                reason: Some("robot".into()),
            },
        )
        .unwrap();
        assert_eq!(
            pipe.store.get("bad1").unwrap().unwrap().review_flag,
            "flagged-training"
        );

        pipe.set_triage("bad1", TriageAction::Keep).unwrap();
        let row = pipe.store.get("bad1").unwrap().unwrap();
        assert_eq!(row.status, "ok");
        assert_eq!(row.review_flag, "keep");

        // Back to failed, then retry re-queues the clip.
        pipe.store.set_triage("bad1", &TriageAction::Retry).unwrap();
        pipe.set_triage("bad1", TriageAction::Retry).unwrap();
        assert_eq!(pipe.queue_len(), 1, "retry must re-queue");

        // Drop removes it from the feed (audio kept: config default).
        pipe.set_triage(
            "bad1",
            TriageAction::Drop {
                delete_audio: false,
            },
        )
        .unwrap();
        let row = pipe.store.get("bad1").unwrap().unwrap();
        assert_eq!(row.status, "dropped");
        assert!(Path::new(row.audio_path.as_ref().unwrap()).exists());

        // Drop with delete removes the file too.
        let doomed = row.audio_path.clone().unwrap();
        pipe.set_triage("bad1", TriageAction::Drop { delete_audio: true })
            .unwrap();
        let row = pipe.store.get("bad1").unwrap().unwrap();
        assert_eq!(row.status, "dropped");
        assert!(row.audio_path.is_none());
        assert!(!Path::new(&doomed).exists(), "audio file must be gone");
    }

    #[test]
    fn file_before_row() {
        // Crash-inject: the clip file lands before any row references it; a
        // leftover .tmp is cleared by the janitor, never referenced (S6).
        let dir = test_dir("atomic");
        let pipe = Pipeline::open_with(test_config(&dir, &test_model())).unwrap();
        let pcm = fixture::tone_ms(440.0, 500, 9_000);
        let meta = hamfeed_ingest::write_clip_atomic(&dir.join("audio"), "crash1", 2000, 500, &pcm)
            .unwrap();
        assert!(meta.path.exists(), "file must land first");
        let page = pipe
            .store
            .search(&SearchQuery {
                limit: 100,
                ..Default::default()
            })
            .unwrap();
        assert!(
            !page.messages.iter().any(|m| m.id == "crash1"),
            "no row may reference a clip that was never inserted"
        );
        drop(pipe);

        // Stale .tmp from the interrupted write clears via janitor.
        let stale = dir.join("audio/2026/01/02/030405006-deadbeef.ogg.tmp");
        std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
        std::fs::write(&stale, b"partial").unwrap();
        let n = Store::tmp_janitor(&dir.join("audio"), 0, 0).unwrap();
        assert_eq!(n, 1);
        assert!(!stale.exists());
    }

    #[test]
    fn missing_model_refuses() {
        // Empty model dir → loud startup error with provisioning text (S11).
        let dir = test_dir("nomodel");
        let err = match Pipeline::open_with(test_config(&dir, &dir.join("models/ggml-tiny.bin"))) {
            Ok(_) => panic!("must refuse a missing model"),
            Err(e) => e,
        };
        let msg = format!("{err:?}");
        assert!(
            msg.contains("scripts/download-model.sh"),
            "must name the helper: {msg}"
        );
        assert!(msg.contains("models"), "must name the drop-in: {msg}");
    }

    #[test]
    fn live_loop_drains_per_segment() {
        // Regression: a never-ending mic stream must still store rows as
        // segments close — draining only at end-of-stream would buffer
        // forever. Three bursts must trigger at least three drains.
        let dir = test_dir("livedrain");
        let mut pipe = Pipeline::open_with(test_config(&dir, &test_model())).unwrap();
        let mut pcm = Vec::new();
        for _ in 0..3 {
            pcm.extend(fixture::speech_like(2));
            pcm.extend(fixture::silence_ms(1200));
        }
        let frames = fixture::chunks(&pcm, 1600);
        let mut src = FakeSource::once(
            frames
                .into_iter()
                .map(|c| hamfeed_source::PcmFrame { samples: c })
                .collect(),
        );
        let n = pipe.run_source(&mut src, 400, 120, true, 150).unwrap();
        assert!(n >= 3, "three bursts, got {n} segments");
        assert!(
            pipe.drain_count() >= 3,
            "must drain per segment, drained {}",
            pipe.drain_count()
        );
    }

    #[test]
    fn fake_source_run_lands_rows() {
        // Synthetic FakeSource transmission through the whole headless loop
        // lands rows whose audio files exist (T9 Done clause).
        let dir = test_dir("fakerun");
        let mut pipe = Pipeline::open_with(test_config(&dir, &test_model())).unwrap();
        let mut pcm = fixture::speech_like(6);
        pcm.extend(fixture::silence_ms(1200));
        let frames = fixture::chunks(&pcm, 1600);
        let mut src = FakeSource::once(
            frames
                .into_iter()
                .map(|c| hamfeed_source::PcmFrame { samples: c })
                .collect(),
        );
        let n = pipe.run_source(&mut src, 400, 120, true, 150).unwrap();
        assert!(n >= 1, "at least one segment must land, got {n}");
        let page = pipe
            .store
            .search(&SearchQuery {
                limit: 100,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page.messages.len(), n);
        for m in &page.messages {
            let p = m.audio_path.as_ref().expect("clip kept");
            assert!(Path::new(p).exists(), "missing {p}");
        }
    }

    #[test]
    fn heard_attaches() {
        let e = enrich(
            "ici VE2DEM vous m'entendez",
            "fr",
            None,
            &|_| None,
            None,
            &[],
        );
        assert_eq!(e.sender_callsign.as_deref(), Some("VE2DEM"));
        assert_eq!(e.sender_source, "heard");
        // Degraded callbook: callsign alone. Alerts land in T9, keys in T8.
        assert_eq!(e.sender_name, None);
        assert_eq!(e.alert, 0);
        assert_eq!(e.speaker_key, None);
    }

    #[test]
    fn name_attaches_when_known() {
        let lookup = |cs: &str| (cs == "VE2DEM").then(|| "Jean Tremblay".to_string());
        let e = enrich("ici VE2DEM", "fr", None, &lookup, None, &[]);
        assert_eq!(e.sender_callsign.as_deref(), Some("VE2DEM"));
        assert_eq!(e.sender_name.as_deref(), Some("Jean Tremblay"));
        // Carried senders get names too.
        let e = enrich("ok merci", "fr", Some("VE2DEM"), &lookup, None, &[]);
        assert_eq!(e.sender_source, "carried");
        assert_eq!(e.sender_name.as_deref(), Some("Jean Tremblay"));
        // Unknown callsign: badge shows the callsign alone.
        let e = enrich("ici VE9ZZZ", "fr", None, &lookup, None, &[]);
        assert_eq!(e.sender_name, None);
    }

    fn cues() -> Vec<String> {
        vec!["mayday".into(), "urgence".into(), "detresse".into()]
    }

    #[test]
    fn me_plain_notifies() {
        let e = enrich(
            "VE2ABC à vous, ici VE2DEM",
            "fr",
            None,
            &|_| None,
            Some("VE2ABC"),
            &cues(),
        );
        assert_eq!(e.alert & ALERT_ME, ALERT_ME);
        assert_eq!(e.alert & ALERT_EMERGENCY, 0);
        // Unconfigured: silence.
        let e = enrich("VE2ABC à vous", "fr", None, &|_| None, None, &cues());
        assert_eq!(e.alert, 0);
    }

    #[test]
    fn me_spelled_notifies() {
        let e = enrich(
            "victor echo two alpha bravo charlie, m'entendez-vous",
            "fr",
            None,
            &|_| None,
            Some("ve2abc"),
            &cues(),
        );
        assert_eq!(e.alert & ALERT_ME, ALERT_ME);
    }

    #[test]
    fn emergency_fr_notifies() {
        let e = enrich(
            "appel d'urgence, je répète, urgence",
            "fr",
            None,
            &|_| None,
            None,
            &cues(),
        );
        assert_eq!(e.alert & ALERT_EMERGENCY, ALERT_EMERGENCY);
        // Accent-folded: "détresse" matches the plain cue.
        let e = enrich(
            "ici VE2DEM en détresse",
            "fr",
            None,
            &|_| None,
            None,
            &cues(),
        );
        assert_eq!(e.alert & ALERT_EMERGENCY, ALERT_EMERGENCY);
        assert_eq!(e.sender_callsign.as_deref(), Some("VE2DEM"));
    }

    #[test]
    fn no_cue_no_alert() {
        let e = enrich(
            "bonjour à tous, bonne soirée",
            "fr",
            None,
            &|_| None,
            Some("VE9ZZ"),
            &cues(),
        );
        assert_eq!(e.alert, 0);
        assert_eq!(e.sender_source, "none");
    }

    #[test]
    fn carry_marks_carried() {
        let mut carry = CarryState::new(30 * 60_000);
        carry.observe(Some("VE2DEM".into()), 1_000);
        // Follow-up 5 min later, no self-ID: carried.
        let e = enrich(
            "ok compris merci",
            "fr",
            carry.carried(301_000),
            &|_| None,
            None,
            &[],
        );
        assert_eq!(e.sender_callsign.as_deref(), Some("VE2DEM"));
        assert_eq!(e.sender_source, "carried");
        // A heard hit beats carry even when carry is fresh.
        let e = enrich(
            "ici VE3MA",
            "fr",
            carry.carried(302_000),
            &|_| None,
            None,
            &[],
        );
        assert_eq!(e.sender_callsign.as_deref(), Some("VE3MA"));
        assert_eq!(e.sender_source, "heard");
    }

    #[test]
    fn expired_window_stays_senderless() {
        let mut carry = CarryState::new(30 * 60_000);
        carry.observe(Some("VE2DEM".into()), 1_000);
        // 31 min later: window expired, no silent link (S4).
        assert_eq!(carry.carried(1_861_000), None);
        let e = enrich(
            "ok compris merci",
            "fr",
            carry.carried(1_861_000),
            &|_| None,
            None,
            &[],
        );
        assert_eq!(e.sender_callsign, None);
        assert_eq!(e.sender_source, "none");
        // Nothing heard yet at all: also sender-less.
        let fresh = CarryState::new(30 * 60_000);
        assert_eq!(fresh.carried(999_999), None);
    }

    #[cfg(feature = "voice")]
    fn test_voice_state() -> VoiceState {
        let emb = std::env::var("HAMFEED_TEST_VOICE_MODEL")
            .ok()
            .and_then(|p| Embedder::open(&p).ok());
        VoiceState::new(0.55, emb)
    }

    #[cfg(not(feature = "voice"))]
    fn test_voice_state() -> VoiceState {
        VoiceState::new(0.55)
    }

    fn new_msg(id: &str) -> NewMessage {
        NewMessage {
            id: id.into(),
            ts_start_ms: 1000,
            ts_end_ms: 1500,
            freq_label: "TEST".into(),
            lang: "fr".into(),
            lang_conf: 0.9,
            transcript: "bonjour".into(),
            stt_conf: 0.8,
            conf_flag: "ok".into(),
            status: "ok".into(),
            fail_reason: None,
            audio_path: None,
            duration_ms: Some(500),
            size_bytes: None,
            short_flag: false,
            group_id: "g".into(),
            seq: 0,
            sender_callsign: None,
            sender_name: None,
            sender_source: "none".into(),
            alert: 0,
            speaker_key: None,
            corrected_text: None,
        }
    }

    #[test]
    fn short_blip_stays_keyless() {
        // Under the minimum embed duration: no key with or without a model.
        let store = Store::open_memory().unwrap();
        let mut v = test_voice_state();
        assert_eq!(
            v.key_for(
                &store,
                &vec![0i16; 32000],
                500,
                "ok",
                "TEST",
                "1",
                1.5,
                "s1"
            ),
            None
        );
        // Failed rows never key either.
        assert_eq!(
            v.key_for(
                &store,
                &vec![0i16; 96000],
                6000,
                "failed",
                "TEST",
                "1",
                1.5,
                "s2"
            ),
            None
        );
    }

    #[test]
    fn link_alias_writes_decayed_conf() {
        let store = Store::open_memory().unwrap();
        link_alias(
            &store,
            &[
                ("k-near".into(), 2000),
                ("k-far".into(), 1000),
                ("k-out".into(), 0),
            ],
            2000,
            "VE2DEM",
            1000,
        );
        // dt=0 → 1.0; dt=1000 at the window edge → 0.5; dt=2000 → skipped.
        assert_eq!(
            store.get_alias("k-near").unwrap(),
            Some(("VE2DEM".to_string(), 1.0))
        );
        assert_eq!(
            store.get_alias("k-far").unwrap(),
            Some(("VE2DEM".to_string(), 0.5))
        );
        assert_eq!(store.get_alias("k-out").unwrap(), None);
    }

    #[test]
    fn expired_carry_suggests_from_alias() {
        let dir = test_dir("suggest");
        let pipe = Pipeline::open_with(test_config(&dir, &test_model())).unwrap();
        // Confident alias → suggestion (never a silent auto-link: the
        // source says suggested, awaiting confirm).
        pipe.store.set_alias("K1", "VE2DEM", 0.9, 1000).unwrap();
        let e = pipe.suggest_from_voice("K1").expect("suggestion");
        assert_eq!(e.sender_callsign.as_deref(), Some("VE2DEM"));
        assert_eq!(e.sender_source, "suggested");
        assert_eq!(e.speaker_key.as_deref(), Some("K1"));
        // Below the suggestion confidence: silence.
        pipe.store.set_alias("K2", "VE3MA", 0.3, 1000).unwrap();
        assert!(pipe.suggest_from_voice("K2").is_none());
        // Unknown key: silence.
        assert!(pipe.suggest_from_voice("K9").is_none());
    }

    #[test]
    fn confirm_refreshes_alias() {
        let dir = test_dir("confirm");
        let pipe = Pipeline::open_with(test_config(&dir, &test_model())).unwrap();
        let mut m = new_msg("v1");
        m.speaker_key = Some("K1".into());
        pipe.store.insert(&m).unwrap();
        pipe.confirm_sender("v1", "ve2dem").unwrap();
        let row = pipe.store.get("v1").unwrap().unwrap();
        assert_eq!(row.sender_callsign.as_deref(), Some("VE2DEM"));
        assert_eq!(row.sender_source, "confirmed");
        assert_eq!(
            pipe.store.get_alias("K1").unwrap(),
            Some(("VE2DEM".to_string(), 1.0))
        );
        // Correct (same call, different callsign): library reassigns.
        pipe.confirm_sender("v1", "VE3MA").unwrap();
        assert_eq!(
            pipe.store.get_alias("K1").unwrap(),
            Some(("VE3MA".to_string(), 1.0))
        );
        // Not a callsign, unknown id: loud errors.
        assert!(pipe.confirm_sender("v1", "XYZ").is_err());
        assert!(pipe.confirm_sender("nope", "VE2DEM").is_err());
    }

    /// Slice-2 stream proof (T11): a synthetic transcript stream through
    /// the real enrich → carry → suggest → confirm → store path lands
    /// linked senders (STT decode and HTTP are proven by their own suites;
    /// this drives everything between).
    #[test]
    fn slice2_stream_links_senders() {
        let dir = test_dir("slice2");
        let pipe = Pipeline::open_with(test_config(&dir, &test_model())).unwrap();
        let lookup = &|_: &str| None;
        let cues = vec!["mayday".to_string()];
        let window = pipe.cfg.identity.link_window_min * 60_000;
        let mut carry = CarryState::new(window);

        // m1: heard self-ID.
        let t0 = 1_000_000u64;
        let e1 = enrich("ici VE2DEM, à vous", "fr", None, lookup, None, &cues);
        assert_eq!(e1.sender_source, "heard");
        carry.observe(e1.sender_callsign.clone(), t0);
        let mut m1 = new_msg("m1");
        m1.ts_start_ms = t0;
        m1.sender_callsign = e1.sender_callsign;
        m1.sender_source = e1.sender_source;
        pipe.store.upsert(&m1).unwrap();

        // m2 five minutes later, no self-ID: carried.
        let t2 = t0 + 5 * 60_000;
        let e2 = enrich(
            "ok, compris, merci",
            "fr",
            carry.carried(t2),
            lookup,
            None,
            &cues,
        );
        assert_eq!(e2.sender_source, "carried");
        let mut m2 = new_msg("m2");
        m2.ts_start_ms = t2;
        m2.sender_callsign = e2.sender_callsign;
        m2.sender_source = e2.sender_source;
        pipe.store.upsert(&m2).unwrap();

        // m3 past the window: sender-less.
        let t3 = t0 + 40 * 60_000;
        let e3 = enrich("toujours là?", "fr", carry.carried(t3), lookup, None, &cues);
        assert_eq!(e3.sender_source, "none");
        let mut m3 = new_msg("m3");
        m3.ts_start_ms = t3;
        pipe.store.upsert(&m3).unwrap();

        // m4: a taught voice returns past the window → suggested, then
        // the operator confirms it.
        pipe.store.set_alias("KX", "VE2DEM", 0.9, t3).unwrap();
        let e4 = pipe.suggest_from_voice("KX").expect("suggestion");
        let mut m4 = new_msg("m4");
        m4.ts_start_ms = t3 + 1000;
        m4.sender_callsign = e4.sender_callsign;
        m4.sender_name = e4.sender_name;
        m4.sender_source = e4.sender_source;
        m4.speaker_key = e4.speaker_key;
        pipe.store.upsert(&m4).unwrap();
        pipe.confirm_sender("m4", "VE2DEM").unwrap();

        // m5: emergency cue raises the bit on a sender-less message.
        let e5 = enrich(
            "mayday, en panne sur la 117",
            "fr",
            None,
            lookup,
            None,
            &cues,
        );
        assert_eq!(e5.alert & ALERT_EMERGENCY, ALERT_EMERGENCY);
        let mut m5 = new_msg("m5");
        m5.ts_start_ms = t3 + 2000;
        m5.alert = e5.alert;
        pipe.store.upsert(&m5).unwrap();

        // Final ledger.
        assert_eq!(
            pipe.store.get("m1").unwrap().unwrap().sender_source,
            "heard"
        );
        assert_eq!(
            pipe.store.get("m2").unwrap().unwrap().sender_source,
            "carried"
        );
        assert_eq!(pipe.store.get("m3").unwrap().unwrap().sender_callsign, None);
        assert_eq!(
            pipe.store.get("m4").unwrap().unwrap().sender_source,
            "confirmed"
        );
        assert_eq!(
            pipe.store.get("m5").unwrap().unwrap().alert,
            ALERT_EMERGENCY
        );
        let page = pipe
            .store
            .search(&SearchQuery {
                sender: Some("VE2DEM".into()),
                limit: 20,
                ..Default::default()
            })
            .unwrap();
        let mut ids: Vec<&str> = page.messages.iter().map(|m| m.id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, vec!["m1", "m2", "m4"]);
    }

    #[test]
    fn day_of_counts_epoch_days() {
        assert_eq!(day_of(0), "0");
        assert_eq!(day_of(86_400_000), "1");
        assert_eq!(day_of(86_400_001), "1");
    }

    #[test]
    fn failed_row_skips_enrich() {
        // Corrupt clip → failed row stays sender-less with alert 0 (S11).
        let dir = test_dir("noenrich");
        let mut pipe = Pipeline::open_with(test_config(&dir, &test_model())).unwrap();
        let item = failed_item(&dir, "bad2");
        pipe.process_item(&item).unwrap();
        let row = pipe.store.get("bad2").unwrap().expect("row stored");
        assert_eq!(row.status, "failed");
        assert_eq!(row.sender_callsign, None);
        assert_eq!(row.sender_source, "none");
        assert_eq!(row.alert, 0);
    }

    #[test]
    fn disaster_cue_sets_bit4_alone() {
        // G2: a disaster-only cue under a disaster profile flags bit 4
        // without touching the Normal emergency bit.
        let disaster = vec!["net control".to_string()];
        let e = enrich_profiled(
            "net control, go ahead with traffic",
            "en",
            None,
            &|_| None,
            None,
            &[],
            &disaster,
        );
        assert_eq!(e.alert & ALERT_DISASTER, ALERT_DISASTER);
        assert_eq!(e.alert & ALERT_EMERGENCY, 0);
    }

    #[test]
    fn baseline_enrich_ignores_disaster_only_cue() {
        // G1: the baseline path never sees disaster cues, so a
        // disaster-only phrase stays silent.
        let e = enrich(
            "net control, go ahead with traffic",
            "en",
            None,
            &|_| None,
            None,
            &[],
        );
        assert_eq!(e.alert, 0);
    }

    #[test]
    fn profiled_enrich_matches_baseline_under_normal() {
        // G5: empty disaster cues (Normal) classify bit-identically to the
        // Slice-2 baseline across sender, emergency, and quiet cases.
        let cases = [
            "ici VE2DEM, a vous",
            "mayday mayday, engine fire",
            "ok merci, a plus tard",
        ];
        for tx in cases {
            let a = enrich(tx, "fr", None, &|_| None, None, &cues());
            let b = enrich_profiled(tx, "fr", None, &|_| None, None, &cues(), &[]);
            assert_eq!(a, b, "divergence on {tx:?}");
        }
    }

    #[test]
    fn live_tap_caps_and_skips_ahead() {
        let tap = LiveTap::new();
        tap.push(&vec![7i16; LIVE_TAP_CAP + 100]);
        // A reader from the start jumps to the retained window (skip-ahead).
        let (seq, first) = tap.read_since(0);
        assert_eq!(seq as usize, LIVE_TAP_CAP + 100);
        assert_eq!(first.len(), LIVE_TAP_CAP);
        // A caught-up reader gets only what is new.
        let (seq2, second) = tap.read_since(seq);
        assert!(second.is_empty());
        assert_eq!(seq2, seq);
        tap.push(&[1, 2, 3]);
        let (seq3, third) = tap.read_since(seq2);
        assert_eq!(third, vec![1, 2, 3]);
        assert_eq!(seq3, seq2 + 3);
    }
}
