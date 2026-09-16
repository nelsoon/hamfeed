//! hamfeed-pipeline: ingest → stt → store orchestration (T9).
//!
//! Owns the queue drain, status mapping (`ok|failed|system|dropped`), triage
//! transitions, system gap markers, and the loud missing-model startup check
//! (S11). Group contract: web may call the `enrich`/triage helpers here, but
//! pipeline never depends on web.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hamfeed_config::Config;
use hamfeed_ingest::{IngestQueue, QueueItem, Segment, Segmenter, SegmenterConfig, SpillDir};
use hamfeed_source::AudioSource;
use hamfeed_store::{Message, NewMessage, SearchQuery, Store, TriageAction, SHORT_MS};
use hamfeed_stt::{SttErr, Transcriber};

/// `stt_conf` at or above this marks a row `ok`, below marks `low`.
pub const CONF_OK: f64 = 0.6;

/// Language recorded when detection was skipped (failed/system rows).
pub const LANG_UNKNOWN: &str = "und";

/// Live pipeline: config + store + transcriber + queue + spill.
pub struct Pipeline {
    cfg: Config,
    store: Store,
    transcriber: Transcriber,
    queue: IngestQueue,
    spill: SpillDir,
    storage_dir: PathBuf,
    drains: u64,
}

impl Pipeline {
    /// Open everything. A missing STT model fails LOUD with provisioning
    /// instructions (S11) — the binary exits non-zero, never loops silently.
    pub fn open(config_path: &Path) -> Result<Self> {
        let cfg = hamfeed_config::load(config_path)?;
        Self::open_with(cfg)
    }

    pub fn open_with(cfg: Config) -> Result<Self> {
        let transcriber =
            Transcriber::open(Path::new(&cfg.stt.model_path), &cfg.stt.lang_whitelist).map_err(
                |e| match e {
                    SttErr::ModelMissing(hint) => anyhow::anyhow!("{hint}"),
                    other => anyhow::anyhow!("stt backend: {other}"),
                },
            )?;
        let store = Store::open(Path::new(&cfg.storage.db_path))?;
        let storage_dir = PathBuf::from(&cfg.storage.dir);
        std::fs::create_dir_all(&storage_dir)
            .with_context(|| format!("cannot create {}", storage_dir.display()))?;
        let spill = SpillDir::open(&storage_dir.join("spill"))?;
        Ok(Self {
            cfg,
            store,
            transcriber,
            queue: IngestQueue::new(100),
            drains: 0,
            spill,
            storage_dir,
        })
    }

    pub fn store(&self) -> &Store {
        &self.store
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
    /// leave a system gap marker when anything was recovered (S9).
    pub fn startup_recovery(&mut self) -> Result<usize> {
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

    fn process_item(&self, item: &QueueItem) -> Result<()> {
        let msg = match self.transcriber.transcribe(Path::new(&item.audio_path)) {
            Ok(out) => NewMessage {
                id: item.id.clone(),
                ts_start_ms: item.ts_start_ms,
                ts_end_ms: item.ts_start_ms + item.duration_ms,
                freq_label: self.cfg.station.freq_label.clone(),
                lang: out.lang,
                lang_conf: out.lang_conf,
                // Spoken phonetics collapse to letter groups ("alpha lima
                // lima oscar" -> "ALLO") so callsigns read and search as
                // written; ham shortcuts ("73", "QTH") pass through.
                transcript: hamfeed_stt::normalize_phonetics(&out.transcript),
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
            },
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
}
