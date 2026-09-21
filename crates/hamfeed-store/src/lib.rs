//! hamfeed-store: SQLite + FTS5 message archive (T7).
//!
//! Owns all SQL (group contract): schema from the plan data model, indices,
//! FTS5 over transcripts, insert / triage transitions / filtered search /
//! retention sweep / `.tmp` janitor / missing-audio audit. No other crate
//! imports rusqlite.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};

/// Seconds of audio below which a row counts as noise (`short_flag`).
pub const SHORT_MS: u64 = 1500;

/// Whisper non-speech tags (`[BLANK_AUDIO]`, `[BELL RINGING]`, …) carry
/// no words; sign-off boilerplate carries no loop-usable information.
/// Filler glue (`you` in "thank you", articles) doesn't count as
/// content either. Anything else — names, places, reports, NATO words,
/// spelled-out or digit callsigns — is content and keeps the row.
const NOISE_FILLER: &[&str] = &[
    "you", "tu", "toi", "vous", "a", "à", "la", "le", "les", "the", "et", "and", "un", "une",
    "des", "du", "au", "ok",
];
const NOISE_BOILER: &[&str] = &[
    "thank",
    "thanks",
    "merci",
    "bye",
    "goodbye",
    "goodnight",
    "night",
    "ciao",
    "73",
    "73s",
    "over",
    "prochaine",
    "qsl",
    "soir",
    "soiree",
    "soirée",
];

/// True when a transcript holds no speech content: only bracketed
/// non-speech tags, sign-off boilerplate, filler, and punctuation.
/// Conservative by construction — a single content word keeps the row,
/// so weak-signal fragments and hallucinations with word-shape stay
/// visible (indistinguishable from real speech, honestly shown).
pub fn transcript_is_noise_text(text: &str) -> bool {
    let mut stripped = String::with_capacity(text.len());
    let mut depth = 0u32;
    for c in text.chars() {
        if c == '[' {
            depth += 1;
        } else if c == ']' {
            depth = depth.saturating_sub(1);
        } else if depth == 0 {
            stripped.push(c);
        }
    }
    let mut any_content = false;
    let mut any_token = false;
    for tok in stripped.split(|c: char| !c.is_alphanumeric()) {
        if tok.is_empty() {
            continue;
        }
        any_token = true;
        let w = tok.to_lowercase();
        if NOISE_FILLER.contains(&w.as_str()) || NOISE_BOILER.contains(&w.as_str()) {
            continue;
        }
        any_content = true;
        break;
    }
    !any_token || !any_content
}

/// A stored message row (plan data model, Slice 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub id: String,
    pub ts_start_ms: u64,
    pub ts_end_ms: u64,
    pub freq_label: String,
    pub lang: String,
    pub lang_conf: f64,
    pub transcript: String,
    pub stt_conf: f64,
    pub conf_flag: String,
    pub status: String,
    pub fail_reason: Option<String>,
    pub audio_path: Option<String>,
    pub audio_purged: bool,
    pub duration_ms: Option<u64>,
    pub size_bytes: Option<u64>,
    pub short_flag: bool,
    pub group_id: String,
    pub seq: u32,
    pub review_flag: String,
    pub flag_reason: Option<String>,
    /// Heard/calling-station callsign, if any (Slice 2).
    pub sender_callsign: Option<String>,
    /// Operator name from the local callbook, if known.
    pub sender_name: Option<String>,
    /// heard|carried|suggested|confirmed|none (Slice 2).
    pub sender_source: String,
    /// Bitmask: 1 = for-you, 2 = emergency (Slice 2).
    pub alert: i32,
    /// Per-segment voice key (`Unknown-N|label|day`), if any (Slice 2).
    pub speaker_key: Option<String>,
    /// Operator correction of the transcript, if any (004). The model's
    /// original stays in `transcript`; this is the label, never a wipe.
    pub corrected_text: Option<String>,
}

/// Insert shape: everything the pipeline knows after STT (or failure).
#[derive(Debug, Clone)]
pub struct NewMessage {
    pub id: String,
    pub ts_start_ms: u64,
    pub ts_end_ms: u64,
    pub freq_label: String,
    pub lang: String,
    pub lang_conf: f64,
    pub transcript: String,
    pub stt_conf: f64,
    pub conf_flag: String,
    pub status: String,
    pub fail_reason: Option<String>,
    pub audio_path: Option<String>,
    pub duration_ms: Option<u64>,
    pub size_bytes: Option<u64>,
    pub short_flag: bool,
    pub group_id: String,
    pub seq: u32,
    pub sender_callsign: Option<String>,
    pub sender_name: Option<String>,
    pub sender_source: String,
    pub alert: i32,
    pub speaker_key: Option<String>,
    pub corrected_text: Option<String>,
}

/// One training-export pair (004): true text + original + clip path.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportRow {
    pub id: String,
    pub lang: String,
    pub original: String,
    pub corrected: String,
    pub audio_path: String,
}

/// Operator triage actions (S5; transitions owned by pipeline, applied here).
#[derive(Debug, Clone)]
pub enum TriageAction {
    /// Accept the clip as-is: failed → ok, kept for the record.
    Keep,
    /// Drop from the feed; optionally delete the audio file too.
    Drop { delete_audio: bool },
    /// Ask for another transcription pass: stays failed, flagged for retry.
    Retry,
    /// Keep for training data review.
    Flag { reason: Option<String> },
}

/// Filtered history query (S8 + Slice 2 sender filter).
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    pub text: Option<String>,
    pub from_ms: Option<u64>,
    pub to_ms: Option<u64>,
    pub hide_noise: bool,
    pub status: Option<String>,
    /// Exact normalized-callsign match (Slice 2). `None` = no sender filter.
    pub sender: Option<String>,
    pub limit: usize,
    pub cursor: Option<String>,
}

/// One search page plus the cursor for the next page ("" = end).
#[derive(Debug)]
pub struct SearchPage {
    pub messages: Vec<Message>,
    pub next_cursor: String,
}

/// Thread-safe handle (web serves concurrent readers).
pub struct Store {
    conn: Mutex<Connection>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS messages(
  id TEXT PRIMARY KEY,
  ts_start INT NOT NULL, ts_end INT NOT NULL,
  freq_label TEXT NOT NULL, lang TEXT NOT NULL, lang_conf REAL NOT NULL,
  transcript TEXT NOT NULL, stt_conf REAL NOT NULL,
  conf_flag TEXT NOT NULL, status TEXT NOT NULL,
  fail_reason TEXT NULL, audio_path TEXT NULL,
  audio_purged BOOL NOT NULL DEFAULT 0,
  duration_ms INT NULL, size_bytes INT NULL,
  short_flag BOOL NOT NULL DEFAULT 0,
  group_id TEXT NOT NULL, seq INT NOT NULL,
  review_flag TEXT NOT NULL DEFAULT 'none',
  flag_reason TEXT NULL,
  sender_callsign TEXT NULL, sender_name TEXT NULL,
  sender_source TEXT NOT NULL DEFAULT 'none',
  alert INT NOT NULL DEFAULT 0,
  speaker_key TEXT NULL,
  corrected_text TEXT NULL
);
CREATE TABLE IF NOT EXISTS speaker_alias(
  key TEXT PRIMARY KEY, callsign TEXT NOT NULL,
  confidence REAL NOT NULL, updated_ts INT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles(name TEXT PRIMARY KEY, cues TEXT NOT NULL DEFAULT '[]');
-- Seeded idempotently: fresh files get them from this batch, pre-existing
-- files (whose profiles table is just created above) from the same lines.
INSERT OR IGNORE INTO profiles(name, cues) VALUES('Normal', '[]');
INSERT OR IGNORE INTO profiles(name, cues) VALUES('ARES Net',
  '[\"emergency traffic\", \"tactical\", \"net control\", \"priority traffic\", \"ares\"]');
INSERT OR IGNORE INTO profiles(name, cues) VALUES('Severe Weather',
  '[\"tornado\", \"flood\", \"spotter\", \"severe thunderstorm\", \"flash flood\"]');
INSERT OR IGNORE INTO settings(key, value) VALUES('active_profile', 'Normal');
CREATE INDEX IF NOT EXISTS idx_ts ON messages(ts_start);
CREATE INDEX IF NOT EXISTS idx_sender ON messages(sender_callsign);
CREATE INDEX IF NOT EXISTS idx_alert ON messages(alert);
CREATE INDEX IF NOT EXISTS idx_lang ON messages(lang);
CREATE INDEX IF NOT EXISTS idx_status ON messages(status);
CREATE INDEX IF NOT EXISTS idx_group ON messages(group_id, seq);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
  USING fts5(transcript, corrected_text, content='messages', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, transcript, corrected_text)
    VALUES (new.rowid, new.transcript, new.corrected_text);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, transcript, corrected_text)
    VALUES ('delete', old.rowid, old.transcript, old.corrected_text);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF transcript, corrected_text ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, transcript, corrected_text)
    VALUES ('delete', old.rowid, old.transcript, old.corrected_text);
  INSERT INTO messages_fts(rowid, transcript, corrected_text)
    VALUES (new.rowid, new.transcript, new.corrected_text);
END;
";

/// Escape user input for a LIKE pattern: an unescaped `%` matches
/// everything, `_` matches any char, and a bare `\` escapes the next
/// char by accident. Used with `ESCAPE '\'`.
fn like_escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Disaster profile: name + extra emergency cue phrases (Slice 3).
/// `Normal` carries no extra cues (baseline behavior unchanged).
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    pub name: String,
    pub cues: Vec<String>,
}

/// `set_active_profile` rejection: the name matches no seeded profile.
/// Web maps this to 404 (verdict pattern); anything else is 500.
#[derive(Debug)]
pub struct UnknownProfile(pub String);

impl std::fmt::Display for UnknownProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown profile: {}", self.0)
    }
}

impl std::error::Error for UnknownProfile {}

impl Store {
    fn init(conn: &Connection) -> Result<()> {
        // Query-time text gate for hide-noise (no schema change, so old
        // databases gain it on open): Rust tokenizer power inside SQL,
        // registered on every connection including :memory: test DBs.
        conn.create_scalar_function(
            "hamfeed_noise_text",
            1,
            rusqlite::functions::FunctionFlags::SQLITE_UTF8
                | rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
            |ctx| {
                let t: Option<String> = ctx.get(0)?;
                Ok(transcript_is_noise_text(t.as_deref().unwrap_or("")))
            },
        )
        .context("cannot register hamfeed_noise_text")?;
        // Migrate BEFORE the schema batch: old files lack the newer
        // columns, and the new indices fail to create without them.
        // (On a fresh file the table is absent and migrate is a no-op.)
        let rebuild_fts = Self::migrate(conn)?;
        conn.execute_batch(SCHEMA).context("cannot create schema")?;
        if rebuild_fts {
            // The FTS index gained a column mid-life: reindex once so old
            // rows are searchable on both texts.
            conn.execute_batch("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
                .context("cannot rebuild search index")?;
        }
        Ok(())
    }

    /// Idempotent migration over older databases: adds each missing
    /// `messages` column (old rows read sender-less / uncorrected).
    /// Returns whether the FTS index needs a one-shot rebuild (its column
    /// set changed, so pre-existing rows must be reindexed).
    fn migrate(conn: &Connection) -> Result<bool> {
        let table: Option<String> = conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        if table.is_none() {
            return Ok(false); // Fresh file: the schema batch creates everything.
        }
        let mut rebuild_fts = false;
        let cols: Vec<String> = conn
            .prepare("PRAGMA table_info(messages)")?
            .query_map([], |r| r.get::<_, String>(1))?
            .collect::<std::result::Result<_, _>>()?;
        for (col, ddl) in [
            (
                "sender_callsign",
                "ALTER TABLE messages ADD COLUMN sender_callsign TEXT",
            ),
            (
                "sender_name",
                "ALTER TABLE messages ADD COLUMN sender_name TEXT",
            ),
            (
                "sender_source",
                "ALTER TABLE messages ADD COLUMN sender_source TEXT NOT NULL DEFAULT 'none'",
            ),
            (
                "alert",
                "ALTER TABLE messages ADD COLUMN alert INT NOT NULL DEFAULT 0",
            ),
            (
                "speaker_key",
                "ALTER TABLE messages ADD COLUMN speaker_key TEXT",
            ),
            (
                "corrected_text",
                "ALTER TABLE messages ADD COLUMN corrected_text TEXT",
            ),
            // Very early Slice-1 files predate grouping and triage.
            (
                "group_id",
                "ALTER TABLE messages ADD COLUMN group_id TEXT NOT NULL DEFAULT ''",
            ),
            (
                "seq",
                "ALTER TABLE messages ADD COLUMN seq INT NOT NULL DEFAULT 0",
            ),
            (
                "review_flag",
                "ALTER TABLE messages ADD COLUMN review_flag TEXT NOT NULL DEFAULT 'none'",
            ),
            (
                "flag_reason",
                "ALTER TABLE messages ADD COLUMN flag_reason TEXT",
            ),
        ] {
            if !cols.iter().any(|c| c == col) {
                conn.execute_batch(ddl)
                    .with_context(|| format!("cannot migrate column {col}"))?;
            }
        }
        // FTS index: virtual tables cannot be ALTERed, so an older
        // single-column index is dropped outright (its content lives in
        // `messages`; the schema batch recreates the two-column index and
        // current triggers via IF NOT EXISTS) and flagged for a one-shot
        // rebuild so old rows index on both texts. Files without any FTS
        // table skip this entirely.
        let fts: Option<String> = conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        if fts.is_some() {
            let fts_cols: Vec<String> = conn
                .prepare("PRAGMA table_info(messages_fts)")?
                .query_map([], |r| r.get::<_, String>(1))?
                .collect::<std::result::Result<_, _>>()?;
            if !fts_cols.iter().any(|c| c == "corrected_text") {
                conn.execute_batch(
                    "DROP TABLE messages_fts;
                     DROP TRIGGER IF EXISTS messages_ai;
                     DROP TRIGGER IF EXISTS messages_ad;
                     DROP TRIGGER IF EXISTS messages_au;",
                )
                .context("cannot drop stale search index")?;
                rebuild_fts = true;
            }
        }
        // The two-column index shipped once with single-column delete legs:
        // a partial-column 'delete' leaves ghost tokens for the omitted
        // column, so refresh those triggers in place. Indexed content is
        // unaffected (no rebuild), only future deletes/updates.
        for trig in ["messages_ad", "messages_au"] {
            let sql: Option<String> = conn
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
                    [trig],
                    |r| r.get(0),
                )
                .optional()?;
            if sql.is_some_and(|s| !s.contains("corrected_text")) {
                conn.execute_batch(
                    "DROP TRIGGER IF EXISTS messages_ai;
                     DROP TRIGGER IF EXISTS messages_ad;
                     DROP TRIGGER IF EXISTS messages_au;",
                )
                .context("cannot refresh search triggers")?;
                break;
            }
        }
        Ok(rebuild_fts)
    }

    /// Open (or create) a file-backed database.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("cannot create {}", parent.display()))?;
            }
        }
        let conn =
            Connection::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        Self::init(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// In-memory database (tests, e2e harness).
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("cannot open :memory:")?;
        Self::init(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn insert(&self, m: &NewMessage) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        conn.execute(
            "INSERT INTO messages(id,ts_start,ts_end,freq_label,lang,lang_conf,
              transcript,stt_conf,conf_flag,status,fail_reason,audio_path,
              duration_ms,size_bytes,short_flag,group_id,seq,
              sender_callsign,sender_name,sender_source,alert,speaker_key,
              corrected_text)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rusqlite::params![
                m.id,
                m.ts_start_ms as i64,
                m.ts_end_ms as i64,
                m.freq_label,
                m.lang,
                m.lang_conf,
                m.transcript,
                m.stt_conf,
                m.conf_flag,
                m.status,
                m.fail_reason,
                m.audio_path,
                m.duration_ms.map(|v| v as i64),
                m.size_bytes.map(|v| v as i64),
                m.short_flag,
                m.group_id,
                m.seq as i64,
                m.sender_callsign,
                m.sender_name,
                m.sender_source,
                m.alert,
                m.speaker_key,
                m.corrected_text,
            ],
        )
        .with_context(|| format!("cannot insert {}", m.id))?;
        Ok(())
    }

    /// Insert, or refresh the transcribed columns when the id already exists
    /// (retry path). `review_flag`/`flag_reason` survive: operator triage is
    /// never wiped by a re-transcription. A `confirmed` sender likewise
    /// survives a sender-less re-transcription (the operator's verdict wins
    /// over a fresh STT pass that heard no self-ID).
    ///
    /// Deliberately NOT refreshed: `group_id`/`seq`. Segment identity is
    /// assigned once by the ingest segmenter (the only allocator) and is
    /// immutable afterwards — a re-transcription must never resequence a
    /// clip into another group, so the upsert leaves both columns alone
    /// and a retry that finds `status = 'dropped'` reports Ok silently.
    pub fn upsert(&self, m: &NewMessage) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        conn.execute(
            "INSERT INTO messages(id,ts_start,ts_end,freq_label,lang,lang_conf,
              transcript,stt_conf,conf_flag,status,fail_reason,audio_path,
              duration_ms,size_bytes,short_flag,group_id,seq,
              sender_callsign,sender_name,sender_source,alert,speaker_key,
              corrected_text)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
             ON CONFLICT(id) DO UPDATE SET
              ts_end=excluded.ts_end, lang=excluded.lang,
              lang_conf=excluded.lang_conf, transcript=excluded.transcript,
              stt_conf=excluded.stt_conf, conf_flag=excluded.conf_flag,
              status=excluded.status, fail_reason=excluded.fail_reason,
              audio_path=excluded.audio_path,
              duration_ms=excluded.duration_ms,
              size_bytes=excluded.size_bytes,
              short_flag=excluded.short_flag,
              sender_callsign=CASE WHEN messages.sender_source='confirmed'
                AND excluded.sender_source='none'
                THEN messages.sender_callsign ELSE excluded.sender_callsign END,
              sender_name=CASE WHEN messages.sender_source='confirmed'
                AND excluded.sender_source='none'
                THEN messages.sender_name ELSE excluded.sender_name END,
              sender_source=CASE WHEN messages.sender_source='confirmed'
                AND excluded.sender_source='none'
                THEN messages.sender_source ELSE excluded.sender_source END,
              alert=excluded.alert, speaker_key=excluded.speaker_key,
              corrected_text=COALESCE(messages.corrected_text, excluded.corrected_text)
             WHERE messages.status <> 'dropped'",
            rusqlite::params![
                m.id,
                m.ts_start_ms as i64,
                m.ts_end_ms as i64,
                m.freq_label,
                m.lang,
                m.lang_conf,
                m.transcript,
                m.stt_conf,
                m.conf_flag,
                m.status,
                m.fail_reason,
                m.audio_path,
                m.duration_ms.map(|v| v as i64),
                m.size_bytes.map(|v| v as i64),
                m.short_flag,
                m.group_id,
                m.seq as i64,
                m.sender_callsign,
                m.sender_name,
                m.sender_source,
                m.alert,
                m.speaker_key,
                m.corrected_text,
            ],
        )
        .with_context(|| format!("cannot upsert {}", m.id))?;
        Ok(())
    }

    /// Total row count (status endpoint; search limits do not apply).
    pub fn count(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("store mutex");
        conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get::<_, i64>(0))
            .map(|n| n as usize)
            .context("cannot count messages")
    }

    pub fn get(&self, id: &str) -> Result<Option<Message>> {
        let conn = self.conn.lock().expect("store mutex");
        conn.query_row("SELECT * FROM messages WHERE id = ?", [id], row_to_msg)
            .optional()
            .context("cannot fetch message")
    }

    /// Apply one triage transition (pipeline owns the decision, store the SQL).
    pub fn set_triage(&self, id: &str, action: &TriageAction) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        match action {
            TriageAction::Keep => {
                conn.execute(
                    "UPDATE messages SET status='ok', review_flag='keep',
                     fail_reason=NULL WHERE id=?",
                    [id],
                )?;
            }
            TriageAction::Drop { delete_audio } => {
                if *delete_audio {
                    let path: Option<String> = conn
                        .query_row("SELECT audio_path FROM messages WHERE id=?", [id], |r| {
                            r.get(0)
                        })
                        .optional()?
                        .flatten();
                    if let Some(p) = path {
                        let _ = std::fs::remove_file(&p);
                    }
                    conn.execute(
                        "UPDATE messages SET status='dropped', audio_purged=1,
                         audio_path=NULL WHERE id=?",
                        [id],
                    )?;
                } else {
                    conn.execute("UPDATE messages SET status='dropped' WHERE id=?", [id])?;
                }
            }
            TriageAction::Retry => {
                conn.execute(
                    "UPDATE messages SET status='failed',
                     fail_reason='retry requested by operator',
                     review_flag='none' WHERE id=?",
                    [id],
                )?;
            }
            TriageAction::Flag { reason } => {
                conn.execute(
                    "UPDATE messages SET review_flag='flagged-training',
                     flag_reason=? WHERE id=?",
                    rusqlite::params![reason, id],
                )?;
            }
        }
        if conn.changes() == 0 {
            anyhow::bail!("triage: no such message {id}");
        }
        Ok(())
    }

    /// Operator transcript correction (004): attach the true text to a
    /// message. `None` (or blank/whitespace-only input) clears it back to
    /// NULL. The model's `transcript` is never touched.
    pub fn set_correction(&self, id: &str, text: Option<&str>) -> Result<()> {
        let cleaned: Option<String> = text.map(|t| t.trim().to_string()).filter(|t| !t.is_empty());
        let conn = self.conn.lock().expect("store mutex");
        conn.execute(
            "UPDATE messages SET corrected_text=? WHERE id=?",
            rusqlite::params![cleaned, id],
        )?;
        if conn.changes() == 0 {
            anyhow::bail!("set_correction: no such message {id}");
        }
        Ok(())
    }

    /// Training-export rows (004): corrected messages that still have a
    /// clip path, oldest first. Audio-less rows are excluded — a pair
    /// needs both halves (unreadable files are skipped at zip time).
    pub fn corrections_export(&self) -> Result<Vec<ExportRow>> {
        let conn = self.conn.lock().expect("store mutex");
        let rows = conn
            .prepare(
                "SELECT id, lang, transcript, corrected_text, audio_path
                 FROM messages
                 WHERE corrected_text IS NOT NULL AND audio_path IS NOT NULL
                 ORDER BY ts_start ASC",
            )?
            .query_map([], |r| {
                Ok(ExportRow {
                    id: r.get(0)?,
                    lang: r.get(1)?,
                    original: r.get(2)?,
                    corrected: r.get(3)?,
                    audio_path: r.get(4)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("cannot read corrections")?;
        Ok(rows)
    }

    /// Operator confirm/correct (Slice 2): attach a sender verdict to a
    /// message. Corrections are the same call with a different callsign.
    pub fn set_sender(
        &self,
        id: &str,
        callsign: Option<&str>,
        name: Option<&str>,
        source: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        conn.execute(
            "UPDATE messages SET sender_callsign=?, sender_name=?,
             sender_source=? WHERE id=?",
            rusqlite::params![callsign, name, source, id],
        )?;
        if conn.changes() == 0 {
            anyhow::bail!("set_sender: no such message {id}");
        }
        Ok(())
    }

    /// Record a voice→callsign link (Slice 2): upsert the alias only when
    /// the new confidence beats the stored one — the library learns, but a
    /// weak re-hearing never downgrades a strong verdict.
    pub fn set_alias(
        &self,
        key: &str,
        callsign: &str,
        confidence: f32,
        updated_ts_ms: u64,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        conn.execute(
            "INSERT INTO speaker_alias(key,callsign,confidence,updated_ts)
             VALUES(?,?,?,?)
             ON CONFLICT(key) DO UPDATE SET
              callsign=excluded.callsign, confidence=excluded.confidence,
              updated_ts=excluded.updated_ts
             WHERE excluded.confidence > speaker_alias.confidence",
            rusqlite::params![key, callsign, confidence, updated_ts_ms as i64],
        )
        .with_context(|| format!("cannot set alias {key}"))?;
        Ok(())
    }

    /// Overwrite a voice→callsign link unconditionally (operator
    /// confirm/correct only — an explicit verdict is authoritative, unlike
    /// a passive re-hearing, which stays behind the `set_alias` guard).
    pub fn overwrite_alias(
        &self,
        key: &str,
        callsign: &str,
        confidence: f32,
        updated_ts_ms: u64,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        conn.execute(
            "INSERT INTO speaker_alias(key,callsign,confidence,updated_ts)
             VALUES(?,?,?,?)
             ON CONFLICT(key) DO UPDATE SET
              callsign=excluded.callsign, confidence=excluded.confidence,
              updated_ts=excluded.updated_ts",
            rusqlite::params![key, callsign, confidence, updated_ts_ms as i64],
        )
        .with_context(|| format!("cannot overwrite alias {key}"))?;
        Ok(())
    }

    /// Look up a voice key's linked callsign, if any.
    pub fn get_alias(&self, key: &str) -> Result<Option<(String, f32)>> {
        let conn = self.conn.lock().expect("store mutex");
        conn.query_row(
            "SELECT callsign, confidence FROM speaker_alias WHERE key=?",
            [key],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .context("cannot read alias")
    }

    /// Allocate the next per-day speaker number for a label
    /// (`speaker_seq_{label}_{day}`); never reused within the store.
    pub fn alloc_speaker_n(&self, label: &str, day: &str) -> Result<u32> {
        let conn = self.conn.lock().expect("store mutex");
        let key = format!("speaker_seq_{label}_{day}");
        let cur: Option<String> = conn
            .query_row("SELECT value FROM settings WHERE key=?", [&key], |r| {
                r.get(0)
            })
            .optional()?;
        let next: u32 = match cur {
            Some(v) => v
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("bad speaker_seq for {key}: {v}"))?
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("speaker_seq overflow for {key}"))?,
            None => 1,
        };
        conn.execute(
            "INSERT INTO settings(key,value) VALUES(?,?)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            rusqlite::params![key, next.to_string()],
        )?;
        Ok(next)
    }

    /// Parse the JSON cue list; a corrupt row fails loudly rather than
    /// silently narrowing disaster detection.
    fn parse_cues(name: &str, raw: &str) -> Result<Vec<String>> {
        serde_json::from_str(raw)
            .with_context(|| format!("bad cues JSON for profile {name}"))
            .and_then(|v: serde_json::Value| {
                v.as_array()
                    .context(format!("cues for profile {name} are not an array"))
                    .map(|a| {
                        a.iter()
                            .filter_map(|c| c.as_str().map(str::to_string))
                            .collect()
                    })
            })
    }

    /// All profiles in seed order. Seeded idempotently at open (T1).
    pub fn list_profiles(&self) -> Result<Vec<Profile>> {
        let conn = self.conn.lock().expect("store mutex");
        let mut stmt = conn.prepare("SELECT name, cues FROM profiles ORDER BY rowid")?;
        let rows = stmt.query_map([], |r| {
            let name: String = r.get(0)?;
            let cues: String = r.get(1)?;
            Ok((name, cues))
        })?;
        rows.map(|r| {
            let (name, cues) = r?;
            Ok(Profile {
                cues: Self::parse_cues(&name, &cues)?,
                name,
            })
        })
        .collect()
    }

    /// Active profile name; always present (seeded `'Normal'` at open).
    pub fn active_profile(&self) -> Result<String> {
        let conn = self.conn.lock().expect("store mutex");
        conn.query_row(
            "SELECT value FROM settings WHERE key='active_profile'",
            [],
            |r| r.get(0),
        )
        .context("active_profile missing")
    }

    /// Extra emergency cues of the active disaster profile, in one round
    /// trip (Slice 3, T2). Empty under Normal. The live loop reads this per
    /// segment so a profile switch applies without restart or cache flush.
    pub fn active_cues(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().expect("store mutex");
        let (name, raw): (String, String) = conn.query_row(
            "SELECT p.name, p.cues FROM profiles p
             JOIN settings s ON s.key='active_profile' AND s.value=p.name",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        Self::parse_cues(&name, &raw)
    }

    /// Switch the active profile (new traffic only; history untouched, R5).
    /// Unknown names are [`UnknownProfile`], never a silent no-op.
    pub fn set_active_profile(&self, name: &str) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM profiles WHERE name=?)",
            [name],
            |r| r.get(0),
        )?;
        if !exists {
            return Err(UnknownProfile(name.to_string()).into());
        }
        conn.execute(
            "INSERT INTO settings(key,value) VALUES('active_profile',?)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [name],
        )?;
        Ok(())
    }

    /// Transcripts occurring more than once (canned repeats). The
    /// hide-noise dup rule and the UI noise badge share this set; NULL
    /// transcripts are excluded (they never match `IN`, and wordless
    /// rows are already covered by the text gate).
    pub fn duplicate_transcripts(&self) -> Result<std::collections::HashSet<String>> {
        let conn = self.conn.lock().expect("store mutex");
        let mut stmt = conn.prepare(
            "SELECT transcript FROM messages WHERE transcript IS NOT NULL \
             GROUP BY transcript HAVING COUNT(*) > 1",
        )?;
        let mut out = std::collections::HashSet::new();
        for t in stmt.query_map([], |r| r.get::<_, String>(0))? {
            out.insert(t?);
        }
        Ok(out)
    }

    /// Drop voice aliases last touched before the retention bound (Slice 2
    /// library purge). Returns removed row count.
    pub fn purge_aliases(&self, retention_days: u64, now_ms: u64) -> Result<usize> {
        let cutoff = now_ms.saturating_sub(retention_days * 86_400_000) as i64;
        let conn = self.conn.lock().expect("store mutex");
        let n = conn.execute("DELETE FROM speaker_alias WHERE updated_ts < ?", [cutoff])?;
        Ok(n)
    }

    /// Mark a message's audio as purged (retention path).
    pub fn mark_audio_purged(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        conn.execute(
            "UPDATE messages SET audio_purged=1, audio_path=NULL WHERE id=?",
            [id],
        )?;
        Ok(())
    }

    /// Newest-first filtered search with keyset pagination (S8).
    /// `cursor` is `"<ts_start>:<id>"`; empty next cursor ends paging.
    pub fn search(&self, q: &SearchQuery) -> Result<SearchPage> {
        let conn = self.conn.lock().expect("store mutex");
        let limit = q.limit.clamp(1, 200) as i64;
        let (cts, cid) = parse_cursor(q.cursor.as_deref());

        // FTS prefilter: rowids matching text (or all rows when no text).
        // A broken MATCH expression falls back to LIKE, never to an error.
        let fts_ids: Option<Vec<i64>> = match q.text.as_deref() {
            Some(t) if !t.trim().is_empty() => Some(self.fts_match(&conn, t)?),
            _ => None,
        };

        let mut sql = String::from("SELECT * FROM messages WHERE 1=1");
        if fts_ids.is_some() {
            sql.push_str(" AND rowid IN (SELECT value FROM json_each(?))");
        }
        if q.hide_noise {
            // Heard senders and failed rows are always traffic, however
            // short: a 1.2 s self-ID is loop-valuable, and error cards
            // are honest states (the pipeline marks failures short, so
            // the exemption must be explicit here, not just below).
            sql.push_str(
                " AND (short_flag = 0 \
                 OR sender_source = 'heard' OR status = 'failed')",
            );
            // Noise hide, query-time (nothing deleted, uncheck reveals):
            // non-heard, non-failed rows whose transcript holds no speech
            // content — wordless shorts, whisper bracket tags
            // (`[BLANK_AUDIO]`, `[BELL RINGING]`), sign-off boilerplate —
            // or verbatim repeats beyond de-duplication (canned network
            // IDs; every copy hides, the text stays searchable). A heard
            // self-ID always keeps its row; failed rows stay visible as
            // honest errors; hallucinations with word-shape stay shown
            // (indistinguishable from weak speech).
            sql.push_str(
                " AND NOT (sender_source != 'heard' AND status != 'failed' \
                 AND (hamfeed_noise_text(transcript) \
                 OR transcript IN (SELECT transcript FROM messages \
                 GROUP BY transcript HAVING COUNT(*) > 1)))",
            );
        }
        // Exact callsign match (Slice 2): callsigns are compact tokens, so
        // `=` is precise where FTS/LIKE would false-positive (VE2DE ⊂ VE2DEM).
        let sender_norm: Option<String> = q.sender.as_deref().map(|s| {
            s.to_ascii_uppercase()
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .collect()
        });
        if sender_norm.is_some() {
            sql.push_str(" AND sender_callsign = ?");
        }
        // Allowlist + bound param: user input never reaches the SQL text
        // (quote-stripping alone admits `OR`-style smuggling past naïve
        // filters). Unknown statuses match nothing (fail closed).
        if let Some(st) = &q.status {
            match st.as_str() {
                "ok" | "failed" | "kept" | "dropped" => {
                    sql.push_str(" AND status = ?");
                }
                _ => {
                    sql.push_str(" AND 1 = 0");
                }
            }
        }
        if q.from_ms.is_some() {
            sql.push_str(" AND ts_start >= ?");
        }
        if q.to_ms.is_some() {
            sql.push_str(" AND ts_start <= ?");
        }
        if q.cursor.is_some() {
            sql.push_str(" AND (ts_start < ? OR (ts_start = ? AND id < ?))");
        }
        sql.push_str(" ORDER BY ts_start DESC, id DESC LIMIT ?");

        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(ids) = &fts_ids {
            params.push(Box::new(format!(
                "[{}]",
                ids.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )));
        }
        if let Some(s) = &sender_norm {
            params.push(Box::new(s.clone()));
        }
        // Mirrors the allowlist above: only known statuses bind a param.
        if let Some(st) = &q.status {
            if matches!(st.as_str(), "ok" | "failed" | "kept" | "dropped") {
                params.push(Box::new(st.clone()));
            }
        }
        if let Some(f) = q.from_ms {
            params.push(Box::new(f as i64));
        }
        if let Some(t) = q.to_ms {
            params.push(Box::new(t as i64));
        }
        if q.cursor.is_some() {
            params.push(Box::new(cts));
            params.push(Box::new(cts));
            params.push(Box::new(cid.clone()));
        }
        params.push(Box::new(limit + 1));
        let refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref()).collect();

        let mut stmt = conn.prepare(&sql).context("cannot prepare search")?;
        let rows = stmt
            .query_map(refs.as_slice(), row_to_msg)
            .context("cannot run search")?
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("cannot read search rows")?;

        let next_cursor = if rows.len() as i64 > limit {
            let last = &rows[limit as usize - 1];
            format!("{}:{}", last.ts_start_ms, last.id)
        } else {
            String::new()
        };
        Ok(SearchPage {
            messages: rows.into_iter().take(limit as usize).collect(),
            next_cursor,
        })
    }

    fn fts_match(&self, conn: &Connection, text: &str) -> Result<Vec<i64>> {
        let try_match = |expr: &str| -> Result<Vec<i64>> {
            let mut stmt =
                conn.prepare("SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?")?;
            let ids = stmt
                .query_map([expr], |r| r.get::<_, i64>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(ids)
        };
        // Phrase-quote each token so FTS5 special characters cannot break
        // the query; on any remaining error fall back to LIKE.
        let quoted = text
            .split_whitespace()
            .map(|t| format!("\"{}\"", t.replace('"', "")))
            .collect::<Vec<_>>()
            .join(" ");
        match try_match(&quoted) {
            Ok(ids) => Ok(ids),
            Err(_) => {
                let mut stmt = conn.prepare(
                    "SELECT rowid FROM messages WHERE transcript LIKE ?1 ESCAPE '\\'
                     OR corrected_text LIKE ?1 ESCAPE '\\'",
                )?;
                let like = format!("%{}%", like_escape(text));
                let ids = stmt
                    .query_map([like], |r| r.get::<_, i64>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(ids)
            }
        }
    }

    /// Purge expired audio per policy (S10): rows older than
    /// `retention_days` with status ok/failed lose their file and move to
    /// `dropped` (dropped-purged), exempt from the missing-audio audit.
    /// Returns purged row count.
    pub fn retention_sweep(&self, retention_days: u64, now_ms: u64) -> Result<usize> {
        let cutoff = now_ms.saturating_sub(retention_days * 86_400_000) as i64;
        let conn = self.conn.lock().expect("store mutex");
        let stale: Vec<(String, Option<String>)> = conn
            .prepare(
                "SELECT id, audio_path FROM messages
                 WHERE ts_start < ? AND status IN ('ok','failed')",
            )?
            .query_map([cutoff], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<std::result::Result<_, _>>()?;
        let n = stale.len();
        for (id, path) in &stale {
            if let Some(p) = path {
                let _ = std::fs::remove_file(p);
            }
            conn.execute(
                "UPDATE messages SET status='dropped', audio_purged=1,
                 audio_path=NULL, fail_reason='retention purged' WHERE id=?",
                [id],
            )?;
        }
        Ok(n)
    }

    /// Delete `*.tmp` files older than `older_than_ms` under `dir` (S10).
    pub fn tmp_janitor(dir: &Path, older_than_ms: u64, now_ms: u64) -> Result<usize> {
        let mut cleared = 0;
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            let entries = match std::fs::read_dir(&d) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                if p.extension().is_some_and(|x| x == "tmp") {
                    let age = e
                        .metadata()
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(u64::MAX);
                    let _ = now_ms;
                    if age >= older_than_ms && std::fs::remove_file(&p).is_ok() {
                        cleared += 1;
                    }
                }
            }
        }
        Ok(cleared)
    }

    /// S10 audit: ok/failed rows whose audio is gone without a purge record.
    /// Dropped and system rows are exempt, as are purged ones.
    pub fn missing_audio_audit(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().expect("store mutex");
        let rows: Vec<(String, Option<String>, bool)> = conn
            .prepare(
                "SELECT id, audio_path, audio_purged FROM messages
                 WHERE status IN ('ok','failed') AND audio_purged = 0",
            )?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<std::result::Result<_, _>>()?;
        Ok(rows
            .into_iter()
            .filter(|(_, path, _)| path.as_deref().is_none_or(|p| !Path::new(p).exists()))
            .map(|(id, _, _)| id)
            .collect())
    }
}

fn parse_cursor(cursor: Option<&str>) -> (i64, String) {
    match cursor.and_then(|c| c.split_once(':')) {
        Some((ts, id)) => (ts.parse().unwrap_or(i64::MAX), id.into()),
        None => (i64::MAX, String::new()),
    }
}

fn row_to_msg(r: &rusqlite::Row) -> rusqlite::Result<Message> {
    Ok(Message {
        id: r.get("id")?,
        ts_start_ms: r.get::<_, i64>("ts_start")? as u64,
        ts_end_ms: r.get::<_, i64>("ts_end")? as u64,
        freq_label: r.get("freq_label")?,
        lang: r.get("lang")?,
        lang_conf: r.get("lang_conf")?,
        transcript: r.get("transcript")?,
        stt_conf: r.get("stt_conf")?,
        conf_flag: r.get("conf_flag")?,
        status: r.get("status")?,
        fail_reason: r.get("fail_reason")?,
        audio_path: r.get("audio_path")?,
        audio_purged: r.get("audio_purged")?,
        duration_ms: r.get::<_, Option<i64>>("duration_ms")?.map(|v| v as u64),
        size_bytes: r.get::<_, Option<i64>>("size_bytes")?.map(|v| v as u64),
        short_flag: r.get("short_flag")?,
        group_id: r.get("group_id")?,
        seq: r.get::<_, i64>("seq")? as u32,
        review_flag: r.get("review_flag")?,
        flag_reason: r.get("flag_reason")?,
        sender_callsign: r.get("sender_callsign")?,
        sender_name: r.get("sender_name")?,
        sender_source: r.get("sender_source")?,
        alert: r.get("alert")?,
        speaker_key: r.get("speaker_key")?,
        corrected_text: r.get("corrected_text")?,
    })
}

/// Seeded-DB helper shared by store tests and later web/e2e tests.
pub mod testutil {
    use super::*;

    pub fn seed_messages(store: &Store) -> Vec<String> {
        let rows = [
            (
                "m-fr-ok",
                9_000u64,
                "fr",
                "bonjour les amis",
                "ok",
                "ok",
                false,
            ),
            (
                "m-en-ok",
                8_000,
                "en",
                "hello radio world",
                "ok",
                "ok",
                false,
            ),
            ("m-fr-low", 7_000, "fr", "appel général", "low", "ok", false),
            ("m-fail", 6_000, "en", "", "ok", "failed", false),
            ("m-noise", 5_000, "en", "uh", "low", "ok", true),
            ("m-old", 1_000, "fr", "ancien message", "ok", "ok", false),
        ];
        let mut ids = Vec::new();
        for (id, ts, lang, text, conf, status, short) in rows {
            store
                .insert(&NewMessage {
                    id: id.into(),
                    ts_start_ms: ts,
                    ts_end_ms: ts + 500,
                    freq_label: "TEST".into(),
                    lang: lang.into(),
                    lang_conf: 0.9,
                    transcript: text.into(),
                    stt_conf: 0.8,
                    conf_flag: conf.into(),
                    status: status.into(),
                    fail_reason: if status == "failed" {
                        Some("decode error".into())
                    } else {
                        None
                    },
                    audio_path: Some(format!("/tmp/{id}.ogg")),
                    duration_ms: Some(500),
                    size_bytes: Some(100),
                    short_flag: short,
                    group_id: "g".into(),
                    seq: 0,
                    sender_callsign: None,
                    sender_name: None,
                    sender_source: "none".into(),
                    alert: 0,
                    speaker_key: None,
                    corrected_text: None,
                })
                .expect("seed insert");
            ids.push(id.to_string());
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::seed_messages;

    #[test]
    fn fts_from_to_hidenoise_pagination() {
        let store = Store::open_memory().unwrap();
        seed_messages(&store);

        // FTS text match.
        let page = store
            .search(&SearchQuery {
                text: Some("bonjour".into()),
                limit: 10,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page.messages.len(), 1);
        assert_eq!(page.messages[0].id, "m-fr-ok");

        // from/to window.
        let page = store
            .search(&SearchQuery {
                from_ms: Some(6_500),
                to_ms: Some(8_500),
                limit: 10,
                ..Default::default()
            })
            .unwrap();
        let ids: Vec<&str> = page.messages.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["m-en-ok", "m-fr-low"]);

        // hide-noise drops the short row.
        let page = store
            .search(&SearchQuery {
                hide_noise: true,
                limit: 10,
                ..Default::default()
            })
            .unwrap();
        assert!(page.messages.iter().all(|m| m.id != "m-noise"));

        // Pagination: 2 per page, newest-first, all six rows reachable.
        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let page = store
                .search(&SearchQuery {
                    limit: 2,
                    cursor: cursor.clone(),
                    ..Default::default()
                })
                .unwrap();
            if page.messages.is_empty() {
                break;
            }
            seen.extend(page.messages.iter().map(|m| m.id.clone()));
            if page.next_cursor.is_empty() {
                break;
            }
            cursor = Some(page.next_cursor);
        }
        assert_eq!(seen.len(), 6);
        assert_eq!(seen[0], "m-fr-ok");
    }

    #[test]
    fn noise_text_gate_cases() {
        // Bracket tags: whisper telling us there was no speech.
        for t in [
            "[BLANK_AUDIO]",
            "[Music] [BLANK_AUDIO]",
            ". [BELL RINGING].",
            "[BEEP]",
            "[inaudible]",
            "",
            ". . .",
        ] {
            assert!(transcript_is_noise_text(t), "{t:?} must read as noise");
        }
        // Sign-off boilerplate (FR+EN) with filler glue.
        for t in [
            "Bye. Thank you.",
            "Thank you. Thank you.",
            "Merci.",
            "73, à la prochaine, over.",
            "QSL, 73.",
            "ok.",
        ] {
            assert!(transcript_is_noise_text(t), "{t:?} must read as noise");
        }
        // A single content word keeps the row: names, places, reports,
        // digits, NATO/number words (spelled callsigns), weak fragments.
        for t in [
            "QTH Montréal, QTH Montréal, 74, à la prochaine, over.",
            "Vous êtes à l'écoute du réseau RTQ.",
            "Minimum Wake and Whitebird requested. [BLANK_AUDIO]",
            "ici VE2DEM",
            "victor echo deux",
            "V12CRS",
            "Yeah.",
            "il pleut",
            "Merci VE2ABC",
        ] {
            assert!(!transcript_is_noise_text(t), "{t:?} must stay visible");
        }
    }

    #[test]
    fn hide_noise_drops_wordless_shorts() {
        let store = Store::open_memory().unwrap();
        let put = |id: &str,
                   dur_ms: u64,
                   text: &str,
                   status: &str,
                   cs: Option<&str>,
                   src: &str,
                   short: bool| {
            store
                .insert(&NewMessage {
                    id: id.into(),
                    ts_start_ms: 1_000,
                    ts_end_ms: 1_000 + dur_ms,
                    freq_label: "TEST".into(),
                    lang: "fr".into(),
                    lang_conf: 0.9,
                    transcript: text.into(),
                    stt_conf: 0.8,
                    conf_flag: "ok".into(),
                    status: status.into(),
                    fail_reason: None,
                    audio_path: Some(format!("/tmp/{id}.ogg")),
                    duration_ms: Some(dur_ms),
                    size_bytes: Some(100),
                    short_flag: short,
                    group_id: "g".into(),
                    seq: 0,
                    sender_callsign: cs.map(str::to_string),
                    sender_name: None,
                    sender_source: src.into(),
                    alert: 0,
                    speaker_key: None,
                    corrected_text: None,
                })
                .expect("seed insert");
        };
        // Beep sliver: wordless, carried badge = carry artifact → hidden.
        put("beep", 128, "", "ok", Some("VE2CRS"), "carried", true);
        // Real quick self-ID: heard sender keeps it, whatever the size —
        // including a genuine sub-1.5 s self-ID the short flag catches.
        put(
            "quick",
            2_500,
            "VE2ABC à l'écoute",
            "ok",
            Some("VE2ABC"),
            "heard",
            false,
        );
        put(
            "quickshort",
            1_200,
            "VE2ABC!",
            "ok",
            Some("VE2ABC"),
            "heard",
            true,
        );
        // Honest error states stay visible: the pipeline marks failures
        // short, so the failed exemption must hold with short_flag set.
        put("fail", 2_500, "", "failed", None, "none", true);
        put(
            "long",
            9_000,
            "un oiseau sur l'antenne",
            "ok",
            None,
            "none",
            false,
        );
        // Whisper non-speech tags and sign-off tails hide.
        put("tag", 3_000, "[BLANK_AUDIO]", "ok", None, "none", false);
        put("tail", 4_000, "Bye. Thank you.", "ok", None, "none", false);
        // Canned repeats hide (every copy; the text stays searchable).
        put(
            "canned1",
            6_500,
            "RTQ network ID.",
            "ok",
            None,
            "none",
            false,
        );
        put(
            "canned2",
            6_600,
            "RTQ network ID.",
            "ok",
            None,
            "none",
            false,
        );
        let ids = |hide_noise: bool| {
            store
                .search(&SearchQuery {
                    hide_noise,
                    limit: 10,
                    ..Default::default()
                })
                .unwrap()
                .messages
                .iter()
                .map(|m| m.id.clone())
                .collect::<Vec<_>>()
        };
        let hidden = ids(true);
        for gone in ["beep", "tag", "tail", "canned1", "canned2"] {
            assert!(!hidden.contains(&gone.to_string()), "{gone} must hide");
        }
        for keep in ["quick", "quickshort", "fail", "long"] {
            assert!(hidden.contains(&keep.to_string()), "{keep} must stay");
        }
        // Off: everything shows (reveal on demand).
        assert_eq!(ids(false).len(), 9);
    }

    #[test]
    fn duplicate_transcripts_lists_repeats() {
        let store = Store::open_memory().unwrap();
        for (id, text) in [
            ("a", "same canned text"),
            ("b", "same canned text"),
            ("c", "unique line"),
        ] {
            store
                .insert(&NewMessage {
                    id: id.into(),
                    ts_start_ms: 1_000,
                    ts_end_ms: 2_000,
                    freq_label: "TEST".into(),
                    lang: "fr".into(),
                    lang_conf: 0.9,
                    transcript: text.into(),
                    stt_conf: 0.8,
                    conf_flag: "ok".into(),
                    status: "ok".into(),
                    fail_reason: None,
                    audio_path: None,
                    duration_ms: Some(2_000),
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
                })
                .expect("seed insert");
        }
        let dups = store.duplicate_transcripts().unwrap();
        assert!(dups.contains("same canned text"));
        assert!(!dups.contains("unique line"));
    }

    #[test]
    fn upsert_keeps_drop() {
        // A late re-transcription must never resurrect a dropped row: the
        // operator's drop wins over an in-flight retry drain.
        let store = Store::open_memory().unwrap();
        seed_messages(&store);
        store
            .set_triage(
                "m-fr-ok",
                &TriageAction::Drop {
                    delete_audio: false,
                },
            )
            .unwrap();
        let mut msg = store.get("m-fr-ok").unwrap().unwrap();
        assert_eq!(msg.status, "dropped");
        store
            .upsert(&NewMessage {
                id: msg.id.clone(),
                ts_start_ms: msg.ts_start_ms,
                ts_end_ms: msg.ts_end_ms,
                freq_label: msg.freq_label.clone(),
                lang: "en".into(),
                lang_conf: 1.0,
                transcript: "late re-transcription".into(),
                stt_conf: 1.0,
                conf_flag: "ok".into(),
                status: "ok".into(),
                fail_reason: None,
                audio_path: msg.audio_path.clone(),
                duration_ms: msg.duration_ms,
                size_bytes: msg.size_bytes,
                short_flag: false,
                group_id: msg.group_id.clone(),
                seq: msg.seq,
                sender_callsign: None,
                sender_name: None,
                sender_source: "none".into(),
                alert: 0,
                speaker_key: None,
                corrected_text: None,
            })
            .unwrap();
        msg = store.get("m-fr-ok").unwrap().unwrap();
        assert_eq!(msg.status, "dropped");
        assert_ne!(msg.transcript, "late re-transcription");
    }

    #[test]
    fn retention_purge() {
        let store = Store::open_memory().unwrap();
        seed_messages(&store);
        // now = 91 days after the 9s-epoch seeds; 90-day policy purges all.
        let now_ms = 91 * 86_400_000u64;
        let n = store.retention_sweep(90, now_ms).unwrap();
        assert_eq!(n, 6, "all six ok/failed rows purge");
        let m = store.get("m-fr-ok").unwrap().expect("row stays");
        assert_eq!(m.status, "dropped");
        assert!(m.audio_purged);
        assert!(store.missing_audio_audit().unwrap().is_empty());
    }

    #[test]
    fn tmp_janitor() {
        let root = std::env::temp_dir().join(format!("hamfeed-janitor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("a.ogg.tmp"), b"x").unwrap();
        std::fs::write(root.join("sub").join("b.ogg.tmp"), b"x").unwrap();
        std::fs::write(root.join("keep.ogg"), b"x").unwrap();
        let n = Store::tmp_janitor(&root, 0, 0).unwrap();
        assert_eq!(n, 2);
        assert!(root.join("keep.ogg").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_audio_is_bug() {
        let store = Store::open_memory().unwrap();
        seed_messages(&store);
        // Seeded audio paths (/tmp/*.ogg) do not exist and are not purged:
        // every ok/failed row flags.
        let mut bugs = store.missing_audio_audit().unwrap();
        bugs.sort();
        assert_eq!(bugs.len(), 6);
        // Dropped rows are exempt.
        store
            .set_triage(
                "m-old",
                &TriageAction::Drop {
                    delete_audio: false,
                },
            )
            .unwrap();
        let bugs = store.missing_audio_audit().unwrap();
        assert!(!bugs.contains(&"m-old".to_string()));
    }

    fn sender_msg(id: &str, ts: u64, callsign: Option<&str>, source: &str) -> NewMessage {
        NewMessage {
            id: id.into(),
            ts_start_ms: ts,
            ts_end_ms: ts + 500,
            freq_label: "TEST".into(),
            lang: "fr".into(),
            lang_conf: 0.9,
            transcript: "ici VE2DEM".into(),
            stt_conf: 0.8,
            conf_flag: "ok".into(),
            status: "ok".into(),
            fail_reason: None,
            audio_path: None,
            duration_ms: Some(500),
            size_bytes: Some(100),
            short_flag: false,
            group_id: "g".into(),
            seq: 0,
            sender_callsign: callsign.map(|s| s.into()),
            sender_name: None,
            sender_source: source.into(),
            alert: 0,
            speaker_key: None,
            corrected_text: None,
        }
    }

    #[test]
    fn migrate_pregroup_db_gains_group_and_triage() {
        // A very early file (no grouping, no triage, no Slice-2/004
        // columns): open adds everything with sane defaults and the row
        // reads back whole.
        let dir = std::env::temp_dir().join(format!("hamfeed-mig0-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE messages(
                   id TEXT PRIMARY KEY, ts_start INT NOT NULL, ts_end INT NOT NULL,
                   freq_label TEXT NOT NULL, lang TEXT NOT NULL, lang_conf REAL NOT NULL,
                   transcript TEXT NOT NULL, stt_conf REAL NOT NULL,
                   conf_flag TEXT NOT NULL, status TEXT NOT NULL,
                   fail_reason TEXT NULL, audio_path TEXT NULL,
                   audio_purged BOOL NOT NULL DEFAULT 0,
                   duration_ms INT NULL, size_bytes INT NULL,
                   short_flag BOOL NOT NULL DEFAULT 0)",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO messages(id,ts_start,ts_end,freq_label,lang,lang_conf,
                 transcript,stt_conf,conf_flag,status)
                 VALUES('old0',1000,1500,'TEST','fr',0.9,'bonjour',0.8,'ok','ok')",
                [],
            )
            .unwrap();
        }
        let store = Store::open(&path).unwrap();
        let m = store.get("old0").unwrap().expect("old row readable");
        assert_eq!(m.group_id, "");
        assert_eq!(m.seq, 0);
        assert_eq!(m.review_flag, "none");
        assert_eq!(m.sender_source, "none");
        assert_eq!(m.corrected_text, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_slice1_db_reads_senderless() {
        // A pre-Slice-2 database file (old column set, no alias tables):
        // open must migrate it and old rows read sender-less.
        let dir = std::env::temp_dir().join(format!("hamfeed-mig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE messages(
                   id TEXT PRIMARY KEY, ts_start INT NOT NULL, ts_end INT NOT NULL,
                   freq_label TEXT NOT NULL, lang TEXT NOT NULL, lang_conf REAL NOT NULL,
                   transcript TEXT NOT NULL, stt_conf REAL NOT NULL,
                   conf_flag TEXT NOT NULL, status TEXT NOT NULL,
                   fail_reason TEXT NULL, audio_path TEXT NULL,
                   audio_purged BOOL NOT NULL DEFAULT 0,
                   duration_ms INT NULL, size_bytes INT NULL,
                   short_flag BOOL NOT NULL DEFAULT 0,
                   group_id TEXT NOT NULL, seq INT NOT NULL,
                   review_flag TEXT NOT NULL DEFAULT 'none', flag_reason TEXT NULL)",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO messages(id,ts_start,ts_end,freq_label,lang,lang_conf,
                 transcript,stt_conf,conf_flag,status,group_id,seq)
                 VALUES('old1',1000,1500,'TEST','fr',0.9,'bonjour',0.8,'ok','ok','g',0)",
                [],
            )
            .unwrap();
        }
        let store = Store::open(&path).unwrap();
        let m = store.get("old1").unwrap().expect("old row readable");
        assert_eq!(m.sender_callsign, None);
        assert_eq!(m.sender_source, "none");
        assert_eq!(m.alert, 0);
        // New sender rows land in the migrated file.
        store
            .insert(&sender_msg("new1", 2000, Some("VE2DEM"), "heard"))
            .unwrap();
        assert_eq!(
            store
                .get("new1")
                .unwrap()
                .unwrap()
                .sender_callsign
                .as_deref(),
            Some("VE2DEM")
        );
        // Migration is idempotent: reopen cleanly.
        drop(store);
        let store = Store::open(&path).unwrap();
        assert!(store.get("old1").unwrap().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sender_filter_exact() {
        let store = Store::open_memory().unwrap();
        store
            .insert(&sender_msg("s-heard", 3000, Some("VE2DEM"), "heard"))
            .unwrap();
        store
            .insert(&sender_msg("s-carried", 2000, Some("VE2DEM"), "carried"))
            .unwrap();
        store
            .insert(&sender_msg("s-other", 1000, Some("VE3MA"), "heard"))
            .unwrap();
        let page = store
            .search(&SearchQuery {
                sender: Some("ve2dem".into()),
                limit: 10,
                ..Default::default()
            })
            .unwrap();
        let ids: Vec<&str> = page.messages.iter().map(|m| m.id.as_str()).collect();
        // Lowercase query normalizes; newest-first.
        assert_eq!(ids, vec!["s-heard", "s-carried"]);
    }

    #[test]
    fn upsert_preserves_confirmed_sender() {
        let store = Store::open_memory().unwrap();
        store
            .insert(&sender_msg("c1", 1000, Some("VE2DEM"), "heard"))
            .unwrap();
        store
            .set_sender("c1", Some("VE2DEM"), Some("Op"), "confirmed")
            .unwrap();
        // A sender-less re-transcription must not wipe the verdict.
        let mut retry = sender_msg("c1", 1000, None, "none");
        retry.transcript = "late re-transcription".into();
        store.upsert(&retry).unwrap();
        let m = store.get("c1").unwrap().unwrap();
        assert_eq!(m.sender_callsign.as_deref(), Some("VE2DEM"));
        assert_eq!(m.sender_source, "confirmed");
        // But a fresh heard sender still replaces a non-confirmed one.
        store
            .insert(&sender_msg("c2", 2000, Some("VE3MA"), "carried"))
            .unwrap();
        let mut heard = sender_msg("c2", 2000, Some("VE2DEM"), "heard");
        heard.transcript = "ici VE2DEM".into();
        store.upsert(&heard).unwrap();
        let m = store.get("c2").unwrap().unwrap();
        assert_eq!(m.sender_callsign.as_deref(), Some("VE2DEM"));
        assert_eq!(m.sender_source, "heard");
    }

    #[test]
    fn alias_roundtrip_confidence() {
        let store = Store::open_memory().unwrap();
        assert_eq!(store.get_alias("Unknown-1|x|d").unwrap(), None);
        store
            .set_alias("Unknown-1|x|d", "VE2DEM", 0.8, 1000)
            .unwrap();
        assert_eq!(
            store.get_alias("Unknown-1|x|d").unwrap(),
            Some(("VE2DEM".to_string(), 0.8))
        );
        // A weaker re-hearing never downgrades a strong verdict.
        store
            .set_alias("Unknown-1|x|d", "VE3MA", 0.6, 2000)
            .unwrap();
        assert_eq!(
            store.get_alias("Unknown-1|x|d").unwrap(),
            Some(("VE2DEM".to_string(), 0.8))
        );
        // A strictly stronger hearing wins.
        store
            .set_alias("Unknown-1|x|d", "VE3MA", 1.0, 3000)
            .unwrap();
        assert_eq!(
            store.get_alias("Unknown-1|x|d").unwrap(),
            Some(("VE3MA".to_string(), 1.0))
        );
    }

    #[test]
    fn overwrite_alias_forces() {
        let store = Store::open_memory().unwrap();
        store.set_alias("k", "VE2DEM", 1.0, 1000).unwrap();
        // Equal confidence via the guarded path: old verdict stands.
        store.set_alias("k", "VE3MA", 1.0, 2000).unwrap();
        assert_eq!(
            store.get_alias("k").unwrap(),
            Some(("VE2DEM".to_string(), 1.0))
        );
        // Operator verdict via the forced path: reassigns.
        store.overwrite_alias("k", "VE3MA", 1.0, 3000).unwrap();
        assert_eq!(
            store.get_alias("k").unwrap(),
            Some(("VE3MA".to_string(), 1.0))
        );
    }

    #[test]
    fn alloc_never_reuses() {
        let store = Store::open_memory().unwrap();
        assert_eq!(store.alloc_speaker_n("TEST", "2026-09-16").unwrap(), 1);
        assert_eq!(store.alloc_speaker_n("TEST", "2026-09-16").unwrap(), 2);
        // A new day starts over; the old day's numbers are never re-minted.
        assert_eq!(store.alloc_speaker_n("TEST", "2026-09-17").unwrap(), 1);
        assert_eq!(store.alloc_speaker_n("TEST", "2026-09-16").unwrap(), 3);
    }

    #[test]
    fn alias_purge_expired() {
        let store = Store::open_memory().unwrap();
        store.set_alias("k-old", "VE2DEM", 1.0, 1_000).unwrap();
        store
            .set_alias("k-new", "VE3MA", 1.0, 8 * 86_400_000)
            .unwrap();
        // 7-day voice retention at now = 8 days: only the stale alias purges.
        let n = store.purge_aliases(7, 8 * 86_400_000).unwrap();
        assert_eq!(n, 1);
        assert!(store.get_alias("k-old").unwrap().is_none());
        assert!(store.get_alias("k-new").unwrap().is_some());
    }

    #[test]
    fn correction_roundtrip() {
        let store = Store::open_memory().unwrap();
        seed_messages(&store);
        // Set: badge text lands, original untouched.
        store
            .set_correction("m-fr-ok", Some("bonjour les vrais amis"))
            .unwrap();
        let m = store.get("m-fr-ok").unwrap().unwrap();
        assert_eq!(m.corrected_text.as_deref(), Some("bonjour les vrais amis"));
        assert_eq!(m.transcript, "bonjour les amis");
        // Overwrite wins (v1: no history).
        store
            .set_correction("m-fr-ok", Some("  bonjour  "))
            .unwrap();
        assert_eq!(
            store
                .get("m-fr-ok")
                .unwrap()
                .unwrap()
                .corrected_text
                .as_deref(),
            Some("bonjour")
        );
        // Blank/None clears back to NULL.
        store.set_correction("m-fr-ok", Some("   ")).unwrap();
        assert_eq!(store.get("m-fr-ok").unwrap().unwrap().corrected_text, None);
        store.set_correction("m-fr-ok", Some("x")).unwrap();
        store.set_correction("m-fr-ok", None).unwrap();
        assert_eq!(store.get("m-fr-ok").unwrap().unwrap().corrected_text, None);
        // Unknown id bails (triage convention).
        assert!(store.set_correction("nope", Some("x")).is_err());
    }

    #[test]
    fn upsert_preserves_correction() {
        let store = Store::open_memory().unwrap();
        seed_messages(&store);
        store.set_correction("m-fr-ok", Some("true text")).unwrap();
        // A re-transcription (which never carries a correction) must not
        // wipe the operator's label.
        let mut retry = sender_msg("m-fr-ok", 9_000, None, "none");
        retry.transcript = "late re-transcription".into();
        store.upsert(&retry).unwrap();
        let m = store.get("m-fr-ok").unwrap().unwrap();
        assert_eq!(m.corrected_text.as_deref(), Some("true text"));
        assert_eq!(m.transcript, "late re-transcription");
    }

    #[test]
    fn search_hits_corrected_only() {
        let store = Store::open_memory().unwrap();
        seed_messages(&store);
        // "sorel" appears in no model transcript.
        let before = store
            .search(&SearchQuery {
                text: Some("sorel".into()),
                limit: 10,
                ..Default::default()
            })
            .unwrap();
        assert!(before.messages.is_empty());
        store
            .set_correction("m-fr-ok", Some("retour à Sorel ce soir"))
            .unwrap();
        let after = store
            .search(&SearchQuery {
                text: Some("sorel".into()),
                limit: 10,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(after.messages.len(), 1);
        assert_eq!(after.messages[0].id, "m-fr-ok");
    }

    #[test]
    fn migrate_pre004_reads_uncorrected() {
        // A pre-004 file (no corrected_text anywhere, single-column FTS):
        // open migrates both, old rows read uncorrected, and both texts
        // index afterwards.
        let dir = std::env::temp_dir().join(format!("hamfeed-mig4-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE messages(
                   id TEXT PRIMARY KEY, ts_start INT NOT NULL, ts_end INT NOT NULL,
                   freq_label TEXT NOT NULL, lang TEXT NOT NULL, lang_conf REAL NOT NULL,
                   transcript TEXT NOT NULL, stt_conf REAL NOT NULL,
                   conf_flag TEXT NOT NULL, status TEXT NOT NULL,
                   fail_reason TEXT NULL, audio_path TEXT NULL,
                   audio_purged BOOL NOT NULL DEFAULT 0,
                   duration_ms INT NULL, size_bytes INT NULL,
                   short_flag BOOL NOT NULL DEFAULT 0,
                   group_id TEXT NOT NULL, seq INT NOT NULL,
                   review_flag TEXT NOT NULL DEFAULT 'none', flag_reason TEXT NULL);
                 CREATE VIRTUAL TABLE messages_fts
                   USING fts5(transcript, content='messages', content_rowid='rowid');
                 CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
                   INSERT INTO messages_fts(rowid, transcript)
                     VALUES (new.rowid, new.transcript);
                 END;",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO messages(id,ts_start,ts_end,freq_label,lang,lang_conf,
                 transcript,stt_conf,conf_flag,status,group_id,seq)
                 VALUES('old1',1000,1500,'TEST','fr',0.9,'bonjour montreal',0.8,'ok','ok','g',0)",
                [],
            )
            .unwrap();
        }
        let store = Store::open(&path).unwrap();
        let m = store.get("old1").unwrap().expect("old row readable");
        assert_eq!(m.corrected_text, None);
        // Old transcript still searchable after the FTS migration.
        let page = store
            .search(&SearchQuery {
                text: Some("montreal".into()),
                limit: 10,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page.messages.len(), 1);
        // ...and a new correction on the old row indexes too.
        store.set_correction("old1", Some("bonjour sorel")).unwrap();
        let page = store
            .search(&SearchQuery {
                text: Some("sorel".into()),
                limit: 10,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page.messages.len(), 1);
        // Reopen: migration is a no-op second time, data intact.
        drop(store);
        let store = Store::open(&path).unwrap();
        assert_eq!(
            store
                .get("old1")
                .unwrap()
                .unwrap()
                .corrected_text
                .as_deref(),
            Some("bonjour sorel")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn like_escape_neutralizes_wildcards() {
        assert_eq!(like_escape("plain"), "plain");
        assert_eq!(like_escape("100%"), "100\\%");
        assert_eq!(like_escape("a_b"), "a\\_b");
        assert_eq!(like_escape("back\\slash"), "back\\\\slash");
        assert_eq!(like_escape("%_%"), "\\%\\_\\%");
    }

    #[test]
    fn status_filter_allowlist() {
        let store = Store::open_memory().unwrap();
        seed_messages(&store);
        let count = |status: Option<&str>| {
            store
                .search(&SearchQuery {
                    status: status.map(|s| s.into()),
                    limit: 20,
                    ..Default::default()
                })
                .unwrap()
                .messages
                .len()
        };
        // Seeded rows are all status ok/failed; the filter narrows.
        assert!(count(Some("ok")) >= 1);
        assert!(count(Some("failed")) >= 1);
        assert_eq!(count(Some("kept")), 0);
        // Injection dead-ends (fail closed): quote-stripping alone would
        // have let `OR '1'='1` smuggle past the filter.
        assert_eq!(count(Some("ok' OR '1'='1")), 0);
        assert_eq!(count(Some("' OR 1=1 --")), 0);
        // No filter: everything.
        assert!(count(None) > count(Some("ok")));
    }

    #[test]
    fn corrections_export_lists_pairs() {
        let store = Store::open_memory().unwrap();
        seed_messages(&store);
        assert!(store.corrections_export().unwrap().is_empty());
        // Corrected but audio-less: excluded (a pair needs both halves).
        store
            .set_correction("m-fr-low", Some("appel général corrigé"))
            .unwrap();
        {
            let conn = store.conn.lock().expect("store mutex");
            conn.execute(
                "UPDATE messages SET audio_path=NULL WHERE id='m-fr-low'",
                [],
            )
            .unwrap();
        }
        store
            .set_correction("m-fr-ok", Some("bonjour les vrais amis"))
            .unwrap();
        let rows = store.corrections_export().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "m-fr-ok");
        assert_eq!(rows[0].corrected, "bonjour les vrais amis");
        assert_eq!(rows[0].original, "bonjour les amis");
        assert_eq!(rows[0].audio_path, "/tmp/m-fr-ok.ogg");
    }

    fn fts_msg(id: &str, transcript: &str, corrected: Option<&str>) -> NewMessage {
        NewMessage {
            id: id.into(),
            ts_start_ms: 1000,
            ts_end_ms: 1500,
            freq_label: "TEST".into(),
            lang: "en".into(),
            lang_conf: 0.9,
            transcript: transcript.into(),
            stt_conf: 0.8,
            conf_flag: "ok".into(),
            status: "ok".into(),
            fail_reason: None,
            audio_path: None,
            duration_ms: Some(500),
            size_bytes: Some(100),
            short_flag: false,
            group_id: "g".into(),
            seq: 0,
            sender_callsign: None,
            sender_name: None,
            sender_source: "none".into(),
            alert: 0,
            speaker_key: None,
            corrected_text: corrected.map(|s| s.into()),
        }
    }

    fn fts_hits(store: &Store, token: &str) -> i64 {
        let conn = store.conn.lock().expect("store mutex");
        conn.query_row(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            [token],
            |r| r.get(0),
        )
        .unwrap()
    }

    #[test]
    fn fts_delete_clears_both_columns() {
        // A partial-column 'delete' leaves ghost tokens for the omitted
        // column, so the delete legs must name both FTS columns.
        let store = Store::open_memory().unwrap();
        store
            .insert(&fts_msg("del1", "zagreb alpha", Some("sorel beta")))
            .unwrap();
        assert_eq!(fts_hits(&store, "zagreb"), 1);
        assert_eq!(fts_hits(&store, "sorel"), 1);
        {
            let conn = store.conn.lock().expect("store mutex");
            conn.execute("DELETE FROM messages WHERE id='del1'", [])
                .unwrap();
        }
        assert_eq!(fts_hits(&store, "zagreb"), 0);
        assert_eq!(fts_hits(&store, "sorel"), 0);
    }

    #[test]
    fn fts_update_clears_old_tokens_of_both_columns() {
        let store = Store::open_memory().unwrap();
        store
            .insert(&fts_msg("upd1", "zagreb alpha", Some("sorel beta")))
            .unwrap();
        // Re-transcription replaces the transcript (the operator's
        // correction survives via COALESCE): the old transcript token
        // must vanish while the correction token stays.
        let mut retry = fts_msg("upd1", "lisbon gamma", None);
        store.upsert(&retry).unwrap();
        assert_eq!(fts_hits(&store, "zagreb"), 0);
        assert_eq!(fts_hits(&store, "lisbon"), 1);
        assert_eq!(fts_hits(&store, "sorel"), 1);
        // A new correction retires the old correction token.
        retry.corrected_text = Some("quebec delta".into());
        store
            .set_correction("upd1", retry.corrected_text.as_deref())
            .unwrap();
        assert_eq!(fts_hits(&store, "sorel"), 0);
        assert_eq!(fts_hits(&store, "quebec"), 1);
        assert_eq!(fts_hits(&store, "lisbon"), 1);
    }

    #[test]
    fn migrate_refreshes_single_column_delete_triggers() {
        // A file built by the buggy two-column code (2-col FTS index,
        // single-column delete legs): open must refresh the triggers so
        // deletes leave no ghost tokens. Indexed content is unaffected.
        let dir = std::env::temp_dir().join(format!("hamfeed-migdel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("buggy.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE messages(
                   id TEXT PRIMARY KEY,
                   ts_start INT NOT NULL, ts_end INT NOT NULL,
                   freq_label TEXT NOT NULL, lang TEXT NOT NULL, lang_conf REAL NOT NULL,
                   transcript TEXT NOT NULL, stt_conf REAL NOT NULL,
                   conf_flag TEXT NOT NULL, status TEXT NOT NULL,
                   fail_reason TEXT NULL, audio_path TEXT NULL,
                   audio_purged BOOL NOT NULL DEFAULT 0,
                   duration_ms INT NULL, size_bytes INT NULL,
                   short_flag BOOL NOT NULL DEFAULT 0,
                   group_id TEXT NOT NULL, seq INT NOT NULL,
                   review_flag TEXT NOT NULL DEFAULT 'none',
                   flag_reason TEXT NULL,
                   sender_callsign TEXT NULL, sender_name TEXT NULL,
                   sender_source TEXT NOT NULL DEFAULT 'none',
                   alert INT NOT NULL DEFAULT 0,
                   speaker_key TEXT NULL,
                   corrected_text TEXT NULL);
                 CREATE VIRTUAL TABLE messages_fts
                   USING fts5(transcript, corrected_text,
                              content='messages', content_rowid='rowid');
                 CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
                   INSERT INTO messages_fts(rowid, transcript, corrected_text)
                     VALUES (new.rowid, new.transcript, new.corrected_text);
                 END;
                 CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
                   INSERT INTO messages_fts(messages_fts, rowid, transcript)
                     VALUES ('delete', old.rowid, old.transcript);
                 END;
                 CREATE TRIGGER messages_au
                   AFTER UPDATE OF transcript, corrected_text ON messages BEGIN
                   INSERT INTO messages_fts(messages_fts, rowid, transcript)
                     VALUES ('delete', old.rowid, old.transcript);
                   INSERT INTO messages_fts(rowid, transcript, corrected_text)
                     VALUES (new.rowid, new.transcript, new.corrected_text);
                 END;",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO messages(id,ts_start,ts_end,freq_label,lang,lang_conf,
                 transcript,stt_conf,conf_flag,status,group_id,seq,corrected_text)
                 VALUES('old1',1000,1500,'TEST','en',0.9,'zagreb alpha',
                        0.8,'ok','ok','g',0,'sorel beta')",
                [],
            )
            .unwrap();
        }
        let store = Store::open(&path).unwrap();
        // Old row still indexed on both texts (no content rebuild).
        assert_eq!(fts_hits(&store, "zagreb"), 1);
        assert_eq!(fts_hits(&store, "sorel"), 1);
        {
            let conn = store.conn.lock().expect("store mutex");
            for trig in ["messages_ad", "messages_au"] {
                let sql: String = conn
                    .query_row(
                        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
                        [trig],
                        |r| r.get(0),
                    )
                    .unwrap();
                assert!(sql.contains("corrected_text"), "{trig} refreshed");
            }
            conn.execute("DELETE FROM messages WHERE id='old1'", [])
                .unwrap();
        }
        assert_eq!(fts_hits(&store, "zagreb"), 0);
        assert_eq!(fts_hits(&store, "sorel"), 0);
        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn profiles_seed_exact_and_normal_default() {
        let store = Store::open_memory().unwrap();
        let names: Vec<String> = store
            .list_profiles()
            .unwrap()
            .iter()
            .map(|p| p.name.clone())
            .collect();
        assert_eq!(names, vec!["Normal", "ARES Net", "Severe Weather"]);
        let normal = &store.list_profiles().unwrap()[0];
        assert!(normal.cues.is_empty());
        let ares = &store.list_profiles().unwrap()[1];
        assert!(ares.cues.contains(&"net control".to_string()));
        assert_eq!(store.active_profile().unwrap(), "Normal");
    }

    #[test]
    fn profile_switch_roundtrip_and_unknown_rejected() {
        let store = Store::open_memory().unwrap();
        store.set_active_profile("ARES Net").unwrap();
        assert_eq!(store.active_profile().unwrap(), "ARES Net");
        store.set_active_profile("Normal").unwrap();
        assert_eq!(store.active_profile().unwrap(), "Normal");
        let err = store.set_active_profile("Nope").unwrap_err();
        assert!(err.downcast_ref::<UnknownProfile>().is_some());
        // Rejected switch leaves the previous profile in place.
        assert_eq!(store.active_profile().unwrap(), "Normal");
    }

    #[test]
    fn active_profile_survives_reopen() {
        // G4: the mode is still on after a restart mid-event.
        let dir = std::env::temp_dir().join(format!("hamfeed-prof-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("feed.db");
        {
            let store = Store::open(&path).unwrap();
            store.set_active_profile("Severe Weather").unwrap();
        }
        {
            let store = Store::open(&path).unwrap();
            assert_eq!(store.active_profile().unwrap(), "Severe Weather");
            // Old file, new code: profiles seed on open, switch works.
            assert_eq!(store.list_profiles().unwrap().len(), 3);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
