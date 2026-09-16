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

/// Filtered history query (S8).
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    pub text: Option<String>,
    pub from_ms: Option<u64>,
    pub to_ms: Option<u64>,
    pub hide_noise: bool,
    pub status: Option<String>,
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
  flag_reason TEXT NULL
);
CREATE INDEX IF NOT EXISTS idx_ts ON messages(ts_start);
CREATE INDEX IF NOT EXISTS idx_lang ON messages(lang);
CREATE INDEX IF NOT EXISTS idx_status ON messages(status);
CREATE INDEX IF NOT EXISTS idx_group ON messages(group_id, seq);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
  USING fts5(transcript, content='messages', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, transcript)
    VALUES (new.rowid, new.transcript);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, transcript)
    VALUES ('delete', old.rowid, old.transcript);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF transcript ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, transcript)
    VALUES ('delete', old.rowid, old.transcript);
  INSERT INTO messages_fts(rowid, transcript)
    VALUES (new.rowid, new.transcript);
END;
";

impl Store {
    fn init(conn: &Connection) -> Result<()> {
        conn.execute_batch(SCHEMA).context("cannot create schema")?;
        Ok(())
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
              duration_ms,size_bytes,short_flag,group_id,seq)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
            ],
        )
        .with_context(|| format!("cannot insert {}", m.id))?;
        Ok(())
    }

    /// Insert, or refresh the transcribed columns when the id already exists
    /// (retry path). `review_flag`/`flag_reason` survive: operator triage is
    /// never wiped by a re-transcription.
    pub fn upsert(&self, m: &NewMessage) -> Result<()> {
        let conn = self.conn.lock().expect("store mutex");
        conn.execute(
            "INSERT INTO messages(id,ts_start,ts_end,freq_label,lang,lang_conf,
              transcript,stt_conf,conf_flag,status,fail_reason,audio_path,
              duration_ms,size_bytes,short_flag,group_id,seq)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
             ON CONFLICT(id) DO UPDATE SET
              ts_end=excluded.ts_end, lang=excluded.lang,
              lang_conf=excluded.lang_conf, transcript=excluded.transcript,
              stt_conf=excluded.stt_conf, conf_flag=excluded.conf_flag,
              status=excluded.status, fail_reason=excluded.fail_reason,
              audio_path=excluded.audio_path,
              duration_ms=excluded.duration_ms,
              size_bytes=excluded.size_bytes,
              short_flag=excluded.short_flag
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
            sql.push_str(" AND short_flag = 0");
        }
        if let Some(st) = &q.status {
            sql.push_str(" AND status = '");
            sql.push_str(&st.replace('\'', ""));
            sql.push('\'');
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
                let mut stmt =
                    conn.prepare("SELECT rowid FROM messages WHERE transcript LIKE ?")?;
                let like = format!("%{text}%");
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
}
