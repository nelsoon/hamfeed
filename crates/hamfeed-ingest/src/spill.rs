//! Spill dir + manifest + replay (T6, S9).
//!
//! When the bounded queue overflows, the evicted item's audio file moves to
//! `<dir>/spill/` and a JSON manifest row is appended. On boot the pipeline
//! replays spilled rows in `(ts_start_ms, id)` order. Manifest appends are
//! fsynced before the move is acknowledged.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::queue::QueueItem;

/// Manifest row: deliberately redundant with the audio filename so a lost
/// file is detectable (pipeline cross-checks, S6/S10).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SpillRow {
    pub id: String,
    pub audio_file: String,
    pub duration_ms: u64,
    pub ts_start_ms: u64,
    #[serde(default)]
    pub group_id: String,
    #[serde(default)]
    pub seq: u32,
}

/// `<storage dir>/spill/`: spilled `.ogg` files + `manifest.jsonl`.
#[derive(Debug)]
pub struct SpillDir {
    dir: PathBuf,
}

impl SpillDir {
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("cannot create spill {}", dir.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
        })
    }

    fn manifest_path(&self) -> PathBuf {
        self.dir.join("manifest.jsonl")
    }

    /// Move the item's audio file into the spill dir and record the row.
    /// The manifest append is flushed to disk before this returns.
    pub fn spill(&self, item: &QueueItem) -> Result<SpillRow> {
        let src = Path::new(&item.audio_path);
        let file = format!("{}-{}.ogg", item.ts_start_ms, item.id);
        let dst = self.dir.join(&file);
        std::fs::rename(src, &dst)
            .with_context(|| format!("cannot spill {} to {}", src.display(), dst.display()))?;
        let row = SpillRow {
            id: item.id.clone(),
            audio_file: file,
            duration_ms: item.duration_ms,
            ts_start_ms: item.ts_start_ms,
            group_id: item.group_id.clone(),
            seq: item.seq,
        };
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.manifest_path())
            .context("cannot open spill manifest")?;
        writeln!(
            f,
            "{}",
            serde_json::to_string(&row).expect("row serializes")
        )?;
        f.sync_all().context("cannot sync spill manifest")?;
        Ok(row)
    }

    /// All spilled rows in `(ts_start_ms, id)` replay order (S9).
    pub fn replay(&self) -> Result<Vec<SpillRow>> {
        let path = self.manifest_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&path).context("cannot read spill manifest")?;
        let mut rows: Vec<SpillRow> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()
            .context("corrupt spill manifest")?;
        rows.sort_by(|a, b| (a.ts_start_ms, &a.id).cmp(&(b.ts_start_ms, &b.id)));
        Ok(rows)
    }

    /// Path of a spilled audio file (for the pipeline drain).
    pub fn audio_path(&self, row: &SpillRow) -> PathBuf {
        self.dir.join(&row.audio_file)
    }

    /// Replay rows as queue items with spill-resolved audio paths.
    pub fn replay_items(&self) -> Result<Vec<QueueItem>> {
        Ok(self
            .replay()?
            .into_iter()
            .map(|r| QueueItem {
                audio_path: self.audio_path(&r).to_string_lossy().into_owned(),
                id: r.id,
                duration_ms: r.duration_ms,
                ts_start_ms: r.ts_start_ms,
                group_id: r.group_id,
                seq: r.seq,
            })
            .collect())
    }

    /// Drop a replayed row and its audio (after successful STT+store).
    /// Rewrites the manifest without the row; missing files are tolerated.
    pub fn remove(&self, id: &str) -> Result<()> {
        let rows: Vec<SpillRow> = self.replay()?.into_iter().filter(|r| r.id != id).collect();
        let gone: Vec<SpillRow> = self.replay()?.into_iter().filter(|r| r.id == id).collect();
        for r in &gone {
            let _ = std::fs::remove_file(self.audio_path(r));
        }
        let mut text = String::new();
        for r in &rows {
            text.push_str(&serde_json::to_string(r).expect("row serializes"));
            text.push('\n');
        }
        std::fs::write(self.manifest_path(), text).context("cannot rewrite manifest")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::tests::test_dir;
    use crate::queue::{IngestQueue, QueueItem};

    fn fake_clip(dir: &Path, id: &str) -> String {
        let p = dir.join(format!("{id}.ogg"));
        std::fs::write(&p, b"fake-opus-bytes").unwrap();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn spill_replay_order() {
        // Fill a cap-100 queue, overflow with out-of-order arrivals, reboot:
        // replay must come back in (ts_start, id) order (S9).
        let root = test_dir("spill-order");
        let clips = root.path().join("clips");
        std::fs::create_dir_all(&clips).unwrap();
        let spill = SpillDir::open(&root.path().join("spill")).unwrap();
        let mut q = IngestQueue::new(100);
        // Descending timestamps: eviction order (q000.. = newest ts) differs
        // from replay order, so the sort is really exercised.
        for i in 0..100 {
            let id = format!("q{i:03}");
            let item = QueueItem {
                id: id.clone(),
                audio_path: fake_clip(&clips, &id),
                duration_ms: 1000,
                ts_start_ms: 10_000 - i,
                group_id: "g".into(),
                seq: i as u32,
            };
            assert_eq!(q.push(item, &spill).unwrap(), None);
        }
        // Overflow evicts the three oldest pushes (q000..q002).
        for (id, ts) in [("z1", 5u64), ("z2", 5000u64), ("z3", 5u64)] {
            let item = QueueItem {
                id: id.into(),
                audio_path: fake_clip(&clips, id),
                duration_ms: 1000,
                ts_start_ms: ts,
                group_id: "g".into(),
                seq: 0,
            };
            q.push(item, &spill).unwrap();
        }
        // Simulated reboot: fresh handle on the same spill dir, replay first.
        let spill2 = SpillDir::open(&root.path().join("spill")).unwrap();
        let replayed = spill2.replay().unwrap();
        assert_eq!(replayed.len(), 3);
        let keys: Vec<(u64, &str)> = replayed
            .iter()
            .map(|r| (r.ts_start_ms, r.id.as_str()))
            .collect();
        let sorted = {
            let mut s = keys.clone();
            s.sort();
            s
        };
        assert_eq!(keys, sorted, "replay must be (ts_start,id) ordered");
        assert_eq!(keys, vec![(9998, "q002"), (9999, "q001"), (10_000, "q000")]);
        // Spilled audio actually moved.
        for r in &replayed {
            assert!(spill2.audio_path(r).exists(), "missing {}", r.audio_file);
        }
        // Removing a consumed row clears row + audio.
        spill2.remove("q001").unwrap();
        let again = spill2.replay().unwrap();
        assert_eq!(again.len(), 2);
        assert!(again.iter().all(|r| r.id != "q001"));
        assert!(!spill2.dir.join("9999-q001.ogg").exists());
    }
}
