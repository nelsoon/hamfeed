//! Bounded ingest→STT queue (T6, S9).
//!
//! Holds metadata only (`QueueItem`); the audio already sits on disk (see
//! `encode::write_clip_atomic`). On overflow the oldest item spills to disk
//! (`spill.rs`) so nothing vanishes silently.

use std::collections::VecDeque;

use crate::spill::SpillDir;

/// One queued clip: metadata only, never audio (plan).
/// `group_id`/`seq` ride along so rows stay grouped across restarts;
/// they are assigned only by the segmenter (group contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueItem {
    pub id: String,
    pub audio_path: String,
    pub duration_ms: u64,
    pub ts_start_ms: u64,
    pub group_id: String,
    pub seq: u32,
}

/// Bounded FIFO. `push` spills the oldest item when full and reports what
/// spilled so the pipeline can account for it.
#[derive(Debug)]
pub struct IngestQueue {
    cap: usize,
    items: VecDeque<QueueItem>,
}

impl IngestQueue {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            items: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Push `item`; when the queue is full the oldest item is spilled first
    /// and returned. The spill row covers the item, so the queue never drops
    /// work silently (S9).
    pub fn push(&mut self, item: QueueItem, spill: &SpillDir) -> anyhow::Result<Option<QueueItem>> {
        let mut spilled = None;
        if self.items.len() >= self.cap {
            let oldest = self.items.pop_front().expect("full but empty");
            spill.spill(&oldest)?;
            spilled = Some(oldest);
        }
        self.items.push_back(item);
        Ok(spilled)
    }

    pub fn pop(&mut self) -> Option<QueueItem> {
        self.items.pop_front()
    }

    /// Drop any queued item with `id` (e.g. operator dropped it mid-flight).
    pub fn remove(&mut self, id: &str) {
        self.items.retain(|i| i.id != id);
    }

    pub fn drain(&mut self) -> Vec<QueueItem> {
        self.items.drain(..).collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn item(dir: &std::path::Path, id: &str, ts: u64) -> QueueItem {
        QueueItem {
            id: id.into(),
            audio_path: dir.join(format!("{id}.ogg")).to_string_lossy().into_owned(),
            duration_ms: 1000,
            ts_start_ms: ts,
            group_id: "g".into(),
            seq: 0,
        }
    }

    #[test]
    fn overflow_spills_oldest() {
        let dir = test_dir("overflow");
        for id in ["a", "b", "c"] {
            std::fs::write(dir.path().join(format!("{id}.ogg")), b"fake").unwrap();
        }
        let spill = SpillDir::open(&dir.path().join("spill")).unwrap();
        let mut q = IngestQueue::new(2);
        assert_eq!(q.push(item(dir.path(), "a", 3), &spill).unwrap(), None);
        assert_eq!(q.push(item(dir.path(), "b", 1), &spill).unwrap(), None);
        let spilled = q
            .push(item(dir.path(), "c", 2), &spill)
            .unwrap()
            .expect("spill");
        assert_eq!(spilled.id, "a");
        assert_eq!(q.len(), 2);
        // Spill dir holds the evicted row for replay.
        let replayed = spill.replay().unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].id, "a");
    }

    pub(crate) fn test_dir(name: &str) -> TestDir {
        TestDir::new(name)
    }

    /// Minimal unique temp dir (no extra deps for tests).
    pub(crate) struct TestDir {
        path: std::path::PathBuf,
    }

    impl TestDir {
        pub(crate) fn new(name: &str) -> Self {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let path =
                std::env::temp_dir().join(format!("hamfeed-{}-{}-{}", name, std::process::id(), n));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        pub(crate) fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
