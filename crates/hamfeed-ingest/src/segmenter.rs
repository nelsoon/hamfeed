//! Energy-VAD segmenter with max-duration split (T5, S1–S3).
//!
//! State machine per push chunk: silence while idle stays shut; voice opens
//! a segment; trailing silence past `hang_ms` closes it at the last voice
//! sample; an open segment hitting `max_segment_ms` splits into a new `seq`
//! under the same `group_id`.

use crate::SAMPLE_RATE;

/// Tunables. `hang_ms` normally comes from the active VAD profile
/// (`Config::hang_ms("repeater"|"simplex")`); `energy_threshold` is a mean
/// absolute amplitude — steady speech (~several thousand) clears it while
/// digital silence and low noise floors do not.
#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    pub hang_ms: u64,
    pub energy_threshold: f32,
    pub max_segment_ms: u64,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            hang_ms: 800,
            energy_threshold: 300.0,
            max_segment_ms: 120_000,
        }
    }
}

/// One closed voice segment with its PCM attached (in-RAM only in T5).
#[derive(Debug, Clone)]
pub struct Segment {
    pub id: String,
    pub group_id: String,
    pub seq: u32,
    pub ts_start_ms: u64,
    pub ts_end_ms: u64,
    pub duration_ms: u64,
    pub pcm: Vec<i16>,
}

#[derive(Debug)]
struct Open {
    group_id: String,
    seq: u32,
    start_sample: u64,
    last_voice_sample: u64,
    pcm: Vec<i16>,
}

/// Energy-VAD segmenter over a sample clock starting at `epoch_ms`.
#[derive(Debug)]
pub struct Segmenter {
    cfg: SegmenterConfig,
    epoch_ms: u64,
    cursor: u64,
    open: Option<Open>,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig, epoch_ms: u64) -> Self {
        Self {
            cfg,
            epoch_ms,
            cursor: 0,
            open: None,
        }
    }

    fn ts(&self, sample: u64) -> u64 {
        self.epoch_ms + sample * 1000 / SAMPLE_RATE as u64
    }

    fn hang_samples(&self) -> u64 {
        self.cfg.hang_ms * SAMPLE_RATE as u64 / 1000
    }

    fn max_samples(&self) -> u64 {
        self.cfg.max_segment_ms * SAMPLE_RATE as u64 / 1000
    }

    fn mean_abs(chunk: &[i16]) -> f32 {
        if chunk.is_empty() {
            return 0.0;
        }
        let sum: i64 = chunk.iter().map(|s| s.unsigned_abs() as i64).sum();
        sum as f32 / chunk.len() as f32
    }

    fn close(&self, o: &Open, end_sample: u64) -> Segment {
        let ts_start_ms = self.ts(o.start_sample);
        let ts_end_ms = self.ts(end_sample);
        Segment {
            id: uuid::Uuid::new_v4().to_string(),
            group_id: o.group_id.clone(),
            seq: o.seq,
            ts_start_ms,
            ts_end_ms,
            duration_ms: ts_end_ms.saturating_sub(ts_start_ms),
            pcm: o.pcm.clone(),
        }
    }

    /// Feed one chunk of 16 kHz mono S16; returns newly closed segments.
    pub fn push(&mut self, chunk: &[i16]) -> Vec<Segment> {
        let mut out = Vec::new();
        if chunk.is_empty() {
            return out;
        }
        let voice = Self::mean_abs(chunk) >= self.cfg.energy_threshold;
        let chunk_start = self.cursor;
        let chunk_end = self.cursor + chunk.len() as u64;

        if voice {
            match self.open.as_mut() {
                Some(o) => {
                    o.last_voice_sample = chunk_end;
                    o.pcm.extend_from_slice(chunk);
                }
                None => {
                    self.open = Some(Open {
                        group_id: uuid::Uuid::new_v4().to_string(),
                        seq: 0,
                        start_sample: chunk_start,
                        last_voice_sample: chunk_end,
                        pcm: chunk.to_vec(),
                    });
                }
            }
        } else if let Some(o) = self.open.as_mut() {
            o.pcm.extend_from_slice(chunk);
            if chunk_end - o.last_voice_sample >= self.hang_samples() {
                let o = self.open.take().expect("just checked");
                out.push(self.close(&o, o.last_voice_sample));
            }
        }

        self.cursor = chunk_end;

        // Max-duration split at push boundaries (chunk-aligned, ≤1 chunk late).
        if let Some(o) = &self.open {
            if chunk_end - o.start_sample >= self.max_samples() {
                let o = self.open.take().expect("just checked");
                let split = self.close(&o, chunk_end);
                let (group_id, seq) = (o.group_id.clone(), o.seq + 1);
                out.push(split);
                self.open = Some(Open {
                    group_id,
                    seq,
                    start_sample: chunk_end,
                    last_voice_sample: chunk_end,
                    pcm: Vec::new(),
                });
            }
        }

        out
    }

    /// End of stream: close any open segment at its last voice sample.
    /// An open segment that never contained voice emits nothing.
    pub fn flush(&mut self) -> Vec<Segment> {
        let mut out = Vec::new();
        if let Some(o) = self.open.take() {
            if o.last_voice_sample > o.start_sample {
                out.push(self.close(&o, o.last_voice_sample));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    /// Push `pcm` in 10 ms chunks, collecting closed segments.
    fn run(cfg: &SegmenterConfig, pcm: &[i16]) -> Vec<Segment> {
        let mut seg = Segmenter::new(cfg.clone(), 0);
        let mut out = Vec::new();
        for c in fixture::chunks(pcm, 160) {
            out.extend(seg.push(&c));
        }
        out.extend(seg.flush());
        out
    }

    #[test]
    fn vad_opens_closes() {
        // Voice 500 ms, then silence well past the 800 ms hang: exactly one
        // segment closing at the voice end (S1).
        let cfg = SegmenterConfig {
            hang_ms: 800,
            ..Default::default()
        };
        let mut pcm = fixture::tone_ms(440.0, 500, 9_000);
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 1, "one segment, got {}", segs.len());
        let s = &segs[0];
        assert_eq!(s.seq, 0);
        assert!((s.ts_start_ms as i64) < 50, "ts_start={}", s.ts_start_ms);
        assert!(
            (s.ts_end_ms as i64 - 500).abs() <= 20,
            "ts_end={}",
            s.ts_end_ms
        );
        assert!((s.duration_ms as i64 - 500).abs() <= 30);
        assert_eq!(s.pcm.len(), (500 + 800) as usize * 16);
    }

    #[test]
    fn simplex_noise_floor_cut() {
        // Simplex profile value (1200 ms hang): voice 300 ms followed by a
        // low noise floor must still cut at the voice end — noise is not
        // voice (S2).
        let cfg = SegmenterConfig {
            hang_ms: 1200,
            ..Default::default()
        };
        assert_eq!(cfg.hang_ms, 1200, "simplex profile hang value");
        let mut pcm = fixture::tone_ms(300.0, 300, 9_000);
        pcm.extend(fixture::noise_floor_ms(2000, 40));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 1, "one segment, got {}", segs.len());
        assert!(
            (segs[0].ts_end_ms as i64 - 300).abs() <= 20,
            "ts_end={}",
            segs[0].ts_end_ms
        );
    }

    #[test]
    fn split_group_seq() {
        // Continuous voice over 2.5x the max: shared group_id, seq from 0 (S3).
        let cfg = SegmenterConfig {
            max_segment_ms: 2000,
            ..Default::default()
        };
        let pcm = fixture::tone_ms(440.0, 5000, 9_000);
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 3, "three splits, got {}", segs.len());
        for (i, s) in segs.iter().enumerate() {
            assert_eq!(s.seq, i as u32);
            assert_eq!(s.group_id, segs[0].group_id);
            assert_ne!(s.id, segs[(i + 1) % 3].id);
        }
        assert!((segs[0].duration_ms as i64 - 2000).abs() <= 30);
    }
}
