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
/// digital silence and low noise floors do not. `beep_split`/`beep_min_ms`
/// come from `[vad]`; a sustained in-band tone after voice then cuts the
/// segment at the turn boundary (B1–B4).
#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    pub hang_ms: u64,
    pub energy_threshold: f32,
    pub max_segment_ms: u64,
    pub beep_split: bool,
    pub beep_min_ms: u64,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            hang_ms: 800,
            energy_threshold: 300.0,
            max_segment_ms: 120_000,
            beep_split: true,
            beep_min_ms: 100,
        }
    }
}

/// Short trailing windows for the idle-drop and `voice_seen` gates run on 50
/// ms; the split evaluation window covers `beep_min_ms`.
const BEEP_SUB_MS: u64 = 50;

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
    /// Set after two consecutive confident-speech windows (loud +
    /// spectrally spread, 100 ms total). A beep with no prior voice never
    /// splits (B5), and a tone-only segment is dropped even when a transient
    /// edge window looks speech-like — one stray window can't mark.
    voice_seen: bool,
    voice_streak: u32,
    pcm: Vec<i16>,
}

/// Goertzel magnitude at `freq_hz` over windowed `win` (single DFT bin).
fn goertzel_mag(win: &[f32], freq_hz: f32) -> f32 {
    let omega = 2.0 * std::f32::consts::PI * freq_hz / SAMPLE_RATE as f32;
    let coeff = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0f32, 0f32);
    for &s in win {
        let s0 = s + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt()
}

/// Hann-windowed copy of `win`: tames rectangular sidelobes so an off-grid
/// beep (880 Hz on a 25 Hz bank) still concentrates in its peak bins
/// instead of smearing across the whole bank.
fn hann(win: &[i16]) -> Vec<f32> {
    let n = win.len();
    win.iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
            s as f32 * w
        })
        .collect()
}

/// Beep band. Multi-tone beeps (e.g. 700+1100 Hz courtesy tones) concentrate
/// here; voice fundamentals sit below it. The 25 Hz grid keeps off-grid
/// beeps (880 Hz, …) inside one bin's main lobe on 50 ms+ windows.
const BANK_LO_HZ: f32 = 600.0;
const BANK_STEP_HZ: f32 = 25.0;
const BANK_BINS: usize = 77; // 600..=2500
/// Guard bins around the band: a strong out-of-band source (440 Hz test
/// tone, mains hum) vetoes the window instead of leaking in.
const GUARD_HZ: [f32; 8] = [100.0, 200.0, 300.0, 400.0, 500.0, 2600.0, 2700.0, 2800.0];

/// In-band spectral peaks of the Hann-windowed `win`: top two (freq,
/// magnitude) plus the total in-band magnitude. One Goertzel bank per
/// evaluation — trivial next to STT.
fn bank_top2(win: &[f32]) -> ((f32, f32), (f32, f32), f32) {
    let mut top = [(0f32, 0f32); 2];
    let mut total = 0f32;
    for i in 0..BANK_BINS {
        let f = BANK_LO_HZ + i as f32 * BANK_STEP_HZ;
        let m = goertzel_mag(win, f);
        total += m;
        if m > top[0].1 {
            top[1] = top[0];
            top[0] = (f, m);
        } else if m > top[1].1 {
            top[1] = (f, m);
        }
    }
    (top[0], top[1], total)
}

/// Bank ratio test on one sub-window: the top two bins dominate, the top
/// peak carries no second harmonic (vowels always do), and no guard bin
/// shows an out-of-band source bleeding in.
fn sub_is_tonal(hw: &[f32], threshold: f32, raw: &[i16]) -> bool {
    if Segmenter::mean_abs(raw) < threshold {
        return false;
    }
    let ((f1, m1), (_, m2), total) = bank_top2(hw);
    // Dual tones only reach ~0.6 (their lobes lift the middle bins);
    // vowels were measured up to ~0.4. The harmonic veto below is what
    // really separates them.
    if total <= 0.0 || m1 < 0.2 * total || m1 + m2 < 0.5 * total {
        return false;
    }
    if 2.0 * f1 <= 3000.0 && goertzel_mag(hw, 2.0 * f1) > 0.4 * m1 {
        return false;
    }
    for g in GUARD_HZ {
        if goertzel_mag(hw, g) > 0.5 * m1 {
            return false;
        }
    }
    true
}

/// True when `win` (a whole number of 50 ms sub-windows) is one sustained
/// beep: every sub-window is one or two steady tones (B3). Single beeps,
/// dual beeps (700+1100 Hz repeater courtesy tones), and test sines pass
/// every window; vowels spread harmonics over many bins, fricatives spread
/// everywhere, and silence fails the level gate — so speech, gaps, and
/// partial fills all fail at least one window.
fn is_tone(win: &[i16], threshold: f32) -> bool {
    let sub = BEEP_SUB_MS as usize * SAMPLE_RATE as usize / 1000;
    if win.is_empty() || !win.len().is_multiple_of(sub) {
        return false;
    }
    win.chunks(sub)
        .all(|s| sub_is_tonal(&hann(s), threshold, s))
}

/// Stride for the tone-run scan: 10 ms pins a beep onset within half a
/// push chunk without changing the 50 ms analysis windows.
const TONE_STRIDE_SAMPLES: usize = 10 * SAMPLE_RATE as usize / 1000;

/// Earliest offset (in samples, relative to `region`) where `need`
/// consecutive 50 ms sub-windows are all tonal, scanning at 10 ms
/// stride. Unlike grid-locked `is_tone`, persistence here is exact: an
/// 80 ms blip can never satisfy `need = 2` no matter how it aligns, so
/// sub-persistence beeps stay glued (B4) on every path.
fn find_tone_run(region: &[i16], need: usize, threshold: f32) -> Option<usize> {
    let sub = BEEP_SUB_MS as usize * SAMPLE_RATE as usize / 1000;
    if region.len() < need * sub || need == 0 {
        return None;
    }
    let last_start = region.len() - need * sub;
    let mut start = 0;
    while start <= last_start {
        let mut ok = true;
        for k in 0..need {
            let w = &region[start + k * sub..start + (k + 1) * sub];
            if !sub_is_tonal(&hann(w), threshold, w) {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(start);
        }
        start += TONE_STRIDE_SAMPLES;
    }
    None
}

/// Gate window for the idle-drop and `voice_seen` tone checks: 50 ms of
/// audio is enough for the bank test, short enough to react fast.
const GATE_SAMPLES: usize = 50 * SAMPLE_RATE as usize / 1000;
/// Trailing audio retained for the gates (covers the longest evaluation
/// window, `beep_min_ms` max 1000 ms).
const RECENT_MAX: usize = 1000 * SAMPLE_RATE as usize / 1000;

/// Energy-VAD segmenter over a sample clock starting at `epoch_ms`.
#[derive(Debug)]
pub struct Segmenter {
    cfg: SegmenterConfig,
    epoch_ms: u64,
    cursor: u64,
    open: Option<Open>,
    recent: Vec<i16>,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig, epoch_ms: u64) -> Self {
        Self {
            cfg,
            epoch_ms,
            cursor: 0,
            open: None,
            recent: Vec::new(),
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

    /// Trailing gate window (last 50 ms of everything pushed), when buffered.
    fn gate_win(&self) -> Option<&[i16]> {
        if self.recent.len() < GATE_SAMPLES {
            return None;
        }
        Some(&self.recent[self.recent.len() - GATE_SAMPLES..])
    }

    /// Beep-split step after a voice chunk was stored (B1/B2). A sustained
    /// in-band tone following voice trims the tone, closes the segment at
    /// the tone start, and leaves nothing open — post-beep speech starts a
    /// NEW group. Tone windows never mark `voice_seen`, so a beep alone can
    /// neither split nor (via the hang-close drop) emit; windows too short
    /// to judge leave it unset, so sub-50 ms blips stay dropped too.
    /// Persistence is stride-exact (`find_tone_run`): only a full
    /// `beep_min_ms` run cuts, so sub-persistence blips glue on any
    /// alignment (B4).
    fn beep_step(&mut self, out: &mut Vec<Segment>) {
        let sub = BEEP_SUB_MS as usize * SAMPLE_RATE as usize / 1000;
        let need = self.cfg.beep_min_ms.div_ceil(BEEP_SUB_MS).max(1) as usize;
        let threshold = self.cfg.energy_threshold;
        let scan = (need + 2) * sub;
        let hit = self.open.as_ref().and_then(|o| {
            if o.pcm.len() < scan {
                return None;
            }
            let base = o.pcm.len() - scan;
            find_tone_run(&o.pcm[base..], need, threshold)
                .map(|off| (o.start_sample + (base + off) as u64, o.voice_seen))
        });
        if let Some((beep_start, voice_seen)) = hit {
            if voice_seen {
                let mut o = self.open.take().expect("just checked");
                o.pcm.truncate((beep_start - o.start_sample) as usize);
                if self.voice_confirmed(&o.pcm) {
                    out.push(self.close(&o, beep_start));
                }
            }
            return;
        }
        // Confident speech only: loud AND spectrally spread, twice running.
        // Loud-but-tonal windows (beeps) break the streak, so a beep-only
        // segment stays unmarked even when a transient edge window looks
        // speech-like — one stray window can't mark.
        let speech = self.gate_win().is_some_and(|w| {
            Segmenter::mean_abs(w) >= self.cfg.energy_threshold
                && !sub_is_tonal(&hann(w), self.cfg.energy_threshold, w)
        });
        if let Some(o) = self.open.as_mut() {
            if speech {
                o.voice_streak += 1;
                if o.voice_streak >= 2 {
                    o.voice_seen = true;
                }
            } else {
                o.voice_streak = 0;
            }
        }
    }

    /// Beep look-back at the voice→silence transition (turn rule). The
    /// live `beep_step` only re-examines trailing audio while voice
    /// chunks flow, so a courtesy beep followed by a carrier drop is
    /// invisible to it; here the drop itself triggers one backward look:
    /// skip the trailing silence (bounded by hang), then require a full
    /// `beep_min_ms` run of sustained tone (`find_tone_run`, stride-exact
    /// like the live path) ending where the voice stopped, extended back
    /// while the tone continues so long beeps trim at onset too. On a hit
    /// the tone is trimmed and the turn closes at the beep start — the
    /// same outcome as `beep_step`, new group on next voice. Runs once
    /// per silence stretch (first non-voice chunk after voice), so
    /// steady-state silence costs one cheap level walk.
    fn beep_lookback(&mut self, chunk_start: u64, out: &mut Vec<Segment>) {
        if !self.cfg.beep_split {
            return;
        }
        let sub = BEEP_SUB_MS as usize * SAMPLE_RATE as usize / 1000;
        let need = self.cfg.beep_min_ms.div_ceil(BEEP_SUB_MS).max(1) as usize;
        let threshold = self.cfg.energy_threshold;
        let hang = self.hang_samples();
        let start_sample = match self.open.as_ref() {
            Some(o) if o.voice_seen && o.last_voice_sample == chunk_start => o.start_sample,
            _ => return,
        };
        let pcm = &self.open.as_ref().expect("just checked").pcm;
        let mut idx = pcm.len();
        let mut skipped: u64 = 0;
        while idx >= sub {
            if Self::mean_abs(&pcm[idx - sub..idx]) >= threshold {
                break;
            }
            idx -= sub;
            skipped += 1;
            if skipped * sub as u64 > hang {
                return;
            }
        }
        let scan_back = (need + 2) * sub;
        let region_base = idx.saturating_sub(scan_back);
        let hit =
            find_tone_run(&pcm[region_base..idx], need, threshold).map(|off| region_base + off);
        let Some(mut run_start) = hit else {
            return;
        };
        // Extend back through a longer beep so the trim lands at onset
        // (bounded by the scanned region: beeps far longer than the
        // window still split, with a rough trim).
        while run_start >= sub && run_start - sub >= region_base && {
            let w = &pcm[run_start - sub..run_start];
            sub_is_tonal(&hann(w), threshold, w)
        } {
            run_start -= sub;
        }
        let beep_start = start_sample + run_start as u64;
        let mut o = self.open.take().expect("just checked");
        o.pcm.truncate((beep_start - o.start_sample) as usize);
        if self.voice_confirmed(&o.pcm) {
            out.push(self.close(&o, beep_start));
        }
    }

    /// Close-time voice confirmation: at least 300 ms (6 windows) of
    /// loud, spectrally-spread audio in the closed PCM. The open-time
    /// latch can be set by a ~150 ms squelch crash before a beep tail,
    /// so a latched segment with almost no voiced content still drops.
    /// Mixed "hello + trailing beep" passes on the speech part; pure
    /// tones fail (tonal windows never count); sub-300 ms real speech is
    /// the accepted loss — whisper hallucinates below that anyway, and
    /// the open gate already demands 100 ms. Weak-signal traffic is
    /// unaffected: anything below the energy gate never opens at all.
    fn voice_confirmed(&self, pcm: &[i16]) -> bool {
        let sub = BEEP_SUB_MS as usize * SAMPLE_RATE as usize / 1000;
        let mut voiced = 0u32;
        for w in pcm.chunks_exact(sub) {
            if Segmenter::mean_abs(w) >= self.cfg.energy_threshold
                && !sub_is_tonal(&hann(w), self.cfg.energy_threshold, w)
            {
                voiced += 1;
                if voiced >= 6 {
                    return true;
                }
            }
        }
        false
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
        // Trailing history for the tone gates (capped well above any window).
        self.recent.extend_from_slice(chunk);
        if self.recent.len() > RECENT_MAX {
            self.recent.drain(..self.recent.len() - RECENT_MAX);
        }

        // While idle, a sustained tone is a beep tail or a kerchunk: drop it
        // instead of opening a tone-only segment, so post-beep speech starts
        // clean (B5). The gate needs 50 ms buffered; stream starts open
        // blind. Computed before the match below to keep borrows disjoint.
        let idle_tonal = self.cfg.beep_split
            && self
                .gate_win()
                .is_some_and(|w| is_tone(w, self.cfg.energy_threshold));

        if voice {
            match self.open.as_mut() {
                Some(o) => {
                    o.last_voice_sample = chunk_end;
                    o.pcm.extend_from_slice(chunk);
                }
                None => {
                    if !idle_tonal {
                        self.open = Some(Open {
                            group_id: uuid::Uuid::new_v4().to_string(),
                            seq: 0,
                            start_sample: chunk_start,
                            last_voice_sample: chunk_end,
                            voice_seen: false,
                            voice_streak: 0,
                            pcm: chunk.to_vec(),
                        });
                    }
                }
            }
            if self.cfg.beep_split {
                self.beep_step(&mut out);
            } else if let Some(o) = self.open.as_mut() {
                o.voice_seen = true;
            }
        } else if self.open.is_some() {
            self.open
                .as_mut()
                .expect("just checked")
                .pcm
                .extend_from_slice(chunk);
            // Turn rule first: a beep ending at the drop cuts here even
            // when the following silence is shorter than hang.
            self.beep_lookback(chunk_start, &mut out);
            if let Some(o) = self.open.as_mut() {
                if chunk_end - o.last_voice_sample >= self.hang_samples() {
                    let o = self.open.take().expect("just checked");
                    // Tone-only segments (beep tails, kerchunks) are
                    // dropped, never transcribed — only voice-bearing ones
                    // close (B5). The latch alone is not enough: a crash
                    // can set it before a beep tail, so the closed audio
                    // must confirm 300 ms of voiced content too.
                    if o.voice_seen && self.voice_confirmed(&o.pcm) {
                        out.push(self.close(&o, o.last_voice_sample));
                    }
                }
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
                    // Same transmission continues: voice was already seen.
                    voice_seen: true,
                    voice_streak: 0,
                    pcm: Vec::new(),
                });
            }
        }

        out
    }

    /// End of stream: close any open segment at its last voice sample.
    /// An open segment that never contained voice emits nothing — a
    /// tone-only tail is voice-less too (`voice_seen` gates it).
    pub fn flush(&mut self) -> Vec<Segment> {
        let mut out = Vec::new();
        if let Some(o) = self.open.take() {
            if o.voice_seen && o.last_voice_sample > o.start_sample && self.voice_confirmed(&o.pcm)
            {
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

    /// Speech + 880 Hz beep + speech: the beep cuts the turn (B1), the tail
    /// starts a NEW group (B2), and the beep itself is trimmed, not stored.
    #[test]
    fn beep_splits_turn() {
        let cfg = SegmenterConfig {
            hang_ms: 800,
            ..Default::default()
        };
        let speech_ms = 6 * (150 + 80); // speech_like(6)
        let mut pcm = fixture::speech_like(6);
        pcm.extend(fixture::tone_ms(880.0, 300, 9_000));
        pcm.extend(fixture::speech_like(6));
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 2, "beep must cut, got {}", segs.len());
        assert_ne!(segs[0].group_id, segs[1].group_id, "new turn, new group");
        assert_eq!(segs[0].seq, 0);
        assert_eq!(segs[1].seq, 0);
        // First segment ends where the speech ends (beep trimmed), within a
        // push chunk of slop.
        assert!(
            (segs[0].ts_end_ms as i64 - speech_ms as i64).abs() <= 30,
            "ts_end={} want ~{speech_ms}",
            segs[0].ts_end_ms
        );
        // Second segment starts after the beep, not inside it.
        let beep_end = speech_ms + 300;
        assert!(
            (segs[1].ts_start_ms as i64 - beep_end as i64).abs() <= 40,
            "ts_start={} want ~{beep_end}",
            segs[1].ts_start_ms
        );
    }

    /// A blip far under `beep_min_ms` never cuts (B4 persistence gate).
    /// 40 ms vs 100 ms: the margin must exceed the 50 ms analysis window,
    /// inside which a tone-adjacent mixed sub-window still reads tonal.
    #[test]
    fn short_beep_ignored() {
        let cfg = SegmenterConfig {
            hang_ms: 800,
            ..Default::default()
        };
        let mut pcm = fixture::speech_like(4);
        pcm.extend(fixture::tone_ms(1000.0, 40, 9_000));
        pcm.extend(fixture::speech_like(4));
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 1, "short beep must not cut, got {}", segs.len());
    }

    /// A beep with no prior voice neither splits nor emits (B5): kerchunks
    /// and tails must not become transcribed clips.
    #[test]
    fn beep_only_dropped() {
        let cfg = SegmenterConfig {
            ..Default::default()
        };
        let mut pcm = fixture::tone_ms(1000.0, 400, 9_000);
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert!(segs.is_empty(), "tone-only must drop, got {}", segs.len());
    }

    /// A squelch crash latched as voice followed by a sub-persistence
    /// beep must not become a clip: the closed audio holds ~150 ms of
    /// voiced content, under the 300 ms confirmation floor. Regression:
    /// prod showed 85–128 ms "Bye. Thank you." slivers (whisper
    /// hallucinating on beeps) wearing carried callsign badges.
    #[test]
    fn crash_then_beep_sliver_dropped() {
        let cfg = SegmenterConfig {
            ..Default::default()
        };
        let mut pcm = fixture::noise_floor_ms(150, 9_000);
        pcm.extend(fixture::tone_ms(1000.0, 80, 9_000));
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert!(
            segs.is_empty(),
            "crash+beep sliver must drop, got {}",
            segs.len()
        );
    }

    /// Three syllables (~700 ms) confirm well above the floor: the
    /// confirmation must not eat short but real speech.
    #[test]
    fn short_speech_confirms() {
        let cfg = SegmenterConfig {
            ..Default::default()
        };
        let mut pcm = fixture::speech_like(3);
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 1, "short speech must keep, got {}", segs.len());
    }

    /// Syllable bursts are irregular: speech-like audio must not false-split
    /// even though each burst is a pure tone (B3 stability gate).
    #[test]
    fn speech_bursts_not_split() {
        let cfg = SegmenterConfig {
            hang_ms: 800,
            ..Default::default()
        };
        let mut pcm = fixture::speech_like(10);
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 1, "bursts must not cut, got {}", segs.len());
    }

    /// A dual-tone courtesy beep (700+1100 Hz, like the real repeater) cuts
    /// just like a single beep (B1/B3).
    #[test]
    fn dual_beep_splits_turn() {
        let cfg = SegmenterConfig {
            hang_ms: 800,
            ..Default::default()
        };
        let speech_ms = 6 * (150 + 80); // speech_like(6)
        let mut pcm = fixture::speech_like(6);
        pcm.extend(fixture::dual_tone_ms(700.0, 1100.0, 300, 9_000));
        pcm.extend(fixture::speech_like(6));
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 2, "dual beep must cut, got {}", segs.len());
        assert_ne!(segs[0].group_id, segs[1].group_id, "new turn, new group");
        assert!(
            (segs[0].ts_end_ms as i64 - speech_ms as i64).abs() <= 30,
            "ts_end={} want ~{speech_ms}",
            segs[0].ts_end_ms
        );
    }

    /// Real repeater audio (10 s trim of a live FR repeater clip: two speech
    /// turns plus courtesy-beep blips). With beep-split the blips are
    /// dropped and only the two turns remain; without it all four land.
    /// Regression test for the operator's 10:27:51 report (B1/B5).
    /// Needs the `opus` feature for fixture decode (on in CI --all-features).
    #[cfg(feature = "opus")]
    #[test]
    fn real_repeater_beeps_dropped() {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/repeater-beeps.ogg"
        ))
        .expect("beep fixture checked in");
        let pcm = crate::decode_ogg_to_pcm(&bytes).expect("fixture decodes");
        let run = |beep_split: bool| {
            let cfg = SegmenterConfig {
                hang_ms: 800,
                beep_split,
                ..Default::default()
            };
            let mut seg = Segmenter::new(cfg, 0);
            let mut out = Vec::new();
            for c in fixture::chunks(&pcm, 1600) {
                out.extend(seg.push(&c));
            }
            out.extend(seg.flush());
            out
        };
        // Close-time confirmation drops tone-only blips with or without
        // split; unsplit, the two turns land whole (beeps glued on).
        // Compact summary: a full PCM dump on failure floods the log.
        let summary = |segs: &[Segment]| {
            segs.iter()
                .map(|s| (s.ts_start_ms, s.ts_end_ms))
                .collect::<Vec<_>>()
        };
        let off = run(false);
        assert_eq!(
            off.len(),
            2,
            "tone-only blips must drop: {:?}",
            summary(&off)
        );
        let on = run(true);
        assert_eq!(on.len(), 2, "beep blips must drop: {:?}", summary(&on));
        assert!(
            (on[0].ts_end_ms as i64 - 1600).abs() <= 200,
            "first turn end: {}",
            on[0].ts_end_ms
        );
        assert!(
            (on[1].ts_start_ms as i64 - 4700).abs() <= 200,
            "second turn start: {}",
            on[1].ts_start_ms
        );
    }

    /// Turn rule: voice, courtesy beep, then a SHORT silence (under hang)
    /// and more voice still cuts at the beep — the live beep_step never
    /// re-examines trailing audio once voice chunks stop, so the silence
    /// transition must look back. Regression test for the 08:48:55
    /// report (four glued transmissions, 1100 Hz beeps).
    #[test]
    fn beep_then_silence_splits() {
        let cfg = SegmenterConfig {
            hang_ms: 800,
            ..Default::default()
        };
        let speech_ms = 4 * (150 + 80); // speech_like(4)
        let mut pcm = fixture::speech_like(4);
        pcm.extend(fixture::tone_ms(1100.0, 120, 9_000));
        pcm.extend(fixture::silence_ms(400));
        pcm.extend(fixture::speech_like(4));
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 2, "beep+silence must cut, got {}", segs.len());
        assert_ne!(segs[0].group_id, segs[1].group_id, "new turn, new group");
        assert_eq!(segs[0].seq, 0);
        assert_eq!(segs[1].seq, 0);
        assert!(
            (segs[0].ts_end_ms as i64 - speech_ms as i64).abs() <= 40,
            "first ends at beep start ~{speech_ms}, got {}",
            segs[0].ts_end_ms
        );
        // Second speech resumes after the 400 ms drop: the beep and the
        // silence belong to neither turn.
        let speech2_start = speech_ms + 120 + 400;
        assert!(
            (segs[1].ts_start_ms as i64 - speech2_start as i64).abs() <= 40,
            "second starts at resumed speech ~{speech2_start}, got {}",
            segs[1].ts_start_ms
        );
    }

    /// A sub-persistence beep before silence still glues: 40 ms cannot
    /// carry a turn boundary even with the look-back (same 50 ms
    /// analysis-window margin as above).
    #[test]
    fn short_beep_then_silence_ignored() {
        let cfg = SegmenterConfig {
            hang_ms: 800,
            ..Default::default()
        };
        let mut pcm = fixture::speech_like(4);
        pcm.extend(fixture::tone_ms(1100.0, 40, 9_000));
        pcm.extend(fixture::silence_ms(400));
        pcm.extend(fixture::speech_like(4));
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 1, "short beep must glue, got {}", segs.len());
    }

    /// A beep with no prior voice followed by speech yields exactly the
    /// speech segment (B5 holds across the look-back too).
    #[test]
    fn beep_only_then_speech_single() {
        let cfg = SegmenterConfig {
            ..Default::default()
        };
        let mut pcm = fixture::tone_ms(1100.0, 400, 9_000);
        pcm.extend(fixture::silence_ms(1500));
        pcm.extend(fixture::speech_like(4));
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 1, "only speech emits, got {}", segs.len());
        assert!(
            (segs[0].ts_start_ms as i64 - 1900).abs() <= 60,
            "starts at speech, got {}",
            segs[0].ts_start_ms
        );
    }

    /// `beep_split = false` restores Slice 1 behavior: the beep glues (B4).
    #[test]
    fn beep_split_disabled() {
        let cfg = SegmenterConfig {
            hang_ms: 800,
            beep_split: false,
            ..Default::default()
        };
        let mut pcm = fixture::speech_like(6);
        pcm.extend(fixture::tone_ms(880.0, 300, 9_000));
        pcm.extend(fixture::speech_like(6));
        pcm.extend(fixture::silence_ms(1500));
        let segs = run(&cfg, &pcm);
        assert_eq!(segs.len(), 1, "disabled must glue, got {}", segs.len());
    }
}
