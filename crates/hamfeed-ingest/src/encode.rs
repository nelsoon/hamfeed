//! Opus-in-Ogg encode/decode + atomic clip persist (T6, S6).
//!
//! Behind the `opus` feature (plan): `audiopus` + `ogg` against the system
//! libopus. Files on disk are self-describing Opus-in-Ogg (OpusHead +
//! OpusTags + audio packets), so browsers play them directly. Writes are
//! atomic (`.tmp` → rename); the queue item for a clip only exists after its
//! file has landed.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use audiopus::coder::{Decoder, Encoder};
use audiopus::{Application, Bitrate, Channels, SampleRate};
use ogg::{PacketReader, PacketWriteEndInfo, PacketWriter};

use crate::SAMPLE_RATE;

/// Opus frame: 20 ms at 16 kHz mono.
pub const FRAME_SAMPLES: usize = 320;
/// Archive bitrate: speech remains intelligible, files stay small.
pub const BITRATE_BPS: i32 = 16_000;
/// Live bitrate (007 ear-test finding): the 16 kbit/s archive setting
/// leaves a speech-locked hash on sibilants over `/api/live` (silence
/// encodes trivially, so it only ever rides on talking). 32 kbit/s
/// costs ~2 KB/s extra on the LAN and never touches the archive.
pub const LIVE_BITRATE_BPS: i32 = 32_000;

fn encoder_with(bitrate: i32) -> Result<Encoder> {
    let mut enc = Encoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip)
        .context("cannot create Opus encoder")?;
    enc.set_bitrate(Bitrate::BitsPerSecond(bitrate))
        .context("cannot set Opus bitrate")?;
    Ok(enc)
}

fn encoder() -> Result<Encoder> {
    encoder_with(BITRATE_BPS)
}

/// Encode 16 kHz mono S16 into a self-describing Opus-in-Ogg byte buffer.
pub fn encode_pcm_to_ogg(pcm: &[i16]) -> Result<Vec<u8>> {
    let enc = encoder()?;
    let mut out = Vec::new();
    let serial: u32 = rand_serial();
    let mut w = PacketWriter::new(&mut out);
    w.write_packet(opus_head(), serial, PacketWriteEndInfo::NormalPacket, 0)
        .context("cannot write OpusHead")?;
    w.write_packet(opus_tags(), serial, PacketWriteEndInfo::NormalPacket, 0)
        .context("cannot write OpusTags")?;

    let mut frame = [0i16; FRAME_SAMPLES];
    let mut packet = [0u8; 4096];
    let mut done = 0usize;
    let mut granule = 0u64;
    while done < pcm.len() {
        let take = (pcm.len() - done).min(FRAME_SAMPLES);
        frame.fill(0);
        frame[..take].copy_from_slice(&pcm[done..done + take]);
        let n = enc
            .encode(&frame, &mut packet)
            .context("Opus encode failed")?;
        let bytes = packet[..n].to_vec();
        done += take;
        granule += take as u64;
        w.write_packet(bytes, serial, PacketWriteEndInfo::NormalPacket, granule)
            .context("cannot write Opus packet")?;
    }
    // End-of-stream on an empty final packet so readers see EOS.
    w.write_packet(Vec::new(), serial, PacketWriteEndInfo::EndStream, granule)
        .context("cannot close Ogg stream")?;
    drop(w);
    Ok(out)
}

/// Streaming Opus-in-Ogg encoder for `/api/live` (Slice 3, R7).
/// Unlike [`encode_pcm_to_ogg`] this never writes EOS: the connection is the
/// stream, and each connection starts with its own OpusHead/Tags/serial, so
/// a late joiner decodes from its first byte. Partial 20 ms frames buffer
/// internally (at most one frame of extra latency).
pub struct LiveEncoder {
    enc: Encoder,
    serial: u32,
    granule: u64,
    pending: Vec<i16>,
}

impl LiveEncoder {
    /// Fresh encoder plus the header bytes the connection must send first.
    pub fn new() -> Result<(Vec<u8>, Self)> {
        let serial = rand_serial();
        let mut head = Vec::new();
        {
            let mut w = PacketWriter::new(&mut head);
            // Solo-paged headers (RFC 7845) double as a flush: the
            // header burst is complete bytes on return, no EOS ever.
            w.write_packet(opus_head(), serial, PacketWriteEndInfo::EndPage, 0)
                .context("cannot write OpusHead")?;
            w.write_packet(opus_tags(), serial, PacketWriteEndInfo::EndPage, 0)
                .context("cannot write OpusTags")?;
        }
        Ok((
            head,
            Self {
                enc: encoder_with(LIVE_BITRATE_BPS)?,
                serial,
                granule: 0,
                pending: Vec::new(),
            },
        ))
    }

    /// Feed 16 kHz mono S16; returns Ogg audio-packet bytes (empty until a
    /// full 20 ms frame accumulates). The granule position runs across calls.
    pub fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>> {
        self.pending.extend_from_slice(pcm);
        let mut out = Vec::new();
        {
            let mut w = PacketWriter::new(&mut out);
            let mut frame = [0i16; FRAME_SAMPLES];
            let mut packet = [0u8; 4096];
            while self.pending.len() >= FRAME_SAMPLES {
                frame.copy_from_slice(&self.pending[..FRAME_SAMPLES]);
                self.pending.drain(..FRAME_SAMPLES);
                let n = self
                    .enc
                    .encode(&frame, &mut packet)
                    .context("Opus encode failed")?;
                self.granule += FRAME_SAMPLES as u64;
                // EndPage: one flushed Ogg page per 20 ms packet, so the
                // browser plays with ~200 ms granularity instead of waiting
                // for a 4 KB page to fill (~2 s of near-silence).
                w.write_packet(
                    packet[..n].to_vec(),
                    self.serial,
                    PacketWriteEndInfo::EndPage,
                    self.granule,
                )
                .context("cannot write Opus packet")?;
            }
        }
        Ok(out)
    }
}

/// Decode an Opus-in-Ogg buffer back to 16 kHz mono S16.
/// Garbage in → error out (callers map this to `failed` + triage, S5).
pub fn decode_ogg_to_pcm(bytes: &[u8]) -> Result<Vec<i16>> {
    let mut r = PacketReader::new(Cursor::new(bytes));
    let head = r.read_packet()?.context("empty Ogg stream (no OpusHead)")?;
    if head.data.len() < 19 || &head.data[..8] != b"OpusHead" {
        anyhow::bail!("not an Opus-in-Ogg stream (bad OpusHead)");
    }
    let _tags = r
        .read_packet()?
        .context("truncated Ogg stream (no OpusTags)")?;

    let mut dec =
        Decoder::new(SampleRate::Hz16000, Channels::Mono).context("cannot create Opus decoder")?;
    let mut pcm = Vec::new();
    let mut frame = [0i16; 4096];
    loop {
        let pkt = r.read_packet()?.context("truncated Ogg stream")?;
        if pkt.data.is_empty() {
            break; // EndStream marker.
        }
        let n = dec
            .decode(Some(&pkt.data), &mut frame[..], false)
            .context("Opus decode failed")?;
        pcm.extend_from_slice(&frame[..n]);
    }
    Ok(pcm)
}

fn rand_serial() -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut h = DefaultHasher::new();
    std::process::id().hash(&mut h);
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
        .hash(&mut h);
    h.finish() as u32
}

fn opus_head() -> Vec<u8> {
    let mut h = Vec::with_capacity(19);
    h.extend_from_slice(b"OpusHead");
    h.push(1); // version
    h.push(1); // mono
    h.extend_from_slice(&0u16.to_le_bytes()); // preskip
    h.extend_from_slice(&(SAMPLE_RATE).to_le_bytes()); // input rate
    h.extend_from_slice(&0i16.to_le_bytes()); // gain
    h.push(0); // mapping family: mono, no mapping table
    h
}

fn opus_tags() -> Vec<u8> {
    let vendor = b"hamfeed";
    let mut t = Vec::new();
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    t.extend_from_slice(vendor);
    t.extend_from_slice(&0u32.to_le_bytes()); // no user comments
    t
}

/// Archive layout: `<dir>/YYYY/MM/DD/HHMMSSmmm-<uuid8>.ogg` (plan).
pub fn clip_path(dir: &Path, ts_start_ms: u64, id: &str) -> PathBuf {
    let secs = (ts_start_ms / 1000) as i64;
    let ms = ts_start_ms % 1000;
    // Calendar math without a date crate: days since epoch → civil date.
    let (y, m, d) = civil_from_days(secs / 86_400);
    let (hh, mm, ss) = ((secs % 86_400) / 3600, (secs % 3600) / 60, secs % 60);
    let short = id.chars().take(8).collect::<String>();
    dir.join(format!("{y:04}/{m:02}/{d:02}"))
        .join(format!("{hh:02}{mm:02}{ss:02}{ms:03}-{short}.ogg"))
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    // Howard Hinnant's civil_from_days.
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Metadata for one persisted clip. The queue item (T6 `queue.rs`) is built
/// from this only after [`write_clip_atomic`] has returned.
#[derive(Debug, Clone)]
pub struct ClipMeta {
    pub id: String,
    pub path: PathBuf,
    pub duration_ms: u64,
    pub ts_start_ms: u64,
    pub size_bytes: u64,
}

/// Persist one segment's PCM as Opus-in-Ogg, atomically (`.tmp` → rename).
/// Returns the clip metadata; callers create the queue item from it, so no
/// queue item can ever reference a missing file (S6, first half).
pub fn write_clip_atomic(
    dir: &Path,
    id: &str,
    ts_start_ms: u64,
    duration_ms: u64,
    pcm: &[i16],
) -> Result<ClipMeta> {
    let bytes = encode_pcm_to_ogg(pcm)?;
    let path = clip_path(dir, ts_start_ms, id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let tmp = path.with_extension("ogg.tmp");
    std::fs::write(&tmp, &bytes).with_context(|| format!("cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("cannot rename {} to {}", tmp.display(), path.display()))?;
    Ok(ClipMeta {
        id: id.into(),
        path,
        duration_ms,
        ts_start_ms,
        size_bytes: bytes.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    #[test]
    fn opus_roundtrip() {
        // PCM → .ogg → decode stays byte-plausible: length preserved within
        // one frame, energy within half, still clearly non-silent (S6 input).
        let pcm = fixture::tone_ms(440.0, 1000, 9_000);
        let ogg = encode_pcm_to_ogg(&pcm).expect("encode");
        assert!(!ogg.is_empty());
        assert_eq!(&ogg[..4], b"OggS", "missing Ogg capture pattern");
        let back = decode_ogg_to_pcm(&ogg).expect("decode");
        assert!(
            back.len() >= pcm.len(),
            "decoded {} < original {}",
            back.len(),
            pcm.len()
        );
        assert!(
            back.len() - pcm.len() <= FRAME_SAMPLES,
            "padding beyond one frame: {}",
            back.len() - pcm.len()
        );
        let energy = |v: &[i16]| v.iter().map(|s| (*s as i64).abs()).sum::<i64>() as f64;
        let (e0, e1) = (energy(&pcm), energy(&back[..pcm.len().min(back.len())]));
        assert!((e1 / e0 - 1.0).abs() < 0.5, "energy drifted: {e0} -> {e1}");
        assert!(back.iter().any(|&s| s.abs() > 1000), "decoded to silence");
    }

    #[test]
    fn undecodable_is_error() {
        assert!(decode_ogg_to_pcm(b"definitely not ogg").is_err());
        assert!(decode_ogg_to_pcm(b"OggS garbage").is_err());
    }

    #[test]
    fn live_encoder_headers_and_cadence() {
        let (head, mut enc) = LiveEncoder::new().unwrap();
        // Ogg page magic up front, OpusHead inside the header burst.
        assert_eq!(&head[..4], b"OggS");
        assert!(head.windows(8).any(|w| w == b"OpusHead"));
        // Sub-frame input buffers silently (no runt packets).
        assert!(enc.push(&[0i16; 100]).unwrap().is_empty());
        // Completing the 20 ms frame flushes audio bytes in Ogg pages.
        let bytes = enc.push(&vec![0i16; FRAME_SAMPLES - 100]).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[..4], b"OggS");
    }

    #[test]
    fn live_encoder_spends_more_bits_than_archive() {
        // The live wire runs hotter than the archive on purpose
        // (LIVE_BITRATE_BPS > BITRATE_BPS): sibilant-heavy input must
        // come out measurably larger through LiveEncoder than through
        // the clip encoder, with margin to spare for page layout.
        let a = fixture::tone_ms(5233.0, 2000, 6000);
        let b = fixture::tone_ms(6117.0, 2000, 6000);
        let pcm: Vec<i16> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x.saturating_add(*y))
            .collect();
        let (_, mut live) = LiveEncoder::new().unwrap();
        let mut live_bytes = 0usize;
        for chunk in pcm.chunks(1600) {
            live_bytes += live.push(chunk).unwrap().len();
        }
        let clip_bytes = encode_pcm_to_ogg(&pcm).unwrap().len();
        assert!(
            live_bytes as f64 > clip_bytes as f64 * 1.3,
            "live {live_bytes} should exceed archive {clip_bytes} with margin"
        );
    }

    #[test]
    fn clip_path_layout() {
        // 2026-01-02 03:04:05.006 UTC.
        let ts = 1_767_323_045_006u64;
        let p = clip_path(Path::new("/data"), ts, "abcdef12-xyz");
        assert_eq!(p, Path::new("/data/2026/01/02/030405006-abcdef12.ogg"));
    }
}
