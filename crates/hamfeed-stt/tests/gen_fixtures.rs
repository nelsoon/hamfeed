//! Fixture generator for T8 (run explicitly, never in normal tests).
//!
//! ```sh
//! sh crates/hamfeed-stt/tests/fixtures/gen.sh
//! ```
//! Downloads the two source clips (English: whisper.cpp `jfk.wav` sample;
//! French: a short TTS rendering), normalizes both to 16 kHz mono S16, and
//! encodes `en.ogg` / `fr.ogg` with the project's own Opus encoder.
//! Total stays under 100 KB. Requires network; ignored by default.

use std::path::{Path, PathBuf};

const EN_URL: &str = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav";
// Short French sentence rendered for fixture use.
const FR_URL: &str = "https://translate.google.com/translate_tts?ie=UTF-8&q=Bonjour%20%C3%A0%20tous%20les%20radioamateurs%2C%20le%20relais%20est%20actif%20ce%20soir&tl=fr&client=tw-ob";

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fetch(url: &str, dest: &Path) {
    if dest.exists() {
        return;
    }
    let status = std::process::Command::new("curl")
        .args([
            "-sSL",
            "-A",
            "Mozilla/5.0",
            "-o",
            &dest.to_string_lossy(),
            url,
        ])
        .status()
        .expect("curl must run");
    assert!(status.success(), "download failed: {url}");
}

fn read_wav_mono16(path: &Path) -> Vec<i16> {
    let bytes = std::fs::read(path).expect("read wav");
    assert!(bytes.len() > 44, "wav too short");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut rate = 0u32;
    let mut channels = 0u16;
    let mut pos = 12;
    let mut data = &[][..];
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"fmt " {
            channels = u16::from_le_bytes(bytes[pos + 8..pos + 10].try_into().unwrap());
            rate = u32::from_le_bytes(bytes[pos + 12..pos + 16].try_into().unwrap());
            let bits = u16::from_le_bytes(bytes[pos + 22..pos + 24].try_into().unwrap());
            assert_eq!(bits, 16, "want 16-bit wav");
        } else if id == b"data" {
            data = &bytes[pos + 8..pos + 8 + len];
        }
        pos += 8 + len;
    }
    assert_eq!(channels, 1, "want mono wav");
    assert_eq!(rate, 16_000, "want 16 kHz wav");
    let (pairs, _) = data.as_chunks::<2>();
    pairs.iter().map(|c| i16::from_le_bytes(*c)).collect()
}

fn decode_mp3(path: &Path) -> (Vec<i16>, i32) {
    let file = std::fs::File::open(path).expect("open mp3");
    let mut dec = minimp3::Decoder::new(file);
    let mut pcm = Vec::new();
    let mut rate = 0;
    loop {
        match dec.next_frame() {
            Ok(f) => {
                assert_eq!(f.channels, 1, "want mono mp3, got {}", f.channels);
                rate = f.sample_rate;
                pcm.extend_from_slice(&f.data);
            }
            Err(minimp3::Error::Eof) => break,
            Err(minimp3::Error::SkippedData) => continue,
            Err(e) => panic!("mp3 frame decodes: {e:?}"),
        }
    }
    assert!(!pcm.is_empty(), "mp3 decoded to nothing");
    (pcm, rate)
}

/// Linear resample mono to 16 kHz.
fn to_16k(pcm: &[i16], from_rate: i32) -> Vec<i16> {
    if from_rate == 16_000 {
        return pcm.to_vec();
    }
    let ratio = from_rate as f64 / 16_000.0;
    let n = (pcm.len() as f64 / ratio) as usize;
    (0..n)
        .map(|i| {
            let pos = i as f64 * ratio;
            let a = pcm[pos as usize];
            let b = *pcm.get(pos as usize + 1).unwrap_or(&a);
            let frac = (pos.fract() * 256.0) as i32;
            ((a as i32 * (256 - frac) + b as i32 * frac) / 256) as i16
        })
        .collect()
}

#[test]
#[ignore]
fn gen_fixtures() {
    let d = dir();
    std::fs::create_dir_all(d.join(".src")).unwrap();

    let en_src = d.join(".src/en-jfk.wav");
    fetch(EN_URL, &en_src);
    let en_pcm = read_wav_mono16(&en_src);
    let en_ogg = hamfeed_ingest::encode_pcm_to_ogg(&en_pcm).expect("encode en");
    std::fs::write(d.join("en.ogg"), &en_ogg).unwrap();

    let fr_src = d.join(".src/fr-tts.mp3");
    fetch(FR_URL, &fr_src);
    let (fr_pcm, fr_rate) = decode_mp3(&fr_src);
    let fr_pcm = to_16k(&fr_pcm, fr_rate);
    let fr_ogg = hamfeed_ingest::encode_pcm_to_ogg(&fr_pcm).expect("encode fr");
    std::fs::write(d.join("fr.ogg"), &fr_ogg).unwrap();

    let total = en_ogg.len() + fr_ogg.len();
    println!(
        "en.ogg={} fr.ogg={} total={total}",
        en_ogg.len(),
        fr_ogg.len()
    );
    assert!(total < 100 * 1024, "fixtures must stay under 100KB");
}
