//! Simulated end-to-loop (T12, S7 full + regression over S1–S11).
//!
//! No hardware, no network: the known EN speech fixture plays through
//! `FakeSource` → pipeline (real VAD, real Opus, real whisper, real SQLite)
//! → web. HTTP GET then shows the newest-first card with playback + badges;
//! a corrupt clip shows as an error card. THE e2e test is
//! `e2e_simulated_feed`.

use std::path::{Path, PathBuf};

use hamfeed_pipeline::Pipeline;
use hamfeed_source::{FakeSource, PcmFrame};
use hamfeed_web::{create_app, AppState};

fn hamfeed_ingest_fixture_tone() -> Vec<i16> {
    // 500 ms tone: encodes to a valid clip, then gets bit-rotted.
    hamfeed_ingest::fixture::tone_ms(440.0, 500, 9_000)
}

fn test_model() -> PathBuf {
    if let Ok(p) = std::env::var("HAMFEED_TEST_MODEL") {
        return p.into();
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ggml-tiny.bin")
}

fn test_pipeline(dir: &Path) -> Pipeline {
    let text = format!(
        r#"
[audio]
device = "default"
sample_rate = 16000
[vad]
engine = "energy"
hang_ms = 800
[segment]
max_s = 120
[stt]
model_path = "{}"
lang_whitelist = ["fr", "en"]
[storage]
dir = "{}"
db_path = "{}"
retention_days = 90
[station]
"#,
        test_model().display(),
        dir.join("audio").display(),
        dir.join("e2e.db").display()
    );
    let cfg = hamfeed_config::parse(&text).expect("e2e config");
    Pipeline::open_with(cfg).expect("pipeline opens")
}

#[tokio::test]
async fn e2e_simulated_feed() {
    let dir = std::env::temp_dir().join(format!("hamfeed-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Real EN speech (the T8 fixture) as source frames: decode once, then
    // play the PCM through FakeSource in 100 ms frames.
    let en_ogg = Path::new(env!("CARGO_MANIFEST_DIR")).join("../hamfeed-stt/tests/fixtures/en.ogg");
    let en_ogg = en_ogg.canonicalize().expect("en fixture exists");
    let bytes = std::fs::read(&en_ogg).unwrap();
    let pcm = hamfeed_ingest::decode_ogg_to_pcm(&bytes).expect("fixture decodes");
    assert!(pcm.len() > 16_000, "fixture holds seconds of speech");
    let frames: Vec<PcmFrame> = pcm
        .chunks(1600)
        .map(|c| PcmFrame {
            samples: c.to_vec(),
        })
        .collect();
    let mut src = FakeSource::once(frames);

    let pipe = test_pipeline(&dir);
    let n = pipe
        .run_source(&mut src, 800, 120, true, 150)
        .expect("loop runs");
    assert!(n >= 1, "speech must produce segments");

    // Corrupt transmission: enqueue a valid segment, then bit-rot its clip
    // before the drain — decoding must fail but the row + clip still land.
    pipe.enqueue_segment(&hamfeed_ingest::Segment {
        id: "e2e-corrupt".into(),
        group_id: "e2e-g".into(),
        seq: 0,
        ts_start_ms: 2,
        ts_end_ms: 502,
        duration_ms: 500,
        pcm: hamfeed_ingest_fixture_tone(),
    })
    .expect("enqueue works");
    let clip = hamfeed_ingest::clip_path(&dir.join("audio"), 2, "e2e-corrupt");
    std::fs::write(&clip, b"corrupt on purpose").expect("bit-rot the clip");
    pipe.drain().expect("drain works");

    // Serve it.
    let state = AppState::new(pipe, dir.join("static").to_string_lossy().into_owned());
    let app = create_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = reqwest::Client::new();

    // Feed shows newest-first cards with playback + badges.
    let feed: serde_json::Value = client
        .get(format!("{base}/api/messages?limit=20"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let msgs = feed["messages"].as_array().unwrap();
    assert!(!msgs.is_empty(), "feed must show cards");
    let ts: Vec<u64> = msgs
        .iter()
        .map(|m| m["ts_start_ms"].as_u64().unwrap())
        .collect();
    let sorted = {
        let mut s = ts.clone();
        s.sort_by(|a, b| b.cmp(a));
        s
    };
    assert_eq!(ts, sorted, "feed must be newest-first");
    for m in msgs {
        assert!(m.get("audio_url").is_some(), "card {m} needs playback");
        assert!(m.get("lang").is_some() && m.get("status").is_some());
    }

    // The real speech transcribed to real English text somewhere in the feed.
    let all_text: String = msgs
        .iter()
        .filter(|m| m["status"] == "ok")
        .map(|m| m["transcript"].as_str().unwrap_or("").to_string())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        all_text.contains("America"),
        "expected transcribed speech, got: {all_text:?}"
    );
    assert!(
        msgs.iter()
            .any(|m| m["lang"] == "en" && m["status"] == "ok"),
        "expected an ok en card with badges"
    );

    // Corrupt clip renders as an error card (failed + reason + kept clip).
    let err_card = msgs
        .iter()
        .find(|m| m["id"] == "e2e-corrupt")
        .expect("error card present");
    assert_eq!(err_card["status"], "failed");
    assert!(err_card["fail_reason"].as_str().is_some());
    assert!(err_card["audio_url"].as_str().is_some());

    let _ = std::fs::remove_dir_all(&dir);
}
