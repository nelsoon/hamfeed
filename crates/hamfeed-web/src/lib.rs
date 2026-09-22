//! hamfeed-web: REST + SSE live feed + static UI (T11).
//!
//! Thin reader/writer over store + pipeline helpers (group contract): no
//! business logic lives here. Serves the T10-approved feed structure against
//! live data; failed rows render error cards with working triage buttons.

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use futures_core::Stream;
use hamfeed_pipeline::{drop_with_config, Pipeline};
use hamfeed_store::{Message, SearchQuery, TriageAction};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex};

/// Shared web state: pipeline behind a mutex, fan-out for live events.
#[derive(Clone)]
pub struct AppState {
    pub pipeline: Arc<Mutex<Pipeline>>,
    pub tx: broadcast::Sender<serde_json::Value>,
    pub static_dir: String,
}

impl AppState {
    pub fn new(pipeline: Pipeline, static_dir: String) -> Self {
        let (tx, _) = broadcast::channel(64);
        Self {
            pipeline: Arc::new(Mutex::new(pipeline)),
            tx,
            static_dir,
        }
    }
}

/// One feed card as JSON (mirrors the T10 card anatomy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMessage {
    pub id: String,
    pub ts_start_ms: u64,
    pub lang: String,
    pub lang_conf: f64,
    pub transcript: String,
    pub stt_conf: f64,
    pub conf_flag: String,
    pub status: String,
    pub fail_reason: Option<String>,
    pub audio_url: Option<String>,
    pub duration_ms: Option<u64>,
    pub short_flag: bool,
    pub review_flag: String,
    /// Operator correction, if any (004). `transcript` stays the model's
    /// original; cards toggle between the two.
    pub corrected_text: Option<String>,
    /// Sender identity (Slice 2, 003): callsign + callbook name + how the
    /// link was made (heard|carried|suggested|confirmed|none).
    pub sender_callsign: Option<String>,
    pub sender_name: Option<String>,
    pub sender_source: String,
    /// Alert bitmask: 1 = for-you, 2 = emergency.
    pub alert: i32,
    /// Noise kind for the UI badge (`short`|`no-words`|`repeat`), so a
    /// revealed junk row advertises why it hides. `None` on traffic.
    pub noise: Option<String>,
}

/// Noise kind: exactly the hide-noise verdict (badge ⟺ hidden when the
/// checkbox is on), so the two can never disagree. Heard senders and
/// failed rows are never noise — same exemptions as the predicate.
pub fn noise_kind(
    short: bool,
    transcript: &str,
    sender_source: &str,
    status: &str,
    dups: &std::collections::HashSet<String>,
) -> Option<&'static str> {
    if sender_source == "heard" || status == "failed" {
        return None;
    }
    if short {
        return Some("short");
    }
    if hamfeed_store::transcript_is_noise_text(transcript) {
        return Some("no-words");
    }
    if dups.contains(transcript) {
        return Some("repeat");
    }
    None
}

impl ApiMessage {
    fn from(m: &Message, dups: &std::collections::HashSet<String>) -> Self {
        Self {
            id: m.id.clone(),
            ts_start_ms: m.ts_start_ms,
            lang: m.lang.clone(),
            lang_conf: m.lang_conf,
            transcript: m.transcript.clone(),
            stt_conf: m.stt_conf,
            conf_flag: m.conf_flag.clone(),
            status: m.status.clone(),
            fail_reason: m.fail_reason.clone(),
            audio_url: m
                .audio_path
                .as_ref()
                .map(|_| format!("/audio/{}/play.wav", m.id)),
            duration_ms: m.duration_ms,
            short_flag: m.short_flag,
            review_flag: m.review_flag.clone(),
            corrected_text: m.corrected_text.clone(),
            sender_callsign: m.sender_callsign.clone(),
            sender_name: m.sender_name.clone(),
            sender_source: m.sender_source.clone(),
            alert: m.alert,
            noise: noise_kind(
                m.short_flag,
                &m.transcript,
                &m.sender_source,
                &m.status,
                dups,
            )
            .map(str::to_string),
        }
    }
}

#[derive(Debug, Deserialize)]
struct MessagesQuery {
    limit: Option<usize>,
    order: Option<String>,
    cursor: Option<String>,
    hide_noise: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SearchParams {
    q: Option<String>,
    from: Option<u64>,
    to: Option<u64>,
    hide_noise: Option<bool>,
    limit: Option<usize>,
    cursor: Option<String>,
    status: Option<String>,
    sender: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ConfirmBody {
    callsign: String,
}

#[derive(Debug, Deserialize)]
struct TriageBody {
    reason: Option<String>,
    delete_audio: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct CorrectBody {
    text: Option<String>,
}

/// Corrections are labels, not essays: a 120 s transmission holds on the
/// order of two thousand characters; beyond this the request is rejected
/// (413) instead of stored.
pub const MAX_CORRECTION_LEN: usize = 4000;

pub fn create_app(state: AppState) -> Router {
    let static_dir = state.static_dir.clone();
    Router::new()
        .route("/api/status", get(api_status))
        .route("/api/level", get(api_level))
        .route("/api/messages", get(api_messages))
        .route("/api/search", get(api_search))
        .route("/api/messages/:id/:action", post(api_triage))
        .route("/api/messages/:id/confirm-sender", post(api_confirm_sender))
        .route("/api/messages/:id/correct", post(api_correct))
        .route("/api/export/training", get(api_export_training))
        .route("/api/events", get(api_events))
        .route("/api/live", get(api_live))
        .route("/api/profile", get(api_get_profile).post(api_set_profile))
        .route("/api/source", get(api_get_source))
        .route("/api/source/channel", post(api_set_source_channel))
        .route("/audio/:id", get(api_audio))
        .route("/audio/:id/play.wav", get(api_audio_wav))
        .fallback_service(tower_http::services::ServeDir::new(static_dir))
        .with_state(state)
}

async fn api_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let pipe = state.pipeline.lock().await;
    let latest = pipe.latest(1).unwrap_or_default();
    let last_voice_ts = latest.first().map(|m| m.ts_start_ms).unwrap_or(0);
    let failed = pipe
        .store()
        .search(&SearchQuery {
            status: Some("failed".into()),
            limit: 1000,
            ..Default::default()
        })
        .map(|p| p.messages.len())
        .unwrap_or(0);
    let count = pipe.store().count().unwrap_or(0);
    Json(serde_json::json!({
        "last_voice_ts": last_voice_ts,
        "message_count": count,
        "failed_count": failed,
        "queue_depth": pipe.queue_len(),
    }))
}

async fn api_level(State(state): State<AppState>) -> Json<serde_json::Value> {
    let pipe = state.pipeline.lock().await;
    let last_voice_ts = pipe
        .latest(1)
        .unwrap_or_default()
        .first()
        .map(|m| m.ts_start_ms)
        .unwrap_or(0);
    let now = hamfeed_pipeline::now_ms();
    let age_s = now.saturating_sub(last_voice_ts) / 1000;
    Json(serde_json::json!({
        "last_voice_ts": last_voice_ts,
        "age_s": age_s,
        "live": last_voice_ts > 0 && age_s < 300,
    }))
}

async fn api_messages(
    State(state): State<AppState>,
    Query(q): Query<MessagesQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(order) = &q.order {
        if order != "new" {
            return Err(StatusCode::BAD_REQUEST);
        }
    }
    let pipe = state.pipeline.lock().await;
    let page = pipe
        .store()
        .search(&SearchQuery {
            limit: q.limit.unwrap_or(20),
            cursor: q.cursor,
            hide_noise: q.hide_noise.unwrap_or(false),
            ..Default::default()
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let dups = pipe.store().duplicate_transcripts().unwrap_or_default();
    Ok(Json(serde_json::json!({
        "messages": page.messages.iter().map(|m| ApiMessage::from(m, &dups)).collect::<Vec<_>>(),
        "next_cursor": page.next_cursor,
    })))
}

async fn api_search(
    State(state): State<AppState>,
    Query(q): Query<SearchParams>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let pipe = state.pipeline.lock().await;
    let page = pipe
        .store()
        .search(&SearchQuery {
            text: q.q,
            from_ms: q.from,
            to_ms: q.to,
            hide_noise: q.hide_noise.unwrap_or(false),
            status: q.status,
            sender: q.sender,
            limit: q.limit.unwrap_or(20),
            cursor: q.cursor,
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let dups = pipe.store().duplicate_transcripts().unwrap_or_default();
    Ok(Json(serde_json::json!({
        "messages": page.messages.iter().map(|m| ApiMessage::from(m, &dups)).collect::<Vec<_>>(),
        "next_cursor": page.next_cursor,
    })))
}

async fn api_triage(
    State(state): State<AppState>,
    Path((id, action)): Path<(String, String)>,
    body: Option<Json<TriageBody>>,
) -> Result<Json<ApiMessage>, (StatusCode, String)> {
    let body = body.map(|b| b.0);
    let (triage, needs_drain) = {
        let pipe = state.pipeline.lock().await;
        let t = match action.as_str() {
            "keep" => TriageAction::Keep,
            "drop" => drop_with_config(pipe.config(), body.as_ref().and_then(|b| b.delete_audio)),
            "retry" => TriageAction::Retry,
            "flag" => TriageAction::Flag {
                reason: body.as_ref().and_then(|b| b.reason.clone()),
            },
            _ => return Err((StatusCode::NOT_FOUND, "unknown action".into())),
        };
        let retry = matches!(t, TriageAction::Retry);
        (t, retry)
    };
    {
        let pipe = state.pipeline.lock().await;
        if let Err(e) = pipe.set_triage(&id, triage) {
            // Drop the guard before mapping: map_store_err re-locks the
            // same mutex and would deadlock on it.
            drop(pipe);
            return Err(map_store_err(&state, &id, e).await);
        }
    }
    if needs_drain {
        // Retry re-transcribes off the async runtime; fresh rows broadcast
        // as SSE events when they land.
        let state2 = state.clone();
        tokio::task::spawn_blocking(move || {
            let pipe = state2.pipeline.blocking_lock();
            if pipe.drain().is_ok() {
                if let Ok(latest) = pipe.latest(64) {
                    let dups = pipe.store().duplicate_transcripts().unwrap_or_default();
                    for m in &latest {
                        let _ = state2
                            .tx
                            .send(serde_json::to_value(ApiMessage::from(m, &dups)).unwrap());
                    }
                }
            }
        });
    }
    let pipe = state.pipeline.lock().await;
    let msg = pipe
        .store()
        .get(&id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "no such message".into()))?;
    let dups = pipe.store().duplicate_transcripts().unwrap_or_default();
    let api = ApiMessage::from(&msg, &dups);
    let _ = state.tx.send(serde_json::to_value(&api).unwrap());
    Ok(Json(api))
}

/// Map a store write failure to 404 (row gone) or 500 (row exists, so
/// the write itself failed). Only runs on the error path.
async fn map_store_err(state: &AppState, id: &str, e: anyhow::Error) -> (StatusCode, String) {
    let exists = state
        .pipeline
        .lock()
        .await
        .store()
        .get(id)
        .map(|o| o.is_some())
        .unwrap_or(false);
    if exists {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    } else {
        (StatusCode::NOT_FOUND, "no such message".into())
    }
}

/// Live monitor (Slice 3, R7): the channel as chained Opus-in-Ogg over
/// chunked HTTP. The pipeline process owns the tap, so each connection
/// subscribes to its relay socket (`<storage.dir>/live.sock`, raw LE i16)
/// and encodes from the current tail — late joiners decode from byte zero
/// via their own OpusHead/serial. No relay (pipeline down or restarting)
/// is 503, never silence: the UI shows stopped instead of hanging.
/// Dropping the connection just ends the stream.
async fn api_live(State(state): State<AppState>) -> Result<Response, (StatusCode, String)> {
    use axum::body::Bytes;
    use tokio::io::AsyncReadExt as _;
    // Socket path mirrors the pipeline relay; the guard drops before any
    // blocking I/O below.
    let sock = {
        let pipe = state.pipeline.lock().await;
        std::path::PathBuf::from(&pipe.config().storage.dir).join("live.sock")
    };
    let relay = match tokio::net::UnixStream::connect(&sock).await {
        Ok(s) => s,
        Err(_) => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "live capture unavailable (pipeline down or restarting)".into(),
            ));
        }
    };
    let (head, enc) = hamfeed_ingest::LiveEncoder::new()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let stream = futures_util::stream::unfold(
        (Some(head), relay, enc),
        move |(mut head, mut relay, mut enc)| async move {
            loop {
                if let Some(h) = head.take() {
                    let st = (None, relay, enc);
                    return Some((Ok::<_, std::io::Error>(Bytes::from(h)), st));
                }
                // One 20 ms frame per read; the relay only ever sends live
                // PCM, so a stall here means the other end went away.
                let mut frame = [0u8; 640];
                if relay.read_exact(&mut frame).await.is_err() {
                    return None;
                }
                let pcm: Vec<i16> = frame
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c))
                    .collect();
                match enc.push(&pcm) {
                    Ok(bytes) if !bytes.is_empty() => {
                        let st = (None, relay, enc);
                        return Some((Ok(Bytes::from(bytes)), st));
                    }
                    Ok(_) => {}
                    // Encoder died mid-stream: end it; the UI shows
                    // stopped rather than hanging on silence.
                    Err(_) => return None,
                }
            }
        },
    );
    Ok((
        [
            (header::CONTENT_TYPE, "audio/ogg"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        Body::from_stream(stream),
    )
        .into_response())
}

/// Disaster profiles (Slice 3, R1/R4): which detector set is live.
/// The banner polls this; a switch broadcasts so open feeds update too.
async fn api_get_profile(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let pipe = state.pipeline.lock().await;
    let active = pipe
        .store()
        .active_profile()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let profiles = pipe
        .store()
        .list_profiles()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(serde_json::json!({
        "active": active,
        "profiles": profiles
            .iter()
            .map(|p| serde_json::json!({"name": p.name, "cues": p.cues}))
            .collect::<Vec<_>>(),
    })))
}

/// SDR source state: input kind, configured channels, and the wanted
/// channel (settings override, else config default — same resolution
/// the pipeline capture loop uses at open time).
async fn api_get_source(State(state): State<AppState>) -> Json<serde_json::Value> {
    let pipe = state.pipeline.lock().await;
    let cfg = pipe.config();
    // Manual UI tune wins when set and in range (same rule as the
    // pipeline loop); otherwise the active preset channel tunes.
    let tune = pipe
        .store()
        .sdr_freq()
        .unwrap_or(None)
        .filter(|f| hamfeed_source::FREQ_RANGE.contains(f))
        .map(|f| {
            let m = pipe.store().sdr_mode().unwrap_or_else(|_| "nbfm".into());
            (None::<String>, f, m)
        });
    let (active, freq_hz, mode) = match tune {
        Some((_, f, m)) => (None::<String>, Some(f), m),
        None => {
            let ch = cfg
                .sdr
                .channel(
                    &pipe
                        .store()
                        .sdr_channel()
                        .unwrap_or(None)
                        .unwrap_or_default(),
                )
                .or_else(|| cfg.sdr.active_channel());
            (
                ch.map(|c| c.name.clone()),
                ch.map(|c| c.freq_hz),
                ch.map(|c| c.mode.clone()).unwrap_or_else(|| "nbfm".into()),
            )
        }
    };
    Json(serde_json::json!({
        "kind": cfg.source.kind,
        "active": active,
        "freq_hz": freq_hz,
        "mode": mode,
        "modes": hamfeed_source::SUPPORTED_MODES,
        "channels": cfg.sdr.channels.iter().map(|c| serde_json::json!({
            "name": c.name, "freq_hz": c.freq_hz, "mode": c.mode,
        })).collect::<Vec<_>>(),
    }))
}

#[derive(Debug, Deserialize)]
struct ChannelBody {
    name: Option<String>,
    freq_hz: Option<f64>,
    mode: Option<String>,
}

/// Retune the SDR (new traffic only; history untouched): either a
/// preset `name` or a manual `freq_hz` (Hz) with optional `mode`
/// (`nbfm` default). Exactly one of name/freq_hz is required (400);
/// unknown names are 404, never a silent no-op; out-of-range
/// frequencies and unknown modes are 400. The live stream ends
/// itself and the pipeline reopens on the new tune. Broadcasts like
/// triage so the selector updates without reload.
async fn api_set_source_channel(
    State(state): State<AppState>,
    Json(body): Json<ChannelBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let pipe = state.pipeline.lock().await;
    match (body.name, body.freq_hz) {
        (Some(name), None) => {
            if pipe.config().sdr.channel(&name).is_none() {
                return Err((StatusCode::NOT_FOUND, "unknown channel".into()));
            }
            pipe.store()
                .set_sdr_channel(&name)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            drop(pipe);
            let _ = state.tx.send(serde_json::json!({"channel": name}));
            Ok(Json(serde_json::json!({"active": name})))
        }
        (None, Some(freq_hz)) => {
            if !freq_hz.is_finite() || !hamfeed_source::FREQ_RANGE.contains(&freq_hz) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "freq_hz out of range (want 1 MHz..=6 GHz)".into(),
                ));
            }
            let mode = body.mode.unwrap_or_else(|| "nbfm".into());
            if !hamfeed_source::SUPPORTED_MODES.contains(&mode.as_str()) {
                return Err((StatusCode::BAD_REQUEST, "unsupported mode".into()));
            }
            pipe.store()
                .set_sdr_freq(freq_hz)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            pipe.store()
                .set_sdr_mode(&mode)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            drop(pipe);
            let _ = state
                .tx
                .send(serde_json::json!({"freq_hz": freq_hz, "mode": mode}));
            Ok(Json(
                serde_json::json!({"active": None::<String>, "freq_hz": freq_hz, "mode": mode}),
            ))
        }
        _ => Err((
            StatusCode::BAD_REQUEST,
            "need exactly one of name, freq_hz".into(),
        )),
    }
}

#[derive(Debug, Deserialize)]
struct ProfileBody {
    name: String,
}

/// Switch the active profile (new traffic only; history untouched, R5).
/// Unknown names are 404 via [`hamfeed_store::UnknownProfile`]; anything
/// else is 500 (verdict pattern). Switches broadcast like triage.
async fn api_set_profile(
    State(state): State<AppState>,
    Json(body): Json<ProfileBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let active = {
        let pipe = state.pipeline.lock().await;
        if let Err(e) = pipe.store().set_active_profile(&body.name) {
            if e.downcast_ref::<hamfeed_store::UnknownProfile>().is_some() {
                return Err((StatusCode::NOT_FOUND, "unknown profile".into()));
            }
            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
        pipe.store()
            .active_profile()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    };
    let _ = state.tx.send(serde_json::json!({"profile": active}));
    Ok(Json(serde_json::json!({"active": active})))
}

/// Operator sender confirm/correct (003 S8): attach the verdict and
/// teach the voice library. Unknown ids are 404, non-callsigns 400.
/// Broadcasts like triage so live cards update without reload.
async fn api_confirm_sender(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ConfirmBody>,
) -> Result<Json<ApiMessage>, (StatusCode, String)> {
    if !hamfeed_callsign::is_valid(&hamfeed_callsign::normalize(&body.callsign)) {
        return Err((StatusCode::BAD_REQUEST, "not a callsign".into()));
    }
    {
        let pipe = state.pipeline.lock().await;
        if let Err(e) = pipe.confirm_sender(&id, &body.callsign) {
            drop(pipe);
            return Err(map_store_err(&state, &id, e).await);
        }
    }
    let pipe = state.pipeline.lock().await;
    let msg = pipe
        .store()
        .get(&id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "no such message".into()))?;
    let dups = pipe.store().duplicate_transcripts().unwrap_or_default();
    let api = ApiMessage::from(&msg, &dups);
    let _ = state.tx.send(serde_json::to_value(&api).unwrap());
    Ok(Json(api))
}

/// Operator transcript correction (004): store the true text beside the
/// model's guess. Missing/blank text clears the correction. Unknown ids
/// are 404 (triage convention). Broadcasts like triage so live cards
/// update without reload.
async fn api_correct(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Option<Json<CorrectBody>>,
) -> Result<Json<ApiMessage>, (StatusCode, String)> {
    if body
        .as_ref()
        .and_then(|b| b.text.as_deref())
        .is_some_and(|t| t.chars().count() > MAX_CORRECTION_LEN)
    {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("correction over {MAX_CORRECTION_LEN} chars"),
        ));
    }
    {
        let pipe = state.pipeline.lock().await;
        if let Err(e) = pipe
            .store()
            .set_correction(&id, body.as_ref().and_then(|b| b.text.as_deref()))
        {
            drop(pipe);
            return Err(map_store_err(&state, &id, e).await);
        }
    }
    let pipe = state.pipeline.lock().await;
    let msg = pipe
        .store()
        .get(&id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "no such message".into()))?;
    let dups = pipe.store().duplicate_transcripts().unwrap_or_default();
    let api = ApiMessage::from(&msg, &dups);
    let _ = state.tx.send(serde_json::to_value(&api).unwrap());
    Ok(Json(api))
}

fn csv_esc(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Training export (004): zip of corrected (audio, true-text) pairs for
/// offline fine-tuning. `Stored` (audio is already compressed). Rows
/// whose clip file is gone are skipped — a pair needs both halves.
async fn api_export_training(State(state): State<AppState>) -> Response {
    let rows = {
        let pipe = state.pipeline.lock().await;
        match pipe.store().corrections_export() {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("export query: {e}"),
                )
                    .into_response();
            }
        }
    };
    // Keep only pairs whose clip file is still on disk — the manifest
    // must never list audio that did not ship.
    let mut pairs: Vec<(&hamfeed_store::ExportRow, Vec<u8>)> = Vec::new();
    for r in &rows {
        if let Ok(bytes) = std::fs::read(&r.audio_path) {
            pairs.push((r, bytes));
        }
    }
    let mut manifest = String::from("id,true_text,lang,original_transcript\n");
    for (r, _) in &pairs {
        manifest.push_str(&format!(
            "{},{},{},{}\n",
            csv_esc(&r.id),
            csv_esc(&r.corrected),
            csv_esc(&r.lang),
            csv_esc(&r.original),
        ));
    }
    let mut zip_buf = std::io::Cursor::new(Vec::new());
    let fail = |what: &str| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("export zip: {what}"),
        )
            .into_response()
    };
    {
        let mut zw = zip::ZipWriter::new(&mut zip_buf);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        use std::io::Write as _;
        if zw.start_file("manifest.csv", opts).is_err()
            || zw.write_all(manifest.as_bytes()).is_err()
        {
            return fail("manifest");
        }
        for (r, bytes) in &pairs {
            let ext = std::path::Path::new(&r.audio_path)
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or("ogg");
            if zw.start_file(format!("{}.txt", r.id), opts).is_err()
                || zw.write_all(r.corrected.as_bytes()).is_err()
                || zw.start_file(format!("{}.{}", r.id, ext), opts).is_err()
                || zw.write_all(bytes).is_err()
            {
                return fail("pair");
            }
        }
        if zw.finish().is_err() {
            return fail("finish");
        }
    }
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/zip")
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=\"training-export.zip\"",
        )
        .body(Body::from(zip_buf.into_inner()))
        .unwrap()
}

/// SSE stream of new/updated messages (S7 live feed).
async fn api_events(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.tx.subscribe();
    let stream = futures_util::stream::unfold(rx, |mut rx| async {
        match rx.recv().await {
            Ok(v) => Some((
                Ok(Event::default().event("message").data(v.to_string())),
                rx,
            )),
            Err(_) => None,
        }
    });
    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new().interval(std::time::Duration::from_secs(15)),
    )
}

/// Playback path: the stored Opus clip decoded to WAV in RAM. WAV plays
/// everywhere (including browsers without Ogg Opus support); the Opus file
/// stays the archive format on disk.
async fn api_audio_wav(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let bytes = {
        let pipe = state.pipeline.lock().await;
        match pipe.store().get(&id) {
            Ok(Some(m)) => m.audio_path,
            _ => None,
        }
    };
    let path = match bytes {
        Some(p) if std::path::Path::new(&p).exists() => p,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    let ogg = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let pcm = match hamfeed_ingest::decode_ogg_to_pcm(&ogg) {
        Ok(p) => p,
        Err(_) => return StatusCode::UNPROCESSABLE_ENTITY.into_response(),
    };
    serve_bytes(encode_wav(&pcm), "audio/wav", headers)
}

/// Encode 16 kHz mono S16 as a WAV file in RAM.
fn encode_wav(pcm: &[i16]) -> Vec<u8> {
    let data_len = pcm.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Serve bytes with single-range support (media players seek).
fn serve_bytes(full: Vec<u8>, content_type: &str, headers: HeaderMap) -> axum::response::Response {
    let total = full.len();
    let mut res_headers = HeaderMap::new();
    res_headers.insert("content-type", content_type.parse().unwrap());
    res_headers.insert("accept-ranges", "bytes".parse().unwrap());
    if let Some(range) = headers.get("range").and_then(|v| v.to_str().ok()) {
        if let Some((start, end)) = parse_range(range, total) {
            res_headers.insert(
                "content-range",
                format!("bytes {start}-{end}/{total}").parse().unwrap(),
            );
            return (
                StatusCode::PARTIAL_CONTENT,
                res_headers,
                full[start..=end].to_vec(),
            )
                .into_response();
        }
        return StatusCode::RANGE_NOT_SATISFIABLE.into_response();
    }
    (StatusCode::OK, res_headers, full).into_response()
}

fn parse_range(header: &str, total: usize) -> Option<(usize, usize)> {
    // `total - 1` underflows on empty bodies: reject first, not last.
    if total == 0 {
        return None;
    }
    let spec = header.strip_prefix("bytes=")?.trim();
    let (start_s, end_s) = spec.split_once('-')?;
    // Suffix form (`bytes=-N`, RFC 7233 §2.1) runs to the end of the body;
    // `bytes=-0` asks for zero bytes and the range check below rejects it.
    let suffix = start_s.is_empty();
    let start: usize = if suffix {
        total.saturating_sub(end_s.parse::<usize>().ok()?)
    } else {
        start_s.parse().ok()?
    };
    let end: usize = if suffix || end_s.is_empty() {
        total - 1
    } else {
        end_s.parse::<usize>().ok()?.min(total - 1)
    };
    if start >= total || start > end {
        return None;
    }
    Some((start, end))
}

async fn api_audio(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    let path = {
        let pipe = state.pipeline.lock().await;
        match pipe.store().get(&id) {
            Ok(Some(m)) => m.audio_path,
            _ => None,
        }
    };
    match path {
        Some(p) if std::path::Path::new(&p).exists() => match tokio::fs::read(&p).await {
            Ok(bytes) => {
                let mut headers = HeaderMap::new();
                headers.insert("content-type", "audio/ogg".parse().unwrap());
                (StatusCode::OK, headers, bytes).into_response()
            }
            Err(_) => StatusCode::NOT_FOUND.into_response(),
        },
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hamfeed_store::testutil::seed_messages;

    fn test_model() -> std::path::PathBuf {
        if let Ok(p) = std::env::var("HAMFEED_TEST_MODEL") {
            return p.into();
        }
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ggml-tiny.bin")
    }

    fn test_pipeline(dir: &std::path::Path) -> Pipeline {
        let text = format!(
            r#"
[audio]
device = "default"
sample_rate = 16000
[vad]
engine = "energy"
hang_ms = 400
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
[source]
kind = "mic"
[sdr]
active = "2m VE2"
[[sdr.channels]]
name = "2m VE2"
freq_hz = 145110000.0
[[sdr.channels]]
name = "marine"
freq_hz = 161750000.0
"#,
            test_model().display(),
            dir.join("audio").display(),
            dir.join("test.db").display()
        );
        let cfg = hamfeed_config::parse(&text).expect("test config");
        Pipeline::open_with(cfg).expect("pipeline opens")
    }

    async fn test_server() -> (String, PipelineHandle) {
        let dir = std::env::temp_dir().join(format!(
            "hamfeed-web-{}-{}",
            std::process::id(),
            rand_suffix()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let pipe = test_pipeline(&dir);
        seed_messages(pipe.store());
        let state = AppState::new(pipe, dir.join("static").to_string_lossy().into_owned());
        let app = create_app(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), PipelineHandle { state, dir })
    }

    struct PipelineHandle {
        // Held to keep the pipeline/server alive for the test lifetime.
        #[allow(dead_code)]
        state: AppState,
        dir: std::path::PathBuf,
    }

    impl Drop for PipelineHandle {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn rand_suffix() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        format!("{}", N.fetch_add(1, Ordering::SeqCst))
    }

    #[tokio::test]
    async fn sse_new_message() {
        // Seed → triage POST → SSE event carries clip URL + badges (S7).
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let mut events = client
            .get(format!("{base}/api/events"))
            .header("Accept", "text/event-stream")
            .send()
            .await
            .expect("sse connects");
        // Trigger a mutation: keep the seeded failed row.
        let kept: ApiMessage = client
            .post(format!("{base}/api/messages/m-fail/keep"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .expect("keep posts")
            .json()
            .await
            .expect("keep json");
        assert_eq!(kept.id, "m-fail");
        assert_eq!(kept.status, "ok");
        // The SSE stream must deliver that update with clip + badges.
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(10), events.chunk())
            .await
            .expect("sse event arrives")
            .expect("chunk reads")
            .expect("non-empty");
        let text = String::from_utf8_lossy(&chunk);
        assert!(text.contains("m-fail"), "event names the row: {text}");
        assert!(text.contains("/audio/m-fail"), "event has clip: {text}");
        assert!(
            text.contains("\"status\":\"ok\""),
            "event has badge: {text}"
        );
    }

    #[tokio::test]
    async fn audio_wav_plays() {
        // Playback path: stored Opus serves as WAV with range support.
        let (base, _h) = test_server().await;
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../hamfeed-stt/tests/fixtures/en.ogg");
        std::fs::copy(&src, "/tmp/m-fr-ok.ogg").unwrap();
        let client = reqwest::Client::new();
        let res = client
            .get(format!("{base}/audio/m-fr-ok/play.wav"))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        assert_eq!(res.headers().get("content-type").unwrap(), "audio/wav");
        let wav = res.bytes().await.unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert!(wav.len() > 1000, "must carry seconds of audio");
        // Single-range request seeks.
        let res = client
            .get(format!("{base}/audio/m-fr-ok/play.wav"))
            .header("Range", "bytes=0-43")
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 206);
        assert_eq!(res.bytes().await.unwrap().len(), 44);
        let _ = std::fs::remove_file("/tmp/m-fr-ok.ogg");
        // Missing clip stays 404 on both paths.
        let r = client
            .get(format!("{base}/audio/nope/play.wav"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404);
    }

    #[tokio::test]
    async fn messages_pagination() {
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let p1: serde_json::Value = client
            .get(format!("{base}/api/messages?limit=2"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(p1["messages"].as_array().unwrap().len(), 2);
        assert_eq!(p1["messages"][0]["id"], "m-fr-ok");
        let cursor = p1["next_cursor"].as_str().unwrap().to_string();
        assert!(!cursor.is_empty());
        let p2: serde_json::Value = client
            .get(format!("{base}/api/messages?limit=2&cursor={cursor}"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(p2["messages"].as_array().unwrap().len(), 2);
        assert_ne!(p1["messages"][0]["id"], p2["messages"][0]["id"]);
    }

    #[tokio::test]
    async fn triage_endpoints() {
        // Every triage action round-trips to the store (S5 via HTTP).
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();

        let flag: ApiMessage = client
            .post(format!("{base}/api/messages/m-fail/flag"))
            .json(&serde_json::json!({"reason": "robot"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(flag.review_flag, "flagged-training");

        // Retry needs the audio file present: create it, then re-queue.
        std::fs::write("/tmp/m-fail.ogg", b"fake").unwrap();
        // NOTE: seed used /tmp/m-fail.ogg as its audio path.
        let retry: ApiMessage = client
            .post(format!("{base}/api/messages/m-fail/retry"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(retry.status, "failed");
        let _ = std::fs::remove_file("/tmp/m-fail.ogg");

        let drop_: ApiMessage = client
            .post(format!("{base}/api/messages/m-fail/drop"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(drop_.status, "dropped");

        // Unknown action + unknown id are 404, not 500.
        let r = client
            .post(format!("{base}/api/messages/m-fail/banish"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404);
        let r = client
            .post(format!("{base}/api/messages/nope/keep"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404);
    }

    #[tokio::test]
    async fn static_ui_serves() {
        // The approved feed UI serves with its markers intact (frontend
        // regression net: runs in CI, no browser needed).
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        // NOTE: test_server points static_dir at an empty temp dir, so seed
        // the approved files for this test.
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("static");
        let dst = _h.dir.join("static");
        std::fs::create_dir_all(&dst).unwrap();
        for f in ["index.html", "app.js", "style.css"] {
            std::fs::copy(src.join(f), dst.join(f)).unwrap();
        }
        let index = client
            .get(format!("{base}/"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(index.contains("id=\"feed\""), "feed root must serve");
        assert!(index.contains("/app.js"), "bundle reference must serve");
        for asset in ["app.js", "style.css"] {
            let r = client.get(format!("{base}/{asset}")).send().await.unwrap();
            assert_eq!(r.status(), 200, "{asset} must serve");
        }
        let js = client
            .get(format!("{base}/app.js"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(js.contains("/api/events"), "SSE wiring must ship");
        assert!(js.contains("/api/messages"), "REST wiring must ship");
        // Correction UI wiring (004): editor posts here, toggle reads this.
        assert!(js.contains("/correct"), "correction endpoint must ship");
        assert!(js.contains("corrected_text"), "toggle needs both texts");
        assert!(
            index.contains("/api/export/training"),
            "export link must ship"
        );
        // Sender UI wiring (003): badges, banners, confirm, filter.
        assert!(
            js.contains("sender_callsign"),
            "sender badge needs identity"
        );
        // esc() must neutralize both quote kinds: card HTML mixes
        // double-quoted attributes with values the model wrote.
        assert!(js.contains("&#39;"), "esc covers single quotes");
        assert!(js.contains("banner foryou"), "for-you banner must ship");
        assert!(
            js.contains("banner emergency"),
            "emergency banner must ship"
        );
        assert!(js.contains("/confirm-sender"), "confirm endpoint must ship");
        assert!(index.contains("id=\"sender\""), "sender filter must ship");
    }

    #[tokio::test]
    async fn status_and_search() {
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let st: serde_json::Value = client
            .get(format!("{base}/api/status"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(st["message_count"], 6);
        assert_eq!(st["failed_count"], 1);
        let lv: serde_json::Value = client
            .get(format!("{base}/api/level"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(lv.get("age_s").is_some());
        let s: serde_json::Value = client
            .get(format!("{base}/api/search?q=bonjour"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(s["messages"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn correct_roundtrip() {
        // Submit a correction: JSON carries both texts, original intact.
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let mut events = client
            .get(format!("{base}/api/events"))
            .header("Accept", "text/event-stream")
            .send()
            .await
            .expect("sse connects");
        let fixed: ApiMessage = client
            .post(format!("{base}/api/messages/m-fr-ok/correct"))
            .json(&serde_json::json!({"text": "bonjour les vrais amis"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(fixed.id, "m-fr-ok");
        assert_eq!(
            fixed.corrected_text.as_deref(),
            Some("bonjour les vrais amis")
        );
        assert_eq!(fixed.transcript, "bonjour les amis");
        // The correction broadcasts like triage (live cards update).
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(10), events.chunk())
            .await
            .expect("sse event arrives")
            .expect("chunk reads")
            .expect("non-empty");
        let text = String::from_utf8_lossy(&chunk);
        assert!(
            text.contains("bonjour les vrais amis"),
            "event carries fix: {text}"
        );
        // Feed search hits the corrected-only word.
        let s: serde_json::Value = client
            .get(format!("{base}/api/search?q=vrais"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(s["messages"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn correct_blank_clears_and_unknown_404() {
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let fixed: ApiMessage = client
            .post(format!("{base}/api/messages/m-fr-ok/correct"))
            .json(&serde_json::json!({"text": "  "}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(fixed.corrected_text, None);
        // Missing body clears too.
        let fixed: ApiMessage = client
            .post(format!("{base}/api/messages/m-fr-ok/correct"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(fixed.corrected_text, None);
        // Unknown id is 404, not 500.
        let r = client
            .post(format!("{base}/api/messages/nope/correct"))
            .json(&serde_json::json!({"text": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404);
    }

    #[tokio::test]
    async fn export_zip_pairs() {
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        // Empty export first: manifest only, no pairs.
        let empty = client
            .get(format!("{base}/api/export/training"))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let names = zip_names(&empty);
        assert_eq!(names, vec!["manifest.csv"]);
        // Correct two rows; only the one with a clip file on disk exports.
        // NOTE: seed audio paths are /tmp/{id}.ogg; m-en-ok is used here
        // because no other test touches it (m-fr-ok races audio_wav_plays).
        std::fs::write("/tmp/m-en-ok.ogg", b"fake-opus-bytes").unwrap();
        for (id, text) in [
            ("m-en-ok", "hello true friends"),
            ("m-fr-low", "appel corrigé"),
        ] {
            client
                .post(format!("{base}/api/messages/{id}/correct"))
                .json(&serde_json::json!({"text": text}))
                .send()
                .await
                .unwrap();
        }
        let body = client
            .get(format!("{base}/api/export/training"))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let _ = std::fs::remove_file("/tmp/m-en-ok.ogg");
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(&body[..])).unwrap();
        let manifest = read_zip(&mut zip, "manifest.csv");
        assert!(
            manifest.contains("m-en-ok"),
            "manifest lists pair: {manifest}"
        );
        assert!(
            manifest.contains("hello true friends"),
            "manifest true text: {manifest}"
        );
        assert!(
            !manifest.contains("m-fr-ok"),
            "uncorrected excluded: {manifest}"
        );
        assert_eq!(read_zip(&mut zip, "m-en-ok.txt"), "hello true friends");
        assert!(
            !manifest.contains("m-fr-low"),
            "audio-less pair excluded: {manifest}"
        );
        assert!(!zip_names(&body).iter().any(|n| n.starts_with("m-fr-low")));
    }

    #[tokio::test]
    async fn correct_oversized_is_413() {
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let big = "x".repeat(MAX_CORRECTION_LEN + 1);
        let r = client
            .post(format!("{base}/api/messages/m-fr-ok/correct"))
            .json(&serde_json::json!({"text": big}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 413);
        // At the boundary: accepted.
        let ok = "y".repeat(MAX_CORRECTION_LEN);
        let r = client
            .post(format!("{base}/api/messages/m-fr-ok/correct"))
            .json(&serde_json::json!({"text": ok}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
    }

    #[tokio::test]
    async fn correction_flow() {
        // Slice proof end to end: correct → feed shows both texts → search
        // hits the corrected-only word → export ships the pair.
        // NOTE: m-old's clip path is exclusive to this test.
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        std::fs::write("/tmp/m-old.ogg", b"fake-opus-bytes").unwrap();
        client
            .post(format!("{base}/api/messages/m-old/correct"))
            .json(&serde_json::json!({"text": "message ancien corrigé"}))
            .send()
            .await
            .unwrap();
        let feed: serde_json::Value = client
            .get(format!("{base}/api/messages?limit=20"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let card = feed["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == "m-old")
            .expect("feed carries m-old");
        assert_eq!(card["corrected_text"], "message ancien corrigé");
        assert_eq!(card["transcript"], "ancien message");
        let s: serde_json::Value = client
            .get(format!("{base}/api/search?q=corrigé"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(s["messages"].as_array().unwrap().len(), 1);
        let body = client
            .get(format!("{base}/api/export/training"))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let _ = std::fs::remove_file("/tmp/m-old.ogg");
        assert!(zip_names(&body).contains(&"m-old.txt".to_string()));
    }

    #[tokio::test]
    async fn sender_badge_in_feed() {
        let (base, _h) = test_server().await;
        {
            let pipe = _h.state.pipeline.lock().await;
            pipe.store()
                .set_sender("m-fr-ok", Some("VE2DEM"), Some("Jean Tremblay"), "heard")
                .unwrap();
            pipe.store()
                .set_sender("m-en-ok", Some("VE2DEM"), None, "carried")
                .unwrap();
        }
        let client = reqwest::Client::new();
        let feed: serde_json::Value = client
            .get(format!("{base}/api/messages?limit=20"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let card = feed["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == "m-fr-ok")
            .expect("feed carries m-fr-ok");
        assert_eq!(card["sender_callsign"], "VE2DEM");
        assert_eq!(card["sender_name"], "Jean Tremblay");
        assert_eq!(card["sender_source"], "heard");
        // Nameless carried badge: callsign alone.
        let card = feed["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == "m-en-ok")
            .expect("feed carries m-en-ok");
        assert_eq!(card["sender_name"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn sender_filter() {
        let (base, _h) = test_server().await;
        {
            let pipe = _h.state.pipeline.lock().await;
            pipe.store()
                .set_sender("m-fr-ok", Some("VE2DEM"), None, "heard")
                .unwrap();
            pipe.store()
                .set_sender("m-en-ok", Some("VE3MA"), None, "heard")
                .unwrap();
        }
        let client = reqwest::Client::new();
        let s: serde_json::Value = client
            .get(format!("{base}/api/search?sender=ve2dem"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let ids: Vec<&str> = s["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["m-fr-ok"]);
    }

    #[tokio::test]
    async fn confirm_roundtrip() {
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let confirmed: ApiMessage = client
            .post(format!("{base}/api/messages/m-fr-ok/confirm-sender"))
            .json(&serde_json::json!({"callsign": "ve2dem"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(confirmed.sender_callsign.as_deref(), Some("VE2DEM"));
        assert_eq!(confirmed.sender_source, "confirmed");
        // Non-callsign is 400, unknown id 404.
        let r = client
            .post(format!("{base}/api/messages/m-fr-ok/confirm-sender"))
            .json(&serde_json::json!({"callsign": "not a call"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400);
        let r = client
            .post(format!("{base}/api/messages/nope/confirm-sender"))
            .json(&serde_json::json!({"callsign": "VE2DEM"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404);
    }

    #[tokio::test]
    async fn alert_banner_payload() {
        // The JSON carries the bits; banners render from them (mockup
        // anatomy, asserted on the shipped JS below).
        let (base, _h) = test_server().await;
        {
            let pipe = _h.state.pipeline.lock().await;
            pipe.store()
                .insert(&hamfeed_store::NewMessage {
                    id: "m-alert".into(),
                    ts_start_ms: 9500,
                    ts_end_ms: 10000,
                    lang: "en".into(),
                    lang_conf: 0.9,
                    transcript: "Mayday, mayday".into(),
                    stt_conf: 0.8,
                    conf_flag: "ok".into(),
                    status: "ok".into(),
                    fail_reason: None,
                    audio_path: None,
                    duration_ms: Some(500),
                    size_bytes: None,
                    short_flag: false,
                    group_id: "g".into(),
                    seq: 0,
                    sender_callsign: Some("W1AW".into()),
                    sender_name: None,
                    sender_source: "heard".into(),
                    alert: 3,
                    speaker_key: None,
                    corrected_text: None,
                })
                .unwrap();
        }
        let client = reqwest::Client::new();
        let feed: serde_json::Value = client
            .get(format!("{base}/api/messages?limit=20"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let card = feed["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == "m-alert")
            .expect("feed carries m-alert");
        assert_eq!(card["alert"], 3);
    }

    #[test]
    fn parse_range_units() {
        assert_eq!(parse_range("bytes=0-9", 100), Some((0, 9)));
        assert_eq!(parse_range("bytes=90-", 100), Some((90, 99)));
        // Suffix form: the LAST N bytes (RFC 7233 §2.1), not total-(N+1).
        assert_eq!(parse_range("bytes=-10", 100), Some((90, 99)));
        assert_eq!(parse_range("bytes=-200", 100), Some((0, 99)));
        assert_eq!(parse_range("bytes=0-999", 100), Some((0, 99)));
        // Rejections: empty body (no underflow), past-the-end,
        // inverted, zero-length, unparsable.
        assert_eq!(parse_range("bytes=0-", 0), None);
        assert_eq!(parse_range("bytes=-5", 0), None);
        assert_eq!(parse_range("bytes=200-300", 100), None);
        assert_eq!(parse_range("bytes=50-40", 100), None);
        assert_eq!(parse_range("bytes=-0", 100), None);
        assert_eq!(parse_range("bytes=abc", 100), None);
        assert_eq!(parse_range("bytes=", 100), None);
    }

    fn zip_names(body: &[u8]) -> Vec<String> {
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(body)).unwrap();
        (0..zip.len())
            .map(|i| zip.by_index(i).unwrap().name().to_string())
            .collect()
    }

    /// Stub the pipeline relay: raw LE i16 frames on the socket path the
    /// handler derives from the test storage dir. Mirrors serve_live_tap's
    /// side of the contract (framing only, no tap involved).
    fn stub_relay(sock_path: std::path::PathBuf, frames: usize) {
        std::thread::spawn(move || {
            use std::io::Write as _;
            use std::os::unix::net::UnixListener;
            let _ = std::fs::remove_file(&sock_path);
            let listener = UnixListener::bind(&sock_path).expect("stub binds");
            let (mut stream, _) = listener.accept().expect("web connects");
            let frame = vec![0u8; 640];
            for _ in 0..frames {
                if stream.write_all(&frame).is_err() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
    }

    async fn relay_sock(state: &AppState) -> std::path::PathBuf {
        let pipe = state.pipeline.lock().await;
        std::path::PathBuf::from(&pipe.config().storage.dir).join("live.sock")
    }

    #[tokio::test]
    async fn live_stream_relays_socket_audio() {
        // Slice 3 fix: with a relay feeding, the second chunk must carry
        // audio pages (headers alone = the old same-process starvation).
        let (base, h) = test_server().await;
        stub_relay(relay_sock(&h.state).await, 200);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let client = reqwest::Client::new();
        let mut res = client
            .get(format!("{base}/api/live"))
            .send()
            .await
            .expect("live connects");
        assert_eq!(res.headers().get("content-type").unwrap(), "audio/ogg");
        let first = res.chunk().await.expect("first chunk").expect("bytes");
        assert_eq!(&first[..4], b"OggS");
        assert!(first.windows(8).any(|w| w == b"OpusHead"));
        let second = res.chunk().await.expect("second chunk").expect("bytes");
        assert!(!second.is_empty(), "relay audio must follow headers");
        assert_eq!(&second[..4], b"OggS");
    }

    #[tokio::test]
    async fn live_stream_503_without_relay() {
        // No socket (pipeline down): honest 503, never hanging silence.
        let (base, _h) = test_server().await;
        let res = reqwest::Client::new()
            .get(format!("{base}/api/live"))
            .send()
            .await
            .expect("live responds");
        assert_eq!(res.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn profile_switch_roundtrip() {
        // Slice 3 T4: list, switch, unknown-name 404, switch back.
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let got: serde_json::Value = client
            .get(format!("{base}/api/profile"))
            .send()
            .await
            .expect("profile gets")
            .json()
            .await
            .expect("profile json");
        assert_eq!(got["active"], "Normal");
        assert_eq!(got["profiles"].as_array().unwrap().len(), 3);
        let switched: serde_json::Value = client
            .post(format!("{base}/api/profile"))
            .json(&serde_json::json!({"name": "ARES Net"}))
            .send()
            .await
            .expect("profile posts")
            .json()
            .await
            .expect("switch json");
        assert_eq!(switched["active"], "ARES Net");
        let missing = client
            .post(format!("{base}/api/profile"))
            .json(&serde_json::json!({"name": "Nope"}))
            .send()
            .await
            .expect("unknown posts");
        assert_eq!(missing.status(), reqwest::StatusCode::NOT_FOUND);
        // Name with a quote: stored/executed as data, still just 404.
        let quoted = client
            .post(format!("{base}/api/profile"))
            .json(&serde_json::json!({"name": "A' OR '1'='1"}))
            .send()
            .await
            .expect("quoted posts");
        assert_eq!(quoted.status(), reqwest::StatusCode::NOT_FOUND);
        let back: serde_json::Value = client
            .post(format!("{base}/api/profile"))
            .json(&serde_json::json!({"name": "Normal"}))
            .send()
            .await
            .expect("back posts")
            .json()
            .await
            .expect("back json");
        assert_eq!(back["active"], "Normal");
    }

    #[tokio::test]
    async fn channel_switch_roundtrip() {
        // SDR selector: list (kind mic here, channels still served so
        // the UI can preview), switch, unknown-name 404, switch back.
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let got: serde_json::Value = client
            .get(format!("{base}/api/source"))
            .send()
            .await
            .expect("source gets")
            .json()
            .await
            .expect("source json");
        assert_eq!(got["kind"], "mic");
        assert_eq!(got["channels"].as_array().unwrap().len(), 2);
        assert_eq!(got["active"], "2m VE2");
        let switched: serde_json::Value = client
            .post(format!("{base}/api/source/channel"))
            .json(&serde_json::json!({"name": "marine"}))
            .send()
            .await
            .expect("channel posts")
            .json()
            .await
            .expect("switch json");
        assert_eq!(switched["active"], "marine");
        let missing = client
            .post(format!("{base}/api/source/channel"))
            .json(&serde_json::json!({"name": "Nope"}))
            .send()
            .await
            .expect("unknown posts");
        assert_eq!(missing.status(), reqwest::StatusCode::NOT_FOUND);
        let back: serde_json::Value = client
            .post(format!("{base}/api/source/channel"))
            .json(&serde_json::json!({"name": "2m VE2"}))
            .send()
            .await
            .expect("back posts")
            .json()
            .await
            .expect("back json");
        assert_eq!(back["active"], "2m VE2");
    }

    #[tokio::test]
    async fn freq_tune_roundtrip() {
        // UI frequency + demod entry: GET serves the effective tune,
        // POST tunes by Hz/mode, bad shapes are 400, and switching
        // back to a preset clears the override.
        async fn get(client: &reqwest::Client, base: &str) -> serde_json::Value {
            client
                .get(format!("{base}/api/source"))
                .send()
                .await
                .expect("source gets")
                .json::<serde_json::Value>()
                .await
                .expect("source json")
        }
        async fn post(
            client: &reqwest::Client,
            base: &str,
            v: serde_json::Value,
        ) -> reqwest::Response {
            client
                .post(format!("{base}/api/source/channel"))
                .json(&v)
                .send()
                .await
                .expect("tune posts")
        }
        let (base, _h) = test_server().await;
        let client = reqwest::Client::new();
        let got = get(&client, &base).await;
        assert_eq!(got["active"], "2m VE2");
        assert_eq!(got["freq_hz"], 145110000.0);
        assert_eq!(got["mode"], "nbfm");
        assert_eq!(got["modes"], serde_json::json!(["nbfm", "am"]));
        let tuned = post(
            &client,
            &base,
            serde_json::json!({"freq_hz": 161775000.0, "mode": "am"}),
        )
        .await
        .json::<serde_json::Value>()
        .await
        .expect("tune json");
        assert!(tuned["active"].is_null());
        assert_eq!(tuned["freq_hz"], 161775000.0);
        assert_eq!(tuned["mode"], "am");
        let got = get(&client, &base).await;
        assert!(got["active"].is_null());
        assert_eq!(got["freq_hz"], 161775000.0);
        assert_eq!(got["mode"], "am");
        for bad in [
            serde_json::json!({}),
            serde_json::json!({"name": "marine", "freq_hz": 1.0}),
            serde_json::json!({"freq_hz": 1.0}),
            serde_json::json!({"freq_hz": 161775000.0, "mode": "ssb"}),
        ] {
            assert_eq!(
                post(&client, &base, bad).await.status(),
                reqwest::StatusCode::BAD_REQUEST
            );
        }
        let back = post(&client, &base, serde_json::json!({"name": "marine"}))
            .await
            .json::<serde_json::Value>()
            .await
            .expect("preset json");
        assert_eq!(back["active"], "marine");
        let got = get(&client, &base).await;
        assert_eq!(got["active"], "marine");
        assert_eq!(got["freq_hz"], 161750000.0);
        assert_eq!(got["mode"], "nbfm");
    }

    fn read_zip(zip: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>, name: &str) -> String {
        use std::io::Read as _;
        let mut f = zip.by_name(name).unwrap();
        let mut s = String::new();
        f.read_to_string(&mut s).unwrap();
        s
    }

    #[test]
    fn noise_kind_mirrors_hide_verdict() {
        use std::collections::HashSet;
        let dups: HashSet<String> = ["canned".into()].into_iter().collect();
        let empty: HashSet<String> = HashSet::new();
        // Exemptions first: heard and failed are never noise.
        assert_eq!(noise_kind(true, "", "heard", "ok", &empty), None);
        assert_eq!(noise_kind(true, "", "none", "failed", &empty), None);
        assert_eq!(
            noise_kind(false, "Bye. Thank you.", "heard", "ok", &empty),
            None
        );
        // Kinds in priority order: short, no-words, repeat.
        assert_eq!(
            noise_kind(true, "VE2ABC!", "none", "ok", &empty),
            Some("short")
        );
        assert_eq!(
            noise_kind(false, "[BLANK_AUDIO]", "none", "ok", &empty),
            Some("no-words")
        );
        assert_eq!(
            noise_kind(false, "Bye. Thank you.", "carried", "ok", &empty),
            Some("no-words")
        );
        assert_eq!(
            noise_kind(false, "canned", "none", "ok", &dups),
            Some("repeat")
        );
        // Traffic: no badge.
        assert_eq!(
            noise_kind(false, "bonjour les amis", "none", "ok", &empty),
            None
        );
    }
}
