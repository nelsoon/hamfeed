//! hamfeed-web: REST + SSE live feed + static UI (T11).
//!
//! Thin reader/writer over store + pipeline helpers (group contract): no
//! business logic lives here. Serves the T10-approved feed structure against
//! live data; failed rows render error cards with working triage buttons.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
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
    pub freq_label: String,
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
}

impl ApiMessage {
    fn from(m: &Message) -> Self {
        Self {
            id: m.id.clone(),
            ts_start_ms: m.ts_start_ms,
            freq_label: m.freq_label.clone(),
            lang: m.lang.clone(),
            lang_conf: m.lang_conf,
            transcript: m.transcript.clone(),
            stt_conf: m.stt_conf,
            conf_flag: m.conf_flag.clone(),
            status: m.status.clone(),
            fail_reason: m.fail_reason.clone(),
            audio_url: m.audio_path.as_ref().map(|_| format!("/audio/{}", m.id)),
            duration_ms: m.duration_ms,
            short_flag: m.short_flag,
            review_flag: m.review_flag.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct MessagesQuery {
    limit: Option<usize>,
    order: Option<String>,
    cursor: Option<String>,
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
}

#[derive(Debug, Deserialize)]
struct TriageBody {
    reason: Option<String>,
    delete_audio: Option<bool>,
}

pub fn create_app(state: AppState) -> Router {
    let static_dir = state.static_dir.clone();
    Router::new()
        .route("/api/status", get(api_status))
        .route("/api/level", get(api_level))
        .route("/api/messages", get(api_messages))
        .route("/api/search", get(api_search))
        .route("/api/messages/:id/:action", post(api_triage))
        .route("/api/events", get(api_events))
        .route("/audio/:id", get(api_audio))
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
            ..Default::default()
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({
        "messages": page.messages.iter().map(ApiMessage::from).collect::<Vec<_>>(),
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
            limit: q.limit.unwrap_or(20),
            cursor: q.cursor,
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({
        "messages": page.messages.iter().map(ApiMessage::from).collect::<Vec<_>>(),
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
        let mut pipe = state.pipeline.lock().await;
        pipe.set_triage(&id, triage)
            .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    }
    if needs_drain {
        // Retry re-transcribes off the async runtime; fresh rows broadcast
        // as SSE events when they land.
        let state2 = state.clone();
        tokio::task::spawn_blocking(move || {
            let mut pipe = state2.pipeline.blocking_lock();
            if pipe.drain().is_ok() {
                if let Ok(latest) = pipe.latest(64) {
                    for m in &latest {
                        let _ = state2
                            .tx
                            .send(serde_json::to_value(ApiMessage::from(m)).unwrap());
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
    let api = ApiMessage::from(&msg);
    let _ = state.tx.send(serde_json::to_value(&api).unwrap());
    Ok(Json(api))
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
freq_label = "TEST"
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
}
