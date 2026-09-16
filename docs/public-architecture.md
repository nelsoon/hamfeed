# Public architecture (hamfeed, Slices 1–2)

## What it does

An operator leaves a receiver feeding the computer microphone on a fixed
frequency. hamfeed captures each transmission, transcribes it locally in its
original language (French or English, never translated), attributes it to
its sender, and serves text + audio clip in a live web feed with searchable
history.

## Components

```text
mic-in → hamfeed-source → hamfeed-ingest → Opus clip (.ogg)
  → hamfeed-stt (local whisper) → hamfeed-store (SQLite + FTS5)
  → hamfeed-web (Axum + SSE + static UI)
hamfeed-pipeline orchestrates ingest → stt → enrich → store, where enrich
  = hamfeed-callsign parse + hamfeed-callbook lookup + voice carry/link
  (hamfeed-speaker, optional) + notify cue match.
```

- **source**: `AudioSource` trait yielding 16 kHz mono S16 frames. v1 ships
  the mic implementation; the trait is shaped for SDR later.
- **ingest**: energy VAD (tunable hang-time, repeater/simplex profiles on one
  engine) → max-duration split sharing one `group_id` → Opus-in-Ogg clips
  written atomically (`.tmp` → rename) → bounded queue (cap 100) with spill
  to disk and `(ts_start, id)`-ordered replay on boot.
- **stt**: whisper model runs locally; language is detected with
  probabilities, then decoding is constrained to that language. Missing
  model refuses loudly at startup with provisioning instructions.
- **store**: one SQLite database, FTS5 over transcripts, retention sweep
  (expired audio purges; rows become dropped-purged), `.tmp` janitor, and a
  missing-audio audit (ok/failed rows without audio flag as bugs).
- **web**: newest-first feed over SSE, per-message `<audio>` playback,
  language/confidence badges, triage buttons on failed cards
  (Keep / Drop / Retry / Flag-for-training), full-text search with
  from/to filters, hide-noise toggle, and cursor pagination.
- **callsign**: self-ID extraction from transcripts — plain (`VE2DEM`)
  and NATO-spelled runs, French and English, accent-tolerant.
- **callbook**: local CA/US operator databases (manual drop-in or
  `callbook-import` over ISED/ULS dumps); a missing file degrades to
  nameless badges, never an error or a network call.
- **speaker**: voiceprints for sender linking (80-bin CMVN mel frontend
  + wespeaker ONNX embeddings behind the `voice` cargo feature).
  Short-term carry auto-links in-window; the long-term library only
  ever *suggests*, and confirm/correct teaches it. The voice model is
  optional (`scripts/download-voice-model.sh`, checksum-verified).
- **notify**: messages addressing the configured callsign raise for-you
  banners; configured emergency cues raise emergency banners.

## Limits (Slices 1–2)

- VAD + courtesy-beep cutting (frequency-agnostic Goertzel detector).
- Opus-in-Ogg archive only.
- Transcription quality follows the whisper model size (tiny for CI,
  small by default); low-confidence rows are badged, not hidden.
- Single operator, single machine; no accounts, no mobile app.
- Transmissions over the max duration split into sequenced parts.
