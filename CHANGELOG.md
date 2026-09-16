# Changelog

## [Unreleased] — Slice 1: live loop

- Fixed-frequency mic capture with energy-VAD segmentation and
  max-duration split (`group_id` / `seq`).
- Opus-in-Ogg archive with atomic writes, bounded queue, spill + replay.
- Local whisper transcription (FR/EN, untranslated) with language +
  transcription confidences; loud refusal when the model is missing.
- SQLite + FTS5 history: search, from/to filters, hide-noise, pagination.
- Live web feed (SSE) with playback, badges, and triage
  (Keep / Drop / Retry / Flag-for-training).
- Retention sweep, `.tmp` janitor, missing-audio audit.
