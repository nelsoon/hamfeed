# hamfeed

Receive-only ham-radio listener: fixed-frequency capture → local FR/EN
transcription → live web feed with audio clips. Pure Rust, local-first —
no audio or text leaves the machine.

Slice 1 ("live loop") captures microphone audio, cuts transmissions with an
energy VAD, archives each as an Opus clip, transcribes locally with whisper,
and serves text + playback in a live web feed with searchable history.

Slice 2 ("know + notify") attributes each message to its sender: callsigns
parsed from self-IDs (plain + NATO-spelled, FR/EN), enriched with local
CA/US callbooks, linked across follow-ups by voiceprint, with for-you and
emergency notifications in the feed.

## Install / Run

Prerequisites: stable Rust, `libopus` + ALSA headers, `cmake`, and a whisper
model (see below).

```sh
# system libraries (Debian/Ubuntu)
sudo apt-get install -y libasound2-dev libopus-dev cmake

# whisper model (tiny is enough to try; small is the default config)
sh scripts/download-model.sh tiny

# build + test (all fixture-backed; no radio hardware needed)
cargo build --workspace
cargo test --workspace --all-features

# configure and run
cp hamfeed.toml.example hamfeed.toml
${EDITOR:-vi} hamfeed.toml
cargo run -p hamfeed-pipeline -- --config hamfeed.toml --fake 10
cargo run -p hamfeed-web -- --config hamfeed.toml --port 8080
```

Open `http://127.0.0.1:8080/` for the live feed.

## Layout

- `crates/hamfeed-config` — TOML config load + validate
- `crates/hamfeed-source` — audio input trait (mic first, SDR later)
- `crates/hamfeed-ingest` — VAD segmentation, Opus clips, spill/replay
- `crates/hamfeed-stt` — local whisper transcription (FR/EN, untranslated)
- `crates/hamfeed-store` — SQLite + FTS5 archive, retention janitors
- `crates/hamfeed-pipeline` — orchestration binary
- `crates/hamfeed-web` — Axum + SSE feed binary + static UI
- `crates/hamfeed-callsign` — self-ID extraction (plain + spelled, FR/EN)
- `crates/hamfeed-callbook` — local CA/US operator lookup + import binary
- `crates/hamfeed-speaker` — voiceprints (mel frontend + ONNX embeddings,
  `voice` cargo feature)
- `scripts/download-model.sh` — fetch a whisper model (tiny/base/small)
- `scripts/download-voice-model.sh` — fetch the wespeaker voice model
  (checksum-verified; optional — voiceprints stay off unless opted in
  with it)

## Privacy

Ham radio is public by design: regulators (ISED in Canada via the
Radiocommunication Act, equivalents elsewhere) prohibit encryption on
amateur bands, require on-air self-identification, and allow anyone to
listen. Recording what anyone may lawfully hear needs no consent — but
what hamfeed *derives and keeps* is a separate matter, and laws about
that vary by jurisdiction:

- Everything stays on the machine: capture → transcribe → store → serve
  makes zero network calls. Audio clips, transcripts, and callsign links
  live in the local SQLite store with time-based retention janitors.
- Voiceprints are off by default (`[voiceprint] enabled = false`) and
  stay off without a model file. A public broadcast is not consent to
  biometric enrollment: enable them only deliberately, keep retention
  short, and purge on request (alias rows are operator-purgeable;
  embeddings never leave RAM).
- If you publish the feed beyond your own household — especially for a
  club — check your local privacy law first (e.g. Québec's Law 25 treats
  biometric data as sensitive). That call belongs to the operator, not
  the software.

## Docs

See `docs/`. Model provisioning, provisioning failures, and triage
semantics are covered in `docs/public-architecture.md`.

## License

TBD — a license file lands before the first tagged release.
