#!/bin/sh
# Regenerate tests/fixtures/{en,fr}.ogg (T8 provenance).
# EN: whisper.cpp jfk.wav sample (11 s, public repo, 16 kHz mono already).
# FR: short French sentence via Google TTS MP3, decoded with minimp3 and
#     linearly resampled to 16 kHz mono. Both encoded with hamfeed's own
#     Opus encoder (encode_pcm_to_ogg); total < 100 KB.
# Requires network. Normal `cargo test` never runs this (#[ignore]).
set -eu
CRATE=$(dirname "$0")/../..
cd "$CRATE"
cargo test --test gen_fixtures -- --ignored --nocapture
ls -la tests/fixtures/*.ogg
