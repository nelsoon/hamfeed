#!/bin/sh
# Fetch the wespeaker voiceprint model into a local directory.
# Usage: download-voice-model.sh [dest-dir]  (default: ./models)
# Idempotent: skips download when a checksum-good file already exists.
# Pinned revision + SHA-256 (see hamfeed-speaker/src/embedder.rs).
set -eu
DEST="${1:-models}"
FILE="wespeaker.onnx"
URL="https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/6a61a1833ff2583aabeba044f5c8221f00b67ceb/onnx/model.onnx"
SHA="3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb"
mkdir -p "$DEST"
if [ -f "$DEST/$FILE" ]; then
  if echo "$SHA  $DEST/$FILE" | sha256sum -c - >/dev/null 2>&1; then
    echo "cached: $DEST/$FILE"
    exit 0
  fi
  echo "stale checksum, re-fetching: $DEST/$FILE"
fi
echo "fetching $URL"
curl -sSL -o "$DEST/$FILE" "$URL"
echo "$SHA  $DEST/$FILE" | sha256sum -c -
echo "saved: $DEST/$FILE"
