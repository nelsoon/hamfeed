#!/bin/sh
# Fetch a whisper.cpp model into a local directory (default: ./models).
# Usage: download-model.sh [tiny|base|small] [dest-dir]
# Idempotent: skips download when the file already exists.
# No private values here: models come from the public whisper.cpp release.
set -eu
SIZE="${1:-tiny}"
DEST="${2:-models}"
case "$SIZE" in
  tiny|base|small) ;;
  *) echo "unknown size '$SIZE' (want tiny|base|small)" >&2; exit 1 ;;
esac
FILE="ggml-$SIZE.bin"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$FILE"
mkdir -p "$DEST"
if [ -f "$DEST/$FILE" ]; then
  echo "cached: $DEST/$FILE"
  exit 0
fi
echo "fetching $URL"
curl -sSL -o "$DEST/$FILE" "$URL"
echo "saved: $DEST/$FILE"
