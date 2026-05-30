#!/bin/bash
set -e

# Sync fonts from image whenever the set changes (checksum-based)
SEED_DIR=/app/fonts_seed
FONTS_DIR=/app/fonts
CHECKSUM_FILE=$FONTS_DIR/.seed_checksum

IMAGE_CHECKSUM=$(find "$SEED_DIR" -type f | sort | xargs md5sum | md5sum | cut -d' ' -f1)
STORED_CHECKSUM=$(cat "$CHECKSUM_FILE" 2>/dev/null || echo "")

if [ "$IMAGE_CHECKSUM" != "$STORED_CHECKSUM" ]; then
    echo "Fonts changed — syncing from image..."
    rsync -a --delete "$SEED_DIR/" "$FONTS_DIR/" 2>/dev/null || cp -r "$SEED_DIR/." "$FONTS_DIR/"
    echo "$IMAGE_CHECKSUM" > "$CHECKSUM_FILE"
    echo "Done — $(ls $FONTS_DIR/*.ttf 2>/dev/null | wc -l) fonts loaded."
fi

mkdir -p /app/fonts_modified

exec uvicorn main:app --host 0.0.0.0 --port 8000
