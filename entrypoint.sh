#!/bin/bash
set -e

# Seed fonts volume on first run
if [ -z "$(ls -A /app/fonts 2>/dev/null)" ]; then
    echo "Seeding fonts from image..."
    cp -r /app/fonts_seed/. /app/fonts/
    echo "Done — $(ls /app/fonts | wc -l) fonts loaded."
fi

mkdir -p /app/fonts_modified

exec uvicorn main:app --host 0.0.0.0 --port 8000
