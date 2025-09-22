#!/usr/bin/env bash
#
# Batch-rotate MP4 videos based on filename patterns.
#
# Rules:
#   - Files containing "51470934" → rotate right (90° clockwise)
#   - Files containing "66920734" → rotate left  (90° counterclockwise)
#
# Skips:
#   - AppleDouble/hidden files (._*)
#   - Files beginning with "_"
#
# --- How to use on macOS ---
# 1. Make sure ffmpeg is installed:
#      brew install ffmpeg
# 2. Open Terminal and cd into the folder containing your videos:
#      cd /path/to/your/folder
# 3. Run this script directly:
#      bash rotate_videos.sh
#      or just copy to terminal and paste
################################## This version not tested, just record keeping

set -euo pipefail

# --- Check for ffmpeg ---
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found. Install with: brew install ffmpeg"
  exit 1
fi

# --- Main loop ---
while IFS= read -r -d '' file; do
  base="$(basename "$file")"
  dir="$(dirname "$file")"

  # Skip AppleDouble and underscore files
  if [[ "$base" == ._* || "$base" == _* ]]; then
    echo "Skipping: $file"
    continue
  fi

  vf=""
  if [[ "$base" == *51470934* ]]; then
    vf="transpose=1"   # rotate right (CW)
  elif [[ "$base" == *66920734* ]]; then
    vf="transpose=2"   # rotate left (CCW)
  fi

  if [[ -n "$vf" ]]; then
    tmp="${dir}/.${base}.rotating.mp4"
    echo "Rotating ($vf): $file"

    ffmpeg -hide_banner -loglevel error -y \
      -i "$file" \
      -vf "$vf" \
      -c:v libx264 -preset veryfast -crf 18 \
      -c:a copy \
      -movflags +faststart \
      "$tmp"

    mv -f "$tmp" "$file"
    echo "Done: $file"
  fi
done < <(find . -type f -iname "*.mp4" -print0)

echo "All matching files processed."