#!/usr/bin/env bash
# Convert a README demo MP4 into a compact, high-quality GIF.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 INPUT.mp4 OUTPUT.gif" >&2
  exit 2
fi

input=$1
output=$2
tmpdir=$(mktemp -d)
palette="$tmpdir/palette.png"
trap 'rm -rf "$tmpdir"' EXIT

gif_fps=${GIF_FPS:-15}
gif_width=${GIF_WIDTH:-1200}
filters="fps=$gif_fps,scale=$gif_width:-1:flags=lanczos"
ffmpeg -y -loglevel error -i "$input" -vf "$filters,palettegen=stats_mode=diff" "$palette"
ffmpeg -y -loglevel error -i "$input" -i "$palette" \
  -lavfi "$filters [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  "$output"
