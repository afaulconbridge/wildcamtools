#!/bin/bash
set -euxo pipefail


# wait_for_rtsp_stream <url> [timeout_seconds=30] [interval=1]
# Returns 0 when the RTSP stream is reachable, nonzero on timeout.
wait_for_rtsp_stream() {
  local url=${1:?URL required}
  local timeout=${2:-30}
  local interval=${3:-1}
  local start now elapsed
  start=$(date +%s)

  while :; do
    # Try to probe the stream with ffprobe (quiet, short timeout)
    if command -v ffprobe >/dev/null 2>&1; then
      if ffprobe -v error -rtsp_transport tcp -timeout 2000000 -show_streams "$url" >/dev/null 2>&1; then
        return 0
      fi
    else
      # Fallback: try opening with ffmpeg for a short period
      if command -v ffmpeg >/dev/null 2>&1; then
        if ffmpeg -v error -rtsp_transport tcp -timeout 2000000 -i "$url" -t 0.1 -f null - >/dev/null 2>&1; then
          return 0
        fi
      else
        printf '%s\n' "ffprobe or ffmpeg required" >&2
        return 2
      fi
    fi

    now=$(date +%s)
    elapsed=$((now - start))
    if [ "$elapsed" -ge "$timeout" ]; then
      printf '%s\n' "timeout waiting for RTSP stream: $url" >&2
      return 1
    fi
    sleep "$interval"
  done
}


segment_dir=ffmpeg/seg/
output_dir=ffmpeg/out/

uv run wildcamtools serve tests/data/04-51-08.mp4 >/dev/null 2>&1 &
pid_serve=$!

if wait_for_rtsp_stream "rtsp://localhost:8554/stream" 30 1; then
  echo "Stream ready"
else
  echo "Stream not available"
  kill "$pid_serve" 2>/dev/null || true
  wait "$pid_serve" 2>/dev/null || true
  exit 1
fi

uv run wildcamtools segment rtsp://localhost:8554/stream "$segment_dir" >/dev/null 2>&1 &
pid_segment=$!

# Cleanup function
# one trap for each event
cleanup() {
  # avoid set -e exiting the trap early
  set +e
  echo "Cleaning up..."
  kill "$pid_segment" 2>/dev/null || true
  kill "$pid_serve" 2>/dev/null || true
  wait "$pid_segment" 2>/dev/null || true
  wait "$pid_serve" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

uv run wildcamtools watch "rtsp://localhost:8554/stream" "$segment_dir" "$output_dir"
