#!/bin/bash
set -euxo pipefail

# Configurable values (edit or export to override)
HOST="${HOST:-localhost}"        # RTSP hostname or IP
PORT="${PORT:-8554}"             # RTSP port
PATH="${PATH:-stream}"           # RTSP stream path (no leading slash)

VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"   # video device
RESOLUTION="${RESOLUTION:-640x480}"           # e.g. 640x480, 1280x720
FRAMERATE="${FRAMERATE:-30}"                  # frames per second
INPUT_FORMAT="${INPUT_FORMAT:-yuyv422}"       # v4l2 input format
PIX_FMT="${PIX_FMT:-yuv420p}"                 # pixel format for ffmpeg filter

# Derived URL
RTSP_URL="rtsp://${HOST}:${PORT}/${PATH}"

# Explanantions:
#    -hide_banner: suppress the initial ffmpeg build/configuration banner
#    -loglevel error: only show error messages
#    -f v4l2: input as Video4Linux2
#    -input_format "${INPUT_FORMAT}": request a specific format from the v4l2 device - if unsupported it will fail
#    -video_size "${RESOLUTION}": resolution (WIDTHxHEIGHT) requested from camera
#    -framerate "${FRAMERATE}": frames per second requested from camera
#    -i "${VIDEO_DEVICE}": path to input device (e.g., /dev/video0)
#    -vf "format=${PIX_FMT}": apply a video filter to convert to the target pixel format expected by the encoder
#    -c:v libx264: select the video encoder; here H.264 via libx264. Change to another encoder (libx265, vp8, etc.) if desired
#    -preset ultrafast: trade-off compression speed vs. quality & CPU
#    -tune zerolatency: for low-latency streaming
#    -f rtsp "${RTSP_URL}": serve an RTSP stream to that address (single client)
ffmpeg -hide_banner \
    -loglevel error \
    -f v4l2 \
    -input_format "${INPUT_FORMAT}" \
    -video_size "${RESOLUTION}" \
    -framerate "${FRAMERATE}" \
    -i "${VIDEO_DEVICE}" \
    -vf "format=${PIX_FMT}" \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -f rtsp "${RTSP_URL}"
