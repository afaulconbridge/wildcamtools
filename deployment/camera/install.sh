#!/bin/bash
set -euxo pipefail

echo "[INFO] Installing script..."
cp streamer.sh /usr/local/bin/streamer.sh

echo "[INFO] Installing service..."
cp streamer.systemd /etc/systemd/system/streamer.service

echo "[INFO] Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable streamer
sudo systemctl restart streamer
