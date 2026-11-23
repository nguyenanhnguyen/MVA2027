#!/usr/bin/env bash
# Track 2:  bash run_track2.sh INPUT_DIR OUTPUT_DIR
# INPUT_DIR: {seq_id}/{window_start}/00001.jpg..00060.jpg (+ imu.csv, không dùng). Không đọc cửa sổ khác.
# OUTPUT_DIR/prediction/{seq_id}.txt  và  OUTPUT_DIR/tracking/{seq_id}.txt
set -euo pipefail
python -m smotp.cli run-track2 "$1" "$2"
