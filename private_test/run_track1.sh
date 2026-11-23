#!/usr/bin/env bash
# Track 1:  bash run_track1.sh INPUT_DIR OUTPUT_DIR
# INPUT_DIR: camera_intrinsics.json + {seq_id}/00001.jpg ... (+ imu.csv, không dùng)
# OUTPUT_DIR/{seq_id}.txt  gồm: frame_id,track_id,x,y,w,h,conf
set -euo pipefail
python -m smotp.cli run-track1 "$1" "$2"
