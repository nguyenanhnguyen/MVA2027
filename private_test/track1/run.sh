#!/usr/bin/env bash
# bash run.sh INPUT_DIR OUTPUT_DIR   (Track 1)
# INPUT_DIR: camera_intrinsics.json + {seq_id}/00001.jpg ... (+ imu.csv)
# OUTPUT_DIR/{seq_id}.txt : frame_id,track_id,x,y,w,h,conf
set -euo pipefail
IN="$1"; OUT="$2"; mkdir -p "$OUT"
echo "TODO: gọi detector + tracker trong src/smotp" >&2
exit 1
