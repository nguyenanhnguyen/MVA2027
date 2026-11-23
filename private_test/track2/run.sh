#!/usr/bin/env bash
# bash run.sh INPUT_DIR OUTPUT_DIR   (Track 2)
# INPUT_DIR: {seq_id}/{window_start}/00001.jpg..00060.jpg (+ imu.csv). Cấm đọc cửa sổ khác.
# OUTPUT_DIR/prediction/{seq_id}.txt  và  OUTPUT_DIR/tracking/{seq_id}.txt
set -euo pipefail
IN="$1"; OUT="$2"; mkdir -p "$OUT/prediction" "$OUT/tracking"
echo "TODO: tracker trên 60 frame -> predictor" >&2
exit 1
