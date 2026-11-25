#!/usr/bin/env bash
# raw Rokoko CSVs -> one folder per recording for clean_mocap.py --rokoko-only:  bash prepare_samples.sh SRC OUT
set -euo pipefail
n=0
while IFS= read -r f; do b=$(basename "$f" .csv); mkdir -p "$2/$b"; cp "$f" "$2/$b/${b}_sync.csv"; n=$((n+1)); done < <(find "$1" -name 'rokoko_recording_*.csv' | sort)
echo "$n recordings in $2"
