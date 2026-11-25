#!/usr/bin/env bash
# Sync + clean + camera for every session (run inside tmux; CPU only):
#   bash run_final.sh
#   1 sync    prepare_mocap.py: pair raw zip <-> Rokoko CSV, fit_rokoko.py measures the clock offset, cuts both to the
#             common span, one Rokoko row per video frame
#   2 clean   clean_mocap.py --force: freezes > 0.5 s cut (session -> sub-sessions, segments.csv), sub-sessions < 3 s
#             dropped, glitches repaired and light smoothing on joint directions; bone lengths stay raw
#   3 camera  camera_adapter.py (our pair in the camera stage's input layout) + rokoko_cam.py (head -> camera geometry):
#             <session>/camera/rokoko_cam.npz = raw Rokoko joints in the camera frame + projection
# Then per session: run_arm2d.sh (detections, SAM masks, arm fit, filter).
# In:  RAW_ROOT (raw)      Out: OUT_ROOT/<date>/<pair>/<session>/
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/.env" ] || { echo "copy .env.example to .env first"; exit 1; }
set -a; . "$HERE/.env"; set +a
RAW=$RAW_ROOT; OUT=$OUT_ROOT
APY=$PY_PIPELINE; PY=${PY_CLEAN:-$PY_PIPELINE}
LOG=$OUT/final_run.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "1/3 sync"
python3 "$HERE/prepare_mocap.py" "$RAW" "$OUT" --ffmpeg-dir "${FFMPEG_DIR:-}" --clean-python "$PY" -j 6 >> "$LOG" 2>&1
log "2/3 clean (freeze split, raw bone lengths)"
"$PY" "$HERE/clean_mocap.py" "$OUT" --force >> "$LOG" 2>&1
log "3/3 camera"
for sd in "$OUT"/*/*/session_*/; do
  sd=${sd%/}; [ -f "$sd/sync_report.json" ] || continue
  if "$APY" "$HERE/camera_adapter.py" "$sd" >> "$LOG" 2>&1 && "$APY" "$HERE/rokoko_cam.py" "$sd" >> "$LOG" 2>&1; then
    log "done $(basename "$sd")"
  else log "FAILED $(basename "$sd")"; fi
done
log "ALL DONE: $(ls "$OUT"/*/*/session_*/camera/rokoko_cam.npz 2>/dev/null | wc -l) sessions with rokoko_cam.npz"
