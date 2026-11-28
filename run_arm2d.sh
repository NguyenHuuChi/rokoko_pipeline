#!/usr/bin/env bash
# Arm labels for ONE session (after run_final.sh: synced, cleaned, <session>/camera/rokoko_cam.npz):
#   JOB=<slurm job id> bash run_arm2d.sh SESSION_DIR [--mode fasam|fah|hand5] [START_FRAME ...]
# Clips of LEN (240) consecutive video frames; default starts: every LEN frames inside each kept sub-session
# (<session>/segments.csv), so no clip crosses a freeze cut.  MODE (or --mode):
#   fasam  (default, M3b) FAH YOLOv8-seg + RTMPose -> SAM 2.1 base+ 2-ID (L=1, R=2) -> WiLoR -> fit -> filter
#   fah    (M3a) FAH YOLOv8-seg + RTMPose L/R lock + gap-fill (1-3f) + SavGol (no SAM) -> WiLoR -> fit -> filter
#   hand5  original repo baseline: RTMPose + Hand5 -> SEG=sam2|sam3 -> WiLoR -> fit -> filter
set -euo pipefail
MODE=${MODE:-fasam}
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE=${2:?--mode requires fasam|fah|hand5}; shift 2 ;;
    --mode=*) MODE=${1#*=}; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]}"
SD=${1%/}; shift
case "$MODE" in
  fasam|fah|hand5) ;;
  *) echo "unknown MODE='$MODE' (expected fasam, fah, or hand5)" >&2; exit 2 ;;
esac
JOB=${JOB:-}; LEN=${LEN:-240}; SEG=${SEG:-sam2}; WRIST=${WRIST:-wilor}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/.env" ] || { echo "copy .env.example to .env first"; exit 1; }
set -a; . "$HERE/.env"; set +a
[ -z "${FFMPEG_DIR:-}" ] || export PATH=$FFMPEG_DIR:$PATH
DPY=$PY_DETECT HPY=$PY_HANDSEG WPY=$PY_WILOR       # RTMPose/FAH/hand5 | SAM, fit, filter | WiLoR
FAH_PY=$DPY
if [ ! -f "$HERE/third_party/FAH/fah/detector.py" ] || ! "$DPY" -c "import torch, torchvision, scipy, cv2" 2>/dev/null; then
  FAH_PY=$WPY
fi
GPU=(); [ -z "$JOB" ] || GPU=(srun --overlap --jobid "$JOB" --export=ALL)      # no JOB: run on this machine
O=$SD/arm2d; mkdir -p "$O/clips" "$O/fit"
V=$(ls "$SD"/*_sync.mp4)
PTS=$O/packet_pts.txt
[ -s "$PTS" ] || ffprobe -v error -select_streams v -show_entries packet=pts_time -of csv=p=0 "$V" | sort -g > "$PTS"
if [ $# -eq 0 ]; then
  mapfile -t STARTS < <(awk -F, -v L=$LEN 'NR>1{for(s=$2; s+L-1<=$3; s+=L) print s}' "$SD/segments.csv")
else STARTS=("$@"); fi
echo "$(basename "$SD"): ${#STARTS[@]} clips of $LEN frames (mode=$MODE)"
CL=()
for f0 in "${STARTS[@]}"; do
  c=$O/clips/$f0; CL+=("$c"); mkdir -p "$c"
  echo "$SD" > "$c/session.txt"; echo "$f0" > "$c/f0.txt"
  if [ ! -s "$c/clip.mp4" ]; then
    read t0 t1 < <(awk -v a=$f0 -v L=$LEN 'NR==a+1{x=$1} NR==a+L{print x, $1; exit}' "$PTS")
    ffmpeg -v error -y -ss "$(awk "BEGIN{print ($t0>3)?$t0-3:0}")" -copyts -i "$V" \
      -vf "select=between(t\,$t0-0.005\,$t1+0.005),setpts=PTS-STARTPTS" -fps_mode passthrough -c:v libx264 -crf 16 -an "$c/clip.mp4"
  fi
  if [ "$MODE" = hand5 ]; then
    [ -s "$c/hands.npz" ] || "${GPU[@]}" "$DPY" "$HERE/hung5_detect.py" "$c/clip.mp4" "$c" > "$c/detect.log" 2>&1
  else
    [ -s "$c/rtm_body.npz" ] || "${GPU[@]}" "$DPY" "$HERE/hung5_detect.py" "$c/clip.mp4" "$c" --rtm-only > "$c/detect.log" 2>&1
  fi
done
if [ "$MODE" = fah ] || [ "$MODE" = fasam ]; then
  req=$([ "$MODE" = fah ] && echo "fah_boxes.npz" || echo "fah_prompts.npz")
  todo=(); for c in "${CL[@]}"; do [ -s "$c/$req" ] || todo+=("$c"); done
  [ ${#todo[@]} -eq 0 ] || TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "${GPU[@]}" "$FAH_PY" "$HERE/fah_detect.py" --mode "$MODE" "${todo[@]}" > "$O/fah_detect.log" 2>&1
fi
if [ "$MODE" = fasam ]; then
  todo=(); for c in "${CL[@]}"; do [ -s "$c/masks.npz" ] && [ -s "$c/fah_boxes.npz" ] || todo+=("$c"); done
  [ ${#todo[@]} -eq 0 ] || HF_HUB_OFFLINE=1 "${GPU[@]}" "$HPY" "$HERE/sam2_clipmask.py" --mode fasam "${todo[@]}" > "$O/masks.log" 2>&1
elif [ "$MODE" = hand5 ]; then
  todo=(); for c in "${CL[@]}"; do [ -s "$c/masks.npz" ] || todo+=("$c"); done
  if [ ${#todo[@]} -gt 0 ]; then
    if [ "$SEG" = sam3 ]; then
      HF_HUB_OFFLINE=1 "${GPU[@]}" "$HPY" "$HERE/handseg_clipmask.py" "${todo[@]}" > "$O/masks.log" 2>&1
    else
      HF_HUB_OFFLINE=1 "${GPU[@]}" "$HPY" "$HERE/sam2_clipmask.py" --mode hand5 "${todo[@]}" > "$O/masks.log" 2>&1
    fi
  fi
fi
if [ "$WRIST" = wilor ]; then
  todo=(); for c in "${CL[@]}"; do [ -s "$c/wilor_wrists.npz" ] || todo+=("$c"); done
  [ ${#todo[@]} -eq 0 ] || TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "${GPU[@]}" "$WPY" "$HERE/wilor_wrists.py" --mode "$MODE" "${todo[@]}" > "$O/wilor.log" 2>&1
fi
"${GPU[@]}" "$HPY" "$HERE/arm2d_fit.py" "${CL[@]}" --out "$O/fit" --wrist "$WRIST" > "$O/fit.log" 2>&1
"$HPY" "$HERE/arm2d_filter.py" "$O/fit" "${CL[@]}" --wrist "$WRIST" > "$O/filter.log" 2>&1
tail -n $(( ${#CL[@]} + 1 )) "$O/filter.log"
