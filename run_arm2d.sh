#!/usr/bin/env bash
# Arm labels for ONE session (after run_final.sh: synced, cleaned, <session>/camera/rokoko_cam.npz):
#   JOB=<slurm job id> bash run_arm2d.sh SESSION_DIR [START_FRAME ...]           (run inside tmux; JOB set: GPU steps via srun --overlap, unset: on this machine)
# Clips of LEN (240) consecutive video frames; default starts: every LEN frames inside each kept sub-session
# (<session>/segments.csv), so no clip crosses a freeze cut.  Per clip, in <session>/arm2d/clips/<start>/:
#   1 cut       exact frames start .. start+LEN-1 of *_sync.mp4 (packet pts, timestamps reset)  -> clip.mp4, f0.txt, session.txt
#   2 detect    hung5_detect.py: hand5 wrists + RTMPose elbows / shoulders                     -> hands.npz, rtm_body.npz
#   3 masks     SEG=sam2 (default): sam2_clipmask.py, fine-tuned SAM 2.1 base+ prompted by the clip's hand5 / RTMPose
#               only (~1 min per clip)                                                          -> masks.npz
#               SEG=sam3: handseg_clipmask.py, SAM 3 rule + SAM 2.1 track (wearer arm + hand), ~5 min per clip
# then for all clips of the session together (one upper arm / forearm length pair per session):
#   4 hands     WRIST=wilor (default): wilor_wrists.py, WiLoR on our masks, tracked sides, wrist smoothing -> wilor_wrists.npz
#               WRIST=hand5: skipped, the fit uses the hand5 wrists of step 2
#   5 fit       arm2d_fit.py  (phone step, arm step with hold of the last detected offsets)    -> <session>/arm2d/fit/
#   6 filter    arm2d_filter.py (drop where the fitted arms still miss the image)              -> fit/<clip>_keep.npy
# Cost (SEG=sam3): SAM masks ~5 min per 240-frame clip on a shared H100; a 40 min session has ~300 clips (~25 h), so pass
# START_FRAMEs to label a sample of a session.
set -euo pipefail
SD=${1%/}; shift
JOB=${JOB:-}; LEN=${LEN:-240}; SEG=${SEG:-sam2}; WRIST=${WRIST:-wilor}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/.env" ] || { echo "copy .env.example to .env first"; exit 1; }
set -a; . "$HERE/.env"; set +a
[ -z "${FFMPEG_DIR:-}" ] || export PATH=$FFMPEG_DIR:$PATH
DPY=$PY_DETECT HPY=$PY_HANDSEG WPY=$PY_WILOR       # hand5 / RTMPose | SAM, fit, filter | WiLoR
GPU=(); [ -z "$JOB" ] || GPU=(srun --overlap --jobid "$JOB" --export=ALL)      # no JOB: run on this machine
O=$SD/arm2d; mkdir -p "$O/clips" "$O/fit"
V=$(ls "$SD"/*_sync.mp4)
PTS=$O/packet_pts.txt
[ -s "$PTS" ] || ffprobe -v error -select_streams v -show_entries packet=pts_time -of csv=p=0 "$V" | sort -g > "$PTS"
if [ $# -eq 0 ]; then
  mapfile -t STARTS < <(awk -F, -v L=$LEN 'NR>1{for(s=$2; s+L-1<=$3; s+=L) print s}' "$SD/segments.csv")
else STARTS=("$@"); fi
echo "$(basename "$SD"): ${#STARTS[@]} clips of $LEN frames"
CL=()
for f0 in "${STARTS[@]}"; do
  c=$O/clips/$f0; CL+=("$c"); mkdir -p "$c"
  echo "$SD" > "$c/session.txt"; echo "$f0" > "$c/f0.txt"
  if [ ! -s "$c/clip.mp4" ]; then
    read t0 t1 < <(awk -v a=$f0 -v L=$LEN 'NR==a+1{x=$1} NR==a+L{print x, $1; exit}' "$PTS")
    ffmpeg -v error -y -ss "$(awk "BEGIN{print ($t0>3)?$t0-3:0}")" -copyts -i "$V" \
      -vf "select=between(t\,$t0-0.005\,$t1+0.005),setpts=PTS-STARTPTS" -fps_mode passthrough -c:v libx264 -crf 16 -an "$c/clip.mp4"
  fi
  [ -s "$c/hands.npz" ] || "${GPU[@]}" "$DPY" "$HERE/hung5_detect.py" "$c/clip.mp4" "$c" > "$c/detect.log" 2>&1
done
todo=(); for c in "${CL[@]}"; do [ -s "$c/masks.npz" ] || todo+=("$c"); done
[ ${#todo[@]} -eq 0 ] || HF_HUB_OFFLINE=1 "${GPU[@]}" "$HPY" "$HERE/$([ "$SEG" = sam3 ] && echo handseg_clipmask.py || echo sam2_clipmask.py)" "${todo[@]}" > "$O/masks.log" 2>&1
if [ "$WRIST" = wilor ]; then
  todo=(); for c in "${CL[@]}"; do [ -s "$c/wilor_wrists.npz" ] || todo+=("$c"); done
  [ ${#todo[@]} -eq 0 ] || TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "${GPU[@]}" "$WPY" "$HERE/wilor_wrists.py" "${todo[@]}" > "$O/wilor.log" 2>&1
fi
"${GPU[@]}" "$HPY" "$HERE/arm2d_fit.py" "${CL[@]}" --out "$O/fit" --wrist "$WRIST" > "$O/fit.log" 2>&1
"$HPY" "$HERE/arm2d_filter.py" "$O/fit" "${CL[@]}" --wrist "$WRIST" > "$O/filter.log" 2>&1
tail -n $(( ${#CL[@]} + 1 )) "$O/filter.log"
