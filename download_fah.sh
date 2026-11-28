#!/usr/bin/env bash
# Fetch FAH standard YOLOv8-seg architecture (third_party/FAH) and detector weights (models/fah/shared/detector.pt).
# Needs: git, hf (huggingface_hub, e.g. from ~/envs/handseg/bin/hf).
# Usage:
#   bash download_fah.sh                  # fetch both architecture and shared/detector.pt (~52 MB)
#   bash download_fah.sh --arch-only      # fetch only official FAH YOLO architecture code (third_party/FAH)
#   bash download_fah.sh --weights-only   # fetch only shared/detector.pt (~52 MB)
#
# Note: FAH detector weights (noahcao/FAH) are governed by FAH's WEIGHTS_LICENSE (research / non-commercial use),
# and its YOLO weights are Ultralytics AGPL-3.0.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }

FAH_DIR=${FAH:-$HERE/third_party/FAH}
FAH_MODELS=${FAH_MODELS:-$HERE/models/fah}
FAH_DET_OUT=${FAH_DET:-$FAH_MODELS/shared/detector.pt}
FAH_REPO=${FAH_REPO:-https://github.com/noahcao/FAH.git}
HF_REPO=${FAH_HF_REPO:-noahcao/FAH}
HF_BIN=${HF_BIN:-}
if [ -z "$HF_BIN" ]; then
  if command -v hf >/dev/null 2>&1; then
    HF_BIN=$(command -v hf)
  elif [ -n "${PY_HANDSEG:-}" ] && [ -x "$(dirname "$PY_HANDSEG")/hf" ]; then
    HF_BIN="$(dirname "$PY_HANDSEG")/hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_BIN=$(command -v huggingface-cli)
  fi
fi

DO_ARCH=1
DO_WEIGHTS=1
for arg in "$@"; do
  case "$arg" in
    --arch-only) DO_WEIGHTS=0 ;;
    --weights-only) DO_ARCH=0 ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$DO_ARCH" -eq 1 ]; then
  echo "[FAH arch] Fetching official YOLO architecture into $FAH_DIR ..."
  if [ ! -d "$FAH_DIR/.git" ]; then
    mkdir -p "$(dirname "$FAH_DIR")"
    git clone --depth 1 --filter=blob:none --sparse "$FAH_REPO" "$FAH_DIR"
  fi
  git -C "$FAH_DIR" sparse-checkout set --no-cone \
    fah/detector.py \
    fah/yolo.py \
    fah/cuda_graph.py \
    fah/frame_cache.py \
    fah/utils.py \
    LICENSE \
    NOTICE \
    THIRD_PARTY_NOTICES.md \
    WEIGHTS_LICENSE
  git -C "$FAH_DIR" pull --ff-only || true
  for f in fah/detector.py fah/yolo.py fah/cuda_graph.py; do
    [ -s "$FAH_DIR/$f" ] || { echo "ERROR: missing $FAH_DIR/$f" >&2; exit 1; }
  done
  echo "[FAH arch] OK: $FAH_DIR/fah/{detector.py,yolo.py,cuda_graph.py}"
fi

if [ "$DO_WEIGHTS" -eq 1 ]; then
  [ -n "$HF_BIN" ] || { echo "ERROR: neither 'hf' nor 'huggingface-cli' found; activate your env or set HF_BIN." >&2; exit 1; }
  mkdir -p "$FAH_MODELS"
  echo "[FAH weights] Downloading shared/detector.pt from $HF_REPO -> $FAH_MODELS ..."
  if [[ "$(basename "$HF_BIN")" == "huggingface-cli" ]]; then
    "$HF_BIN" download "$HF_REPO" shared/detector.pt --local-dir "$FAH_MODELS"
  else
    "$HF_BIN" download "$HF_REPO" shared/detector.pt --local-dir "$FAH_MODELS"
  fi
  if [ "$FAH_DET_OUT" != "$FAH_MODELS/shared/detector.pt" ] && [ ! -s "$FAH_DET_OUT" ]; then
    mkdir -p "$(dirname "$FAH_DET_OUT")"
    cp "$FAH_MODELS/shared/detector.pt" "$FAH_DET_OUT"
  fi
  [ -s "$FAH_DET_OUT" ] || { echo "ERROR: missing weights at $FAH_DET_OUT" >&2; exit 1; }
  echo "[FAH weights] OK: $FAH_DET_OUT ($(du -h "$FAH_DET_OUT" | cut -f1))"
fi

echo "Done. Run: python check_setup.py"
