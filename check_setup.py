#!/usr/bin/env python3
"""List what is missing before running the pipeline."""
import os
import shutil
import subprocess
from pathlib import Path

import settings as S

bad = 0


def need(ok, what):
    global bad
    bad += not ok
    print("ok     " if ok else "MISSING", what)


need((S.HERE / ".env").exists(), ".env (cp .env.example .env)")
need(Path(S.RAW_ROOT).is_dir(), f"RAW_ROOT {S.RAW_ROOT}")
need(bool(shutil.which(S.FFMPEG)), f"ffmpeg ({S.FFMPEG})")
for k, mods in (("PY_PIPELINE", "numpy,pandas,scipy,cv2"), ("PY_DETECT", "onnxruntime,torch,cv2,pandas"),
                ("PY_HANDSEG", "torch,sam2,sam3,cv2,pandas"), ("PY_WILOR", "torch,smplx,yacs,pytorch_lightning,ultralytics,cv2,pandas")):
    py = S.get(k, "python")
    r = subprocess.run([py, "-c", f"import {mods}"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""}) if shutil.which(py) else None
    need(bool(r) and r.returncode == 0, f"{k} {py}: {mods}" + (f"  ({r.stderr.strip().splitlines()[-1]})" if r and r.returncode else ""))
M = S.HERE / "models"
for f in (M / "hand5/hand5_r17d.pt", M / "hand5/rtmpose_m_hand5_clip2max.onnx", M / "rtmpose/rtmpose_l_body12_ft4c.onnx", S.SAM2_FT, S.SAM2_BASE):
    need(Path(f).is_file(), f"weights {f}")
for f in ("wilor/models/wilor.py", "pretrained_models/wilor_final.ckpt", "mano_data/MANO_RIGHT.pkl"):
    need((Path(S.WILOR) / f).is_file(), f"WiLoR {f} (not needed with WRIST=hand5)")
for f in (S.SAM2_LARGE, S.SAM3):
    print("ok     " if Path(f).is_file() else "absent ", f"{f} (only MODE=hand5 SEG=sam3)")
print("ok     " if Path(S.FAH_DET).is_file() else "absent ", f"FAH weights {S.FAH_DET} (bash download_fah.sh; required for MODE=fasam|fah)")
print("ok     " if (Path(S.FAH) / "fah/detector.py").is_file() else "absent ", f"FAH arch {Path(S.FAH) / 'fah/detector.py'} (bash download_fah.sh)")
print("all set" if not bad else f"{bad} missing")
raise SystemExit(bool(bad))
