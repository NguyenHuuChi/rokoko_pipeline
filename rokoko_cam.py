#!/usr/bin/env python3
"""Raw Rokoko joints in the reference camera frame, rebuilt with hand_fit/session.py's Scene (head -> camera geometry, same
settings as the original skeleton stage: rng seed 0, smooth_s 0.033): no hand fit, no elbow correction.

    python rokoko_cam.py SESSION_DIR          (~/envs/pipeline)  -> SESSION_DIR/camera/rokoko_cam.npz
npz: <joint> (n, 3) metres, reference camera; Ku, Ruw, tuw = ultra-wide projection (Cams.uw).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_fit.session import Inputs, Scene                              # noqa: E402

sd = Path(sys.argv[1])
sc = Scene(Inputs.find((sd / "camera/session").resolve(), (sd / "camera").resolve()), smooth_s=0.033)
np.savez_compressed(sd / "camera/rokoko_cam.npz", **{j: sc.cam[j].astype(np.float32) for j in sc.joint_names},
                    Ku=sc.cams.Ku, Ruw=sc.cams.Ruw, tuw=sc.cams.tuw)
print(sd.name, sc.n, "rows")
