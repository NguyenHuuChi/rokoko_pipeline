#!/usr/bin/env python3
"""Sync check of every finished session: does the Rokoko head turn when the phone camera (ARKit) turns?

    python sync_check.py [OUT_ROOT]        (~/envs/pipeline)

Per WIN_S window (hop HOP_S) the head and camera rotation speeds (deg/frame) are correlated at every lag within
+-MAX_LAG frames; a correct sync peaks at lag 0.  Writes <session>/camera/sync_check.csv:
  f0, f1        video frames of the window
  lag           best lag (frames; > 0: Rokoko late), corr = correlation there, corr0 = correlation at lag 0
  cam_deg_s     camera rotation speed p90 (deg/s): low = little motion to compare, the window says nothing
and <session>/camera/sync_check.json: the head -> camera calibration residual (median deg; ~2 on a good take).
Frames clean_mocap marked invalid are left out.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_fit.session import Inputs, Scene        # noqa: E402

WIN_S, HOP_S, MAX_LAG, FPS = 30.0, 15.0, 90, 30.0


def speed(R):
    return np.r_[Rotation.from_matrix(np.swapaxes(R[:-1], 1, 2) @ R[1:]).magnitude(), 0.0]


def check(sd: Path):
    sc = Scene(Inputs.find(sd / "camera/session", sd / "camera"))
    man = pd.read_csv(sc.inp.manifest)
    a, b = speed(sc.R_H), speed(sc.R_AC)
    ok = man["noise_quality"].to_numpy() != "invalid"
    W, Hp = int(WIN_S * FPS), int(HOP_S * FPS)
    rows = []
    for s in range(0, max(len(a) - W, 0) + 1, Hp):
        i = np.arange(max(s, MAX_LAG), min(s + W, len(a) - MAX_LAG))
        best, c0 = (np.nan, 0), np.nan
        for lag in range(-MAX_LAG, MAX_LAG + 1):
            m = ok[i] & ok[i - lag]
            if m.sum() < W // 3:
                continue
            c = np.corrcoef(a[i - lag][m], b[i][m])[0, 1]
            c0 = c if lag == 0 else c0
            if not c <= best[0]:
                best = (c, lag)
        rows.append(dict(f0=s, f1=min(s + W, len(a)) - 1, lag=best[1], corr=round(best[0], 3), corr0=round(c0, 3),
                         cam_deg_s=round(float(np.degrees(np.percentile(b[s:s + W], 90)) * FPS), 1)))
    pd.DataFrame(rows).to_csv(sd / "camera/sync_check.csv", index=False)
    g = sc.geometry
    (sd / "camera/sync_check.json").write_text(json.dumps({"calib_residual_deg": round(g["R_HC_pair_residual_deg"]["median"], 2),
                                                         "t_HC_mm": g["t_HC_mm"]}))
    return g["R_HC_pair_residual_deg"]["median"], pd.DataFrame(rows)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else __import__("settings").OUT_ROOT
    for j in sorted(glob.glob(f"{root}/*/*/session_*/camera/rokoko_cam.npz")):     # sessions past stage 3 (camera)
        sd = Path(j).parent.parent
        if (sd / "camera/sync_check.csv").exists():
            continue
        res, d = check(sd)
        good = d[d["corr"] >= 0.5]
        print(f"{sd.name[8:23]} calib {res:5.1f} deg  windows {len(d)}  measured {len(good)}  "
              f"off(|lag|>2) {(good.lag.abs() > 2).sum()}", flush=True)


if __name__ == "__main__":
    main()
