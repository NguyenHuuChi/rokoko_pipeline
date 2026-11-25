#!/usr/bin/env python3
"""Last stage of the arm labels: drop the frames where Rokoko is wrong beyond what the small arm correction
(arm2d_fit.py) can fix (the fitted skeleton still disagrees with the detections), then light smoothing of the arms.

    python arm2d_filter.py FIT_DIR CLIP_DIR [CLIP_DIR ...] [--wrist wilor|hand5]          (~/envs/handseg, CPU)

Per frame, only where hand5 sees a wrist (frames without one hold the last detected offsets and are kept):
  any hand seen     a confident hand5 wrist and the fitted wrist more than RES_PX apart (the fit cannot bring the arm
                    onto the real hand: Rokoko wrong beyond the allowed correction)
  both hands seen   also: hand-to-hand vector (left -> right wrist) of the fitted skeleton vs the detected one, off > MIS_PX
                    (a camera shift moves both hands alike and cancels; what is left is arm pose)
A bad run of >= MIN_BAD_S is dropped, widened by PAD_S on both sides; shorter blips are kept, except frames where a
confident wrist is missed by more than BIG_PX (dropped on their own, no widening).  Also dropped: every frame
clean_mocap.py marked valid = 0 (freeze, short sub-session, or at >= 31 fps a video frame without its own Rokoko sample),
so the keep mask is the one place that decides which frames become labels.
Smoothing (always): the fitted arm joints (upper arm, forearm, hand, fingers) are smoothed as directions from their
parent (Savitzky-Golay over ARM_SMOOTH_S, order 2), then rebuilt from the shoulder with their own bone lengths
(unchanged); each kept run on its own, never across a dropped stretch; the body stays as clean_mocap left it.
Writes FIT_DIR/<clip>_labels.npz (the labels: joints (names), xyz (n, J, 3) corrected camera frame m, uv (n, J, 2) video
pixels, keep, frame = video frame ids), <clip>_keep.npy, <clip>_filtered.mp4 (fit video, dropped frames marked) and
filter_summary.csv (+ wrist jitter / wrist error before -> after smoothing).
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm2d_fit import Proj, wrists                                    # noqa: E402
from clean_mocap import FINGER, depth_of, odd, parent_of                    # noqa: E402
from scipy.signal import savgol_filter                                      # noqa: E402
from handseg_pilot import FF                                                # noqa: E402

MIS_PX, RES_PX, MIN_BAD_S, PAD_S, FPS = 100.0, 40.0, 0.5, 0.2, 30.0
BIG_PX = 150.0           # a confident wrist missed by more than this: frame dropped however short (never a usable label)
ARM_SMOOTH_S = 0.13      # 5 frames at 30 fps: shorter than the cleaner's 0.23 s, a fast swing must not lag the hand


def runs(b):
    """(start, end) of True runs, end exclusive."""
    e = np.diff(np.r_[0, b.astype(int), 0])
    return list(zip(np.where(e == 1)[0], np.where(e == -1)[0]))


def smooth_arms(X, keep, fps):
    """X: joint -> (n, 3).  Arm joints smoothed as directions from their parent inside each kept run, lengths kept."""
    arm = [j for j in X if j.startswith(("left", "right")) and (j.endswith(("UpperArm", "LowerArm", "Hand")) or FINGER.search(j))]
    sw = odd(ARM_SMOOTH_S * fps)
    Y = {j: v.copy() for j, v in X.items()}
    for a, b in runs(keep):
        if b - a <= sw:
            continue
        for j in sorted(arm, key=lambda j: depth_of(j, X)):                   # parents first
            p = parent_of(j, X)
            if p is None:
                continue
            v = X[j][a:b] - X[p][a:b]
            L = np.linalg.norm(v, axis=1, keepdims=True)
            u = savgol_filter(v / np.maximum(L, 1e-9), sw, 2, axis=0)
            Y[j][a:b] = Y[p][a:b] + L * u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return Y


def jitter(uv, keep):
    """Mean |second difference| (px / frame^2) of a projected track over kept frames with kept neighbours."""
    acc = np.linalg.norm(uv[2:] - 2 * uv[1:-1] + uv[:-2], axis=1)
    ok = keep[2:] & keep[1:-1] & keep[:-2] & np.isfinite(acc)
    return float(acc[ok].mean()) if ok.any() else np.nan


def keep_mask(fit, z, w, conf):
    P = Proj(z)
    pj = lambda X: P(torch.tensor(X)).cpu().numpy()
    hw = np.stack([pj(fit["seg_leftHand"]), pj(fit["seg_rightHand"])], 1)       # (n, 2, 2) fitted wrists, pixels
    seen = np.isfinite(w).all(-1)
    both = seen.all(1)
    mis = np.linalg.norm((hw[:, 0] - hw[:, 1]) - (w[:, 0] - w[:, 1]), axis=1)
    res = np.linalg.norm(hw - w, axis=2)
    wrist_off = np.where(seen & conf, res, 0).max(1) > RES_PX                     # any confident wrist the fit misses
    bad = (both & (mis > MIS_PX)) | wrist_off
    keep = np.ones(len(w), bool)
    pad = int(PAD_S * FPS)
    for a, b in runs(bad):
        if b - a >= MIN_BAD_S * FPS:
            keep[max(a - pad, 0):b + pad] = False
    keep &= ~(np.where(seen & conf, res, 0).max(1) > BIG_PX)                    # big miss: dropped however short
    return keep, bad, mis, both


def main():
    args, src = sys.argv[1:], "wilor"                                          # same wrist target as arm2d_fit.py
    if "--wrist" in args:
        i = args.index("--wrist")
        src = args[i + 1]
        del args[i:i + 2]
    out, clips = Path(args[0]), [Path(c) for c in args[1:]]
    rows = []
    for cd in clips:
        sd = Path((cd / "session.txt").read_text().strip())
        f0 = int((cd / "f0.txt").read_text())
        name = f"{sd.name[8:23]}_{f0}"
        fit, z = np.load(out / f"{name}.npz"), np.load(sd / "camera/rokoko_cam.npz")
        n = len(fit["seg_leftHand"])
        w, conf, _, _ = wrists(cd, n, src)
        keep, bad, mis, both = keep_mask(fit, z, w, conf)
        valid = pd.read_csv(next(sd.glob("*_perframe_clean.csv")), usecols=["valid"], skiprows=range(1, f0 + 1), nrows=n)["valid"]
        keep &= valid.to_numpy() == 1
        np.save(out / f"{name}_keep.npy", keep)
        names = [k[4:] for k in fit.files if k.startswith("seg_") and fit[k].ndim == 2 and fit[k].shape[1] == 3 and k[4:] in z.files]
        X = {j: fit[f"seg_{j}"].astype(float) for j in names}
        Y = smooth_arms(X, keep, FPS)
        P = Proj(z)
        pj = lambda A: P(torch.tensor(A)).cpu().numpy()
        xyz = np.stack([Y[j] for j in names], 1)
        uv = pj(xyz)
        np.savez_compressed(out / f"{name}_labels.npz", joints=np.array(names), xyz=xyz.astype(np.float32), uv=uv.astype(np.float32),
                            keep=keep, frame=np.arange(f0, f0 + n))
        sm = {}
        for tag, S in (("before", X), ("after", Y)):
            wu = np.stack([pj(S[f"{s}Hand"]) for s in ("left", "right")], 1)
            sm[f"wrist_jitter_{tag}"] = round(np.nanmean([jitter(wu[:, i], keep) for i in range(2)]), 2)
            e = np.linalg.norm(wu - w, axis=2)[conf & keep[:, None]]
            sm[f"wrist_px_{tag}"] = round(float(np.median(e)), 2) if len(e) else np.nan
        rows.append(dict(clip=name, frames=n, both_hands=int(both.sum()), no_hand=int((~np.isfinite(w).all(-1)).all(1).sum()),
                         bad_frames=int(bad.sum()), invalid=int((valid != 1).sum()), dropped=int((~keep).sum()), dropped_pct=round(100 * (~keep).mean(), 1),
                         mismatch_med_px=round(float(np.median(mis[both])), 1) if both.any() else np.nan, **sm))
        print(rows[-1], flush=True)
        cap = cv2.VideoCapture(str(out / f"{name}.mp4"))
        Wv, Hv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wv}x{Hv}", "-r", "30", "-i", "-",
                                "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(out / f"{name}_filtered.mp4")], stdin=subprocess.PIPE)
        for k in range(n):
            ok, im = cap.read()
            if not ok:
                break
            if not keep[k]:
                cv2.rectangle(im, (0, 0), (Wv - 1, Hv - 1), (0, 0, 255), 12)
                why = ("no Rokoko data (clean_mocap valid = 0)" if valid.iloc[k] != 1 else
                       f"Rokoko wrong (hand-to-hand off {mis[k]:.0f} px)" if both[k] else "Rokoko wrong")
                cv2.putText(im, "DROPPED: " + why,
                            (20, Hv // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            enc.stdin.write(im.tobytes())
        enc.stdin.close(); enc.wait()
    df = pd.DataFrame(rows)
    df.to_csv(out / "filter_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
