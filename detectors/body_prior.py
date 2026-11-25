"""10/10: hand prior WITHOUT Rokoko, from a body pose model on the video itself (RTMPose-l body7, elbow fine-tune,
rtmpose_local.py video): same npz layout as make_prior.py (wrist (n,2,2) px, centre (n,2,2), size (n,2) hand length px,
valid (n,2)), so detect_hands_student.py --prior / --wrist-gate use it unchanged.  Side = the body model's left / right
(COCO 9 / 10 wrists, 7 / 8 elbows), i.e. the wearer's arm the hand hangs on, not YOLO's handedness guess.
  size    K_SIZE x forearm px, clipped to 0.6-1.6 x the take median (foreshortened forearms).  K_SIZE 1.15: the crop side the
          Rokoko prior uses (19 cm at the wrist depth) is 1.55-1.67 x the image forearm on 17:46 (top-down view shortens it)
  centre  wrist + 0.42 x size along elbow -> wrist (Rokoko prior: 8 cm of a 19 cm hand)
  valid   wrist score >= WRIST_MIN and elbow score >= ELBOW_MIN, inside the picture; gaps <= GAP frames interpolated;
          positions median-filtered over 3 frames
    python body_prior.py <rtm npz (video mode)> <out prior.npz> [--wrist-min 0.3] [--elbow-min 0.2] [--gap 5]
"""
import argparse

import numpy as np

KP = {0: (9, 7), 1: (10, 8)}                     # side -> (wrist, elbow) COCO-17
K_SIZE = 1.15


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rtm"); ap.add_argument("out")
    ap.add_argument("--wrist-min", type=float, default=0.3); ap.add_argument("--elbow-min", type=float, default=0.2)
    ap.add_argument("--gap", type=int, default=5); ap.add_argument("--wh", default="1920,1440")
    a = ap.parse_args()
    Z = np.load(a.rtm); n = int(Z["n_frames"]); W, H = (int(v) for v in a.wh.split(","))
    kp = np.full((n, 17, 2), np.nan); cf = np.zeros((n, 17))
    kp[Z["frame"]] = Z["kp"]; cf[Z["frame"]] = Z["conf"]
    wrist = np.full((n, 2, 2), np.nan); centre = np.full((n, 2, 2), np.nan); size = np.full((n, 2), np.nan); valid = np.zeros((n, 2), bool)
    t = np.arange(n)
    for h, (wi, ei) in KP.items():
        w, e = kp[:, wi].copy(), kp[:, ei].copy()
        ok = (cf[:, wi] >= a.wrist_min) & (cf[:, ei] >= a.elbow_min) & np.isfinite(w).all(1) & (w[:, 0] >= 0) & (w[:, 0] < W) & (w[:, 1] >= 0) & (w[:, 1] < H)
        for arr in (w, e):                                                   # 3-frame median on the good rows
            m = np.stack([np.roll(arr, s, 0) for s in (-1, 0, 1)]); okm = np.stack([np.roll(ok, s) for s in (-1, 0, 1)])
            m[~okm] = np.nan; arr[ok] = np.nanmedian(m[:, ok], 0)
        # interpolate short gaps
        good = np.nonzero(ok)[0]; fill = ok.copy()
        for g0, g1 in zip(good[:-1], good[1:]):
            if 1 < g1 - g0 <= a.gap + 1:
                fill[g0 + 1:g1] = True
        if len(good) >= 2:
            for arr in (w, e):
                for d in (0, 1):
                    arr[:, d] = np.where(fill, np.interp(t, good, arr[good, d]), np.nan)
        fore = np.linalg.norm(w - e, axis=1)
        med = np.nanmedian(fore[ok]) if ok.any() else np.nan
        L = np.clip(K_SIZE * fore, 0.6 * K_SIZE * med, 1.6 * K_SIZE * med)
        d = (w - e) / np.maximum(fore, 1e-6)[:, None]
        wrist[:, h] = w; size[:, h] = L; centre[:, h] = w + 0.42 * L[:, None] * d; valid[:, h] = fill & np.isfinite(L)
        print(f"{'LR'[h]}: valid {100 * valid[:, h].mean():.1f} % of {n} rows (measured {100 * ok.mean():.1f} %), hand length px median {np.nanmedian(L[valid[:, h]]):.0f}")
    np.savez(a.out, wrist=wrist.astype(np.float32), centre=centre.astype(np.float32), size=size.astype(np.float32), valid=valid)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
