#!/usr/bin/env python3
"""Rokoko arms bent onto the 2D evidence, Rokoko kept as reference.  Two steps: the phone first, then the arms.
Pipeline version: session2 (two steps) + a small clavicle offset in the arm step + hold of the last detected offsets.

    python arm2d_fit.py CLIP_DIR [CLIP_DIR ...] --out OUT [--wrist wilor|hand5]          (~/envs/handseg, CPU)

CLIP_DIR as hung5_detect.py (+ masks.npz from sam2_clipmask.py, + wilor_wrists.npz from wilor_wrists.py); Rokoko from
<session>/camera/rokoko_cam.npz.  Wrist target ("hand5 wrist" below): --wrist wilor (default) = wilor_wrists.npz, confident where
hand5 is confident and WiLoR agrees within RTM_AGREE px; --wrist hand5 = hand5 alone.
  1 phone  X' = C_k X + c_k for every joint: small change of the camera stage's (ARKit) camera (<= R_MAX, T_MAX, smooth) so the raw
           Rokoko arms lie in the SAM arm mask (seg), reach the SAM hands (reach) and the hand5 wrists; arms untouched
  2 arms   camera of step 1 fixed.  E' = S + a R_k (E - S),  W' = E' + b R_k D_k (W - E)   (S = UpperArm, E = LowerArm,
           W = Hand): R_k shoulder offset, <= SH_DET on frames with a hand5 wrist of that side, <= SH_HI where that wrist is
           also on a SAM hand blob (two detectors agree, within AGREE_PX: high confidence); frames without a hand5 wrist of that side HOLD the offsets (shoulder, clavicle,
           elbow) of the last frame where it was detected, until it is detected again (before the first detection: the
           first detection's; never detected in the clip: raw Rokoko); the arm step fits on detected frames only (SAM
           mask / reach act through the phone step);
           K_k clavicle about the Shoulder joint (moves S and the whole arm), <= CL_DET near a hand5 wrist easing to
           CL_NODET like the shoulder, prior SIG_CLP;  D_k elbow flex about the hinge <= EL_MAX; upper arm / forearm LENGTHS (not scales): one pair per session, the
           same for left and right, as arm length U + F in TOT_RANGE and ratio U / F in RATIO_RANGE (hard; the upper arm is
           never longer than the forearm), pulled weakly to Rokoko's own total / ratio (clipped into the ranges); each clip
           fits them on its representative frames (a confident wrist and a SAM arm mask; the RTM elbow is not trusted for
           lengths), the session takes the mean weighted by those frames, then every clip of the session is fitted with
           them fixed.  EgoSuite: arm 50-55 cm, ratio 0.99-1.05; Rokoko ~0.83.  Fingers (glove takes) move rigidly with the wrist
Residuals (robust soft-L1):
  wrist   hand5 wrist: SIG_W px if confident (pass 1 / 2 agree < P12 palm and RTM wrist within RTM_AGREE px), else SIG_W2
  elbow   RTM elbow (conf >= EL_CONF; hidden hand: >= EL_CONF_NODET and inside the picture) SIG_E px (loose: sleeves)
  seg     whole projected arm inside the SAM arm mask (masks.npz wearer: hand + arm + glove of the wearer): ARM_T samples
          along shoulder -> elbow -> wrist + palm, those inside the picture, mean distance outside / SIG_SEG x SEG_W (phone),
          x SEG_W_ARM on the arm step and only for a side without a hand5 wrist (the wrist leads where detected)
  reach   every hand blob (masks.npz hand, >= BLOB_MIN px at 1/4 res) has a projected wrist or palm (either side) near it
  prior   offsets / SIG_SHP, SIG_ELP; camera / SIG_R, SIG_T;  smooth: frame-to-frame change of every offset
Variants: noseg = the camera stage's camera, arms from the detections only; seg = phone step + arms with detections, seg and reach.
Holdout (seg worth check): wrist detections in every third 20-frame block hidden, error there.
Writes OUT/<clip>.mp4 (raw Rokoko | noseg | seg), OUT/<clip>.npz (fitted joints), OUT/summary.csv.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handseg_pilot import FF                                                    # noqa: E402

TORSO = ["hip", "spine", "chest", "neck", "head"]                                # drawing chains
ARM = ["neck", "{s}Shoulder", "{s}UpperArm", "{s}LowerArm", "{s}Hand"]
LEG = ["hip", "{s}UpLeg", "{s}Leg", "{s}Foot", "{s}Toe"]
COL = {"": (255, 255, 255), "left": (0, 140, 255), "right": (255, 120, 0)}

SIG_W, SIG_W2, SIG_E = 8.0, 25.0, 80.0                                        # elbow loose: trust wrist + mask
P12, RTM_AGREE, EL_CONF, SH_CONF = 0.08, 60.0, 0.5, 0.5
EL_CONF_NODET, RAMP = 0.7, 15                                               # hidden hand: strict elbow; cap 17 -> 10 deg over RAMP frames
SH_DET, SH_NODET, EL_MAX = np.radians(17), np.radians(10), np.radians(10)
SH_HI, AGREE_PX = np.radians(60), 60.0          # hand5 wrist on a SAM hand blob (both detectors agree): shoulder <= SH_HI
CL_DET, CL_NODET, SIG_CLP = np.radians(8), np.radians(3), np.radians(3)       # clavicle: the shoulder point rarely moves
SIG_SHP, SIG_ELP, SIG_R, SIG_T = np.radians(8), np.radians(3), np.radians(1), 0.015
R_MAX, T_MAX = np.radians(3), 0.05                                            # camera stays near the camera stage's
SIG_DSH, SIG_DEL, SIG_DR, SIG_DT = np.radians(1.5), np.radians(1), np.radians(0.3), 0.005
TOT_RANGE, RATIO_RANGE, SIG_TOT, SIG_RATIO = (0.46, 0.56), (0.80, 1.00), 0.02, 0.05   # arm = upper + fore (m), upper / fore
PALM = 0.08
SIG_COV, BLOB_MIN, BLOB_PTS, COV_OK = 15.0, 150, 64, 30.0
SEG_W_ARM = 5.0
ARM_T, SEG_W, SIG_SEG = np.r_[np.linspace(0.3, 1, 4), 1 + np.linspace(0.25, 1, 4)], 20.0, 5.0   # samples shoulder -> elbow (0-1) -> wrist (1-2)
SIDES = ("left", "right")
torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


def rod(v):
    """Rotation vectors (..., 3) -> matrices (..., 3, 3)."""
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = v / th
    K = torch.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = th.sin()[..., None], th.cos()[..., None]
    return torch.eye(3) + s * K + (1 - c) * K @ K


def capped(x, m):
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return x * (m * torch.tanh(n / m) / n)


def observations(cd, n):
    r = np.load(cd / "rtm_body.npz")
    kp, cf = np.full((n, 17, 2), np.nan), np.zeros((n, 17))
    keep = r["frame"] < n
    kp[r["frame"][keep]], cf[r["frame"][keep]] = r["kp"][keep], r["conf"][keep]
    w, conf = np.full((n, 2, 2), np.nan), np.zeros((n, 2), bool)
    if (cd / "hands.npz").exists():
        h = np.load(cd / "hands.npz")
        p1, p2 = h["pass"] == 1, h["pass"] == 2
        kp2d = h["kp2d"].reshape(len(h["pass"]), 21, 2)                            # no hand in the clip: saved as (0,)
        k1, k2, fr, hd = kp2d[p1], kp2d[p2], h["frame"][p2].astype(int), h["hand"][p2].astype(int)
        in_n = (fr >= 0) & (fr < n)
        k1, k2, fr, hd = k1[in_n], k2[in_n], fr[in_n], hd[in_n]
        w[fr, hd] = k2[:, 0]
        agree = np.linalg.norm(k1[:, 0] - k2[:, 0], axis=1) / np.linalg.norm(k2[:, 0] - k2[:, 9], axis=1).clip(1) < P12
        conf[fr, hd] = agree & (np.linalg.norm(kp[fr, 9 + hd] - k2[:, 0], axis=1) < RTM_AGREE)
    inside = (kp[:, 7:9] >= 0).all(-1) & (kp[:, 7:9, 0] < 1920) & (kp[:, 7:9, 1] < 1440)
    hidden = ~np.isfinite(w).all(-1)                                            # no hand5 wrist on that side
    ok = (cf[:, 7:9] >= EL_CONF) & (~hidden | ((cf[:, 7:9] >= EL_CONF_NODET) & inside))
    e = np.where(ok[..., None], kp[:, 7:9], np.nan)
    sh = np.where((cf[:, 5:7] >= SH_CONF)[..., None], kp[:, 5:7], np.nan)
    return w, conf, e, sh


def wrists(cd, n, src):
    """observations() with the wrist target of --wrist (falls back to wilor_wrists + RTMPose when hands.npz is absent)."""
    if src != "hand5" and not (cd / "hands.npz").exists():
        r = np.load(cd / "rtm_body.npz")
        kp, cf = np.full((n, 17, 2), np.nan), np.zeros((n, 17))
        keep = r["frame"] < n
        kp[r["frame"][keep]], cf[r["frame"][keep]] = r["kp"][keep], r["conf"][keep]
        V = np.full((n, 2, 2), np.nan)
        k = np.load(cd / "wilor_wrists.npz")["wrist"][:n]
        V[:len(k)] = k
        rtm_w = np.where((cf[:, 9:11] >= 0.3)[..., None], kp[:, 9:11], np.nan)
        rtm_ok = np.isfinite(rtm_w).all(-1)
        conf = np.isfinite(V).all(-1) & (~rtm_ok | (np.linalg.norm(V - rtm_w, axis=-1) < RTM_AGREE))
        inside = (kp[:, 7:9] >= 0).all(-1) & (kp[:, 7:9, 0] < 1920) & (kp[:, 7:9, 1] < 1440)
        ok = (cf[:, 7:9] >= EL_CONF) & (np.isfinite(V).all(-1) | ((cf[:, 7:9] >= EL_CONF_NODET) & inside))
        sh = np.where((cf[:, 5:7] >= SH_CONF)[..., None], kp[:, 5:7], np.nan)
        return V, conf, np.where(ok[..., None], kp[:, 7:9], np.nan), sh
    w, conf, e, sh = observations(cd, n)
    if src == "hand5":
        return w, conf, e, sh
    V, k = np.full((n, 2, 2), np.nan), np.load(cd / "wilor_wrists.npz")["wrist"][:n]
    V[:len(k)] = k
    conf = conf & (np.linalg.norm(V - w, axis=-1) < RTM_AGREE)
    r = np.load(cd / "rtm_body.npz")
    kp, cf, keep = np.full((n, 17, 2), np.nan), np.zeros((n, 17)), r["frame"] < n
    kp[r["frame"][keep]], cf[r["frame"][keep]] = r["kp"][keep], r["conf"][keep]
    inside = (kp[:, 7:9] >= 0).all(-1) & (kp[:, 7:9, 0] < 1920) & (kp[:, 7:9, 1] < 1440)
    ok = (cf[:, 7:9] >= EL_CONF) & (np.isfinite(V).all(-1) | ((cf[:, 7:9] >= EL_CONF_NODET) & inside))
    return V, conf, np.where(ok[..., None], kp[:, 7:9], np.nan), sh


class Proj:
    def __init__(self, z):
        self.R, self.t, self.K = (torch.tensor(np.asarray(z[k], float)) for k in ("Ruw", "tuw", "Ku"))

    def __call__(self, X):
        p = (X - self.t) @ self.R
        return p[..., :2] / p[..., 2:].clamp_min(0.05) * torch.diagonal(self.K)[:2] + self.K[:2, 2]


def held(det):
    """(n, 2) wrist seen -> (n, 2) index of the frame whose arm offsets each frame uses: itself when seen, else the last
    seen frame before it (the first seen one before the first detection; itself when never seen = no data, raw Rokoko)."""
    n = len(det)
    out = np.tile(np.arange(n)[:, None], (1, 2))
    for i in range(2):
        f = np.where(det[:, i])[0]
        if len(f):
            last = np.maximum.accumulate(np.where(det[:, i], np.arange(n), -1))
            out[:, i] = np.where(last >= 0, last, f[0])
    return out


def shoulder_cap(det, hi=SH_DET, lo=SH_NODET):
    """(n, 2) wrist seen -> cap (rad): hi on seen frames, easing to lo over RAMP frames away from the nearest seen frame
    of that side (shoulder; clavicle with CL_DET, CL_NODET)."""
    n = len(det)
    SH_DET, SH_NODET = hi, lo
    cap = np.full(det.shape, SH_NODET)
    for i in range(2):
        f = np.where(det[:, i])[0]
        if len(f):
            dist = np.abs(np.arange(n)[:, None] - f[None]).min(1)
            cap[:, i] = SH_NODET + (SH_DET - SH_NODET) * np.clip(1 - dist / RAMP, 0, 1)
    return cap


def fit(J, P, w, conf, e, mode, rep=None, lengths=None, cam=None, seg_dt=None, blobs=None, iters=1500, agree=None):
    """mode "cam": phone correction only (arms raw), from seg / reach; mode "arm": arm offsets on the fixed camera cam
    ((C, c) tensors, None = the camera stage's).  J: joint -> (n, 3) tensor; seg_dt (n, h, w) distance outside the arm mask (px);
    blobs from hand_blobs; rep: frames used (scale step) or None (all)."""
    n = len(w)
    K, S, E, W = ({s: J[f"{s}{j}"] for s in SIDES} for j in ("Shoulder", "UpperArm", "LowerArm", "Hand"))
    th = {k: torch.zeros(sz, requires_grad=True) for k, sz in
          (("q", (n, 2, 3)), ("k", (n, 2, 3)), ("d", (n, 2)), ("r", (n, 3)), ("t", (n, 3)), ("len", (2,)))}
    free = ["r", "t"] if mode == "cam" else ["q", "k", "d"] + (["len"] if lengths is None else [])
    lo, hi = torch.tensor([TOT_RANGE[0], RATIO_RANGE[0]]), torch.tensor([TOT_RANGE[1], RATIO_RANGE[1]])
    ru, rf = (torch.median(torch.cat([(X[s] - Y[s]).norm(dim=-1) for s in SIDES])) for X, Y in ((E, S), (W, E)))
    prior = torch.minimum(torch.maximum(torch.stack([ru + rf, ru / rf]), lo + 1e-3), hi - 1e-3)   # Rokoko's, clipped
    l0 = torch.logit((prior - lo) / (hi - lo))                                  # raw 0 = prior
    W_, C_, E_ = (torch.tensor(np.nan_to_num(x)) for x in (w, conf, e))
    hw, he = (torch.tensor(np.isfinite(x).all(-1)) for x in (w, e))
    use = torch.ones(n, dtype=torch.bool) if rep is None else torch.tensor(rep)
    shcap = shoulder_cap(np.isfinite(w).all(-1))
    if agree is not None:                                                       # hand5 + SAM agree: high confidence
        shcap = np.where(agree & np.isfinite(w).all(-1), SH_HI, shcap)
    shcap = torch.tensor(shcap)[..., None]                                      # (n, 2, 1)
    clcap = torch.tensor(shoulder_cap(np.isfinite(w).all(-1), CL_DET, CL_NODET))[..., None]
    src = torch.tensor(held(np.isfinite(w).all(-1)))                            # (n, 2) frame whose offsets each frame uses
    if seg_dt is not None:
        dt = torch.tensor(seg_dt)[:, None]                                      # (n, 1, h, w)
        has_mask = torch.tensor((seg_dt == 0).any((1, 2)))
    soft = lambda x: 2 * ((1 + x ** 2).sqrt() - 1)
    opt = torch.optim.Adam([th[k] for k in free], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)

    def forward():
        q, d = capped(th["q"] * SIG_SHP, shcap), EL_MAX * torch.tanh(th["d"] * SIG_ELP / EL_MAX)
        kc = capped(th["k"] * SIG_CLP, clcap)
        if mode == "arm":                                                       # hidden hand: hold the last detected offsets
            q, kc = (torch.stack([X[src[:, i], i] for i in range(2)], 1) for X in (q, kc))
            d = torch.stack([d[src[:, i], i] for i in range(2)], 1)
        tr = lo + (hi - lo) * torch.sigmoid(l0 + th["len"])                     # (total, ratio)
        L = torch.stack([tr[0] * tr[1] / (1 + tr[1]), tr[0] / (1 + tr[1])]) if lengths is None else lengths   # (U, F) m
        if mode == "cam":
            C, c = rod(capped(th["r"] * SIG_R, R_MAX)), capped(th["t"] * SIG_T, T_MAX)
        else:
            C, c = cam if cam is not None else (torch.eye(3).expand(n, 3, 3), torch.zeros(n, 3))
        out = {}
        for i, s in enumerate(SIDES):
            u, l = E[s] - S[s], W[s] - E[s]
            ax = torch.cross(u, l, dim=-1)
            ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            Rk = rod(kc[:, i])
            Rq = rod(q[:, i]) @ Rk
            s2 = S[s] if mode == "cam" else K[s] + (Rk @ (S[s] - K[s])[..., None])[..., 0]
            un, ln = u / u.norm(dim=-1, keepdim=True), l / l.norm(dim=-1, keepdim=True)
            if mode == "cam":                                                   # phone step: Rokoko's own arm
                e2, l2 = S[s] + u, l
            else:
                e2 = s2 + L[0] * (Rq @ un[..., None])[..., 0]
                l2 = L[1] * (Rq @ rod(ax * d[:, i:i + 1]) @ ln[..., None])[..., 0]
            w2 = e2 + l2
            for nm, X in (("S", s2), ("E", e2), ("W", w2), ("P", w2 + PALM * l2 / l2.norm(dim=-1, keepdim=True))):
                out[nm, i] = (C @ X[..., None])[..., 0] + c
        return out, q, d, L, C, c, kc

    for it in range(iters):
        out, q, d, L, C, c, kc = forward()
        loss = 0
        for i in range(2):
            pw, pe, ps = P(out["W", i]), P(out["E", i]), P(out["S", i])
            sig = torch.where(C_[:, i], SIG_W, SIG_W2)
            loss = loss + (soft((pw - W_[:, i]).norm(dim=-1) / sig) * (hw[:, i] & use)).sum()
            if mode == "arm":
                loss = loss + (soft((pe - E_[:, i]).norm(dim=-1) / SIG_E) * (he[:, i] & hw[:, i] & use)).sum()
            if seg_dt is not None:
                pp = torch.cat([arm_samples(ps, pe, pw), P(out["P", i])[:, None]], 1)              # (n, k, 2)
                g = torch.stack([pp[..., 0] / (dt.shape[-1] * 4) * 2 - 1, pp[..., 1] / (dt.shape[-2] * 4) * 2 - 1], -1)
                inside = (g.abs() < 1).all(-1)
                dd = torch.nn.functional.grid_sample(dt, g[:, None], align_corners=False, padding_mode="border")[:, 0, 0]
                per = (soft(dd / SIG_SEG) * inside).sum(1) / inside.sum(1).clamp_min(1)
                if mode == "cam":                                               # arm step: offsets come from detections
                    loss = loss + (per * SEG_W * (has_mask & use)).sum()
        if blobs is not None:                                                   # arm step: only where both hands are seen
            fw = use if mode == "cam" else use & hw.all(1)
            loss = loss + (soft(reach(blobs, [P(out[j, i]) for j in ("W", "P") for i in range(2)]) / SIG_COV) * blobs[1] * fw[:, None]).sum()
        if mode == "cam":
            loss = loss + (th["r"] ** 2).sum() / 2 + (th["t"] ** 2).sum() / 2
            loss = loss + ((th["r"][1:] - th["r"][:-1]) ** 2).sum() * (SIG_R / SIG_DR) ** 2 + ((th["t"][1:] - th["t"][:-1]) ** 2).sum() * (SIG_T / SIG_DT) ** 2
        else:
            pw_ = hw[..., None].double()                                        # prior only where the hand is seen:
            loss = loss + (q ** 2 * pw_).sum() / SIG_SHP ** 2 / 2 + (d ** 2 * pw_[..., 0]).sum() / SIG_ELP ** 2 / 2   # hidden = hold
            loss = loss + (kc ** 2 * pw_).sum() / SIG_CLP ** 2 / 2 + ((kc[1:] - kc[:-1]) ** 2).sum() / SIG_DSH ** 2
            loss = loss + ((q[1:] - q[:-1]) ** 2).sum() / SIG_DSH ** 2 + ((d[1:] - d[:-1]) ** 2).sum() / SIG_DEL ** 2
            if lengths is None:
                loss = loss + (((torch.stack([L[0] + L[1], L[0] / L[1]]) - prior) / torch.tensor([SIG_TOT, SIG_RATIO])) ** 2).sum() / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        out, q, d, L, C, c, kc = forward()
    return dict(out={k: v.cpu().numpy() for k, v in out.items()}, q=q.cpu().numpy(), kc=kc.cpu().numpy(), d=d.cpu().numpy(), len=L.detach(), cam=(C.detach(), c.detach()),
                C=C.cpu().numpy(), c=c.cpu().numpy())


def arm_samples(ps, pe, pw):
    """Points along the projected shoulder -> elbow -> wrist, (n, len(ARM_T), 2); works on tensors and arrays."""
    t = torch.tensor(ARM_T)[None, :, None] if torch.is_tensor(ps) else ARM_T[None, :, None]
    a, b = ps[:, None] + (pe - ps)[:, None] * t.clip(0, 1), pe[:, None] + (pw - pe)[:, None] * (t - 1).clip(0, 1)
    return a * (t <= 1) + b * (t > 1)


def hand_blobs(cd, n):
    """Hand / glove blobs per frame: points (n, B, BLOB_PTS, 2) full res, valid (n, B)."""
    m = np.load(cd / "masks.npz")["hand"][:n]
    per = []
    for x in m:
        k, lab, st, _ = cv2.connectedComponentsWithStats(x.astype(np.uint8))
        bl = []
        for b in range(1, k):
            if st[b, cv2.CC_STAT_AREA] >= BLOB_MIN:
                v, u = np.nonzero(lab == b)
                i = np.linspace(0, len(u) - 1, BLOB_PTS).astype(int)
                bl.append(np.stack([u[i], v[i]], 1) * 4 + 2.0)
        per.append(bl)
    B = max(1, max(map(len, per)))
    pts, ok = np.zeros((n, B, BLOB_PTS, 2)), np.zeros((n, B), bool)
    for f, bl in enumerate(per):
        for b, q in enumerate(bl):
            pts[f, b], ok[f, b] = q, True
    return torch.tensor(pts), torch.tensor(ok)


def reach(blobs, ends):
    """Distance (n, B) from each blob to the nearest projected end point (list of (n, 2))."""
    E = torch.stack(ends, 1)                                                    # (n, k, 2)
    d = (blobs[0][:, :, :, None] - E[:, None, None]).norm(dim=-1)              # (n, B, P, k)
    return d.flatten(2).min(-1).values


def covered(blobs, ends):
    with torch.no_grad():
        d = reach(blobs, [torch.tensor(np.nan_to_num(x, nan=1e6)) for x in ends])
    return (d[blobs[1]] < COV_OK).double().mean().item()


def in_mask(pts, m):
    """Share of the in-picture points inside the (5 px dilated, 1/4 res) mask, frames with a mask."""
    hit, tot = 0, 0
    for f in range(len(m)):
        if not m[f].any():
            continue
        md = cv2.dilate(m[f].astype(np.uint8), np.ones((3, 3)))
        q = pts[f].reshape(-1, 2)
        q = (q[np.isfinite(q).all(1)].clip(-1e5, 1e5) / 4).astype(int)
        q = q[(q[:, 0] >= 0) & (q[:, 0] < m.shape[2]) & (q[:, 1] >= 0) & (q[:, 1] < m.shape[1])]
        hit, tot = hit + md[q[:, 1], q[:, 0]].sum(), tot + len(q)
    return hit / max(tot, 1)


def sam_agree(cd, w):
    """(n, 2) hand5 wrist lies on (within AGREE_PX of) a SAM hand blob of the wearer: both detectors see the hand."""
    m = np.load(cd / "masks.npz")["hand"][:len(w)]
    out = np.zeros(w.shape[:2], bool)
    for k in range(len(w)):
        if not m[k].any():
            continue
        dt = cv2.distanceTransform((~m[k]).astype(np.uint8), cv2.DIST_L2, 3) * 4
        for i in range(2):
            if np.isfinite(w[k, i]).all():
                u, v = (w[k, i] / 4).astype(int)
                out[k, i] = 0 <= u < dt.shape[1] and 0 <= v < dt.shape[0] and dt[v, u] <= AGREE_PX
    return out


def distance_outside(cd, n):
    m = np.load(cd / "masks.npz")["wearer"][:n]
    return np.stack([cv2.distanceTransform((~x).astype(np.uint8), cv2.DIST_L2, 3) * 4 if x.any() else np.zeros(x.shape, np.float32)
                     for x in m]).astype(float), m


def skeleton(z, J, f, Pnp, res=None):
    """All joints (n, 3) projected, arms from res when given (camera correction applied to every joint)."""
    out = {}
    for j, X in J.items():
        X = X.cpu().numpy()
        if res is not None:
            X = np.einsum("nij,nj->ni", res["C"], X) + res["c"]
        out[j] = X
    if res is not None:
        for i, s in enumerate(SIDES):
            out[f"{s}UpperArm"], out[f"{s}LowerArm"], out[f"{s}Hand"] = res["out"]["S", i], res["out"]["E", i], res["out"]["W", i]
            fingers = [f"{s}{d}{p}" for d in ("Thumb", "Index", "Middle", "Ring", "Little")
                       for p in ("Proximal", "Medial", "Distal", "Tip") if f"{s}{d}{p}" in J]
            if fingers:                                                         # rigid with the wrist: same rotation as the forearm
                S_, E_, W_ = (J[f"{s}{j}"].cpu().numpy() for j in ("UpperArm", "LowerArm", "Hand"))
                ax = np.cross(E_ - S_, W_ - E_)
                ax /= np.linalg.norm(ax, axis=1, keepdims=True).clip(1e-6)
                Rt = Rotation.from_rotvec(res["q"][:, i]).as_matrix() @ Rotation.from_rotvec(res["kc"][:, i]).as_matrix() \
                    @ Rotation.from_rotvec(ax * res["d"][:, i:i + 1]).as_matrix()
                w_new = np.einsum("nji,nj->ni", res["C"], res["out"]["W", i] - res["c"])          # camera correction undone
                for j in fingers:
                    x = w_new + np.einsum("nij,nj->ni", Rt, J[j].cpu().numpy() - W_)
                    out[j] = np.einsum("nij,nj->ni", res["C"], x) + res["c"]
    return {j: Pnp(X) for j, X in out.items()}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wrist", choices=["wilor", "hand5"], default="wilor")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, D = [], {}
    for cd in a.clips:                                                          # pass 1: phone step + lengths per clip
        sd = Path((cd / "session.txt").read_text().strip())
        f0 = int((cd / "f0.txt").read_text())
        z = np.load(sd / "camera/rokoko_cam.npz")
        cap = cv2.VideoCapture(str(cd / "clip.mp4"))
        Hc = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1440
        Wc = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        has_masks = (cd / "masks.npz").exists()
        if has_masks:
            mz = np.load(cd / "masks.npz")
            n = min(nf, len(mz["hand"]))                                             # header can count a last packet flagged discard
            J = {j: torch.tensor(z[j][f0:f0 + n].astype(float)) for j in z.files if j not in ("Ku", "Ruw", "tuw")}
            P = Proj(z)
            w, conf, e, _ = wrists(cd, n, a.wrist)
            dt, hm = distance_outside(cd, n)
            bl = hand_blobs(cd, n)
            agree = sam_agree(cd, w)
            hand_any = mz["hand"][:n].any((1, 2))
            rep = conf.any(1) & (hm.any((1, 2)) if hm.any() else (hand_any if hand_any.any() else True))
        else:
            n = nf
            J = {j: torch.tensor(z[j][f0:f0 + n].astype(float)) for j in z.files if j not in ("Ku", "Ruw", "tuw")}
            P = Proj(z)
            w, conf, e, _ = wrists(cd, n, a.wrist)
            dt = np.zeros((n, Hc // 4, Wc // 4), float)
            hm = np.zeros((n, Hc // 4, Wc // 4), bool)
            bl = (torch.zeros((n, 1, BLOB_PTS, 2)), torch.zeros((n, 1), dtype=torch.bool))
            agree = np.zeros(conf.shape, bool)
            rep = conf.any(1)
        cam_seg = fit(J, P, w, conf, e, "cam", seg_dt=dt if hm.any() else None, blobs=bl if bl[1].any() else None)["cam"]  # 1 phone onto segmentation
        L = fit(J, P, w, conf, e, "arm", rep=rep, cam=cam_seg, seg_dt=dt if hm.any() else None, blobs=bl if bl[1].any() else None, agree=agree)["len"]
        D[cd] = dict(sd=sd, f0=f0, z=z, n=n, J=J, P=P, w=w, conf=conf, e=e, dt=dt, hm=hm, bl=bl, rep=rep, cam=cam_seg, L=L, agree=agree)
        print(cd.name, "clip lengths (cm)", (100 * L).cpu().numpy().round(1), "from", int(rep.sum()), "frames", flush=True)
    for sd in {v["sd"] for v in D.values()}:                                    # one pair of lengths per session
        V = [v for v in D.values() if v["sd"] == sd]
        wt = torch.tensor([max(int(v["rep"].sum()), 1) for v in V], dtype=torch.float64)
        Ls = (torch.stack([v["L"] for v in V]) * wt[:, None]).sum(0) / wt.sum()
        for v in V:
            v["Ls"] = Ls
        print(sd.name[8:23], "session lengths (cm)", (100 * Ls).cpu().numpy().round(1), flush=True)
    for cd in a.clips:                                                          # pass 2: arms with the session lengths
        v = D[cd]
        sd, f0, z, n, J, P, w, conf, e, dt, hm, bl, rep, cam_seg = (v[k] for k in ("sd", "f0", "z", "n", "J", "P", "w", "conf", "e", "dt",
                                                                                  "hm", "bl", "rep", "cam"))
        cap = cv2.VideoCapture(str(cd / "clip.mp4"))
        Pnp = lambda X, P=P: P(torch.tensor(X)).cpu().numpy()
        name = f"{sd.name[8:23]}_{f0}"
        hide = ((np.arange(n) // 20) % 3 == 1)[:, None] & np.isfinite(w).all(-1)
        wh, ch = np.where(hide[..., None], np.nan, w), conf & ~hide
        res = {}
        for seg in (False, True):
            kw = dict(cam=cam_seg, seg_dt=dt if hm.any() else None, blobs=bl if bl[1].any() else None) if seg else {}
            res[seg] = fit(J, P, w, conf, e, "arm", lengths=v["Ls"], agree=v["agree"], **kw)
            ho = fit(J, P, wh, ch, e, "arm", lengths=v["Ls"], agree=v["agree"] & ~hide, **kw)
            pw = np.stack([Pnp(ho["out"]["W", i]) for i in range(2)], 1)
            res[seg, "ho"] = np.linalg.norm(pw - w, axis=-1)[hide & conf]
        raw_w = np.stack([Pnp(J[f"{s}Hand"].cpu().numpy()) for s in SIDES], 1)
        raw_e = np.stack([Pnp(J[f"{s}LowerArm"].cpu().numpy()) for s in SIDES], 1)
        raw_s = np.stack([Pnp(J[f"{s}UpperArm"].cpu().numpy()) for s in SIDES], 1)
        row = dict(clip=name, agree_frames=int(v["agree"].sum()), upper_cm=100 * v["Ls"][0].item(), fore_cm=100 * v["Ls"][1].item(), rep_frames=int(rep.sum()), conf_wrist_frames=int(conf.sum()), wrist_frames=int(np.isfinite(w).all(-1).sum()),
                   arm_mask_frames=int(hm.any((1, 2)).sum()),
                   hand_reached_raw=covered(bl, [raw_w[:, 0], raw_w[:, 1]]),
                   arm_in_mask_raw=in_mask(np.stack([arm_samples(raw_s[:, i], raw_e[:, i], raw_w[:, i]) for i in range(2)], 1), hm),
                   wrist_px_raw=np.nanmedian(np.linalg.norm(raw_w - w, axis=-1)[conf]),
                   elbow_px_raw=np.nanmedian(np.linalg.norm(raw_e - e, axis=-1)),
                   heldout_px_raw=np.median(np.linalg.norm(raw_w - w, axis=-1)[hide & conf]))
        for seg in (False, True):
            r, k = res[seg], "seg" if seg else "noseg"
            pw = np.stack([Pnp(r["out"]["W", i]) for i in range(2)], 1)
            pe = np.stack([Pnp(r["out"]["E", i]) for i in range(2)], 1)
            ps = np.stack([Pnp(r["out"]["S", i]) for i in range(2)], 1)
            row |= {f"hand_reached_{k}": covered(bl, [pw[:, 0], pw[:, 1], Pnp(r["out"]["P", 0]), Pnp(r["out"]["P", 1])]),
                    f"arm_in_mask_{k}": in_mask(np.stack([arm_samples(ps[:, i], pe[:, i], pw[:, i]) for i in range(2)], 1), hm),
                    f"wrist_px_{k}": np.nanmedian(np.linalg.norm(pw - w, axis=-1)[conf]),
                    f"elbow_px_{k}": np.nanmedian(np.linalg.norm(pe - e, axis=-1)),
                    f"heldout_px_{k}": np.median(res[seg, "ho"]),
                    f"sh_deg_med_{k}": np.degrees(np.median(np.linalg.norm(r["q"], axis=-1))),
                    f"sh_deg_max_det_{k}": np.degrees(np.linalg.norm(r["q"], axis=-1)[np.isfinite(w).all(-1)].max(initial=0)),
                    f"sh_deg_max_nodet_{k}": np.degrees(np.linalg.norm(r["q"], axis=-1)[~np.isfinite(w).all(-1)].max(initial=0)),
                    f"el_deg_max_{k}": np.degrees(np.abs(r["d"]).max()),
                    f"cl_deg_med_{k}": np.degrees(np.median(np.linalg.norm(r["kc"], axis=-1))),
                    f"cl_deg_max_{k}": np.degrees(np.linalg.norm(r["kc"], axis=-1).max()),
                    f"sh_point_cm_max_{k}": 100 * np.linalg.norm(np.stack([r["out"]["S", i] for i in range(2)], 1)
                                                               - np.einsum("nij,nsj->nsi", r["C"], np.stack([J[f"{s}UpperArm"].cpu().numpy() for s in SIDES], 1)) - r["c"][:, None], axis=-1).max(),
                    f"cam_deg_max_{k}": np.degrees(np.arccos(np.clip((np.trace(r["C"], axis1=1, axis2=2) - 1) / 2, -1, 1)).max()),
                    f"cam_cm_max_{k}": 100 * np.linalg.norm(r["c"], axis=1).max()}
        rows.append(row)
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        sk = [skeleton(z, J, f0, Pnp)[0]] + [skeleton(z, J, f0, Pnp, res[s])[0] for s in (False, True)]
        np.savez_compressed(a.out / f"{name}.npz", f0=f0, lengths=v["Ls"].cpu().numpy(),
                            det=np.isfinite(w).all(-1), **{f"{k}_{p}": res[s][p] for k, s in (("noseg", False), ("seg", True)) for p in ("q", "kc", "d", "C", "c")},
                            **{f"{k}_{j}": v for k, s in (("noseg", False), ("seg", True)) for j, v in skeleton(z, J, f0, Pnp, res[s])[1].items()})
        draw(cap, n, sk, w, conf, e, hm, name, f0, a.out / f"{name}.mp4")
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "summary.csv", index=False)
    print(df.round(2).T.to_string())


def draw(cap, n, sk, w, conf, e, hm, name, f0, path):
    chains = [("", TORSO)] + [(s, c) for s in SIDES for c in (ARM, LEG)]
    labels = ["raw Rokoko", "fit, no seg", "fit + SAM arm mask"]
    enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x480", "-r", "30", "-i", "-",
                            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for k in range(n):
        ok, im = cap.read()
        if not ok:
            break
        panels = []
        for p, (S, lab) in enumerate(zip(sk, labels)):
            A = im.copy()
            if p == 2 and hm[k].any():
                m = cv2.resize(hm[k].astype(np.uint8), A.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
                A[m] = (0.5 * A[m] + (0, 127, 0)).astype(np.uint8)
            for s, c in chains:
                pts = [S.get(j.format(s=s)) for j in c]
                for u, v in zip(pts[:-1], pts[1:]):
                    if u is not None and v is not None and np.isfinite(np.r_[u[k], v[k]]).all() and np.abs(np.r_[u[k], v[k]]).max() < 1e5:
                        cv2.line(A, tuple(int(x) for x in u[k]), tuple(int(x) for x in v[k]), COL[s], 8, cv2.LINE_AA)
            for i in range(2):
                if np.isfinite(e[k, i]).all():
                    cv2.drawMarker(A, tuple(int(x) for x in e[k, i]), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 40, 6)
                if np.isfinite(w[k, i]).all():
                    cv2.circle(A, tuple(int(x) for x in w[k, i]), 14, (0, 255, 255) if conf[k, i] else (0, 160, 160), 4 if not conf[k, i] else -1)
            A = cv2.resize(A, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.rectangle(A, (0, 0), (640, 22), (0, 0, 0), -1)
            cv2.putText(A, f"{lab}  f{f0 + k}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(A)
        fr = np.hstack(panels)
        cv2.putText(fr, f"{name}  yellow dot wrist target (filled = confident)  cyan x RTM elbow  green = SAM arm", (6, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        enc.stdin.write(fr.tobytes())
    enc.stdin.close(); enc.wait()


if __name__ == "__main__":
    main()
