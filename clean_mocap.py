#!/usr/bin/env python3
"""Remove spikes from synced Rokoko mocap and resample it onto video frames.

    python clean_mocap.py PATH [PATH ...] [--force] [--rokoko-only]

--rokoko-only: a dir with a Rokoko CSV named *_sync.csv and NO *_perframe.csv (a take without video): its own rows
are the frames (frame = row, video_time = time from the first row, match_error_ms = 0); everything else is the same.

PATH is a session dir or any parent dir; every dir below it holding *_sync.csv +
*_perframe.csv (fit_rokoko.py --per-frame output) is cleaned. Writes
*_perframe_clean.csv: same columns, plus
  repaired  number of joints that were spike-repaired in this frame
  gap_ms    length of the mocap gap this frame was interpolated across (time between
            the two real, non-repeated samples around it; ~33 normally)
  valid     0 inside a stream freeze (no new sample for > FREEZE_S), an implausible posture or a sub-session shorter than
            MIN_SEG_S: values there are a straight-line guess, not measurements - drop these frames;
            video >= ROKOKO_DRIVEN_FPS: also 0 on the frames no Rokoko sample was paired with (see below)
  pairing   video slower than ROKOKO_DRIVEN_FPS (all our takes, 30 fps): every video frame gets Rokoko
            interpolated at its exact time.  Faster video (more frames than Rokoko's ~30 Hz samples):
            each Rokoko sample takes its nearest video frame (within PAIR_MAX_S) and that frame gets
            the sample's own (cleaned) value, no interpolated in-between poses; the other frames are
            valid = 0.  Freeze split, repair, smoothing, raw bone lengths: the same in both
  segment   sub-session id (0, 1, ...; -1 = dropped): a freeze, or a held posture no person has (hips twisted
            > 60 deg against the shoulders, or seated with feet > 1 m apart / a thigh pointing backward: suit drift, see implausible()), cuts the session, nothing is filled or
            smoothed across a cut; <session>/segments.csv lists them (frame range, duration)

Noise in the raw stream and the fix:
  repeated packets (same row again ~1 sample later, then a catch-up jump)
      -> drop repeats; interpolate to each frame's exact time, not nearest row
  short glitches (body teleports, legs fold up, feet fly and snap back), detectors below; a sample right
  after a stall gets the stall's time (within CATCHUP_S) as credit: the stream catches up in a burst of
  ~33 ms steps carrying the motion it missed, real motion, not a jump:
      a) reachability: a sample further from the last good sample than VLIM / ROTLIM
         allows in the elapsed time is bad; run forward and backward, and a bad run
         longer than MAX_RUN_S is a real level change, not a spike, so it is kept.
         Works inside fast periodic motion, where a median window is blind.
      b) Hampel (rolling median over WIN_S) gated by speed: an outlier counts only
         within GATE_S of an impossible jump; catches the slow ramp-up part of a
         glitch that (a) accepts. Real fast swings slow down at their peak and
         never jump, so neither detector touches them.
      c) snap-back: a ramp-hold-snap glitch (feet drift up 30 cm at a normal
         speed, hold, then drop back in ONE frame) has a plausible ramp, so (a)/(b)
         only catch the snap and the refill keeps the ramp as a bump. A snap is a
         one-frame step > SNAP_V that is > SNAP_RATIO x both neighbouring steps,
         found on the stream before repeats are dropped (a stall during the hold
         must not stretch the snap's time step and hide it; a real dropout must)
         (real motion speeds up and slows down; it never steps once). From the
         snap, walk back (and forward) to where the joint was last at the level it
         snaps to, only while the distance shrinks steadily (a real swing turns
         around, a glitch ramp does not); the whole excursion between is bad. The
         refill (below) then runs
         through the good samples on both sides: flat when the body was still,
         following the motion when it was moving.
      d) legs fold up: both feet more than FOLD_Z above the floor while the hip neither rises nor drops (a jump moves
         the hip; standing, both feet cannot leave the floor otherwise), the person upright (hip >= FOLD_STAND x the
         standing hip height: sitting with the feet up is real), for FOLD_MIN_S..FOLD_MAX_S.  Extended both ways while
         the feet are still off the floor (> FOLD_EDGE), both whole legs flagged.  Speeds stay plausible, so (a) - (c)
         miss it.
      A flagged joint flags every joint below it in the skeleton on the same samples
      (knee -> ankle -> toe ...), and a flagged arm / leg joint flags its whole limb (upper arm ->
      hand, thigh -> toe), so a limb is repaired as a whole and keeps its shape.
      Flagged samples are refilled (direction / root position / rotation) through the good samples
      of the same segment: cubic spline for holes
      up to SPLINE_MAX_S (a straight line would cut the top off a fast swing under a
      glitch), PCHIP for longer holes (cubic overshoots across long holes).
  finger joints (Smartgloves) flex much faster and noisier than the body
      -> same detectors with looser limits (FINGER_*), so only real glitches go
  quaternion sign flips / 150-180 deg one-frame rotations
      -> sign continuity, same gated Hampel on components, renormalise
  residual jitter -> Savitzky-Golay over SMOOTH_S after resampling
  bone lengths -> NEVER changed: a joint is stored as its raw bone length + direction from its parent
      (roots: position); glitch refill, resampling and smoothing touch only directions, roots and
      rotations, then positions are rebuilt from the root with the raw lengths (interpolated between
      the raw samples, so a rigid Rokoko bone keeps its exact length)
  joints that never move (Rokoko ToeEnd = constant 0) -> left untouched, skipped

ponytail: glitches with no single-frame snap that last longer than ~WIN_S/2, and
excursions longer than SNAP_MAX_S, are not removed; permanent steps are only softened.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import re

from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import savgol_filter

WIN_S = 0.5         # Hampel window
K = 4.0             # outlier threshold in robust sigmas
MIN_POS = 0.03      # m: never flag position deviations below this
MIN_QUAT = 0.05     # quaternion component units (~6 deg)
VLIM = 7.0          # m/s: faster jumps are not human motion
ROTLIM = 1200.0     # deg/s (40 deg per 33 ms)
GATE_S = 0.17       # a Hampel outlier counts only this close to an impossible jump
SMOOTH_S = 0.23     # Savitzky-Golay window (7 frames at 30 fps)
FREEZE_S = 0.5      # no new mocap sample for longer than this = stream freeze: cut, the session splits there
MIN_SEG_S = 3.0     # sub-sessions shorter than this are dropped (valid = 0)
ROKOKO_DRIVEN_FPS = 31.0   # video at least this fast: more frames than Rokoko samples -> pairing driven by Rokoko
PAIR_MAX_S = 0.025         # ... each sample takes its nearest video frame within this (as the camera stage's 1:1 map)
CATCHUP_S = 0.5     # stall time in this window before a sample counts as time it could move (late catch-up burst)
SPLINE_MAX_S = 0.4  # cubic spline up to this (MAX_RUN_S + edge samples), PCHIP beyond
FINGER = re.compile(r"Thumb|Index|Middle|Ring|Little")
FINGER_VLIM, FINGER_ROTLIM, FINGER_MIN_QUAT = 12.0, 3000.0, 0.15
MAX_RUN_S = 0.3     # reachability: longer unreachable runs are real level changes
TOL_POS, TOL_DEG = 0.02, 5.0    # slack on reachability for sensor jitter
SNAP_V, SNAP_W = 3.0, 600.0     # snap: one-frame step faster than this (m/s, deg/s) ...
SNAP_RATIO = 3.0                # ... and this many times both neighbouring steps
SNAP_MAX_S = 1.0                # longest excursion a snap may close
SNAP_TOL = 0.25                 # "back at the snap level" = within this x snap size
FOLD_Z, FOLD_EDGE = 0.06, 0.01  # m: both toes above the floor = folded (real glitches lift ~10 cm); extended while above FOLD_EDGE
FOLD_HIP = 0.03                 # m: hip height change allowed over the fold (a jump moves it more)
FOLD_STAND = 0.85               # upright: hip >= this x standing hip height (90th percentile)
FOLD_MIN_S, FOLD_MAX_S = 0.15, 1.5
POSE_WIN_S = 10.0               # implausible(): medians over this window = a held posture, not a quick move
TWIST_HI, TWIST_LO = 60.0, 45.0 # deg, hip line vs shoulder line about the trunk: starts above HI, extends while above LO
SPLIT_M, SEATED_M = 1.0, 0.55   # m: feet further apart than SPLIT_M while the hip is below SEATED_M (seated)
KNEE_BACK_M = -0.05             # m: a knee this far BEHIND its hip joint while seated (kneeling is ~0)



def time_shift(raw, pf, cols):
    """*_perframe.csv keeps the ORIGINAL CSV time, *_sync.csv is rebased by fit_rokoko (0 at the first video frame):
    they differ by a constant whenever the CSV head was trimmed (Rokoko started before the video).  Measured from the
    rows both files carry with identical values (perframe copies raw rows); 0 with --keep-time."""
    m = pf[cols + ["time"]].drop_duplicates(cols, keep=False).merge(raw[cols + ["time"]].drop_duplicates(cols, keep=False), on=cols)
    if len(m) < 10:
        raise ValueError(f"cannot align *_perframe.csv to *_sync.csv time ({len(m)} common rows)")
    d = (m["time_y"] - m["time_x"]).to_numpy()
    assert np.ptp(np.percentile(d, [10, 90])) < 1e-3, "perframe / sync time offset is not a constant"
    return float(np.median(d))

def odd(n: float, lo: int = 3) -> int:
    n = max(lo, int(round(n)))
    return n if n % 2 else n + 1


def hampel(x: np.ndarray, win: int, floor) -> np.ndarray:
    """Boolean outlier mask per element of a (T, C) array."""
    df = pd.DataFrame(x)
    med = df.rolling(win, center=True, min_periods=1).median()
    mad = (df - med).abs().rolling(win, center=True, min_periods=1).median()
    return ((df - med).abs() > np.maximum(K * 1.4826 * mad, floor)).to_numpy()


def ang_speed(Q: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """(T, J, 4) quaternions -> (T-1, J) deg/s."""
    return 2 * np.degrees(np.arccos(np.clip(np.abs((Q[1:] * Q[:-1]).sum(2)), 0, 1))) / dt


def stall_credit(t: np.ndarray) -> np.ndarray:
    """Per sample: stall time (steps beyond 1.5 x the normal one) within CATCHUP_S before it.  After a stall the stream
    delivers the motion it missed in a burst of ~33 ms steps: that motion had the stall's time to happen."""
    ex = np.r_[0, np.maximum(np.diff(t) - 1.5 * np.median(np.diff(t)), 0)]
    cs = np.cumsum(ex)
    j = np.searchsorted(t, t - CATCHUP_S, side="right")             # first sample inside the window
    return cs - np.where(j > 0, cs[np.maximum(j - 1, 0)], 0)


def unreachable(t: np.ndarray, X: np.ndarray, dist, vmax: float, tol: float) -> np.ndarray:
    """(T, J) mask of samples farther from the last good sample than vmax*(dt + stall credit) + tol.
    X is (T, J, C); dist(a, b) -> (J,) distance. Runs longer than MAX_RUN_S are unflagged."""
    T, J = X.shape[:2]
    credit = stall_credit(t)
    bad = np.zeros((T, J), bool)
    g = np.zeros(J, int)                          # index of last good sample per joint
    run = np.full(J, -1)
    for i in range(1, T):
        b = dist(X[i], X[g, np.arange(J)]) > vmax * (t[i] - t[g] + credit[i]) + tol
        bad[i] = b
        start = b & (run < 0)
        run[start] = i
        long = b & (t[i] - t[np.maximum(run, 0)] > MAX_RUN_S)
        for k in np.where(long)[0]:               # level change: accept the new level
            bad[run[k]:i + 1, k] = False
        ok = ~b | long
        g[ok], run[ok] = i, -1
    return bad


def snap_excursions(t: np.ndarray, X: np.ndarray, dist, vmin: float, floor: float) -> np.ndarray:
    """(T, J) mask of ramp-hold-snap excursions. X is (T, J, C)."""
    T, J = X.shape[:2]
    bad = np.zeros((T, J), bool)
    if T < 4:
        return bad
    dt = np.maximum(np.diff(t), 1e-3)
    step = np.stack([dist(X[1:, k], X[:-1, k]) for k in range(J)], 1)      # (T-1, J): step i -> i+1
    v = step / dt[:, None]
    nb = np.maximum(np.r_[v[:1] * 0, v[:-1]], np.r_[v[1:], v[-1:] * 0])     # neighbouring steps
    maxn = max(2, int(SNAP_MAX_S / np.median(dt)))
    for i, k in zip(*np.where((v > vmin) & (v > SNAP_RATIO * nb))):
        s = i + 1                                       # first sample after the snap
        tol = max(floor, SNAP_TOL * step[i, k])
        jit = max(floor, 0.1 * step[i, k])
        seg = None
        for rng, ref, side in ((range(s - 1, max(-1, s - 1 - maxn), -1), X[s, k], "back"),
                               (range(s, min(T, s + maxn)), X[s - 1, k], "fwd")):
            prev = np.inf
            for a in rng:            # walk away from the snap toward the level it snaps to
                d = dist(X[a, k][None], ref[None])[0]
                if d > prev + jit:   # turned around: oscillating real motion, not a ramp
                    break
                if d < tol:
                    cand = (a + 1, s) if side == "back" else (s, a)
                    if cand[1] > cand[0] and (seg is None or cand[1] - cand[0] < seg[1] - seg[0]):
                        seg = cand
                    break
                prev = min(prev, d)
        if seg:
            bad[seg[0]:seg[1], k] = True
    return bad


def legs_folded(t: np.ndarray, X: np.ndarray, js: list) -> np.ndarray:
    """(T, J) mask of (d) legs-fold-up runs; X (T, J, 3) Rokoko positions, y up."""
    bad = np.zeros(X.shape[:2], bool)
    k = {j: i for i, j in enumerate(js)}
    if not all(n in k for n in ("hip", "leftToe", "rightToe", "leftUpLeg", "rightUpLeg")):
        return bad
    toe = np.minimum(X[:, k["leftToe"], 1], X[:, k["rightToe"], 1])
    hip = X[:, k["hip"], 1]
    win = max(3, int(30.0 / max(float(np.median(np.diff(t))), 1e-3)))     # floor: low toe level over ~30 s
    floor = pd.Series(toe).rolling(win, center=True, min_periods=1).quantile(0.02).to_numpy()
    lift = toe - floor
    stand = np.percentile(hip, 90)
    up = (lift > FOLD_Z) & (hip >= FOLD_STAND * stand)
    edges = np.flatnonzero(np.diff(np.r_[0, up.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        while a > 0 and lift[a - 1] > FOLD_EDGE:
            a -= 1
        while b < len(t) and lift[b] > FOLD_EDGE:
            b += 1
        dur = t[min(b, len(t) - 1)] - t[a]
        h = hip[max(a - 1, 0):min(b + 1, len(t))]
        if FOLD_MIN_S <= dur <= FOLD_MAX_S and h.max() - h.min() <= FOLD_HIP:
            bad[a:b, [k["leftUpLeg"], k["rightUpLeg"]]] = True                   # propagate_limb takes the whole legs
    return bad


def implausible(X: np.ndarray, js: list, seg: np.ndarray, fps: float) -> np.ndarray:
    """(T,) frames whose HELD posture no person has: the suit drifted (slow, nothing jumps, so no glitch detector sees it and
    nothing can be refilled): hips turned against the shoulders, or, while seated, the feet far apart or a thigh pointing backward.  X (T, J, 3) y up;
    medians inside each segment."""
    k = {j: i for i, j in enumerate(js)}
    need = ("hip", "neck", "leftUpLeg", "rightUpLeg", "leftLeg", "rightLeg", "leftUpperArm", "rightUpperArm", "leftFoot", "rightFoot")
    bad = np.zeros(len(X), bool)
    if not all(n in k for n in need):
        return bad
    p = {n: X[:, k[n]] for n in need}
    ax = p["neck"] - p["hip"]
    ax = ax / np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-9)
    flat = lambda v: v - (v * ax).sum(1, keepdims=True) * ax
    u, w = flat(p["leftUpLeg"] - p["rightUpLeg"]), flat(p["leftUpperArm"] - p["rightUpperArm"])
    c = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-9)
    twist = np.degrees(np.arccos(np.clip(c, -1, 1)))
    split = np.linalg.norm(p["leftFoot"] - p["rightFoot"], axis=1)
    lat = (p["leftUpLeg"] - p["rightUpLeg"]) * [-1, 0, 1]                      # x mirrored: Rokoko is left-handed
    fwd = np.cross(lat / np.maximum(np.linalg.norm(lat, axis=1, keepdims=True), 1e-9), [0, 1, 0]) * [-1, 1, 1]
    knee = {s: ((p[f"{s}Leg"] - p[f"{s}UpLeg"]) * fwd).sum(1) for s in ("left", "right")}
    win = odd(POSE_WIN_S * fps)
    for g in np.unique(seg[seg >= 0]):
        f = np.where(seg == g)[0]
        med = lambda x: pd.Series(x[f]).rolling(win, center=True, min_periods=1).median().to_numpy()
        t, hi = med(twist), med(p["hip"][:, 1])
        run = t > TWIST_LO
        e = np.flatnonzero(np.diff(np.r_[0, run.astype(int), 0]))
        for a, b in zip(e[::2], e[1::2]):
            if (t[a:b] > TWIST_HI).any():
                bad[f[a:b]] = True
        bad[f[((med(split) > SPLIT_M) | (np.minimum(med(knee["left"]), med(knee["right"])) < KNEE_BACK_M)) & (hi < SEATED_M)]] = True
    return bad


def parent_of(j: str, joints) -> str | None:
    """Rokoko skeleton parent (hip -> spine -> chest -> neck -> head; chest -> shoulder -> upper/lower arm ->
    hand -> finger proximal -> medial -> distal -> tip; hip -> up leg -> leg -> foot -> toe -> toe end)."""
    side = "left" if j.startswith("left") else "right" if j.startswith("right") else ""
    tail = j[len(side):]
    chain = {"spine": "hip", "chest": "spine", "neck": "chest", "head": "neck",
             "Shoulder": "chest", "UpperArm": side + "Shoulder", "LowerArm": side + "UpperArm",
             "Hand": side + "LowerArm", "UpLeg": "hip", "Leg": side + "UpLeg", "Foot": side + "Leg",
             "Toe": side + "Foot", "ToeEnd": side + "Toe"}
    p = chain.get(tail)
    if p is None and FINGER.search(tail):
        for a, b in (("Medial", "Proximal"), ("Distal", "Medial"), ("Tip", "Distal")):
            if tail.endswith(a):
                p = side + tail[:-len(a)] + b
        p = p or side + "Hand"                                  # *Proximal
    return p if p in joints else None


def propagate_limb(bad: np.ndarray, js: list) -> np.ndarray:
    """A flagged arm (upper arm / forearm / hand) or leg (thigh / shin / foot / toe) joint flags the whole limb: refilled
    as one unit it keeps its shape (a hand refilled alone bends the elbow; e.g. a fast swing caught up after a stall)."""
    for side in ("left", "right"):
        for limb in (("UpperArm", "LowerArm", "Hand"), ("UpLeg", "Leg", "Foot", "Toe")):
            c = [k for k, j in enumerate(js) if j in {side + n for n in limb}]
            if c:
                bad[:, c] |= bad[:, c].any(1, keepdims=True)
    return bad


def propagate_down(bad: np.ndarray, js: list) -> np.ndarray:
    """A repaired joint repairs everything below it on the same samples (keeps bone lengths)."""
    idx = {j: k for k, j in enumerate(js)}
    for j in sorted(js, key=lambda j: depth_of(j, idx)):        # parents before children
        p = parent_of(j, idx)
        if p is not None:
            bad[:, idx[j]] |= bad[:, idx[p]]
    return bad


def depth_of(j: str, joints) -> int:
    p = parent_of(j, joints)
    return 0 if p is None else depth_of(p, joints) + 1


def unreachable_both(t, X, dist, vmax, tol):
    fwd = unreachable(t, X, dist, vmax, tol)
    bwd = unreachable(-t[::-1], X[::-1], dist, vmax, tol)[::-1]
    return fwd | bwd


def pos_dist(a, b):
    return np.linalg.norm(a - b, axis=-1)


def rot_dist(a, b):
    return 2 * np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))


def find_sessions(paths, rokoko_only=False) -> list:
    out = []
    for p in map(Path, paths):
        cands = [p] + [d for d in p.rglob("*") if d.is_dir()] if p.is_dir() else []
        out += [d for d in cands if any(d.glob("*_sync.csv")) and (rokoko_only or any(d.glob("*_perframe.csv")))
                and not d.name.startswith(".")]
    return sorted(set(out))


def clean(session: Path, rokoko_only=False) -> dict:
    sync_path = next(session.glob("*_sync.csv"))
    raw = pd.read_csv(sync_path)
    if rokoko_only:                                   # no video: the take's own rows are the frames
        pf_path = sync_path.with_name(sync_path.name.replace("_sync.csv", "_perframe.csv"))
        pf = raw.copy()
        pf.insert(0, "frame", np.arange(len(pf)))
        pf.insert(1, "video_time", pf["time"] - pf["time"].iloc[0])
        pf.insert(2, "match_error_ms", 0.0)
    else:
        pf_path = next(p for p in session.glob("*_perframe.csv"))
        pf = pd.read_csv(pf_path)
    joints = [c[:-6] for c in raw.columns if c.endswith("_pos_x")]
    moving = [j for j in joints if raw[[f"{j}_pos_{a}" for a in "xyz"]].to_numpy().std(0).max() > 0]
    js = [j for j in moving if all(f"{j}_quat_{a}" in raw for a in "xyzw")]
    pos_c = [f"{j}_pos_{a}" for j in js for a in "xyz"]
    quat_c = [f"{j}_quat_{a}" for j in js for a in "xyzw"]
    J = len(js)

    # 1. drop repeated packets
    vals = raw[pos_c].to_numpy()
    rep = np.r_[False, np.abs(np.diff(vals, axis=0)).max(1) == 0]
    # snaps are found on the stream WITH its repeats: a stall (held rows) then a jump is still a
    # one-sample snap, while a real dropout (missing rows) keeps its true time gap
    t_all = raw["time"].to_numpy()
    snap = (snap_excursions(t_all, vals.reshape(len(raw), -1, 3), pos_dist, SNAP_V, TOL_POS)
            | snap_excursions(t_all, raw[quat_c].to_numpy().reshape(len(raw), -1, 4), rot_dist, SNAP_W, TOL_DEG))[~rep]
    raw = raw[~rep].reset_index(drop=True)
    t = raw["time"].to_numpy()
    if len(t) < 3 or len(pf) == 0:
        raise ValueError(f"too few mocap samples ({len(t)})")
    dt_med = float(np.median(np.diff(t)))
    win, gate = odd(WIN_S / dt_med, 5), max(1, int(round(GATE_S / dt_med)))
    P = raw[pos_c].to_numpy().copy()
    Q = raw[quat_c].to_numpy().reshape(len(raw), J, 4).copy()

    # 2. quaternion sign continuity
    for i in range(1, len(Q)):
        Q[i, (Q[i] * Q[i - 1]).sum(1) < 0] *= -1

    # 3. speed-gated Hampel + reachability; a flagged channel flags its whole joint
    fing = np.array([bool(FINGER.search(j)) for j in js])
    vlim = np.where(fing, FINGER_VLIM, VLIM)
    rotlim = np.where(fing, FINGER_ROTLIM, ROTLIM)
    bad = (hampel(P, win, MIN_POS).reshape(-1, J, 3).any(2)
           | hampel(Q.reshape(len(raw), -1), win,
                    np.repeat(np.where(fing, FINGER_MIN_QUAT, MIN_QUAT), 4)).reshape(-1, J, 4).any(2))
    dt = (np.maximum(np.diff(t), 1e-3) + stall_credit(t)[1:])[:, None]  # catch-up after a stall is not a jump
    v = np.linalg.norm(np.diff(P.reshape(-1, J, 3), axis=0), axis=2) / dt
    jump = np.r_[np.zeros((1, J), bool), (v > vlim) | (ang_speed(Q, dt) > rotlim)]
    bad &= pd.DataFrame(jump).rolling(2 * gate + 1, center=True, min_periods=1).max().to_numpy() > 0
    bad |= unreachable_both(t, P.reshape(-1, J, 3), pos_dist, vlim, TOL_POS)
    bad |= unreachable_both(t, Q, rot_dist, rotlim, TOL_DEG)
    bad |= snap
    bad |= legs_folded(t, P.reshape(-1, J, 3), js)
    bad |= np.r_[bad[1:], bad[-1:]] | np.r_[bad[:1], bad[:-1]]     # glitch edges are half-way samples
    bad = propagate_down(propagate_limb(bad, js), js)
    # 4. segments: a freeze (no new sample for > FREEZE_S) cuts the stream; nothing is filled or smoothed across it
    sid = np.r_[0, np.cumsum(np.diff(t) > FREEZE_S)]                # segment id per mocap sample

    # 5. bones: a root joint keeps its position, every other joint = parent + raw length x direction.  Repair and
    #    smoothing touch only directions (and roots / rotations); lengths are the raw ones, never changed
    idx = {j: k for k, j in enumerate(js)}
    par = np.array([idx[p] if (p := parent_of(j, idx)) else -1 for j in js])
    order = sorted(range(J), key=lambda k: depth_of(js[k], idx))   # parents before children
    X = P.reshape(-1, J, 3)
    child = par >= 0
    vec = X - X[:, np.maximum(par, 0)]
    L = np.linalg.norm(vec, axis=2)                                   # raw bone length per sample
    D = np.where(child[None, :, None], vec / np.maximum(L, 1e-9)[..., None], X)   # direction, or root position
    Q = Q.reshape(len(raw), -1)
    for g in np.unique(sid):                                          # glitch refill inside each segment
        sg = np.where(sid == g)[0]
        ts = t[sg]
        for k in range(J):
            b = bad[sg, k]
            if not b.any() or (~b).sum() < 4:
                continue
            rid = np.cumsum(np.r_[b[0], b[1:] & ~b[:-1]]) * b          # run id per bad sample
            first = {r: i for i, r in reversed(list(enumerate(rid))) if r}
            last = {r: i for i, r in enumerate(rid) if r}
            span = np.array([ts[min(last[r] + 1, len(ts) - 1)] - ts[max(first[r] - 1, 0)] for r in rid[b]])
            short = span <= SPLINE_MAX_S
            for arr in (D[:, k], Q[:, 4 * k:4 * k + 4]):
                fill = PchipInterpolator(ts[~b], arr[sg][~b])(ts[b])
                fill[short] = CubicSpline(ts[~b], arr[sg][~b])(ts[b][short])
                arr[sg[b]] = fill
            L[sg[b], k] = np.interp(ts[b], ts[~b], L[sg][~b, k])       # raw length of the good samples around

    # 6. resample to the exact mocap time of each video frame (not nearest row), light smoothing, per segment
    tf = (pf["time"] - pf["match_error_ms"] / 1000).to_numpy() + time_shift(raw, pf, pos_c[:3] + quat_c[:4])
    j = np.clip(np.searchsorted(t, tf), 1, len(t) - 1)
    gap = t[j] - t[j - 1]
    fseg = np.where((tf >= t[0]) & (tf <= t[-1]) & (sid[j] == sid[j - 1]), sid[j], -1)   # frame inside a freeze: -1
    fps = (len(pf) - 1) / max(pf["video_time"].iloc[-1], 1e-6) if len(pf) > 1 else 30.0
    paired = np.ones(len(tf), bool)
    if fps >= ROKOKO_DRIVEN_FPS:                                       # Rokoko-driven: sample -> nearest frame
        jn = np.where(np.abs(t[j] - tf) < np.abs(tf - t[j - 1]), j, j - 1)
        err = np.abs(t[jn] - tf)
        best = pd.Series(err).groupby(jn).idxmin()                     # one frame per sample: the closest
        paired = np.zeros(len(tf), bool)
        paired[best.to_numpy()] = True
        paired &= err <= PAIR_MAX_S
        tf = np.where(paired, t[jn], tf)                               # the sample's own time -> its own value
    sw = odd(SMOOTH_S * fps)
    Df, Lf, Qf = (np.stack([np.interp(tf, t, A[:, c]) for c in range(A.shape[1])], 1)          # fallback (invalid
                  for A in (D.reshape(len(t), -1), L, Q))                                     # frames only)
    for g in np.unique(fseg[fseg >= 0]):
        f, sg = np.where(fseg == g)[0], np.where(sid == g)[0]
        for A, Af in ((D.reshape(len(t), -1), Df), (L, Lf), (Q, Qf)):
            Af[f] = np.stack([np.interp(tf[f], t[sg], A[sg, c]) for c in range(A.shape[1])], 1)
        if len(f) > sw:                                               # frames of a segment are consecutive
            Df[f] = savgol_filter(Df[f], sw, 2, axis=0)
            Qf[f] = savgol_filter(Qf[f], sw, 2, axis=0)
    badf = np.stack([np.interp(tf, t, bad[:, k].astype(float)) > 0 for k in range(J)], 1)
    Df = Df.reshape(len(tf), J, 3)
    Xf = np.zeros_like(Df)
    for k in order:                                                   # rebuild from the root, raw lengths
        if child[k]:
            u = Df[:, k] / np.maximum(np.linalg.norm(Df[:, k], axis=1, keepdims=True), 1e-9)
            Xf[:, k] = Xf[:, par[k]] + Lf[:, k:k + 1] * u
        else:
            Xf[:, k] = Df[:, k]
    Pf = Xf.reshape(len(tf), -1)
    Qf = Qf.reshape(len(tf), J, 4)
    Qf /= np.linalg.norm(Qf, axis=2, keepdims=True)

    # 7. frames with an implausible held posture cut the session like a freeze does
    wrong = implausible(Xf, js, fseg, fps)
    fseg = np.where(wrong, -1, fseg)
    fseg = np.where(fseg >= 0, np.cumsum(np.r_[True, np.diff(fseg) != 0]), -1)          # each remaining run its own id

    seg_len = {g: (tf[fseg == g][-1] - tf[fseg == g][0]) for g in np.unique(fseg[fseg >= 0])}
    keep = {g for g, d in seg_len.items() if d >= MIN_SEG_S}
    fseg = np.where(np.isin(fseg, list(keep)), fseg, -1)
    ren = {g: i for i, g in enumerate(sorted(keep))}                  # sub-sessions numbered 0, 1, ...
    fseg = np.array([ren.get(g, -1) for g in fseg])
    out = pf.copy()
    out[pos_c] = Pf
    out[quat_c] = Qf.reshape(len(tf), -1)
    out = pd.concat([out, pd.DataFrame({"repaired": badf.sum(1), "gap_ms": (gap * 1000).round(1),
                                        "valid": ((fseg >= 0) & paired).astype(int), "segment": fseg})], axis=1)
    segs = pd.DataFrame([dict(segment=i, frame_start=int(out.frame[fseg == i].iloc[0]), frame_end=int(out.frame[fseg == i].iloc[-1]),
                              frames=int((fseg == i).sum()), duration_s=round(float(tf[fseg == i][-1] - tf[fseg == i][0]), 2))
                         for i in range(len(keep))])
    segs.to_csv(session / "segments.csv", index=False)
    out.to_csv(pf_path.with_name(pf_path.stem + "_clean.csv"), index=False, float_format="%.6g")
    return {"session": session.name, "frames": len(out), "repeats_dropped": int(rep.sum()),
            "frames_repaired": int((out["repaired"] > 0).sum()),
            "frames_invalid": int((out["valid"] == 0).sum()), "frames_implausible": int(wrong.sum()),
            "static_joints": [j for j in joints if j not in moving],
            **{f"{tag}_{k}": v for tag, df in (("before", pf), ("after", out))
               for k, v in spike_stats(df, js, fps).items()}}


def spike_stats(df, js, fps) -> dict:
    """Jumps no human joint makes in one frame: > 8 m/s, > 60 deg/frame (at 30 fps)."""
    if len(df) < 2 or not js:
        return {"v>8": 0, "rot>60": 0}
    X = df[[f"{j}_pos_{a}" for j in js for a in "xyz"]].to_numpy().reshape(len(df), len(js), 3)
    q = df[[f"{j}_quat_{a}" for j in js for a in "xyzw"]].to_numpy().reshape(len(df), len(js), 4)
    v = np.linalg.norm(np.diff(X, axis=0), axis=2) * fps
    return {"v>8": int((v > 8).sum()), "rot>60": int((ang_speed(q, 1 / fps) > 60 * fps).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--force", action="store_true", help="redo sessions that already have _clean.csv")
    p.add_argument("--rokoko-only", action="store_true", help="takes without video: *_sync.csv only (see top)")
    a = p.parse_args()
    for s in find_sessions(a.paths, a.rokoko_only):
        if not a.force and any(s.glob("*_perframe_clean.csv")):
            continue
        try:
            print(clean(s, a.rokoko_only), flush=True)
        except Exception as e:                       # one bad session must not stop the batch
            print({"session": s.name, "error": f"{type(e).__name__}: {e}"}, flush=True)


if __name__ == "__main__":
    main()
