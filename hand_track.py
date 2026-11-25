"""Left / right tracking, hand boxes and wrist smoothing on the SAM 2.1 masks, for wilor_wrists.py (no WiLoR import here)."""
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter


MIN_ARM, MIN_ARM_BOTTOM = 1000, 300    # mask px: an arm piece; smaller if it comes in through the bottom edge


def arm_end(arm, hand):
    """Cuff of an arm piece at a hand (mask res): (point where the arm meets the hand, unit forearm direction toward
    the hand), or None when they do not touch."""
    edge = arm & (cv2.dilate(hand.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    if edge.sum() < 5:
        return None
    ys, xs = np.nonzero(edge)
    q = np.array([xs.mean(), ys.mean()])
    ya, xa = np.nonzero(arm)
    near = np.hypot(xa - q[0], ya - q[1]) < 60                                 # the forearm just behind the cuff
    v = q - np.array([xa[near].mean(), ya[near].mean()])
    n = np.linalg.norm(v)
    return q, (v / n if n > 1e-6 else np.array([0.0, -1.0]))


def blobs(hm, wm=None, min_cuffs=20):
    """Hand mask (H/4, W/4) -> [(box x1 y1 x2 y2, centroid, cuff or None, merged)] at full res, slivers (< ~640 px) left
    out.  With the wearer mask wm (arm + hand) a blob with TWO arms at it (hands touching / crossed) gives one box per
    arm, centred ahead of that arm's cuff along the forearm: a box around a piece of the blob would give WiLoR two
    almost equal crops (it pads x2), and both hands land on the same hand.  Two cuffs closer than min_cuffs (mask px)
    are one arm broken in two."""
    n, cc, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(hm.astype(np.uint8), np.ones((5, 5), np.uint8)))
    arms = []
    if wm is not None:
        na, ca, sa, _ = cv2.connectedComponentsWithStats((wm & ~hm).astype(np.uint8))
        arms = [ca == i for i in range(1, na)                                  # real arms, not fragments next to a hand
                if sa[i, cv2.CC_STAT_AREA] >= MIN_ARM or (sa[i, cv2.CC_STAT_AREA] >= MIN_ARM_BOTTOM and (ca[-1] == i).any())]
    out = []
    for i, (x, y, w, h, a) in enumerate(st):
        if not i or a < 40:
            continue
        part = cc == i
        ends = [e for e in (arm_end(m, part) for m in arms) if e is not None]
        if len(ends) >= 2:
            ends = sorted(ends, key=lambda e: -abs((e[0] - cen[i]) @ e[1]))[:2]
            if np.linalg.norm(ends[0][0] - ends[1][0]) >= min_cuffs:
                L = 1.5 * np.sqrt(a / 2)                                       # one hand's size, mask px
                for q, v in ends:
                    c = q + 0.5 * L * v
                    box = np.array([c[0] - L / 2, c[1] - L / 2, c[0] + L / 2, c[1] + L / 2]) * 4
                    out.append((box, c * 4, q * 4, True))
                continue
        out.append((np.array([x, y, x + w, y + h], float) * 4, cen[i] * 4, ends[0][0] * 4 if len(ends) == 1 else None, False))
    return out


def track_sides(hm, wr, W, gate=300, lost=30, wm=None):
    """Per frame [(box, side)]: two tracks (0 left, 1 right), started on a frame with two blobs and both RTMPose
    wrists, then walked forward and backward.  Cost blob -> side = distance to the track's last position (gated,
    gate grows 20 px per unseen frame) + 0.5 x side prior (RTMPose wrist of that side, else 300 px for the wrong image
    half); a track unseen for > lost frames is restarted from the prior.  Hungarian per frame: a lone blob keeps its
    own track, extra blobs (a bystander's arm) are dropped."""
    n = len(hm)
    B = [blobs(hm[f], None if wm is None else wm[f]) for f in range(n)]
    pos = lambda b: b[2] if b[2] is not None else b[1]                         # a hand's place: its cuff (box centres of
                                                                               # touching hands cross over), else centroid
    prior = lambda f, c, s: np.linalg.norm(wr[f, s] - c) if np.isfinite(wr[f, s]).all() else (0.0 if (c[0] > W / 2) == bool(s) else 300.0)

    def assign(f, last):
        if not B[f]:
            return {}
        C = np.full((len(B[f]), 2), 1e6)
        for i, b in enumerate(B[f]):
            c = pos(b)
            for s in (0, 1):
                p, g = last[s]
                if p is None or g > lost:
                    C[i, s] = 100 + 0.5 * prior(f, c, s)                       # (re)start: prior only
                elif (d := np.linalg.norm(c - p)) <= gate + 20 * g:
                    C[i, s] = d + 0.5 * prior(f, c, s)
        r, k = linear_sum_assignment(C)
        return {i: s for i, s in zip(r, k) if C[i, s] < 1e6}

    cand = [f for f in range(n) if len(B[f]) == 2 and np.isfinite(wr[f]).all()]
    f0 = cand[0] if cand else int(np.argmax([len(b) for b in B]))
    side = {}
    for rng in (range(f0, n), range(f0, -1, -1)):
        last = [(None, 0), (None, 0)]
        for f in rng:
            a = side[f] = assign(f, last)
            for s in (0, 1):
                hit = [i for i, q in a.items() if q == s]
                last[s] = (pos(B[f][hit[0]]), 0) if hit else (last[s][0], last[s][1] + 1)
    return [[(B[f][i][0], s, B[f][i][2], B[f][i][3]) for i, s in side.get(f, {}).items()] for f in range(n)]


def apply_cuffs(K, sides, rel=0.5):
    """Touching hands (merged blob, two arms): WiLoR may put both hands on one.  Rejected: a wrist nearer the OTHER arm's
    cuff than its own, and, when the two wrists are closer than rel x hand size, the one further from its own cuff.
    The distance to the own cuff alone is no test: on long gloves the cuff lies far down the forearm.  A rejected
    hand keeps its cuff as the wrist, no fingers.  -> K, wrist_only (n, 2)."""
    K, wo = K.copy(), np.zeros(K.shape[:2], bool)
    for f, a in enumerate(sides):
        cuffs = {s: cuff for _, s, cuff, merged in a if merged and cuff is not None}
        if len(cuffs) < 2:
            continue
        bad = set()
        ok = {s: np.isfinite(K[f, s]).all() for s in (0, 1)}
        d = {s: np.linalg.norm(K[f, s, 0] - cuffs[s]) if ok[s] else np.inf for s in (0, 1)}
        for s in (0, 1):
            if not ok[s] or d[s] > np.linalg.norm(K[f, s, 0] - cuffs[1 - s]):
                bad.add(s)
        if ok[0] and ok[1]:
            size = max(np.linalg.norm(K[f, s, 9] - K[f, s, 0]) for s in (0, 1))
            if np.linalg.norm(K[f, 0, 0] - K[f, 1, 0]) < rel * size:
                bad.add(int(d[1] > d[0]))
        for s in bad:
            K[f, s] = np.nan
            K[f, s, 0] = cuffs[s]
            wo[f, s] = True
    return K, wo


def hand_boxes(hm, sides, w, ok, palm, rtm, elbow):
    """v8: hand boxes placed from the wrist, the arm only says which hand is which.  w, palm (n, 2, 2): hand5 wrist / middle
    knuckle, ok (n, 2) trusted (arm2d_fit.observations); rtm, elbow (n, 2, 2) RTMPose fallback.  A prompt wrist must lie on the hand
    mask (a bystander's does not).  Separate hand: the blob is cut behind the wrist (a bare forearm is in the hand mask);
    almost nothing left ahead of it = hand out of view, no box.  Merged blob: a box of the hand's last size ahead of the wrist.
    No prompt: the box stays.  -> sides with the prompt wrist (else cuff) as reference, prompted (n, 2)."""
    near = [cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0 for m in hm]
    H, Wm = hm.shape[1:]
    out, prompted, last = [], np.zeros(ok.shape, bool), [None, None]
    for f, a in enumerate(sides):
        row = []
        for box, s, cuff, merged in a:
            p, d = None, None
            if ok[f, s]:
                p, d = w[f, s], palm[f, s] - w[f, s]
            elif np.isfinite(rtm[f, s]).all():
                p = rtm[f, s]
                if np.isfinite(elbow[f, s]).all():
                    d = p - elbow[f, s]
                elif cuff is not None:
                    d = p - cuff
            cell = None if p is None else (p / 4).astype(int)
            if p is None or not (0 <= cell[0] < Wm and 0 <= cell[1] < H and near[f][cell[1], cell[0]]):
                row.append((box, s, cuff, merged))
                continue
            if not merged and not (box[0] - 40 <= p[0] <= box[2] + 40 and box[1] - 40 <= p[1] <= box[3] + 40):
                row.append((box, s, cuff, merged))                             # the other hand's wrist
                continue
            prompted[f, s] = True
            n = np.linalg.norm(d) if d is not None else 0
            if n < 1:
                row.append((box, s, p, merged))
                continue
            d = d / n
            if merged:
                L = last[s] if last[s] is not None else box[2] - box[0]
                c = p + 0.5 * L * d
                row.append((np.array([c[0] - L / 2, c[1] - L / 2, c[0] + L / 2, c[1] + L / 2]), s, p, True))
                continue
            x1, y1, x2, y2 = (box / 4).astype(int)
            ys, xs = np.nonzero(hm[f, max(y1, 0):y2, max(x1, 0):x2])
            xs, ys = xs + max(x1, 0), ys + max(y1, 0)
            ahead = (xs * 4 - p[0]) * d[0] + (ys * 4 - p[1]) * d[1]
            k = ahead > -30
            if (ahead > 0).sum() < 40:                                         # same floor as blobs(): a sliver
                continue
            b = np.array([xs[k].min(), ys[k].min(), xs[k].max() + 1, ys[k].max() + 1], float) * 4
            last[s] = max(b[2] - b[0], b[3] - b[1])
            row.append((b, s, p, False))
        out.append(row)
    return out, prompted


def smooth_wrist(K, jump=60, gap=3, win=7):
    """Smooth the WRIST trajectory only: per hand, wrist frames jumping > jump px from the 5-frame running median are
    dropped, gaps <= gap frames interpolated, Savitzky-Golay (win, order 2) on each run; every hand is then shifted
    by its wrist correction, so the finger pose of each frame is WiLoR's own.  (n, 2, 21, 2) -> same, dropped."""
    S, dropped = K.copy(), np.zeros(K.shape[:2], bool)
    for s in (0, 1):
        X = K[:, s, 0].copy()
        med = pd.DataFrame(X).rolling(5, center=True, min_periods=3).median().to_numpy()
        bad = np.nan_to_num(np.linalg.norm(X - med, axis=1), nan=0) > jump
        dropped[:, s] = bad
        X[bad] = np.nan
        X = pd.DataFrame(X).interpolate(limit=gap, limit_area="inside").to_numpy().copy()
        idx = np.flatnonzero(np.isfinite(X).all(1))
        for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(r) >= win:
                X[r] = savgol_filter(X[r], win, 2, axis=0)
        has = np.isfinite(K[:, s, 0]).all(1)
        S[has, s] = K[has, s] + (X[has] - K[has, s, 0])[:, None]
        S[has & ~np.isfinite(X).all(1), s] = np.nan
    return S, dropped


def smooth_stretches(S, M):
    """v9: smooth_wrist() on each run of merged frames M on its own, so no wrist is smoothed across the edge of a touch.
    Runs shorter than its window keep their positions (outlier frames still replaced)."""
    out, idx = S.copy(), np.flatnonzero(M)
    for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []:
        Z = np.full_like(S, np.nan)
        Z[r] = S[r]
        out[r] = smooth_wrist(Z)[0][r]
    return out
