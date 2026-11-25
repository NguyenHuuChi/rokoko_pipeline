#!/usr/bin/env python3
"""Wearer arm / hand masks on a cut clip with the fine-tuned SAM 2.1 base+ (handseg_model/sam2_ft.py), prompted only
from the clip's detections (hand5 + RTMPose, hung5_detect.py) - no SAM 3.

    python sam2_clipmask.py CLIP_DIR [CLIP_DIR ...] [--ckpt BEST.pth]          (~/envs/handseg, GPU)

Prompt points of a frame (confident sides only, arm2d_fit.observations): hand object = hand5 wrist + palm (middle
knuckle); wearer object = those + RTMPose elbow + forearm midpoint.  Each object is tracked on its own (the learned type
token of sam2_ft tells the decoder which one), from its own memory once prompted:
  1 keyframe = the frame with the most confident sides: prompt, track forward and backward to the clip ends;
  2 walking away from the keyframe (both ways), a confident detection re-prompts when its wrist is > AGREE_PX from the
    hand mask (drift, or a hand never prompted) or the hand mask was empty for >= GAP frames before it (hand came back);
    a re-prompt replaces the masks from there on and is also tracked back, OR-ed into the earlier frames until its hand
    mask is empty for STOP frames (the hand was already there before the detector became confident).
Wearer check: a keyframe / re-prompt counts only if its wrist's piece touches the bottom edge or lies within ROKOKO_PX of the
Rokoko wrist of that side (validate(); a "confident" hand5 on a bystander is rejected, the next candidate tried); then a
piece stays only if it holds such a prompt, continues a kept piece of the neighbouring frame within GROW px, or comes in
through the bottom edge (temporal_wearer()): a mask jumping onto a bystander is dropped.
Writes CLIP_DIR/masks.npz (--name): hand, wearer (n, H/4, W/4) bool (as handseg_clipmask.py); no confident detection -> empty.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings                                                             # noqa: E402
from arm2d_fit import observations                                          # noqa: E402

torch.set_default_dtype(torch.float32)                                      # arm2d_fit sets float64 / cuda globally
torch.set_default_device("cpu")

CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CKPT = settings.SAM2_FT
W, H = 640, 480                                                             # tracker frames (as sam2_ft training)
AGREE_PX, GAP, BACK, MIN_SEP = 40, 15, 15, 10
STOP, KEY_TRIES, GROW, ROKOKO_PX = 5, 8, 15, 250
WEARER, HAND = 0, 1


def model(ckpt):
    p = build_sam2_video_predictor(CFG, settings.SAM2_BASE)
    sd = torch.load(ckpt, map_location="cuda")
    p.type_emb, p.typ = sd.pop("type_emb").cuda(), WEARER
    p.load_state_dict(sd)

    def hook(mod, inp, out):                                                # type token on every frame, as in training
        sp, de = out
        return torch.cat([sp, p.type_emb[p.typ].expand(sp.shape[0], 1, -1).to(sp.dtype)], 1), de
    p.sam_prompt_encoder.register_forward_hook(hook)
    return p


def prompts(cd, n):
    """Per frame None or (wearer points, hand points), (P, 2) in tracker pixels."""
    w, conf, e, _ = observations(cd, n)
    h = np.load(cd / "hands.npz")
    p2 = h["pass"] == 2
    palm = np.full((n, 2, 2), np.nan)
    palm[h["frame"][p2].astype(int), h["hand"][p2].astype(int)] = h["kp2d"].reshape(len(h["pass"]), 21, 2)[p2][:, 9]
    s = W / 1920
    out = []
    for f in range(n):
        hp, wp = [], []
        for i in np.flatnonzero(conf[f]):
            hp += [w[f, i] * s, palm[f, i] * s]
            wp += [w[f, i] * s, palm[f, i] * s] + ([e[f, i] * s, (w[f, i] + e[f, i]) * s / 2] if np.isfinite(e[f, i]).all() else [])
        ok = lambda q: np.array([x for x in q if np.isfinite(x).all()])
        out.append((ok(wp), ok(hp)) if len(ok(hp)) else None)
    return out, w * s, conf


def track(p, st, f, pts, ahead, d, back=BACK):
    """Both objects prompted on frame f, tracked `ahead` frames in direction d and `back` frames the other way."""
    res = {}
    for typ in (WEARER, HAND):
        p.typ = typ
        p.reset_state(st)
        p.add_new_points_or_box(st, frame_idx=f, obj_id=1, points=pts[typ], labels=np.ones(len(pts[typ]), np.int32))
        for rev, m in ((d < 0, ahead), (d > 0, back)):
            for k, _, lg in p.propagate_in_video(st, start_frame_idx=f, max_frame_num_to_track=m, reverse=rev):
                res.setdefault(k, [None, None])[typ] = (lg[0, 0] > 0).cpu().numpy()
    return res


def rokoko_wrists(cd, n):
    """(n, 2, 2) projected raw Rokoko left / right wrist in tracker px; nan without <session>/camera/rokoko_cam.npz."""
    out = np.full((n, 2, 2), np.nan)
    try:
        sd, f0 = Path((cd / "session.txt").read_text().strip()), int((cd / "f0.txt").read_text())
        z = np.load(sd / "camera/rokoko_cam.npz")
    except (FileNotFoundError, ValueError):
        return out
    R, t, K = (np.asarray(z[k], float) for k in ("Ruw", "tuw", "Ku"))
    for s, j in enumerate(("leftHand", "rightHand")):
        q = (z[j][f0:f0 + n].astype(float) - t) @ R
        out[:len(q), s] = (q[:, :2] / np.maximum(q[:, 2:], 0.05) * np.diag(K)[:2] + K[:2, 2]) * W / 1920
    return out


def validate(f, a, b, wr, conf, rk):
    """Prompt check (v6): the pieces of the prompt-frame masks (a, b) holding a confident hand5 wrist that is the
    wearer's: its piece touches the bottom edge OR the wrist is within ROKOKO_PX of the Rokoko wrist of that side.
    -> bool mask of those pieces, or None (prompt rejected)."""
    _, cc = cv2.connectedComponents(cv2.dilate((a | b).astype(np.uint8), np.ones((7, 7), np.uint8)))
    bottom = set(np.unique(cc[-1, :])) - {0}
    ok = set()
    for s in np.flatnonzero(conf[f]):
        x, y = wr[f, s]
        if not (0 <= x < W and 0 <= y < H):
            continue
        labs = set(np.unique(cc[max(0, int(y) - 10):int(y) + 11, max(0, int(x) - 10):int(x) + 11])) - {0}
        near = np.isfinite(rk[f, s]).all() and np.linalg.norm(wr[f, s] - rk[f, s]) < ROKOKO_PX * W / 1920
        ok |= {l for l in labs if l in bottom or near}
    return np.isin(cc, list(ok)) if ok else None


def temporal_wearer(M, anchors):
    """v6 wearer check over time: a piece is kept if it holds a validated prompt (anchors), continues a kept piece of
    the neighbouring frame (within GROW px), or comes in through the bottom edge; swept forward and backward, a piece
    kept in either sweep stays.  A piece appearing from nowhere (the mask jumping onto a bystander) is dropped."""
    n = len(M)
    cc = [cv2.connectedComponents(cv2.dilate((M[t, 0] | M[t, 1]).astype(np.uint8), np.ones((7, 7), np.uint8)))[1] for t in range(n)]
    ker = np.ones((2 * GROW + 1, 2 * GROW + 1), np.uint8)
    keep = np.zeros((n, H, W), bool)
    for order in (range(n), range(n - 1, -1, -1)):
        prev = None
        for t in order:
            ls = set(np.unique(cc[t][-1, :])) - {0}
            if prev is not None and prev.any():
                ls |= set(np.unique(cc[t][cv2.dilate(prev.astype(np.uint8), ker) > 0])) - {0}
            k = np.isin(cc[t], list(ls)) if ls else np.zeros((H, W), bool)
            if t in anchors:
                k |= anchors[t]
            keep[t] |= k
            prev = k
    M[:, 0] &= keep
    M[:, 1] &= keep
    return M


def clip_masks(p, cd, wd):
    cap = cv2.VideoCapture(str(cd / "clip.mp4"))
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    n = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(wd / f"{n:05d}.jpg"), cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))
        n += 1
    P, wr, conf = prompts(cd, n)
    rk = rokoko_wrists(cd, n)
    cand = sorted([f for f in range(n) if P[f] is not None], key=lambda f: (-conf[f].sum(), f))
    M = np.zeros((n, 2, H, W), bool)
    log, anchors = dict(tried=0, accepted=0, rejected=0), {}
    if not cand:
        shutil.rmtree(wd)
        return M, log
    st = p.init_state(video_path=str(wd))
    k0 = None
    for f in cand[:KEY_TRIES]:
        log["tried"] += 1
        R = track(p, st, f, P[f], n, 1, back=n)
        kp = validate(f, *R[f], wr, conf, rk)
        if kp is None:
            log["rejected"] += 1
            continue
        k0, anchors[f] = f, kp
        for k, (a, b) in R.items():
            M[k] = a, b
        log["accepted"] += 1
        break
    if k0 is None:
        p.reset_state(st)
        shutil.rmtree(wd)
        return M, log
    for d in (1, -1):
        last = k0
        for f in (range(k0 + 1, n) if d > 0 else range(k0 - 1, -1, -1)):
            if P[f] is None or abs(f - last) < MIN_SEP:
                continue
            dt = cv2.distanceTransform((~M[f, HAND]).astype(np.uint8), cv2.DIST_L2, 3)
            far = any(dt[min(int(y), H - 1), min(int(x), W - 1)] > AGREE_PX for x, y in wr[f][conf[f]] if 0 <= x < W and 0 <= y < H)
            prev = [f - d * j for j in range(1, GAP + 1) if 0 <= f - d * j < n]
            back = len(prev) == GAP and not M[prev, HAND].any()
            if not (far or back):
                continue
            log["tried"] += 1
            R = track(p, st, f, P[f], n, d, back=n)
            kp = validate(f, *R[f], wr, conf, rk)
            if kp is None:
                log["rejected"] += 1
                continue
            anchors[f] = kp
            for k in (range(f, n) if d > 0 else range(f, -1, -1)):
                if k in R:
                    M[k] = R[k]
            empty = 0
            for k in (range(f - 1, -1, -1) if d > 0 else range(f + 1, n)):
                if k not in R:
                    break
                a, b = R[k]
                empty = 0 if b.any() else empty + 1
                if empty >= STOP:
                    break
                M[k, WEARER] |= a
                M[k, HAND] |= b
            last = f
            log["accepted"] += 1
    p.reset_state(st)
    shutil.rmtree(wd)
    return temporal_wearer(M, anchors), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--name", default="masks", help="output CLIP_DIR/<name>.npz (keep masks.npz when comparing models)")
    a = ap.parse_args()
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    p = model(a.ckpt)
    for cd in a.clips:
        M, log = clip_masks(p, cd, cd / "frames_tmp")
        cap = cv2.VideoCapture(str(cd / "clip.mp4"))
        Hc, Wc = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        q = lambda m: cv2.resize(m.astype(np.uint8), (Wc // 4, Hc // 4), interpolation=cv2.INTER_NEAREST) > 0
        hand = np.stack([q(h & w) for w, h in M])
        np.savez_compressed(cd / f"{a.name}.npz", hand=hand, wearer=np.stack([q(w) for w, _ in M]))
        print(cd.name, f"prompts tried {log['tried']} accepted {log['accepted']} rejected {log['rejected']} | frames with hand {hand.any((1, 2)).mean():.2f}",
              flush=True)


if __name__ == "__main__":
    main()
