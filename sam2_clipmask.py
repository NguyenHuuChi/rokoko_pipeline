#!/usr/bin/env python3
"""Wearer arm / hand masks on a cut clip with the fine-tuned SAM 2.1 base+ (handseg_model/sam2_ft.py), prompted only
from the clip's detections (hand5 + RTMPose, hung5_detect.py) or FAH YOLO-seg (`--mode fasam`, default) - no SAM 3.

    python sam2_clipmask.py CLIP_DIR [CLIP_DIR ...] [--mode fasam|hand5] [--ckpt BEST.pth]  (~/envs/handseg, GPU)

Modes:
  fasam  (default, M3b) SAM 2.1 base+ (same 640x480 / bfloat16 / init_state config) prompted by FAH YOLO-seg apart L/R
         masks (fah_prompts.npz) with 2 IDs (L=1, R=2) -> masks.npz (hand, hand_lr, empty wearer) + smoothed fah_boxes.npz
  hand5  original main behavior: fine-tuned SAM 2.1 base+ prompted from hands.npz + RTMPose (WEARER/HAND 2-pass)
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from scipy.signal import savgol_filter

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


def select_anchors(apart_idx, min_step=15):
    """Choose clean apart anchor frames covering contiguous segment boundaries and spaced by min_step."""
    if not len(apart_idx):
        return []
    anchors = [int(apart_idx[0])]
    for i in range(1, len(apart_idx)):
        cur, prev = int(apart_idx[i]), int(apart_idx[i - 1])
        if (cur - prev > 1) or (cur - anchors[-1] >= min_step):
            if prev not in anchors and cur - prev > 1:
                anchors.append(prev)
            if cur not in anchors:
                anchors.append(cur)
    if int(apart_idx[-1]) not in anchors:
        anchors.append(int(apart_idx[-1]))
    return sorted(set(anchors))


def smooth_boxes_from_masks(M, frame_wh, min_area_trk=40, win=5):
    """M: (n, 2, Ht, Wt) bool -> (n, 2, 4) SavGol-smoothed full-res square boxes, plus valid (n, 2)."""
    n, _, Ht, Wt = M.shape
    Wc, Hc = frame_wh
    sx, sy = float(Wc) / Wt, float(Hc) / Ht
    cxs = np.full((n, 2, 3), np.nan, np.float64)
    raw_valid = np.zeros((n, 2), bool)
    for f in range(n):
        for s in (0, 1):
            ys, xs = np.nonzero(M[f, s])
            if len(xs) < min_area_trk:
                continue
            raw_valid[f, s] = True
            x1, x2 = float(xs.min()) * sx, (float(xs.max()) + 1.0) * sx
            y1, y2 = float(ys.min()) * sy, (float(ys.max()) + 1.0) * sy
            cxs[f, s, 0] = 0.5 * (x1 + x2)
            cxs[f, s, 1] = 0.5 * (y1 + y2)
            cxs[f, s, 2] = max(x2 - x1, y2 - y1)

    for s in (0, 1):
        ok = np.isfinite(cxs[:, s, 0])
        f_idx = 0
        while f_idx < n:
            if not ok[f_idx]:
                f_idx += 1
                continue
            j_idx = f_idx
            while j_idx < n and ok[j_idx]:
                j_idx += 1
            seg = j_idx - f_idx
            if seg >= win:
                for d_i in range(3):
                    cxs[f_idx:j_idx, s, d_i] = savgol_filter(cxs[f_idx:j_idx, s, d_i], window_length=win, polyorder=2)
            elif seg >= 3:
                for d_i in range(3):
                    cxs[f_idx:j_idx, s, d_i] = savgol_filter(cxs[f_idx:j_idx, s, d_i], window_length=3, polyorder=1)
            f_idx = j_idx

    boxes = np.full((n, 2, 4), np.nan, np.float32)
    for s in (0, 1):
        ok = np.isfinite(cxs[:, s, 0])
        sz = np.maximum(cxs[ok, s, 2], 20.0)
        boxes[ok, s, 0] = cxs[ok, s, 0] - 0.5 * sz
        boxes[ok, s, 1] = cxs[ok, s, 1] - 0.5 * sz
        boxes[ok, s, 2] = cxs[ok, s, 0] + 0.5 * sz
        boxes[ok, s, 3] = cxs[ok, s, 1] + 0.5 * sz
    return boxes, raw_valid


def clip_masks_fasam(p, cd, wd):
    """M3b: Track L (obj_id=1) and R (obj_id=2) masks across clip with SAM 2.1 prompted from fah_prompts.npz."""
    pz = np.load(cd / "fah_prompts.npz")
    anchor_frames = pz["anchor_frames"].astype(int)
    anchor_masks = pz["anchor_masks"]
    Wt, Ht = (int(x) for x in pz["tracker_wh"]) if "tracker_wh" in pz else (W, H)
    frame_to_idx = {int(f): idx for idx, f in enumerate(anchor_frames)}
    anchors = select_anchors(anchor_frames, min_step=15)

    cap = cv2.VideoCapture(str(cd / "clip.mp4"))
    Hc, Wc = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    n = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if anchors:
            cv2.imwrite(str(wd / f"{n:05d}.jpg"), cv2.resize(im, (Wt, Ht), interpolation=cv2.INTER_AREA))
        n += 1
    cap.release()

    M = np.zeros((n, 2, Ht, Wt), bool)
    if anchors and n > 0:
        st = p.init_state(video_path=str(wd))
        for fa in anchors:
            idx = frame_to_idx[fa]
            p.add_new_mask(st, frame_idx=fa, obj_id=1, mask=anchor_masks[idx, 0])
            p.add_new_mask(st, frame_idx=fa, obj_id=2, mask=anchor_masks[idx, 1])
        f0 = anchors[0]
        for k, obj_ids, logits in p.propagate_in_video(st, start_frame_idx=f0, reverse=False):
            for j, oid in enumerate(obj_ids):
                M[k, int(oid) - 1] = (logits[j, 0] > 0).cpu().numpy()
        if f0 > 0:
            for k, obj_ids, logits in p.propagate_in_video(st, start_frame_idx=f0, reverse=True):
                for j, oid in enumerate(obj_ids):
                    M[k, int(oid) - 1] = (logits[j, 0] > 0).cpu().numpy()
        p.reset_state(st)
    shutil.rmtree(wd, ignore_errors=True)
    return M, (Wc, Hc), anchors, pz["scores"][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--mode", choices=["fasam", "hand5"], default="fasam")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--name", default="masks", help="output CLIP_DIR/<name>.npz (keep masks.npz when comparing models)")
    a = ap.parse_args()
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if a.mode == "fasam":
        p = build_sam2_video_predictor(CFG, a.ckpt or settings.SAM2_BASE)
        for cd in a.clips:
            M, (Wc, Hc), anchors, raw_scores = clip_masks_fasam(p, cd, cd / "frames_tmp")
            sm_boxes, valid = smooth_boxes_from_masks(M, (Wc, Hc))
            np.savez_compressed(
                cd / "fah_boxes.npz",
                boxes=sm_boxes,
                raw_valid=valid,
                fill_valid=valid,
                scores=np.where(valid, np.where(np.isfinite(raw_scores), raw_scores, np.float32(1.0)), np.float32(np.nan)),
                anchors=np.array(anchors, np.int32),
                mode=np.array("fasam"),
            )
            q = lambda m: cv2.resize(m.astype(np.uint8), (Wc // 4, Hc // 4), interpolation=cv2.INTER_NEAREST) > 0
            hand_lr = np.stack([np.stack([q(mL), q(mR)], axis=0) for mL, mR in M]) if len(M) else np.zeros((0, 2, Hc // 4, Wc // 4), bool)
            hand = hand_lr.any(1) if len(hand_lr) else np.zeros((0, Hc // 4, Wc // 4), bool)
            wearer = np.zeros_like(hand)
            np.savez_compressed(cd / f"{a.name}.npz", hand=hand, hand_lr=hand_lr, wearer=wearer)
            print(
                cd.name,
                f"[M3b fasam] anchors={len(anchors)} | L={100 * valid[:, 0].mean():.1f}% R={100 * valid[:, 1].mean():.1f}%",
                flush=True,
            )
    else:
        p = model(a.ckpt or CKPT)
        for cd in a.clips:
            M, log = clip_masks(p, cd, cd / "frames_tmp")
            cap = cv2.VideoCapture(str(cd / "clip.mp4"))
            Hc, Wc = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            q = lambda m: cv2.resize(m.astype(np.uint8), (Wc // 4, Hc // 4), interpolation=cv2.INTER_NEAREST) > 0
            hand = np.stack([q(h & w) for w, h in M])
            np.savez_compressed(cd / f"{a.name}.npz", hand=hand, wearer=np.stack([q(w) for w, _ in M]))
            print(cd.name, f"prompts tried {log['tried']} accepted {log['accepted']} rejected {log['rejected']} | frames with hand {hand.any((1, 2)).mean():.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
