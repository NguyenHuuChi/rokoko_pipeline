#!/usr/bin/env python3
"""Hand keypoints + wrist target of the arm fit: WiLoR (ViT-L).

    python wilor_wrists.py CLIP_DIR [CLIP_DIR ...] [--mode fasam|fah|hand5] [--rescale 2.0]   (~/envs/wilor, GPU)

Modes:
  fasam  (default, M3b) runs WiLoR on FAH + SAM 2.1 2-ID smoothed boxes (CLIP_DIR/fah_boxes.npz)
  fah    (M3a) runs WiLoR on FAH gap-filled + SavGol-smoothed boxes (CLIP_DIR/fah_boxes.npz, no SAM)
  hand5  original pipeline: uses masks.npz + hands.npz + rtm_body.npz + hand_track

Writes CLIP_DIR/wilor_wrists.npz: kp2d (n, 2 left / right, 21, 2) full-res px, nan = none; wrist (n, 2, 2);
wrist_only (n, 2); merged (n,); kp3d (n, 2, 21, 3) m camera-aligned 3D joints; cam_t (n, 2, 3) m camera translation.
"""
import argparse
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import settings                                                              # noqa: E402
WD = Path(settings.WILOR).resolve()
sys.path.insert(0, str(WD))
os.chdir(WD)                                                                 # its configs use relative paths
from wilor.datasets.vitdet_dataset import ViTDetDataset                      # noqa: E402
from wilor.models import load_wilor                                          # noqa: E402
from wilor.utils import recursive_to                                         # noqa: E402
from wilor.utils.renderer import cam_crop_to_full                            # noqa: E402

sys.modules["handseg_pilot"] = types.SimpleNamespace(FF="")                  # arm2d_fit needs only its ffmpeg path; this env has no sam3
from arm2d_fit import observations                                           # noqa: E402
from hand_track import apply_cuffs, hand_boxes, smooth_stretches, smooth_wrist, track_sides  # noqa: E402

torch.set_default_dtype(torch.float32)                                       # arm2d_fit sets float64 + GPU at import
torch.set_default_device("cpu")


def _run_crops(model, cfg, im, boxes, right, rescale_factor=2.0):
    """Single WiLoR inference pass -> [(right, kp (21, 2) px, kp3d (21, 3) m, cam_t (3,) m)]."""
    if len(boxes) == 0:
        return []
    out = []
    for b in torch.utils.data.DataLoader(ViTDetDataset(cfg, im, boxes, right, rescale_factor=rescale_factor), batch_size=16, shuffle=False, num_workers=0):
        b = recursive_to(b, "cuda")
        with torch.no_grad():
            o = model(b)
        cam = o["pred_cam"]
        cam[:, 1] = (2 * b["right"] - 1) * cam[:, 1]
        size = b["img_size"].float()
        f = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * size.max()
        t = cam_crop_to_full(cam, b["box_center"].float(), b["box_size"].float(), size, f).cpu().numpy()
        for i in range(len(t)):
            j = o["pred_keypoints_3d"][i].cpu().numpy().copy()
            r = b["right"][i].item()
            j[:, 0] = (2 * r - 1) * j[:, 0]
            p = j + t[i]
            w, h = size[i].cpu().numpy()
            kp = np.c_[f.item() * p[:, 0] / p[:, 2] + w / 2, f.item() * p[:, 1] / p[:, 2] + h / 2]
            out.append((int(r), kp, j.astype(np.float32), t[i].astype(np.float32)))
    return out


def predict(model, cfg, im, given, rescale_factor=2.0, return_3d=False):
    """given [(box x1 y1 x2 y2, right)] -> [(right, kp (21, 2) px)] or with (kp3d, cam_t) when return_3d=True."""
    if not given:
        return []
    boxes = np.stack([b for b, *_ in given]).astype(np.float32)
    right = np.array([r for _, r, *_ in given], float)
    res = _run_crops(model, cfg, im, boxes, right, rescale_factor=rescale_factor)
    if return_3d:
        return res
    return [(r, kp) for r, kp, _, _ in res]


def _run_fah_clip(model, cfg, cd: Path, rescale: float):
    """Run WiLoR on smoothed L/R boxes saved by M3a (fah) or M3b (fasam)."""
    bz = np.load(cd / "fah_boxes.npz")
    sm_boxes = bz["boxes"]
    scores = bz["scores"] if "scores" in bz else np.full(sm_boxes.shape[:2], np.nan, np.float32)
    cap = cv2.VideoCapture(str(cd / "clip.mp4"))
    n = len(sm_boxes)
    K = np.full((n, 2, 21, 2), np.nan, np.float32)
    K3 = np.full((n, 2, 21, 3), np.nan, np.float32)
    T3 = np.full((n, 2, 3), np.nan, np.float32)
    f = 0
    while f < n:
        good, im = cap.read()
        if not good:
            break
        given = [(sm_boxes[f, s], s) for s in (0, 1) if np.isfinite(sm_boxes[f, s, 0])]
        for s, kp, kp3, ct in predict(model, cfg, im, given, rescale_factor=rescale, return_3d=True):
            K[f, s] = kp
            K3[f, s] = kp3
            T3[f, s] = ct
        f += 1
    cap.release()
    K, K3, T3 = K[:f], K3[:f], T3[:f]
    np.savez_compressed(
        cd / "wilor_wrists.npz",
        kp2d=K,
        wrist=K[:, :, 0],
        wrist_only=np.zeros((f, 2), bool),
        merged=np.zeros((f,), bool),
        kp3d=K3,
        cam_t=T3,
        det_conf=scores[:f],
    )
    print(cd.name, f"wrists L={np.isfinite(K[:, 0, 0, 0]).sum()}/{f} R={np.isfinite(K[:, 1, 0, 0]).sum()}/{f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--mode", choices=["fasam", "fah", "hand5"], default="fasam")
    ap.add_argument("--rescale", type=float, default=2.0)
    a = ap.parse_args()

    model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt", cfg_path="./pretrained_models/model_config.yaml")
    model = model.cuda().eval()
    for cd in a.clips:
        if a.mode in ("fasam", "fah"):
            _run_fah_clip(model, cfg, cd, rescale=a.rescale)
            continue
        z = np.load(cd / "masks.npz")
        hm, wm = z["hand"], z["wearer"]
        n = len(hm)
        w, ok, elbow, _ = observations(cd, n)
        h = np.load(cd / "hands.npz")
        p2 = h["pass"] == 2
        palm = np.full((n, 2, 2), np.nan)
        palm[h["frame"][p2].astype(int), h["hand"][p2].astype(int)] = h["kp2d"].reshape(len(h["pass"]), 21, 2)[p2][:, 9]
        r = np.load(cd / "rtm_body.npz")
        rtm, c, k = np.full((n, 2, 2), np.nan), r["conf"][:, 9:11] >= 0.3, r["frame"] < n
        rtm[r["frame"][k].astype(int)] = np.where(c[k][..., None], r["kp"][k][:, 9:11], np.nan)
        cap = cv2.VideoCapture(str(cd / "clip.mp4"))
        sides, prompted = hand_boxes(hm, track_sides(hm, rtm, int(cap.get(3)), wm=wm), w, ok, palm, rtm, elbow)
        K = np.full((n, 2, 21, 2), np.nan)
        K3 = np.full((n, 2, 21, 3), np.nan, np.float32)
        T3 = np.full((n, 2, 3), np.nan, np.float32)
        for f in range(n):
            good, im = cap.read()
            if not good:
                break
            for s, kp, kp3, ct in predict(model, cfg, im, [(b, s) for b, s, *_ in sides[f]], rescale_factor=a.rescale, return_3d=True):
                K[f, s] = kp
                K3[f, s] = kp3
                T3[f, s] = ct
        cap.release()
        C, wo = apply_cuffs(K, sides)
        C[wo & ~prompted] = np.nan                                           # no prompt wrist: blank
        K3[wo] = np.nan
        T3[wo] = np.nan
        M = np.array([any(e[3] for e in a_row) for a_row in sides])
        Z = C.copy()
        Z[M] = np.nan
        S = smooth_wrist(Z)[0]
        S[M] = C[M]
        S = smooth_stretches(S, M)
        np.savez_compressed(cd / "wilor_wrists.npz", kp2d=S, wrist=S[:, :, 0], wrist_only=wo & prompted, merged=M, kp3d=K3, cam_t=T3)
        print(cd.name, f"wrists {np.isfinite(S[:, :, 0]).all(-1).sum()}, with fingers {np.isfinite(S).all((2, 3)).sum()}, touching frames {M.sum()}", flush=True)


if __name__ == "__main__":
    main()
