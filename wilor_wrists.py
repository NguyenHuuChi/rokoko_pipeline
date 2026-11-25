#!/usr/bin/env python3
"""Hand keypoints + wrist target of the arm fit: WiLoR (ViT-L) on our SAM 2.1 hands.

    python wilor_wrists.py CLIP_DIR [CLIP_DIR ...]          (~/envs/wilor, GPU)

CLIP_DIR: clip.mp4, masks.npz, hands.npz, rtm_body.npz.  For each clip:
  sides    two tracks, each hand followed by its arm (hand_track.track_sides); RTMPose wrists only as a hint.  WiLoR models
           right hands, left crops are mirrored, so the side must be right before inference
  boxes    from the wrist (hand_track.hand_boxes): hand5 where trusted, else RTMPose, and on the hand mask.  A separate hand: the
           blob cut behind the wrist (a bare forearm is in the hand mask), nothing ahead of it = hand out of view.  Touching
           hands (one blob, two arms): a box of that hand's last size ahead of its wrist.  No prompt: the mask box
  check    touching hands: a wrist nearer the other hand's reference, or two wrists collapsed: the worse one is rejected (prompt
           wrist without fingers, or nothing)
  smooth   wrist path only (fingers as predicted, each hand moved with its wrist); separated stretches and each touch on their
           own, never across the edge of a touch
Writes CLIP_DIR/wilor_wrists.npz: kp2d (n, 2 left / right, 21, 2) full-res px, nan = none; wrist (n, 2, 2); wrist_only (n, 2);
merged (n,).
"""
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


def predict(model, cfg, im, given):
    """given [(box x1 y1 x2 y2, right)] -> [(right, kp (21, 2) full-res px)]; steps of WiLoR's demo.py, box x2."""
    if not given:
        return []
    boxes, right = np.stack([b for b, _ in given]), np.array([r for _, r in given], float)
    out = []
    for b in torch.utils.data.DataLoader(ViTDetDataset(cfg, im, boxes, right, rescale_factor=2.0), batch_size=16, shuffle=False, num_workers=0):
        b = recursive_to(b, "cuda")
        with torch.no_grad():
            o = model(b)
        cam = o["pred_cam"]
        cam[:, 1] = (2 * b["right"] - 1) * cam[:, 1]
        size = b["img_size"].float()
        f = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * size.max()
        t = cam_crop_to_full(cam, b["box_center"].float(), b["box_size"].float(), size, f).cpu().numpy()
        for i in range(len(t)):
            j = o["pred_keypoints_3d"][i].cpu().numpy()
            r = b["right"][i].item()
            j[:, 0] = (2 * r - 1) * j[:, 0]
            p = j + t[i]
            w, h = size[i].cpu().numpy()
            out.append((int(r), np.c_[f.item() * p[:, 0] / p[:, 2] + w / 2, f.item() * p[:, 1] / p[:, 2] + h / 2]))
    return out


def main():
    model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt", cfg_path="./pretrained_models/model_config.yaml")
    model = model.cuda().eval()
    for cd in map(Path, sys.argv[1:]):
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
        for f in range(n):
            good, im = cap.read()
            if not good:
                break
            for s, kp in predict(model, cfg, im, [(b, s) for b, s, *_ in sides[f]]):
                K[f, s] = kp
        C, wo = apply_cuffs(K, sides)
        C[wo & ~prompted] = np.nan                                           # no prompt wrist: blank
        M = np.array([any(e[3] for e in a) for a in sides])
        Z = C.copy()
        Z[M] = np.nan
        S = smooth_wrist(Z)[0]
        S[M] = C[M]
        S = smooth_stretches(S, M)
        np.savez_compressed(cd / "wilor_wrists.npz", kp2d=S, wrist=S[:, :, 0], wrist_only=wo & prompted, merged=M)
        print(cd.name, f"wrists {np.isfinite(S[:, :, 0]).all(-1).sum()}, with fingers {np.isfinite(S).all((2, 3)).sum()}, touching frames {M.sum()}", flush=True)


if __name__ == "__main__":
    main()
