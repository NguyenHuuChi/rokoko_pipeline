#!/usr/bin/env python3
"""Wearer hand masks on a cut clip: the SAM 3 rule (handseg_pilot.wearer) on every frame, SAM 2.1-L track from the seed
frames, final hand mask = SAM 3 "hand" / "glove" instances inside the (dilated) track.

    python handseg_clipmask.py CLIP_DIR [CLIP_DIR ...]          (~/envs/handseg, GPU)

CLIP_DIR as hung5_detect.py (clip.mp4).  Writes CLIP_DIR/masks.npz: hand, wearer (n, H/4, W/4) bool; rule frames
without a track keep the rule's own wearer instances; no seed in the clip -> empty masks (no constraint).
track_clip() is also used by handseg_gen_clips.py (temporal segmentation labels).
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handseg_pilot import MIN_IN, SAM3_CKPT, wearer                         # noqa: E402

import settings                                                             # noqa: E402
SAM2_CKPT = settings.SAM2_LARGE
TW, TH = 960, 720                                                            # tracker frame size

NO_PTS = {"left": np.zeros((0, 2)), "right": np.zeros((0, 2))}


def models():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    proc = Sam3Processor(build_sam3_image_model(checkpoint_path=SAM3_CKPT, load_from_HF=False), confidence_threshold=0.4)
    return proc, build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", SAM2_CKPT)


def track_clip(proc, pred, ims, wd):
    """ims: consecutive BGR frames.  Returns per frame (wearer, hand, other people's) full-res bool masks and the number
    of seed frames.  wd: scratch dir for the tracker's frames (removed)."""
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    rule = []
    for k, im in enumerate(ims):
        cv2.imwrite(str(wd / f"{k:05d}.jpg"), cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA))
        keep, _, _, me, ni, inst = wearer(proc, im, NO_PTS)
        hand = [m for i, m in enumerate(inst) if not ni["hand"] <= i < ni["hand"] + ni["arm"]]   # hand + glove prompts
        rule.append((keep, me, inst, hand))
    H, W = rule[0][0].shape
    seeds = [k for k, (keep, me, _, _) in enumerate(rule) if me is not None and keep.any()]
    track = {}
    if seeds:
        st = pred.init_state(video_path=str(wd))
        for k in seeds:
            pred.add_new_mask(st, frame_idx=k, obj_id=1, mask=cv2.resize(rule[k][0].astype(np.uint8), (TW, TH),
                                                                          interpolation=cv2.INTER_NEAREST) > 0)
        for rev in (False, True):
            for k, _, lg in pred.propagate_in_video(st, start_frame_idx=seeds[0], reverse=rev):
                track[k] = (lg[0, 0] > 0).cpu().numpy()
        pred.reset_state(st)
    shutil.rmtree(wd)
    out = []
    for k, (keep, _, inst, hand) in enumerate(rule):
        T = cv2.dilate(cv2.resize(track[k].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST), np.ones((25, 25))) > 0 \
            if k in track else None
        inside = (lambda m: (m & T).sum() >= MIN_IN * m.sum()) if T is not None else (lambda m: (m & keep).sum() >= MIN_IN * m.sum())
        h, w, o = (np.zeros((H, W), bool) for _ in range(3))
        for m in inst:
            if inside(m):
                w |= m
            else:
                o |= m
        for m in hand:
            if inside(m):
                h |= m
        out.append((w, h, o & ~w))
    return out, len(seeds)


def main():
    proc, pred = models()
    for cd in map(Path, sys.argv[1:]):
        cap = cv2.VideoCapture(str(cd / "clip.mp4"))
        ims = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            ims.append(im)
        res, ns = track_clip(proc, pred, ims, cd / "frames_tmp")
        H, W = ims[0].shape[:2]
        q = lambda m: cv2.resize(m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST) > 0
        HM, WM = [q(h) for _, h, _ in res], [q(w) for w, _, _ in res]
        np.savez_compressed(cd / "masks.npz", hand=np.stack(HM), wearer=np.stack(WM))
        print(cd.name, f"{ns} seeds / {len(ims)} frames, hand px {np.mean(HM):.4f}, frames with hand {np.mean([m.any() for m in HM]):.2f}", flush=True)


if __name__ == "__main__":
    main()
