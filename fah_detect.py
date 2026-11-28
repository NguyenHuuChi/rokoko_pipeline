#!/usr/bin/env python3
"""FAH YOLOv8-seg hand detector + RTMPose left/right locking for M3a (`fah`) and M3b (`fasam`).

    python fah_detect.py CLIP_DIR [CLIP_DIR ...] [--mode fasam|fah] [--weights PATH]

Reads CLIP_DIR/{clip.mp4, rtm_body.npz}:
  fasam  (default, M3b) saves CLIP_DIR/fah_prompts.npz with apart L/R segmentation anchor masks (768x576) for sam2_clipmask.py
  fah    (M3a) locks L/R boxes with RTMPose + IoU, fills short gaps (<= 3 frames), SavGol-smooths -> CLIP_DIR/fah_boxes.npz
"""
import argparse
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # ultralytics 8.1.34 calls torch.load without weights_only

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import settings  # noqa: E402

TW, TH = 640, 480  # SAM 2.1 tracker resolution (matching sam2_clipmask.py)


def box_iou(a, b):
    """Intersection-over-Union of two [x1, y1, x2, y2] boxes."""
    if a is None or b is None or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = (
        max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        + max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        - inter
    )
    return inter / ua if ua > 1e-6 else 0.0


def load_rtm_wrists(rtm_path: Path, n_frames: int, min_conf: float = 0.3):
    """Return (n_frames, 2, 2) left/right RTMPose wrist px (COCO 9, 10), nan if missing."""
    wr = np.full((n_frames, 2, 2), np.nan, np.float32)
    if not rtm_path.is_file():
        return wr
    z = np.load(rtm_path)
    fr = z["frame"].astype(int)
    k = (fr >= 0) & (fr < n_frames)
    fr = fr[k]
    kp = z["kp"][k][:, 9:11].astype(np.float32)
    cf = z["conf"][k][:, 9:11] >= min_conf
    wr[fr] = np.where(cf[..., None], kp, np.nan)
    return wr


class _UltralyticsDetector:
    """Fallback adapter when third_party/FAH is not cloned locally, matching WiLoRHandDetector's API."""

    def __init__(self, weights: Path, device: str, conf: float, iou: float, imgsz: int):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        if self.model.task != "segment" and hasattr(self.model.model, "model") and len(self.model.model.model):
            if "Segment" in type(self.model.model.model[-1]).__name__:
                self.model = YOLO(str(weights), task="segment")
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.seg = getattr(self.model, "task", "detect") == "segment"
        self._last = None

    def __call__(self, img_bgr):
        det = self.model(
            img_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            **({"retina_masks": True} if self.seg else {}),
        )[0]
        boxes = det.boxes.xyxy.cpu().numpy().astype(np.float32) if len(det.boxes) else np.zeros((0, 4), np.float32)
        scores = det.boxes.conf.cpu().numpy().astype(np.float32) if len(det.boxes) else np.zeros((0,), np.float32)
        is_right = det.boxes.cls.cpu().numpy().astype(bool) if len(det.boxes) else np.zeros((0,), bool)
        if self.seg:
            if getattr(det, "masks", None) is not None and det.masks is not None and len(det.masks.data):
                m = det.masks.data.cpu().numpy() > 0.5
            else:
                m = np.zeros((len(boxes), *img_bgr.shape[:2]), bool)
                for k, b in enumerate(boxes):
                    x0, y0, x1, y1 = np.clip(np.round(b).astype(int), 0, [img_bgr.shape[1], img_bgr.shape[0], img_bgr.shape[1], img_bgr.shape[0]])
                    if x1 > x0 and y1 > y0:
                        m[k, y0:y1, x0:x1] = True
            self._last = (boxes, m, img_bgr.shape[:2])
        else:
            m = np.zeros((len(boxes), *img_bgr.shape[:2]), bool)
            for k, b in enumerate(boxes):
                x0, y0, x1, y1 = np.clip(np.round(b).astype(int), 0, [img_bgr.shape[1], img_bgr.shape[0], img_bgr.shape[1], img_bgr.shape[0]])
                if x1 > x0 and y1 > y0:
                    m[k, y0:y1, x0:x1] = True
            self._last = (boxes, m, img_bgr.shape[:2])
        return boxes, scores, is_right

    def masks(self, keep=None):
        _, m, _ = self._last
        idx = np.arange(len(m)) if keep is None else np.asarray(keep, dtype=int).reshape(-1)
        return m[idx]


def load_detector(weights: Path, device: str, conf: float, iou: float, imgsz: int):
    """Load official FAH WiLoRHandDetector if third_party/FAH exists, else ultralytics adapter."""
    if not weights.is_file():
        raise FileNotFoundError(
            f"FAH detector weights not found: {weights}\n"
            "Run `bash download_fah.sh` (or set FAH_DET=/path/to/detector.pt)."
        )
    fah_dir = Path(settings.FAH).resolve()
    if (fah_dir / "fah" / "detector.py").is_file() and (fah_dir / "fah" / "yolo.py").is_file():
        fah_pkg = types.ModuleType("fah")
        fah_pkg.__path__ = [str((fah_dir / "fah").resolve())]
        sys.modules["fah"] = fah_pkg
        from fah.detector import WiLoRHandDetector  # noqa: E402

        use_fast = (fah_dir / "fah" / "cuda_graph.py").is_file()
        return WiLoRHandDetector(
            weights=str(weights.resolve()),
            device=device,
            conf=conf,
            iou=iou,
            fast=use_fast,
            half=False,
            imgsz=imgsz,
        )
    return _UltralyticsDetector(weights.resolve(), device=device, conf=conf, iou=iou, imgsz=imgsz)


def match_lr_frame(boxes, scores, is_right, wr_f, prev_box, prev_age, w1=1.0, w2=1.5, w3=1.0):
    """Assign frame detections to left (0) / right (1) tracks using RTMPose wrist + IoU + side score."""
    m = len(boxes)
    assigned = {}
    if m > 0:
        cost = np.full((m, 2), 1e6, np.float64)
        for i in range(m):
            b = boxes[i]
            cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
            side_i = int(bool(is_right[i]))
            for s in (0, 1):
                has_wr = bool(np.isfinite(wr_f[s]).all())
                d = float(np.hypot(cx - wr_f[s, 0], cy - wr_f[s, 1])) if has_wr else 0.0
                pb = prev_box[s] if prev_age[s] <= 5 else None
                iou = box_iou(b, pb) if pb is not None else 0.0
                if has_wr and d > 350.0 and iou < 0.05:
                    continue
                cost[i, s] = (
                    (w1 * (d / 100.0) if has_wr else 0.0)
                    + (w2 * (1.0 - iou) if pb is not None else 0.0)
                    + (w3 if side_i != s else 0.0)
                )
        ri, ci = linear_sum_assignment(cost)
        for i, s in zip(ri, ci):
            if cost[i, s] < 1e5:
                assigned[int(s)] = (int(i), boxes[i].astype(np.float32), float(scores[i]))
                prev_box[s] = boxes[i].copy()
                prev_age[s] = 0
    for s in (0, 1):
        if s not in assigned:
            prev_age[s] += 1
    return assigned


def lock_and_smooth_boxes(matched_boxes, wr, max_gap=3, win=5):
    """Fill gaps <= max_gap frames and SavGol-smooth square (cx, cy, size) trajectories -> (n, 2, 4) xyxy."""
    n = len(matched_boxes)
    cxs = np.full((n, 2, 3), np.nan, np.float64)
    raw_valid = np.isfinite(matched_boxes[..., 0])

    for s in (0, 1):
        ok = raw_valid[:, s]
        cxs[ok, s, 0] = 0.5 * (matched_boxes[ok, s, 0] + matched_boxes[ok, s, 2])
        cxs[ok, s, 1] = 0.5 * (matched_boxes[ok, s, 1] + matched_boxes[ok, s, 3])
        cxs[ok, s, 2] = np.maximum(matched_boxes[ok, s, 2] - matched_boxes[ok, s, 0], matched_boxes[ok, s, 3] - matched_boxes[ok, s, 1])

        idx = np.flatnonzero(np.isfinite(cxs[:, s, 0]))
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if 1 <= gap <= max_gap:
                for k in range(1, gap + 1):
                    t = k / (gap + 1.0)
                    cxs[a + k, s] = (1.0 - t) * cxs[a, s] + t * cxs[b, s]

        if len(idx) > 0:
            first = int(idx[0])
            for k in range(1, min(max_gap, first) + 1):
                f_cur = first - k
                if np.isfinite(wr[f_cur, s]).all() and np.isfinite(wr[first, s]).all():
                    cxs[f_cur, s, :2] = cxs[first, s, :2] + (wr[f_cur, s] - wr[first, s])
                    cxs[f_cur, s, 2] = cxs[first, s, 2]
            last = int(idx[-1])
            for k in range(1, min(max_gap, n - 1 - last) + 1):
                f_cur = last + k
                if np.isfinite(wr[f_cur, s]).all() and np.isfinite(wr[last, s]).all():
                    cxs[f_cur, s, :2] = cxs[last, s, :2] + (wr[f_cur, s] - wr[last, s])
                    cxs[f_cur, s, 2] = cxs[last, s, 2]

        ok2 = np.isfinite(cxs[:, s, 0])
        f_idx = 0
        while f_idx < n:
            if not ok2[f_idx]:
                f_idx += 1
                continue
            j_idx = f_idx
            while j_idx < n and ok2[j_idx]:
                j_idx += 1
            seg_len = j_idx - f_idx
            if seg_len >= win:
                for d_i in range(3):
                    cxs[f_idx:j_idx, s, d_i] = savgol_filter(cxs[f_idx:j_idx, s, d_i], window_length=win, polyorder=2)
            elif seg_len >= 3:
                for d_i in range(3):
                    cxs[f_idx:j_idx, s, d_i] = savgol_filter(cxs[f_idx:j_idx, s, d_i], window_length=3, polyorder=1)
            f_idx = j_idx

    out_boxes = np.full((n, 2, 4), np.nan, np.float32)
    for s in (0, 1):
        ok = np.isfinite(cxs[:, s, 0])
        sz = np.maximum(cxs[ok, s, 2], 20.0)
        out_boxes[ok, s, 0] = cxs[ok, s, 0] - 0.5 * sz
        out_boxes[ok, s, 1] = cxs[ok, s, 1] - 0.5 * sz
        out_boxes[ok, s, 2] = cxs[ok, s, 0] + 0.5 * sz
        out_boxes[ok, s, 3] = cxs[ok, s, 1] + 0.5 * sz
    return out_boxes, raw_valid, np.isfinite(out_boxes[..., 0])


def process_clip(det, cd: Path, mode: str, min_anchor_conf: float = 0.45):
    """Run FAH detector + L/R lock on a clip directory for `fah` (M3a) or `fasam` (M3b)."""
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(cd / "clip.mp4"))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    n = len(frames)
    wr = load_rtm_wrists(cd / "rtm_body.npz", n)

    matched = np.full((n, 2, 4), np.nan, np.float32)
    scores_arr = np.full((n, 2), np.nan, np.float32)
    prev_box, prev_age = [None, None], [999, 999]
    anchor_frames, anchor_masks = [], []

    for f, im in enumerate(frames):
        boxes, scores, is_right = det(im)
        assigned = match_lr_frame(boxes, scores, is_right, wr[f], prev_box, prev_age)
        for s, (_, box_s, sc_s) in assigned.items():
            matched[f, s] = box_s
            scores_arr[f, s] = sc_s

        if mode == "fasam" and 0 in assigned and 1 in assigned:
            bL, bR = assigned[0][1], assigned[1][1]
            if box_iou(bL, bR) < 0.12 and min(assigned[0][2], assigned[1][2]) >= min_anchor_conf:
                ms = det.masks([assigned[0][0], assigned[1][0]])
                mL = cv2.resize(ms[0].astype(np.uint8), (TW, TH), interpolation=cv2.INTER_NEAREST) > 0
                mR = cv2.resize(ms[1].astype(np.uint8), (TW, TH), interpolation=cv2.INTER_NEAREST) > 0
                if mL.sum() >= 60 and mR.sum() >= 60:
                    anchor_frames.append(f)
                    anchor_masks.append(np.stack([mL, mR], axis=0))

    dt = time.perf_counter() - t0
    if mode == "fah":
        sm_boxes, raw_valid, fill_valid = lock_and_smooth_boxes(matched, wr)
        np.savez_compressed(
            cd / "fah_boxes.npz",
            boxes=sm_boxes,
            raw_valid=raw_valid,
            fill_valid=fill_valid,
            scores=scores_arr,
            mode=np.array("fah"),
        )
        print(
            f"[FAH M3a] {cd.name}: {n}f in {dt:.1f}s | raw L={100 * raw_valid[:, 0].mean():.1f}% R={100 * raw_valid[:, 1].mean():.1f}% "
            f"-> filled L={100 * fill_valid[:, 0].mean():.1f}% R={100 * fill_valid[:, 1].mean():.1f}%",
            flush=True,
        )
    else:
        raw_valid = np.isfinite(matched[..., 0])
        am = np.stack(anchor_masks, axis=0) if anchor_masks else np.zeros((0, 2, TH, TW), bool)
        np.savez_compressed(
            cd / "fah_prompts.npz",
            anchor_frames=np.array(anchor_frames, np.int32),
            anchor_masks=am,
            raw_boxes=matched,
            raw_valid=raw_valid,
            scores=scores_arr,
            tracker_wh=np.array([TW, TH], np.int32),
            mode=np.array("fasam"),
        )
        print(
            f"[FAH M3b prompts] {cd.name}: {n}f in {dt:.1f}s | raw L={100 * raw_valid[:, 0].mean():.1f}% R={100 * raw_valid[:, 1].mean():.1f}% "
            f"| clean anchors={len(anchor_frames)}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--mode", choices=["fasam", "fah"], default="fasam")
    ap.add_argument("--weights", type=Path, default=Path(settings.FAH_DET))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    det = load_detector(a.weights, device=dev, conf=a.conf, iou=a.iou, imgsz=a.imgsz)
    for cd in a.clips:
        process_clip(det, cd, mode=a.mode)


if __name__ == "__main__":
    main()
