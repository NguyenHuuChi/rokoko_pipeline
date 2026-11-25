"""RTMPose-l body7 (OpenMMLab, Apache-2.0) on the Mac, onnxruntime, no rtmlib (06/10).

Same pre / post-processing as rtmlib 0.0.13 (what bodypose_modal.py ran): box = the whole picture (padding 1.25),
top-down affine to 288x384, (BGR - mean) / std, SimCC argmax (split 2), score = min(max_x, max_y).  The picture is
rotated 180 deg first (wearer upright); keypoints come back in the video's own pixels.  COCO-17.
    python rtmpose_local.py labels                       -> compares with bodypose_out.npz (the Modal run)
    python rtmpose_local.py video <mp4> <out.npz> [every] -> kp (n, 17, 2), conf (n, 17), frame (n,) for frames 0, every, ...
    [ROI_CSV=<pass-1 body_fix csv>] python rtmpose_local.py frames <mp4> <out.npz> <rows.npy>  (07/10: ROI = wearer box)
    env ORT_COREML=1 = CoreML EP (Mac GPU / ANE, 31 vs 117 ms per frame; 80 test elbows p50/p90 46/144 vs CPU 43/154 px);
        RTM_READER_THREAD=1 = video decoded in a thread (frames mode); RTM_ONNX = another ONNX (fine-tuned elbows).
Model: experiments/fast_pipeline/models/rtmpose/.../end2end.onnx (downloaded 06/10 from download.openmmlab.com).
"""
from __future__ import annotations
from package_paths import PACKAGE_ROOT,SOURCE_ROOT,MODEL_ROOT,SMPLH_MODEL

import json
import sys
import threading
import queue
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ONNX = Path(__import__('os').environ.get('RTM_ONNX',MODEL_ROOT/'rtmpose/rtmpose_l_body12_ft4c.onnx'))
IN_W, IN_H = 288, 384
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class RTMPose:
    def __init__(self, onnx=ONNX):
        so = ort.SessionOptions(); so.intra_op_num_threads = 0
        prov = (["CoreMLExecutionProvider"] if ORT_COREML and "CoreMLExecutionProvider" in ort.get_available_providers() else []) + ["CPUExecutionProvider"]
        self.s = ort.InferenceSession(str(onnx), so, providers=prov)
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, im, box=None):
        """im: BGR picture as filmed (not rotated) -> kp (17, 2) in its pixels, conf (17,).
        box (x0, y0, x1, y1) in the picture's pixels (07/10: wearer ROI, may reach outside the picture); None = whole."""
        H, W = im.shape[:2]
        r = im[::-1, ::-1]
        if box is None:
            c = np.array([W / 2, H / 2]); sw, sh = W * 1.25, H * 1.25
        else:
            x0, y0, x1, y1 = box
            c = np.array([W - 1 - (x0 + x1) / 2, H - 1 - (y0 + y1) / 2]); sw, sh = (x1 - x0) * 1.25, (y1 - y0) * 1.25
        aspect = IN_W / IN_H
        sw, sh = (sw, sw / aspect) if sw > sh * aspect else (sh * aspect, sh)
        k = IN_W / sw
        M = np.array([[k, 0, IN_W / 2 - k * c[0]], [0, k, IN_H / 2 - k * c[1]]], np.float32)
        x = cv2.warpAffine(np.ascontiguousarray(r), M, (IN_W, IN_H), flags=cv2.INTER_LINEAR).astype(np.float32)
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        sx, sy = self.s.run(None, {self.inp: x})
        ix, iy = sx[0].argmax(1), sy[0].argmax(1)
        conf = np.minimum(sx[0].max(1), sy[0].max(1))
        loc = np.stack([ix, iy], 1).astype(np.float64) / 2.0                     # simcc split ratio 2
        kp = loc / np.array([IN_W, IN_H]) * np.array([sw, sh]) + c - np.array([sw, sh]) / 2
        kp = np.stack([W - 1 - kp[:, 0], H - 1 - kp[:, 1]], 1)                    # back from rot180
        return kp, conf


ORT_COREML = __import__("os").environ.get("ORT_COREML") == "1"
ROI_JOINTS = ("leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm", "leftHand", "rightHand")


def wearer_boxes(csv, n, pad=0.15, min_wh=(640, 480), clip=True, wh=(1920, 1440)):
    """07/10: per-row wearer box from a body_fix pass-1 CSV (Rokoko shoulders + IK elbows + wrists, ultra-wide px):
    bounding box of those points (wrists optional), padded by pad x size, at least min_wh; rows without points -> None.
    Keeps RTMPose on the wearer instead of a bystander next to them (box = whole picture).  clip: box cut to the picture
    (shoulders are below / beside it with the head camera); on the 80 clicked hard elbows: whole picture p50/p90 51/317 px,
    box 72/349, box clipped 43/155 (pad 0.4 clipped 51/182) -> clipped, pad 0.15."""
    import pandas as pd
    cols = [f"{j}_{c}" for j in ROI_JOINTS for c in "uv"]
    P = pd.read_csv(csv, usecols=cols)[cols].to_numpy(float).reshape(-1, len(ROI_JOINTS), 2)[:n]
    out = []
    for p in P:
        p = p[np.isfinite(p).all(1)]
        if len(p) < 3:
            out.append(None); continue
        x0, y0 = p.min(0); x1, y1 = p.max(0); w, h = x1 - x0, y1 - y0
        w2, h2 = max(w * (1 + 2 * pad), min_wh[0]) / 2, max(h * (1 + 2 * pad), min_wh[1]) / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        b = (cx - w2, cy - h2, cx + w2, cy + h2)
        if clip:
            b = (max(b[0], 0), max(b[1], 0), min(b[2], wh[0]), min(b[3], wh[1]))
            b = b if b[2] - b[0] > 300 and b[3] - b[1] > 300 else None
        out.append(b)
    return out + [None] * (n - len(out))


def labels_check():
    js = json.loads((HERE / "bodypose_out.json").read_text()); Z = np.load(HERE / "bodypose_out.npz")
    m = RTMPose(); d = []; t0 = time.perf_counter()
    for i, fid in enumerate(js["ids"]):
        im = cv2.resize(cv2.imread(str(HERE / "frames" / f"{fid}.jpg")), (1920, 1440), interpolation=cv2.INTER_LINEAR)
        kp, cf = m(im)
        d.append(np.abs(kp - Z["rtmpose-l_kp"][i]).max())
    print(f"max |local - Modal| px: p50 {np.median(d):.2f}  max {np.max(d):.2f}   {1e3 * (time.perf_counter() - t0) / len(d):.0f} ms/frame")


def video(mp4, out, every=1):
    every = int(every); cap = cv2.VideoCapture(str(mp4)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    q: queue.Queue = queue.Queue(64)

    def reader():
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if i % every == 0:
                q.put((i, im))
            i += 1
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()
    m = RTMPose(); K, C, F = [], [], []; t0 = time.perf_counter()
    while (it := q.get()) is not None:
        i, im = it; kp, cf = m(im); K.append(kp); C.append(cf); F.append(i)
        if len(F) % 500 == 0:
            print(f"{i}/{n}  {len(F) / (time.perf_counter() - t0):.1f} fps", flush=True)
    dt = time.perf_counter() - t0
    np.savez_compressed(out, kp=np.array(K, np.float32), conf=np.array(C, np.float32), frame=np.array(F), n_frames=n)
    print(f"{len(F)} frames in {dt:.0f} s ({len(F) / dt:.1f} fps) -> {out}")


def frames(mp4, out, rows_npy):
    """07/10: RTMPose only on the rows in rows_npy (bool mask or ids), decoding forward (grab() skips the others).
    Output has ALL rows (frame = 0..n-1) with kp NaN / conf 0 where not run, so arm_direct3.py works unchanged."""
    cap = cv2.VideoCapture(str(mp4)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    R = np.load(rows_npy); want = np.zeros(n, bool)
    if R.dtype == bool:
        want[:len(R)] = R[:n]
    else:
        want[R[R < n]] = True
    K = np.full((n, 17, 2), np.nan, np.float32); C = np.zeros((n, 17), np.float32)
    roi = __import__("os").environ.get("ROI_CSV")                        # 07/10: wearer box per row (wearer_boxes)
    boxes = wearer_boxes(roi, n) if roi else [None] * n
    m = RTMPose(); t0 = time.perf_counter(); done = 0

    def frames_iter():                                # decode in order; grab() skips the rows not run
        for i in range(n):
            if not want[i]:
                cap.grab(); continue
            ok, im = cap.read()
            if not ok:
                return
            yield i, im
    it = frames_iter()
    if __import__("os").environ.get("RTM_READER_THREAD") == "1":   # 07/10: decode in a thread, overlapped with the model
        q: queue.Queue = queue.Queue(32)
        threading.Thread(target=lambda: ([q.put(x) for x in frames_iter()], q.put(None)), daemon=True).start()
        it = iter(q.get, None)
    for i, im in it:
        K[i], C[i] = m(im, boxes[i]); done += 1
        if done % 500 == 0:
            print(f"{i}/{n}  {done} run  {done / (time.perf_counter() - t0):.1f} fps", flush=True)
    dt = time.perf_counter() - t0
    np.savez_compressed(out, kp=K, conf=C, frame=np.arange(n), n_frames=n, ran=want)
    print(f"{done} of {n} frames ({100 * done / n:.0f}%) in {dt:.0f} s -> {out}  [providers {m.s.get_providers()[0]}, "
          f"reader thread {__import__('os').environ.get('RTM_READER_THREAD') == '1'}]")


if __name__ == "__main__":
    if sys.argv[1] == "labels":
        labels_check()
    elif sys.argv[1] == "frames":
        frames(*sys.argv[2:5])
    else:
        video(*sys.argv[2:5])
