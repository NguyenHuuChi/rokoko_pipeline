#!/usr/bin/env python3
"""hand5 + RTMPose detectors (detectors/, weights models/) on a video; CUDA for RTMPose and no YOLO import.

    PATH=<ffmpeg >= 5>:$PATH python hung5_detect.py VIDEO OUT_DIR       (~/envs/detect, GPU)

  rtm    RTMPose-l body12 ft4c (whole picture, COCO-17: shoulders 5/6, elbows 7/8, wrists 9/10) -> OUT_DIR/rtm_body.npz
  prior  body_prior.py: hand crop prior from RTMPose wrist + elbow (no Rokoko)              -> OUT_DIR/body_prior.npz
  hand5  detect_hands_student.py --no-yolo --track 10 --segments 4 (as pipeline.py)          -> OUT_DIR/hands.npz
         (21 keypoints per hand, kp2d[:, 0] = wrist; hand 0 = left, 1 = right)
"""
import argparse
import sys
import types
from pathlib import Path

import onnxruntime as ort

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG / "detectors"))
import package_paths as PP                                                   # noqa: E402
PP.activate()
import rtmpose_local as R                                                    # noqa: E402


def cuda_init(self, onnx=R.ONNX):
    ort.preload_dlls()                                                       # cuDNN 9 from the venv's nvidia wheels
    self.s = ort.InferenceSession(str(onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    self.inp = self.s.get_inputs()[0].name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rtm-only", action="store_true", help="only run RTMPose + body_prior (skip Hand5)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    R.RTMPose.__init__ = cuda_init
    R.video(a.video, a.out / "rtm_body.npz")
    import cv2
    cap = cv2.VideoCapture(str(a.video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1440
    cap.release()
    sys.argv = ["body_prior.py", str(a.out / "rtm_body.npz"), str(a.out / "body_prior.npz"), "--wh", f"{w},{h}"]
    import body_prior
    body_prior.main()
    if a.rtm_only:
        return
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=lambda *_: None)   # --no-yolo: the model is never called
    import detect_hands_student as D
    D.main([str(a.video), str(a.out / "hands.npz"), "--ckpt", str(PP.MODEL_ROOT / "hand5/hand5_r17d.pt"), "--prior",
            str(a.out / "body_prior.npz"), "--no-yolo", "--track", "10", "--segments", "4"])


if __name__ == "__main__":
    main()
