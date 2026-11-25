#!/usr/bin/env python3
"""Pilot: SAM 3 text-prompted hand/arm masks on sampled frames, keep the wearer's, score our labels against them.

    python handseg_pilot.py OUT_DIR SESSION_DIR [SESSION_DIR ...] [--n 500]      (~/envs/handseg, GPU)

Per session, N frames evenly over <session>/*_sync.mp4:
  SAM 3 image model, prompts "hand", "arm", "glove" -> instances; "person" -> people
  depth   LiDAR png of the same raw frame (camera/materialized manifest source_real_frame_id -> depth_frames.csv
          primary_frame_index), mm, confidence >= 1. It is aligned to the WIDE camera, the video is the ULTRAWIDE:
          back-project, move by calibration.json transforms.ultrawide_to_reference, project -> covers only the
          middle ~half of the video (no depth near the borders, where the wearer's arms usually are)
  wearer  the person instance covering most of the bottom-centre box (BOX: the wearer's torso under the head-mounted
          camera; SAM 3 gives torso + both arms + gloved hands of the wearer as one person, bystanders as others);
          keep the hand / arm / glove instances >= MIN_IN inside it (dilated). No person on the box (camera looking
          away from the body): frame not dumped. Our labels only flag rejected instances as unsure (see _n.png), the
          wearer mask itself does not use them
  score   share of our projected forearm + hand points (camera/joints.csv u,v; inside the image) that fall in the
          wearer mask -> camera / sync / hand-fit check
Writes OUT_DIR/<session>/{overlay.mp4, frames.csv} and <session>/handseg_pilot/masks.npz (wearer mask, 1/4 res, packed).
--dump DIR: also training triplets for handseg_train.py, 640x480: DIR/<session>_<frame>.jpg (RGB), _d.png (depth on the
video raster, uint16 mm, 0 = none), _m.png (wearer mask: 0 background, 1 sleeve / arm, 2 hand or glove = SAM 3 "hand" /
"glove" instances inside it), _n.png (1 = rejected hand / arm instances = other
people's, hard negatives; 2 = unsure: outside the wearer person but entering from the bottom / lower sides with our
projected hand / forearm points on it -> may be the wearer's, ignored in training); split of the frame's chunk (split/chunks.csv) in frames.csv.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import settings
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

FF = settings.FFMPEG
SAM3_CKPT = settings.SAM3
PROMPTS = ["hand", "arm", "glove"]
BOX = (0.8, 0.2, 0.8)                                                       # rows below 80 %, columns 20-80 %
MIN_IN = 0.5
PW = 640
ARM = {s: [f"{s}LowerArm", f"{s}Hand"] for s in ("left", "right")}        # elbow, wrist
TIPS = [f"{d}Tip" for d in ("Thumb", "Index", "Middle", "Ring", "Little")]


def arm_points(r, s, W, H):
    """Our projected forearm (elbow -> wrist, 10 samples) + fingertips of side s, those inside the image."""
    uv = lambda n: np.array([r.get(f"{n}_u", np.nan), r.get(f"{n}_v", np.nan)], float)
    e, w = uv(ARM[s][0]), uv(ARM[s][1])
    P = [e + (w - e) * t for t in np.linspace(0, 1, 10)] + [uv(f"{s}{t}") for t in TIPS]
    P = np.array([p for p in P if np.isfinite(p).all() and 0 <= p[0] < W and 0 <= p[1] < H])
    return P.reshape(-1, 2)


def depth_to_video(sd, W, H):
    """LiDAR png (wide-camera aligned) -> depth on the video raster (mm, 0 = none), splat at 1/4 res: the ultrawide video, or
    the wide (primary) video of Wide-only takes."""
    c = json.load(open(sd / "calibration.json"))
    Kd = np.array(c["depth"]["K_row_major"]).reshape(3, 3)
    if "ultrawide_camera" in c:
        cam = c["ultrawide_camera"]
        T = np.array(c["transforms"]["ultrawide_to_reference"]["matrix_3x4_row_major"]).reshape(3, 4)  # ref <- uw, mm
        R, t = T[:, :3], T[:, 3]
    else:                                     # Wide-only take (ios_arkit_wide_v1): the video IS the primary camera
        cam = c["primary_camera"]
        tr = c["depth"]["depth_to_primary_transform"]                         # primary <- depth, m
        Rd, td = np.array(tr["rotation_row_major"], float).reshape(3, 3), np.array(tr["translation_metres"], float) * 1000
        R, t = Rd.T, -Rd.T @ td                                               # same form as above: X = (X_d - t) @ R
    Ku = np.array(cam["K_encoded_row_major"]).reshape(3, 3)
    if cam["sensor_to_encoded_operation"].startswith("rotate_180"):           # extrinsic is in sensor axes
        F = np.diag([-1.0, -1, 1])
        R, t = F @ R @ F, F @ t
    v, u = np.mgrid[0:192, 0:256]
    ray = np.stack([u.ravel(), v.ravel(), np.ones(u.size)]) .T @ np.linalg.inv(Kd).T
    def f(d):
        X = (ray * d.reshape(-1, 1) - t) @ R                                     # uw camera, mm
        ok = (d.ravel() > 0) & (X[:, 2] > 1)
        p = X[ok] @ Ku.T
        q = (p[:, :2] / p[:, 2:] / 4).astype(int)
        g = np.zeros((H // 4, W // 4), np.float32)
        i = (q[:, 0] >= 0) & (q[:, 0] < W // 4) & (q[:, 1] >= 0) & (q[:, 1] < H // 4)
        g[q[i, 1], q[i, 0]] = X[ok][i, 2]
        g = np.where(g > 0, g, cv2.dilate(g, np.ones((3, 3))))                   # close the splat holes
        return cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
    return f


def wearer(proc, im, P):
    """SAM 3 on one BGR frame -> wearer / other people's / unsure masks, wearer person, counts, hand+arm+glove instances.
    P: our projected hand / forearm points per side (only to flag unsure)."""
    H, W = im.shape[:2]
    st = proc.set_image(Image.fromarray(im[..., ::-1]))
    ninst, inst = {}, []
    for pr in PROMPTS + ["person"]:
        o = proc.set_text_prompt(state=st, prompt=pr)
        M = o["masks"].squeeze(1).cpu().numpy().astype(bool)
        ninst[pr] = len(M)
        inst += [(pr, m) for m in M]
    box = np.zeros((H, W), bool)
    box[int(BOX[0] * H):, int(BOX[1] * W):int(BOX[2] * W)] = True
    ppl = [m for pr, m in inst if pr == "person"]
    cov = [(m & box).sum() for m in ppl]
    me = cv2.dilate(ppl[int(np.argmax(cov))].astype(np.uint8), np.ones((25, 25))) > 0 if ppl and max(cov) > 0 \
        else None                                     # the wearer's person mask
    edge = np.zeros((H, W), bool)                     # where the wearer's own arms enter the image
    edge[-H // 50:] = True
    edge[int(0.6 * H):, :W // 50] = edge[int(0.6 * H):, -W // 50:] = True
    pts = np.concatenate([q for q in P.values()]).astype(int)
    keep, drop, unsure = (np.zeros((H, W), bool) for _ in range(3))
    for pr, m in inst:
        if pr != "person":
            if me is not None and (m & me).sum() >= MIN_IN * m.sum():
                keep |= m
            elif (m & edge).any() and len(pts) and m[pts[:, 1], pts[:, 0]].any():
                unsure |= m                           # may be the wearer's (person split / torso unseen)
            else:
                drop |= m
    unsure &= ~keep
    drop &= ~keep & ~unsure
    return keep, drop, unsure, me, ninst, [m for pr, m in inst if pr != "person"]


def session(proc, sd, out, n, dump, chunks, frames=None):
    vid = next(sd.glob("*_sync.mp4"))
    man = pd.read_csv(sd / "camera/materialized/filtered_synced_manifest.csv")
    dep = pd.read_csv(sd / "camera/session/depth/depth_frames.csv").set_index("primary_frame_index")["index"]
    dep = dep[~dep.index.duplicated()]                                       # some sessions repeat a frame index
    j = pd.read_csv(sd / "camera/joints.csv").set_index("output_row_id")
    cap = cv2.VideoCapture(str(vid))
    F = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    od = out / sd.name
    od.mkdir(parents=True, exist_ok=True)
    enc, rows, masks, warp = None, [], {}, None
    ch = chunks[chunks.session == sd.name[8:23]]
    t0 = time.time()
    for f in frames if frames else np.linspace(0, F - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            continue
        H, W = im.shape[:2]
        raw = int(man.source_real_frame_id.iloc[f])
        D = np.zeros((H, W), np.float32)
        warp = warp or depth_to_video(sd, W, H)
        if raw in dep.index:
            p = sd / f"camera/session/depth/{int(dep[raw]):06d}"
            d, c = cv2.imread(f"{p}.png", -1), cv2.imread(f"{p}_confidence.png", -1)
            D = warp(np.where(c >= 1, d, 0).astype(np.float32))
        r = j.loc[f] if f in j.index else pd.Series(dtype=float)
        P = {s: arm_points(r, s, W, H) for s in ARM}
        keep, drop, unsure, me, ninst, inst = wearer(proc, im, P)
        hand = np.zeros_like(keep)                                          # "hand" / "glove" instances (not "arm")
        for i, m in enumerate(inst):
            if not ninst["hand"] <= i < ninst["hand"] + ninst["arm"]:
                hand |= m
        hand &= keep
        sc = {s: (keep[q[:, 1].astype(int), q[:, 0].astype(int)].mean() if len(q) else np.nan) for s, q in P.items()}
        rows.append(dict(frame=f, raw_frame=raw, n_hand=ninst["hand"], n_arm=ninst["arm"], n_glove=ninst["glove"],
                         wearer_found=me is not None,
                         wearer_px=keep.mean(), dropped_px=drop.mean(), unsure_px=unsure.mean(), in_mask_left=sc["left"],
                         in_mask_right=sc["right"], L_source=r.get("L_source", ""), R_source=r.get("R_source", "")))
        sp = ch.split[(ch.t0 <= f / fps) & (f / fps <= ch.t1)]
        rows[-1]["split"] = sp.iloc[0] if len(sp) else ""
        if dump and rows[-1]["wearer_found"]:
            nm = dump / f"{sd.name[8:23]}_{f:06d}"
            cv2.imwrite(f"{nm}.jpg", cv2.resize(im, (640, 480), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(f"{nm}_d.png", cv2.resize(D, (640, 480), interpolation=cv2.INTER_NEAREST).astype(np.uint16))
            cv2.imwrite(f"{nm}_m.png", cv2.resize(keep.astype(np.uint8) + hand, (640, 480), interpolation=cv2.INTER_NEAREST))
            cv2.imwrite(f"{nm}_n.png", cv2.resize(drop.astype(np.uint8) + 2 * unsure, (640, 480), interpolation=cv2.INTER_NEAREST))
        masks[f] = np.packbits(cv2.resize(keep.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST))
        # overlay: video | depth, wearer mask green, rejected red, unsure orange, our forearm + tips points (yellow in, magenta out)
        k = PW / W
        A = cv2.resize(im, (PW, round(H * k)))
        kk, dd = cv2.resize(keep.astype(np.uint8), A.shape[1::-1]) > 0, cv2.resize(drop.astype(np.uint8), A.shape[1::-1]) > 0
        A[kk] = (0.5 * A[kk] + (0, 127, 0)).astype(np.uint8)
        A[dd & ~kk] = (0.5 * A[dd & ~kk] + (0, 0, 127)).astype(np.uint8)
        uu = cv2.resize(unsure.astype(np.uint8), A.shape[1::-1]) > 0
        A[uu] = (0.5 * A[uu] + (0, 110, 127)).astype(np.uint8)  # unsure: orange
        if me is not None:                                # wearer person, white outline
            cs, _ = cv2.findContours(cv2.resize(me.astype(np.uint8), A.shape[1::-1]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(A, cs, -1, (255, 255, 255), 1)
        for s, q in P.items():
            for u, v in q:
                cv2.circle(A, (int(u * k), int(v * k)), 3, (0, 255, 255) if keep[int(v), int(u)] else (255, 0, 255), -1)
        B = cv2.applyColorMap(cv2.convertScaleAbs(cv2.resize(D, A.shape[1::-1]), alpha=255 / 3000), cv2.COLORMAP_TURBO)
        B[cv2.resize(D, A.shape[1::-1]) == 0] = 0
        B[kk] = (0.5 * B[kk] + 127).astype(np.uint8)
        t = f"{sd.name[8:23]} f{f}  hand:{ninst['hand']} arm:{ninst['arm']}  in-mask L {sc['left']:.2f} R {sc['right']:.2f}"
        fr = np.hstack([A, B])
        cv2.rectangle(fr, (0, 0), (fr.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(fr, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if enc is None:
            enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                    f"{fr.shape[1]}x{fr.shape[0]}", "-r", "2", "-i", "-", "-c:v", "libx264",
                                    "-crf", "23", "-pix_fmt", "yuv420p", str(od / "overlay.mp4")], stdin=subprocess.PIPE)
        enc.stdin.write(fr.tobytes())
    enc.stdin.close()
    enc.wait()
    df = pd.DataFrame(rows)
    df.to_csv(od / "frames.csv", index=False)
    (sd / "handseg_pilot").mkdir(exist_ok=True)
    np.savez_compressed(sd / "handseg_pilot/masks.npz", frames=np.array(list(masks)), shape=(H // 4, W // 4),
                        packed=np.stack(list(masks.values())))
    print(sd.name, f"fps {fps:.1f}  {(time.time() - t0) / len(df):.2f} s/frame", df[["n_hand", "n_arm", "wearer_px", "in_mask_left", "in_mask_right"]]
          .describe().loc[["mean", "50%"]].round(3).to_string(), sep="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("sessions", type=Path, nargs="+")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--frames", type=lambda x: [int(v) for v in x.split(",")], help="these frames instead of --n")
    ap.add_argument("--root", default=settings.OUT_ROOT)
    a = ap.parse_args()
    chunks = pd.read_csv(f"{a.root}/split/chunks.csv", dtype={"session": str})
    if a.dump:
        a.dump.mkdir(parents=True, exist_ok=True)
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    proc = Sam3Processor(build_sam3_image_model(checkpoint_path=SAM3_CKPT, load_from_HF=False), confidence_threshold=0.4)
    for sd in a.sessions:
        session(proc, sd, a.out, a.n, a.dump, chunks, a.frames)


if __name__ == "__main__":
    main()
