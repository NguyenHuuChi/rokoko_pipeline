"""Packaged hand detector for no-glove takes (06/10): YOLO (WiLoR's detector.pt, YOLOv8-pose, classes left / right) +
the fine-tuned student (r12 by default), on the Mac (MPS), no WiLoR, no Modal.  Output npz = the layout of
hand_fit.detect_hands (frame, pass, hand, box, kp2d, j3d, det_conf), so hand_fit.noglove / body_fix / the glove-layout
CSV run on it unchanged.

Pipeline, three threads joined by bounded queues (the student starts on a crop as soon as YOLO returns it):
  reader   decodes the video
  yolo     batches of YOLO_BATCH frames, conf >= CONF, imgsz 640, at most MAX_PER_CLASS boxes per class (as on Modal)
  student  batches of up to STUDENT_BATCH crops: pass 1 on the YOLO box, pass 2 re-centred on the pass-1 keypoints x 1.5
           (as distill_2d/student_infer.py); left hands mirrored inside the runner
Tracking mode (--track K [--segments S]): the video is cut into S segments tracked in lockstep (each read by its own
ffmpeg, seeking on the packet timestamp), so YOLO and the student run in batches across segments; per segment one track
per class (left / right); the crop of frame t+1 comes from the student's own
keypoints of frame t (box of the 21 points x 1.5, as pass 2), YOLO runs only every K frames, when a track is lost, or
every MISS_EVERY frames while a class has no track.  Lost = pass 1 / pass 2 disagree (> 0.25 palm, the noglove gate),
palm size off the track's YOLO start (outside 0.6-1.6 x), student score < SCORE_REL x the median score of YOLO-started
crops, or both tracks on one hand.  A tracked hand YOLO did not report is kept (npz key src: 0 = YOLO crop, 1 = tracked),
det_conf = the YOLO confidence that started the track.
Other people's hands (08/10, --prior P --wrist-gate R): YOLO keeps 2 boxes per class, so a bystander's hand can win a
side; with the gate a hand of side h is taken only if its wrist is within R x Rokoko hand length (px) of Rokoko's wrist
pixel of side h (rows where Rokoko's hand is valid).  R = 2.0: the wearer's hands sit < 1.5 on 11:07 (bad head->camera
calibration, ~1 hand length off) and < 1.1 on 10:20.
    python detect_hands_student.py <video> <out.npz> [--ckpt last.pt] [--yolo detector.pt] [--max-frames n] [--serial]
                                   [--track K] [--prior rokoko_prior.npz --wrist-gate 2.0]
(light_hand venv: torch, onnx2torch, ultralytics 8.1.34).  Prints the time per stage.
"""
from __future__ import annotations
from package_paths import PACKAGE_ROOT,SOURCE_ROOT,MODEL_ROOT,SMPLH_MODEL

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE_ROOT))
import eval_student as E                         # noqa: E402
from student_infer import grow_box, BOX_X        # noqa: E402

CKPT = MODEL_ROOT/'hand5/hand5_r17d.pt'
YOLO_PT = MODEL_ROOT/'NOT_INCLUDED_NO_YOLO_MODE'
CONF, MAX_PER_CLASS, YOLO_BATCH, STUDENT_BATCH = 0.3, 2, 16, 128
MISS_EVERY, DISAGREE, PALM_RANGE, SCORE_REL = 3, 0.25, (0.6, 1.6), 0.6
PRIOR_CONF = 0.5                                 # det_conf written for prior crops (noglove keeps conf >= 0.4); src = 2
PRED_GROW_MAX = 1.5                              # --predict: crop widened up to this x when the hand moves fast
PRIOR_SEP = 0.6                                  # a prior hand closer than this x palm to the other hand's track is dropped
GAP_SEEK = 15                                    # --frame-map: excluded raw spans longer than this are skipped by a new seek
W5 = [0, 5, 9, 13, 17]
END = object()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--ckpt", type=Path, default=CKPT); ap.add_argument("--yolo", type=Path, default=YOLO_PT)
    ap.add_argument("--max-frames", type=int, default=0); ap.add_argument("--serial", action="store_true",
                                                                            help="no threads (timing reference)")
    ap.add_argument("--track", type=int, default=0, help="tracking mode: YOLO every K frames (0 = YOLO on every frame)")
    ap.add_argument("--segments", type=int, default=8, help="tracking mode: video segments tracked in lockstep (batching)")
    ap.add_argument("--frame-map", type=Path, default=None,
                    help="tracking mode on the RAW video: .npy of raw frame ids (ascending); output frame i = raw frame map[i]")
    ap.add_argument("--colors", action="store_true", help="also write <out>_glove_color.npz (hand_fit.glove_color layout)")
    ap.add_argument("--no-yolo", action="store_true", help="tracking mode with --prior: no YOLO at all, tracks start / "
                    "restart only from the Rokoko prior crops")
    ap.add_argument("--prior", type=Path, default=None, help="tracking mode: Rokoko hand prior (make_prior.py); when YOLO "
                    "runs and a hand still has no track, the student tries a large crop at Rokoko's hand (src 2)")
    ap.add_argument("--wrist-gate", type=float, default=0.0, help="tracking mode with --prior (08/10, other people's hands): "
                    "a hand whose wrist is farther than R x Rokoko hand length (px) from Rokoko's wrist pixel of that side is "
                    "not taken (YOLO crop not emitted / track dropped / prior crop refused); rows where Rokoko's hand is not "
                    "valid are not gated.  0 = off (default)")
    ap.add_argument("--prior-scale", type=float, default=1.0, help="tracking mode with --prior (08/10, hands reaching far "
                    "past Rokoko's short arm): Rokoko crop box side x S (default 1)")
    ap.add_argument("--prior-shift", type=float, default=0.0, help="tracking mode with --prior: move the Rokoko crop centre by "
                    "F x hand length along the forearm direction (wrist -> hand centre), towards the fingers (default 0)")
    ap.add_argument("--predict", action="store_true", help="tracking mode: move the next crop by the hand's expected motion "
                    "(Rokoko prior centre t-1 -> t, else the track's own velocity) and widen it with the speed (07/10, fast hands)")
    a = ap.parse_args(argv)
    from ultralytics import YOLO
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    t_load = time.perf_counter()
    yolo = YOLO(str(a.yolo))
    import os as _os                                 # 08/10 speed test: DET_RUNNER=cpu|gpucrop|coreml (fast_runner.py), unset = as before
    if _os.environ.get("DET_RUNNER"):
        raise RuntimeError('This release uses the current standard Hand5 Runner; DET_RUNNER accelerators are not packaged')
        runner = FastRunner(str(a.ckpt), dev, _os.environ["DET_RUNNER"])
    else:
        runner = E.Runner(str(a.ckpt), dev)
    HWDEC = ["-hwaccel", "videotoolbox"] if _os.environ.get("DET_HWDEC") == "1" else []   # 08/10: Mac hardware decode
    cap = cv2.VideoCapture(str(a.video)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4)); E.F_W = 5000.0 / 256 * max(W, H)
    FM = None
    PRIOR = dict(np.load(a.prior)) if a.prior is not None else None
    assert not a.wrist_gate or PRIOR is not None, "--wrist-gate needs --prior"
    if a.frame_map is not None:
        assert a.track, "--frame-map needs --track"
        FM = np.load(a.frame_map).astype(int); assert (np.diff(FM) > 0).all(); n = len(FM)
    if a.max_frames:
        n = min(n, a.max_frames)
    T = {"load": time.perf_counter() - t_load, "decode": 0.0, "yolo": 0.0, "student": 0.0}
    PTS = None
    if a.track:                                                      # packet timestamps for exact segment starts (no B-frames)
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "packet=pts_time", "-of", "csv=p=0",
                              str(a.video)], capture_output=True, text=True, check=True).stdout
        PTS = np.sort(np.array([float(x) for x in out.split()]))
    rec = {k: [] for k in ("frame", "pass", "hand", "box", "kp2d", "j3d", "det_conf", "src", "col")}
    NAN7 = np.full(7, np.nan, np.float32)
    ccache = {}

    def color(fi, im, kp):
        """hand_fit.glove_color.hand_colors for one pass-2 hand (half resolution, hull eroded 1 px, >= 30 px)."""
        if fi not in ccache:
            if len(ccache) > 64:
                ccache.clear()
            h = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            ccache[fi] = (h.shape[:2], cv2.cvtColor(h, cv2.COLOR_BGR2HSV), cv2.cvtColor(h, cv2.COLOR_BGR2LAB))
        shp, hsv, lab = ccache[fi]
        m = np.zeros(shp, np.uint8); cv2.fillConvexPoly(m, cv2.convexHull((kp * 0.5).astype(np.int32)), 1)
        sel = cv2.erode(m, np.ones((3, 3), np.uint8)).astype(bool)
        out = NAN7.copy()
        if sel.sum() >= 30:
            out[:3] = np.median(hsv[sel], 0); out[3:6] = np.median(lab[sel], 0); out[6] = sel.sum()
        return out

    def frames():
        fi = 0
        while fi < n:
            t0 = time.perf_counter(); ok, im = cap.read(); T["decode"] += time.perf_counter() - t0
            if not ok:
                break
            yield fi, im; fi += 1

    def detect(batch):
        t0 = time.perf_counter()
        res = yolo([im for _, im in batch], conf=CONF, imgsz=640, device=dev.type, verbose=False)
        out = []
        for (fi, im), r in zip(batch, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b, c, cf = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
            for cl in (0, 1):
                for k in [k for k in np.argsort(-cf) if c[k] == cl][:MAX_PER_CLASS]:
                    out.append((fi, im, cl, b[k].astype(np.float32), float(cf[k])))
        T["yolo"] += time.perf_counter() - t0
        return out

    def student(items):
        t0 = time.perf_counter()
        ims = [it[1] for it in items]; hs = [it[2] for it in items]
        b1 = [grow_box(it[3], 1.0) for it in items]
        c1, S1 = zip(*[E.square(b) for b in b1]); kp1, _, r1 = runner(ims, np.array(c1), S1, hs)
        b2 = [E.grow(p, BOX_X) for p in kp1]
        c2, S2 = zip(*[E.square(b) for b in b2]); kp2, _, r2 = runner(ims, np.array(c2), S2, hs)
        j1, j2 = E.to_j3d(kp1[:, 0], r1, (W, H)), E.to_j3d(kp2[:, 0], r2, (W, H))
        for ps, bx, kp, j in ((1, b1, kp1, j1), (2, b2, kp2, j2)):
            for k, it in enumerate(items):
                for key, v in (("frame", it[0]), ("pass", ps), ("hand", it[2]), ("box", bx[k]), ("kp2d", kp[k]), ("j3d", j[k]),
                               ("det_conf", it[4]), ("src", 0), ("col", NAN7)):
                    rec[key].append(v)
        T["student"] += time.perf_counter() - t0

    def r12(ims, boxes, hands, square_from_kp):
        """Student pass 1 on `boxes` (YOLO boxes, or keypoint boxes), pass 2 re-centred: kp1, kp2, j1, j2, score, b1, b2."""
        t0 = time.perf_counter()
        b1 = [b if sq else grow_box(b, 1.0) for b, sq in zip(boxes, square_from_kp)]
        c1, S1 = zip(*[E.square(b) for b in b1]); kp1, _, r1 = runner(ims, np.array(c1), S1, hands)
        b2 = [E.grow(p, BOX_X) for p in kp1]
        c2, S2 = zip(*[E.square(b) for b in b2]); kp2, sc2, r2 = runner(ims, np.array(c2), S2, hands)
        T["student"] += time.perf_counter() - t0
        return kp1, kp2, E.to_j3d(kp1[:, 0], r1, (W, H)), E.to_j3d(kp2[:, 0], r2, (W, H)), sc2.mean(1), b1, b2

    def emit(fi, hand, conf, src, b1, b2, kp1, kp2, j1, j2, im=None):
        for ps, bx, kp, j in ((1, b1, kp1, j1), (2, b2, kp2, j2)):          # pass 1 / pass 2 pairs stay in the same order
            col = color(fi, im, kp) if (a.colors and ps == 2 and im is not None) else NAN7
            for key, v in (("frame", fi), ("pass", ps), ("hand", hand), ("box", bx), ("kp2d", kp), ("j3d", j), ("det_conf", conf),
                           ("src", src), ("col", col)):
                rec[key].append(v)

    palm = lambda kp: max(float(np.linalg.norm(kp[9] - kp[0])), 1.0)

    def far(fi, h, kp):
        """--wrist-gate: the hand's wrist is too far from Rokoko's wrist of side h (someone else's hand)."""
        if not a.wrist_gate or fi >= len(PRIOR["valid"]) or not PRIOR["valid"][fi, h]:
            return False
        return float(np.linalg.norm(kp[0] - PRIOR["wrist"][fi, h])) > a.wrist_gate * float(PRIOR["size"][fi, h])

    def segment_reader(start, length, q):
        if FM is None:
            return segment_reader_1(start, length, q)
        sel = FM[start:start + length]                              # raw ids of output frames start .. start+length-1
        cut = np.nonzero(np.diff(sel) > GAP_SEEK)[0] + 1              # long excluded spans: seek again instead of decoding
        o = start
        for run in np.split(sel, cut):
            want = {int(r): o + k for k, r in enumerate(run)}; o += len(run)
            r0 = int(run[0]); ln = int(run[-1]) - r0 + 1
            cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(PTS[r0] - 0.002, 0):.6f}", *HWDEC, "-i", str(a.video), "-frames:v", str(ln),
                   "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
            pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 2); sz = W * H * 3
            for k in range(ln):
                t0 = time.perf_counter(); buf = pr.stdout.read(sz); T["decode"] += time.perf_counter() - t0
                if len(buf) < sz:
                    break
                if r0 + k in want:
                    q.put((want[r0 + k], np.frombuffer(buf, np.uint8).reshape(H, W, 3)))
            pr.stdout.close(); pr.wait()
        q.put(END)

    def segment_reader_1(start, length, q):
        """Exact frames start .. start+length-1 (ffmpeg seek on the packet timestamp; OpenCV seeks wrongly on these VFR
        videos), raw BGR into q, then END."""
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(PTS[start] - 0.002, 0):.6f}", *HWDEC, "-i", str(a.video), "-frames:v", str(length),
               "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 2); sz = W * H * 3
        for k in range(length):
            t0 = time.perf_counter(); buf = pr.stdout.read(sz); T["decode"] += time.perf_counter() - t0
            if len(buf) < sz:
                break
            q.put((start + k, np.frombuffer(buf, np.uint8).reshape(H, W, 3)))
        pr.stdout.close(); pr.wait(); q.put(END)

    def track_box(t, fi, h):
        """Crop for frame fi from the track's last keypoints (frame t['fi']).  --predict: shifted by the expected hand
        motion (Rokoko prior centre, which follows the hand without lag; else the track's own velocity) and widened
        by up to PRED_GROW_MAX x when the shift is large against the hand box."""
        kp = t["kp"]
        if not a.predict or t.get("fi") is None:
            return E.grow(kp, BOX_X)
        f0 = t["fi"]; d = None
        if PRIOR is not None and f0 < len(PRIOR["valid"]) and fi < len(PRIOR["valid"]) and PRIOR["valid"][f0, h] and PRIOR["valid"][fi, h]:
            d = PRIOR["centre"][fi, h].astype(float) - PRIOR["centre"][f0, h].astype(float)
        elif t.get("v") is not None:
            d = t["v"] * (fi - f0)
        if d is None or not np.isfinite(d).all():
            return E.grow(kp, BOX_X)
        size = float(np.max(kp.max(0) - kp.min(0))) + 1e-6
        g = 1.0 + min(np.linalg.norm(d) / size, PRED_GROW_MAX - 1.0)
        return E.grow(kp + d, BOX_X * g)

    def run_track(K, n_seg):
        """Tracking mode, n_seg segments in lockstep (one YOLO call and one student call pair per step for all of them)."""
        edges = np.linspace(0, n, n_seg + 1).round().astype(int)
        qs = [queue.Queue(maxsize=6) for _ in range(n_seg)]
        for k in range(n_seg):
            threading.Thread(target=segment_reader, args=(int(edges[k]), int(edges[k + 1] - edges[k]), qs[k]), daemon=True).start()
        seg = [{"tracks": {}, "last": -10 ** 9, "done": False} for _ in range(n_seg)]
        init_scores = []
        cnt = {"steps": 0, "yolo_frames": 0, "tracked": 0, "yolo_crops": 0, "lost": 0, "rescued": 0, "prior_tried": 0, "prior_ok": 0,
               "gate_yolo": 0, "gate_track": 0, "gate_prior": 0}
        while not all(sg["done"] for sg in seg):
            cnt["steps"] += 1
            cur = []                                                     # (segment, frame, image)
            for k, sg in enumerate(seg):
                if sg["done"]:
                    continue
                it = qs[k].get()
                if it is END:
                    sg["done"] = True; continue
                cur.append((k, it[0], it[1]))
            if not cur:
                break
            yl = [(k, fi, im) for k, fi, im in cur if fi - seg[k]["last"] >= K or len(seg[k]["tracks"]) < 2 and fi - seg[k]["last"] >= MISS_EVERY]
            dets = []
            if yl:
                for k, fi, _ in yl:
                    seg[k]["last"] = fi
                cnt["yolo_frames"] += len(yl)
                seg_of = {fi: k for k, fi, _ in yl}
                if not a.no_yolo:
                    dets = [(seg_of[d[0]],) + d for d in detect([(fi, im) for _, fi, im in yl])]  # (seg, frame, im, hand, box, conf)
            got = {(d[0], d[3]) for d in dets}
            trk = [(k, fi, im, h) for k, fi, im in cur for h in seg[k]["tracks"] if (k, h) not in got]
            items = [(d[0], d[1], d[2], d[3], d[4], False, d[5], 0) for d in dets] + \
                    [(k, fi, im, h, track_box(seg[k]["tracks"][h], fi, h), True, seg[k]["tracks"][h]["conf"], 1) for k, fi, im, h in trk]
            if items:                                                    # (no YOLO and no track yet: prior only)
                kp1, kp2, j1, j2, sc, b1, b2 = r12([it[2] for it in items], [it[4] for it in items], [it[3] for it in items],
                                                   [it[5] for it in items])
                if init_scores:
                    smin = SCORE_REL * float(np.median(init_scores))
                best = {}
                for i, (k, fi, im, h, box, tracked, cf, src) in enumerate(items):
                    dis = np.mean(np.linalg.norm(kp1[i][W5] - kp2[i][W5], axis=1)) / palm(kp2[i])
                    if far(fi, h, kp2[i]):                             # --wrist-gate: not the wearer's hand
                        if tracked:
                            del seg[k]["tracks"][h]; seg[k]["last"] = -10 ** 9; cnt["gate_track"] += 1
                        else:
                            cnt["gate_yolo"] += 1
                        continue
                    if not tracked:
                        emit(fi, h, cf, 0, b1[i], b2[i], kp1[i], kp2[i], j1[i], j2[i], im); cnt["yolo_crops"] += 1
                        if dis <= DISAGREE and ((k, h) not in best or cf > best[(k, h)][0]):
                            best[(k, h)] = (cf, i)
                        continue
                    ratio = palm(kp2[i]) / seg[k]["tracks"][h]["palm0"]
                    if dis > DISAGREE or not PALM_RANGE[0] <= ratio <= PALM_RANGE[1] or (init_scores and sc[i] < smin):
                        del seg[k]["tracks"][h]; seg[k]["last"] = -10 ** 9; cnt["lost"] += 1; continue
                    emit(fi, h, cf, 1, b1[i], b2[i], kp1[i], kp2[i], j1[i], j2[i], im); cnt["tracked"] += 1
                    cnt["rescued"] += int(any(y[0] == k for y in yl))           # YOLO ran on this frame and missed this hand
                    t_ = seg[k]["tracks"][h]
                    if t_.get("fi") is not None and fi > t_["fi"]:
                        t_["v"] = (kp2[i].mean(0) - t_["kp"].mean(0)) / (fi - t_["fi"])
                    t_["kp"] = kp2[i]; t_["fi"] = fi
                for (k, h), (cf, i) in best.items():
                    seg[k]["tracks"][h] = {"kp": kp2[i], "conf": cf, "palm0": palm(kp2[i]), "fi": items[i][1]}; init_scores.append(float(sc[i]))
            if PRIOR is not None and yl:                                 # YOLO ran here and a hand is still missing
                pit = []
                for k, fi, im in yl:
                    for h in (0, 1):
                        if h in seg[k]["tracks"] or not PRIOR["valid"][fi, h]:
                            continue
                        c, L = PRIOR["centre"][fi, h].astype(float), float(PRIOR["size"][fi, h])
                        if a.prior_shift:
                            d = c - PRIOR["wrist"][fi, h]; nd = float(np.linalg.norm(d))
                            if nd > 1e-6:
                                c = c + a.prior_shift * L * d / nd
                        L *= a.prior_scale
                        pit.append((k, fi, im, h, np.array([c[0] - L / 2, c[1] - L / 2, c[0] + L / 2, c[1] + L / 2], np.float32)))
                if pit:
                    cnt["prior_tried"] += len(pit)
                    pk1, pk2, pj1, pj2, psc, pb1, pb2 = r12([it[2] for it in pit], [it[4] for it in pit], [it[3] for it in pit],
                                                            [False] * len(pit))
                    for i, (k, fi, im, h, _) in enumerate(pit):
                        dis = np.mean(np.linalg.norm(pk1[i][W5] - pk2[i][W5], axis=1)) / palm(pk2[i])
                        if dis > DISAGREE or (init_scores and psc[i] < SCORE_REL * float(np.median(init_scores))):
                            continue
                        if far(fi, h, pk2[i]):
                            cnt["gate_prior"] += 1; continue
                        o = seg[k]["tracks"].get(1 - h)
                        if o is not None and np.linalg.norm(pk2[i][0] - o["kp"][0]) < PRIOR_SEP * palm(o["kp"]):
                            continue                                     # the crop found the other hand
                        emit(fi, h, PRIOR_CONF, 2, pb1[i], pb2[i], pk1[i], pk2[i], pj1[i], pj2[i], im); cnt["prior_ok"] += 1
                        seg[k]["tracks"][h] = {"kp": pk2[i], "conf": PRIOR_CONF, "palm0": palm(pk2[i]), "fi": fi}
                        if a.no_yolo:
                            init_scores.append(float(psc[i]))                # score gate reference without YOLO starts
            for k, sg in enumerate(seg):
                tr = sg["tracks"]
                if len(tr) == 2 and np.linalg.norm(tr[0]["kp"][0] - tr[1]["kp"][0]) < 0.3 * palm(tr[0]["kp"]):
                    tr.clear(); sg["last"] = -10 ** 9; cnt["lost"] += 2            # both tracks on one hand
        return cnt

    t_all = time.perf_counter()
    cnt = None
    if a.track:
        cnt = run_track(a.track, a.segments)
    elif a.serial:
        pend, crops = [], []
        for fr in frames():
            pend.append(fr)
            if len(pend) == YOLO_BATCH:
                crops += detect(pend); pend = []
                if len(crops) >= STUDENT_BATCH:
                    student(crops); crops = []
        crops += detect(pend) if pend else []
        if crops:
            student(crops)
    else:
        qf, qc = queue.Queue(maxsize=4), queue.Queue(maxsize=8)
        err, stop = [], threading.Event()                            # any failure stops every stage (no thread left waiting)

        def put(q, x):
            while not stop.is_set():
                try:
                    q.put(x, timeout=0.5); return
                except queue.Full:
                    pass

        def reader():
            try:
                pend = []
                for fr in frames():
                    if stop.is_set():
                        break
                    pend.append(fr)
                    if len(pend) == YOLO_BATCH:
                        put(qf, pend); pend = []
                if pend:
                    put(qf, pend)
            except Exception as e:                                   # noqa: BLE001
                err.append(e); stop.set()
            put(qf, END)

        def yolo_worker():
            try:
                while (b := qf.get()) is not END and not stop.is_set():
                    items = detect(b)
                    if items:
                        put(qc, items)
            except Exception as e:                                   # noqa: BLE001
                err.append(e); stop.set()
            qc.put(END)

        th = [threading.Thread(target=reader, daemon=True), threading.Thread(target=yolo_worker, daemon=True)]
        for t in th:
            t.start()
        crops = []
        try:
            while (items := qc.get()) is not END:
                crops += items
                if len(crops) >= STUDENT_BATCH:
                    student(crops); crops = []
            if crops and not err:
                student(crops)
        finally:
            if err:
                stop.set()
            for t in th:
                t.join(timeout=5)
        if err:
            raise err[0]
    wall = time.perf_counter() - t_all
    out = {"n_frames": n, **{k: np.array(v) for k, v in rec.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    col = out.pop("col").reshape(-1, 7).astype(np.float32)
    np.savez(a.out, **out)
    if a.colors:
        np.savez(a.out.with_name(a.out.stem + "_glove_color.npz"), col=col,
                 cols=np.array(("H", "S", "V", "L", "a", "b", "n_px")))
    p2 = out["pass"] == 2 if len(rec["pass"]) else np.zeros(0, bool)
    print(f"{n} frames, {int(p2.sum())} hands (L {int((out['hand'][p2] == 0).sum())}, R {int((out['hand'][p2] == 1).sum())}) "
          f"-> {a.out}\n{'tracking' if a.track else 'serial' if a.serial else 'pipelined'} on {dev}: wall {wall:.0f} s = {n / wall:.1f} frames/s | "
          f"stage busy time: decode {T['decode']:.0f} s, yolo {T['yolo']:.0f} s ({1000 * T['yolo'] / max(n, 1):.1f} ms/frame), "
          f"student {T['student']:.0f} s ({1000 * T['student'] / max(int(p2.sum()), 1):.1f} ms/hand, 2 passes) | load {T['load']:.0f} s"
          + (f"\nrunner {getattr(runner, 'mode', 'default')}: {runner.n} crops, whole call {getattr(runner, 'tt', float('nan')):.0f} s, "
             f"model part {runner.t:.0f} s, hwdec {bool(HWDEC)}" if hasattr(runner, 'n') else "")
          + (f"\ntracking K={a.track}: {cnt}" if cnt else ""), flush=True)


if __name__ == "__main__":
    main()
