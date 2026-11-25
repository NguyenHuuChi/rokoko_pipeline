"""Inputs of the hand-fit stage, all on the 1:1 output rows of the noise -> sync pipeline.

Output row i = materialized/filtered_synced.mp4 frame i = materialized/filtered_synced_rokoko.csv row i.  Per row:
  Rokoko joints (R'), the primary (ultra-wide) source frame, its ARKit camera pose (slerp of pose.csv on the shared
  monotonic clock) and its LiDAR depth map (depth/depth_frames.csv, primary_frame_index).
Head -> camera geometry from head motion only (no markers): R_HC from relative rotations, R_R'A = chordal mean,
t_HC from rigid head motion (pairwise form).  Then every Rokoko joint goes to the reference camera frame:
  p_C = R_C^T (p - c),  R_C = R_R'A R_AC,  c = p_head + R_head t_HC.
Optional smoothing (smooth_s > 0): every Rokoko joint POSITION is smoothed in time (Gaussian, real time, gaps
respected) before anything else; Rokoko rotations and the ARKit camera rotation stay raw (the picture shakes with
the camera, so the drawing must too; the fast part of the Rokoko positions is mostly not in the picture).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .common import M, SIDES, back_normal, chordal_mean, kabsch, log, exp, angle_deg, project, tsmooth

BLOCK_S = 6.0


# --------------------------------------------------------------------------- cameras
class Cams:
    def __init__(self, cal: dict):
        self.K = np.array(cal["reference_camera"]["K_encoded_row_major"], float).reshape(3, 3)
        self.Ku = np.array(cal["ultrawide_camera"]["K_encoded_row_major"], float).reshape(3, 3)
        T = np.array(cal["transforms"]["ultrawide_to_reference"]["matrix_3x4_row_major"], float).reshape(3, 4)
        self.Ruw, self.tuw = T[:, :3], T[:, 3] / 1000.0
        self.wh = (int(cal["reference_camera"]["encoded_width"]), int(cal["reference_camera"]["encoded_height"]))
        self.uw_wh = (int(cal["ultrawide_camera"]["encoded_width"]), int(cal["ultrawide_camera"]["encoded_height"]))
        self.depth_wh = (int(cal["depth"]["width"]), int(cal["depth"]["height"]))

    def uw(self, X):
        """Reference-camera points -> ultra-wide (primary video) pixels."""
        return project(self.Ku, (np.asarray(X) - self.tuw) @ self.Ruw)

    def uw_ray(self, uv):
        """Ultra-wide pixel -> ray in the reference camera frame: origin, unit direction."""
        d = self.Ruw @ (np.linalg.inv(self.Ku) @ np.r_[uv, 1.0])
        return self.tuw, d / np.linalg.norm(d)


# --------------------------------------------------------------------------- loading
def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@dataclass
class Inputs:
    session: Path
    sync_out: Path
    video: Path                  # materialized/filtered_synced.mp4
    rokoko_csv: Path             # materialized/filtered_synced_rokoko.csv
    manifest: Path               # materialized/filtered_synced_manifest.csv
    primary_name: str

    @classmethod
    def find(cls, session: Path, sync_out: Path):
        m = sync_out / "materialized"
        man = json.loads((session / "sync_manifest.json").read_text())
        prim = [v for v in man["videos"] if v.get("role") == "primary"]
        if len(prim) != 1:
            raise RuntimeError("sync_manifest.json must name exactly one primary video")
        return cls(session, sync_out, m / "filtered_synced.mp4", m / "filtered_synced_rokoko.csv",
                   m / "filtered_synced_manifest.csv", Path(prim[0]["archive_path"]).name)


class Scene:
    """Everything per output row.  Joint dicts are in the reference camera frame (metres)."""

    def __init__(self, inp: Inputs, rng_seed: int = 0, smooth_s: float = 0.0):
        self.inp = inp
        self.smooth_s = float(smooth_s)
        self.cal = json.loads((inp.session / "recording" / "calibration.json").read_text())
        self.cams = Cams(self.cal)
        man = pd.read_csv(inp.manifest)
        self.n = len(man)
        self.t = man["output_time_s"].to_numpy(float)
        self.frame = man["source_real_frame_id"].to_numpy(int)          # primary frame index
        self.src_row = man["source_rokoko_row_id"].to_numpy(int)
        self.blk = (np.floor((self.t - self.t[0]) / BLOCK_S).astype(int)) % 2
        self._load_rokoko()
        self._load_camera()
        self._load_depth_index()
        self.geometry = self._estimate_geometry(np.random.default_rng(rng_seed))
        self._to_camera()

    # ---- Rokoko (R')
    def _load_rokoko(self):
        df = pd.read_csv(self.inp.rokoko_csv, low_memory=False)
        if len(df) != self.n:
            raise RuntimeError(f"Rokoko CSV has {len(df)} rows, manifest {self.n}")
        joints = sorted({c[:-6] for c in df.columns if c.endswith("_pos_x")})
        self.P = {j: tsmooth(df[[f"{j}_pos_{a}" for a in "xyz"]].to_numpy(float) @ M.T, self.t, self.smooth_s) for j in joints}
        qd = lambda j: df[[f"{j}_quat_{a}" for a in "xyzw"]].to_numpy(float)
        rot = lambda j: M @ Rotation.from_quat(qd(j)).as_matrix() @ M
        self.R_H, self.R_chest = rot("head"), rot("chest")
        self.joint_names = joints

    # ---- ARKit camera pose at each primary frame (monotonic clock)
    def _load_camera(self):
        s = self.inp.session
        mono = {int(r["frame_index"]): int(r["monotonic_ns"]) for r in _rows(s / "frame_timestamps.csv")
                if r["role"] == "primary" and r["video_file"] == self.inp.primary_name}
        self.mono = np.array([mono[k] for k in self.frame], np.int64)
        pose = [r for r in _rows(s / "pose.csv")]
        pm = np.array([int(r["monotonic_ns"]) for r in pose], np.int64)
        o = np.argsort(pm); pose = [pose[i] for i in o]; pm = pm[o]
        pq = np.array([[float(r[f"orientation_{a}"]) for a in "xyzw"] for r in pose])
        pp = np.array([[float(r[f"position_{a}"]) for a in "xyz"] for r in pose])
        ok = np.array([r["tracking_state"] == "normal" for r in pose])
        tq = (self.mono - pm[0]) / 1e9; tp = (pm - pm[0]) / 1e9
        self.R_AC = Slerp(tp, Rotation.from_quat(pq))(np.clip(tq, tp[0], tp[-1])).as_matrix()
        self.p_AC = np.stack([np.interp(tq, tp, pp[:, a]) for a in range(3)], 1)
        j = np.clip(np.searchsorted(pm, self.mono), 1, len(pm) - 1)
        self.cam_ok = ok[j] & ok[j - 1] & (self.mono >= pm[0]) & (self.mono <= pm[-1])

    def _load_depth_index(self):
        dfr = _rows(self.inp.session / "depth" / "depth_frames.csv")
        dm = {int(r["primary_frame_index"]): int(r["index"]) for r in dfr if r.get("primary_frame_index", "") != ""}
        self.depth_index = np.array([dm.get(int(k), -1) for k in self.frame])

    def depth(self, i):
        """(depth m, confidence) of output row i, or (None, None)."""
        import cv2
        di = int(self.depth_index[i])
        if di < 0:
            return None, None
        d = self.inp.session / "depth"
        D = cv2.imread(str(d / f"{di:06d}.png"), cv2.IMREAD_UNCHANGED)
        C = cv2.imread(str(d / f"{di:06d}_confidence.png"), cv2.IMREAD_UNCHANGED)
        return (None, None) if D is None or C is None else (D.astype(np.float32) / 1000.0, C)

    # ---- head -> camera geometry
    def _estimate_geometry(self, rng):
        t, R_H, R_AC, p_H, p_AC = self.t, self.R_H, self.R_AC, self.P["head"], self.p_AC
        idx = np.nonzero(self.cam_ok)[0]
        I, J = _pairs(t, idx, 0.3, 2.0)
        ang = angle_deg(np.swapaxes(R_AC[I], 1, 2) @ R_AC[J])
        k = np.nonzero(ang >= 5.0)[0]
        if len(k) > 3000:
            k = np.sort(rng.choice(k, 3000, replace=False))
        I, J = I[k], J[k]
        if len(I) < 100:
            raise RuntimeError(f"only {len(I)} head-rotation pairs: not enough head motion to calibrate the camera")
        dH = np.swapaxes(R_H[I], 1, 2) @ R_H[J]
        dC = np.swapaxes(R_AC[I], 1, 2) @ R_AC[J]
        R_HC = _solve_rhc(dH, dC)
        res = angle_deg(np.swapaxes(R_HC, -1, -2) @ dH @ R_HC @ np.swapaxes(dC, 1, 2))
        R_RA = chordal_mean(R_H[idx] @ R_HC @ np.swapaxes(R_AC[idx], 1, 2))
        dev = angle_deg(np.swapaxes(R_H[idx] @ R_HC, 1, 2) @ R_RA @ R_AC[idx])
        I2, J2 = _pairs(t, idx[::2], 0.2, 2.0)
        A = R_H[J2] - R_H[I2]
        b = (p_AC[J2] - p_AC[I2]) @ R_RA.T - (p_H[J2] - p_H[I2])
        solve_t = lambda m: least_squares(lambda x: (np.einsum("nij,j->ni", A[m], x) - b[m]).ravel(), np.zeros(3),
                                          loss="huber", f_scale=0.01).x
        t_HC = solve_t(np.ones(len(I2), bool))
        half = t[I2] < np.median(t[I2])
        self.R_HC, self.R_RA, self.t_HC = R_HC, R_RA, t_HC
        return {"rotation_pairs": int(len(I)), "R_HC_pair_residual_deg": {"median": float(np.median(res)), "p90": float(np.percentile(res, 90))},
                "R_HC_per_frame_dev_deg": {"median": float(np.median(dev)), "p90": float(np.percentile(dev, 90))},
                "R_HC_rotvec_deg": np.degrees(log(R_HC)).round(2).tolist(), "R_RA_rotvec_deg": np.degrees(log(R_RA)).round(2).tolist(),
                "translation_pairs": int(len(I2)), "t_HC_mm": (1000 * t_HC).round(1).tolist(),
                "t_HC_first_half_mm": (1000 * solve_t(half)).round(1).tolist(),
                "t_HC_second_half_mm": (1000 * solve_t(~half)).round(1).tolist(),
                "camera_rows_tracking_normal_pct": round(100 * float(self.cam_ok.mean()), 1)}

    # ---- every joint to the reference camera frame
    def _to_camera(self):
        self.R_C = self.R_RA @ self.R_AC
        self.c = self.P["head"] + np.einsum("nij,j->ni", self.R_H, self.t_HC)
        to_cam = lambda p: np.einsum("nji,nj->ni", self.R_C, p - self.c)
        self.cam = {j: to_cam(p) for j, p in self.P.items()}
        self.cam2ch = np.einsum("nij,njk->nik", np.swapaxes(self.R_chest, 1, 2), self.R_C)
        self.ch2cam = np.swapaxes(self.cam2ch, 1, 2)
        self.base = {}
        self.sign = {}
        self.no_glove = {}
        for h, s in SIDES.items():
            self.no_glove[h] = f"{s}ThumbProximal" not in self.cam
            if self.no_glove[h]:                                       # Rokoko recorded without gloves
                self.cam.update(_placeholder_hand(self.cam[f"{s}LowerArm"], self.cam[f"{s}Hand"], s, h))
            J = {n: self.cam[f"{s}{n}"] for n in ["UpperArm", "LowerArm", "Hand"] + [f"{d}{p}" for d in ("Thumb", "Index", "Middle", "Ring", "Little") for p in ("Proximal", "Medial", "Distal", "Tip")]}
            self.base[h] = J
            nz = back_normal(J)[:, 2]
            self.sign[h] = -1.0 if np.median(nz) > 0 else 1.0        # back normal mostly faces the camera


# flat right hand in a frame x = forearm direction, y = towards the thumb, z = back normal (metres); left hand: y mirrored
_FLAT = {"ThumbProximal": (0.020, 0.025), "ThumbMedial": (0.045, 0.045), "ThumbDistal": (0.070, 0.060), "ThumbTip": (0.090, 0.070),
         **{f"{d}{p}": (x0 + dx, y) for d, x0, y in (("Index", 0.085, 0.025), ("Middle", 0.090, 0.0), ("Ring", 0.085, -0.020), ("Little", 0.078, -0.038))
            for p, dx in (("Proximal", 0.0), ("Medial", 0.045), ("Distal", 0.072), ("Tip", 0.092))}}


def _placeholder_hand(lower_arm, hand, s, h):
    """Rokoko without gloves has no finger joints: a flat hand continuing the forearm (only so that the scene can be built;
    the no-glove mode replaces the hands with measured ones).  Roll: back normal towards the camera (egocentric view)."""
    x = hand - lower_arm; x /= np.linalg.norm(x, axis=1, keepdims=True)
    to_cam = -hand / np.linalg.norm(hand, axis=1, keepdims=True)
    z = to_cam - (to_cam * x).sum(1, keepdims=True) * x; z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = np.cross(z, x) * (1.0 if h == "R" else -1.0)
    return {f"{s}{n}": hand + a * x + b * y for n, (a, b) in _FLAT.items()}


def _pairs(t, idx, dt_min, dt_max):
    idx = np.asarray(idx); ti = t[idx]
    I, J = [], []
    for off in range(1, len(idx)):
        d = ti[off:] - ti[:-off]
        if d.size == 0 or d.min() > dt_max:
            break
        m = (d >= dt_min) & (d <= dt_max)
        I.append(idx[:-off][m]); J.append(idx[off:][m])
    if not I:
        return np.array([], int), np.array([], int)
    I, J = np.concatenate(I), np.concatenate(J)
    o = np.lexsort((J, I))
    return I[o], J[o]


def _solve_rhc(dH, dC):
    """dH = R_HC dC R_HC^T: Kabsch on the rotation axes, then soft-L1 refinement."""
    a, b = log(dH), log(dC)
    R0 = kabsch(b, a)
    fun = lambda v: log(exp(v).T @ dH @ exp(v) @ np.swapaxes(dC, 1, 2)).ravel()
    return exp(least_squares(fun, log(R0), loss="soft_l1", f_scale=np.radians(2.0)).x)
