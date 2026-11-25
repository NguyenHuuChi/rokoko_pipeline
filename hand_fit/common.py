"""Shared constants, small geometry helpers and the frame conventions of the hand-fit stage.

Frames (column vectors, x_dst = R_dst_src x_src):
  R'  right-handed Rokoko world: R' = M * (CSV world), M = diag(1, 1, -1)
  A   ARKit world;  C  reference (wide) camera optical frame (x right, y down, z forward), metres
  U   ultra-wide camera = the PRIMARY video: p_C = Ruw p_U + tuw  (calibration.json, ultrawide_to_reference)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

M = np.diag([1.0, 1.0, -1.0])
SIDES = {"L": "left", "R": "right"}
DIGITS = ("Thumb", "Index", "Middle", "Ring", "Little")
FINGERS = [f"{d}{p}" for d in DIGITS for p in ("Proximal", "Medial", "Distal", "Tip")]
HAND_JOINTS = ["Hand"] + FINGERS
ARM_JOINTS = ["UpperArm", "LowerArm", "Hand"] + FINGERS
MCP = ["IndexProximal", "MiddleProximal", "RingProximal", "LittleProximal"]
W_IDX = [0, 5, 9, 13, 17]                     # WiLoR (OpenPose order): wrist + 4 MCP
BODY_BONES = [("hip", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"),
              ("chest", "leftShoulder"), ("leftShoulder", "leftUpperArm"),
              ("chest", "rightShoulder"), ("rightShoulder", "rightUpperArm"),
              ("hip", "leftUpLeg"), ("leftUpLeg", "leftLeg"), ("leftLeg", "leftFoot"), ("leftFoot", "leftToe"),
              ("hip", "rightUpLeg"), ("rightUpLeg", "rightLeg"), ("rightLeg", "rightFoot"), ("rightFoot", "rightToe")]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def tsmooth(x, t, sig):
    """Zero-phase Gaussian of width sig seconds along rows at times t (rows may be unevenly spaced; +-3 sig window)."""
    x = np.asarray(x, float)
    if sig <= 0:
        return x.copy()
    n = len(t); i = np.arange(n)
    K = 1
    while K < n and np.min(t[K:] - t[:-K]) <= 3 * sig:
        K += 1
    num = np.zeros_like(x); den = np.zeros(n)
    for k in range(-K, K + 1):
        j = np.clip(i + k, 0, n - 1)
        w = np.exp(-0.5 * ((t[j] - t) / sig) ** 2) * ((i + k >= 0) & (i + k < n) & (np.abs(t[j] - t) <= 3 * sig))
        num += w.reshape(-1, *[1] * (x.ndim - 1)) * x[j]; den += w
    return num / den.reshape(-1, *[1] * (x.ndim - 1))


def project(K, p):
    z = p[..., 2]
    return np.stack([K[0, 0] * p[..., 0] / z + K[0, 2], K[1, 1] * p[..., 1] / z + K[1, 2]], -1)


def back_normal(J):
    """Back-of-hand normal of a hand joint dict (sign fixed by the caller)."""
    return unit(np.cross(J["MiddleProximal"] - J["Hand"], J["IndexProximal"] - J["LittleProximal"]))


def hand_frames(J):
    """Per-row hand frame: x = wrist -> middle MCP, y = little -> index MCP (orthogonalised), z = x cross y."""
    a1 = unit(J["MiddleProximal"] - J["Hand"])
    a2 = J["IndexProximal"] - J["LittleProximal"]
    a2 = unit(a2 - (a2 * a1).sum(-1, keepdims=True) * a1)
    return np.stack([a1, a2, np.cross(a1, a2)], -1)


def rot_a_to_b(a, b, max_deg=180.0):
    a, b = unit(a), unit(b)
    ax = np.cross(a, b); s = np.linalg.norm(ax)
    if s < 1e-12:
        return np.zeros(3)
    return ax / s * min(np.arctan2(s, float(np.clip(a @ b, -1, 1))), np.radians(max_deg))


def rotvec_matrix(v):
    """Rotation vector(s) (..., 3) -> matrix (..., 3, 3), Rodrigues in numpy (same values as
    Rotation.from_rotvec(v).as_matrix() to ~1e-16, ~10x faster for the many small calls of the fits)."""
    v = np.asarray(v, float)
    th2 = np.sum(v * v, -1)[..., None, None]
    th = np.sqrt(th2)
    small = th < 1e-4
    ths = np.where(small, 1.0, th)
    A = np.where(small, 1.0 - th2 / 6.0, np.sin(ths) / ths)                          # sin(th) / th
    B = np.where(small, 0.5 - th2 / 24.0, 2.0 * np.sin(ths / 2.0) ** 2 / (ths * ths))   # (1 - cos th) / th^2
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    return np.eye(3) + A * K + B * (K @ K)


FD_EPS = np.finfo(float).eps ** 0.5


def fd_jacobian(res_batch):
    """jac= for least_squares: scipy's own '2-point' forward differences (diff_step=None: step sqrt(eps) * sign(x) *
    max(1, |x|), rounded to a representable step), with the 1 + n residual evaluations done in ONE vectorised call
    res_batch(P (k, n), *args) -> (k, m).  Same Jacobian as jac='2-point', fewer Python round trips."""
    def jac(p, *args):
        h = FD_EPS * np.where(p >= 0, 1.0, -1.0) * np.maximum(1.0, np.abs(p))
        h = (p + h) - p
        F = res_batch(np.vstack([p, p + np.diag(h)]), *args)
        return (F[1:] - F[0]).T / h
    return jac


def q(v):
    """Median / p90 / n summary (finite values only)."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return None
    return {"median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2), "n": int(len(v))}


# --------------------------------------------------------------------------- SO(3)
def project_to_so3(A):
    U, _, Vt = np.linalg.svd(A)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def kabsch(src, dst):
    return project_to_so3(((src).T @ dst).T)


def chordal_mean(Rs):
    return project_to_so3(Rs.sum(0))


def log(R):
    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


def exp(v):
    return Rotation.from_rotvec(np.asarray(v, float)).as_matrix()


def angle_deg(R):
    return np.degrees(np.linalg.norm(log(R), axis=-1))
