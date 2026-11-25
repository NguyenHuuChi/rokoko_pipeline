#!/usr/bin/env python3
"""Self-check for clean_mocap.py on synthetic data with known truth.  python test_clean_mocap.py"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from clean_mocap import clean

rng = np.random.default_rng(0)
t = np.cumsum(rng.uniform(0.030, 0.037, 1800))            # ~30 Hz Rokoko with jitter, 60 s
t -= t[0]
truth_x = 0.5 * np.sin(2 * np.pi * 2.0 * t)                # real fast swing: 2 Hz, 0.5 m, peak 6.3 m/s
yaw = 0.6 * np.sin(2 * np.pi * 0.3 * t)
quat = np.stack([np.zeros_like(t), np.sin(yaw / 2), np.zeros_like(t), np.cos(yaw / 2)], 1)

cols = {"time": t}
for j, x, y in (("hand", truth_x.copy(), 1.0), ("hip", 0.1 * np.sin(2 * np.pi * 0.2 * t), 1.0),
                ("toeEnd", 0 * t, 1.0), ("leftLeg", 0 * t, 0.44), ("leftFoot", 0 * t, 0.07)):
    cols |= {f"{j}_pos_x": x, f"{j}_pos_y": y + 0 * t, f"{j}_pos_z": 0 * t}
    q = quat.copy() if j != "toeEnd" else np.tile([0, 0, 0, 1.0], (len(t), 1))
    cols |= {f"{j}_quat_{a}": q[:, i] for i, a in enumerate("xyzw")}
raw = pd.DataFrame(cols)
raw.loc[600:602, "hand_pos_x"] += 0.6                       # 3-sample teleport glitch
raw.loc[900, [f"hand_quat_{a}" for a in "xyzw"]] = [0.99, 0, 0.1, 0.1]   # ~170 deg one-sample flip
raw.loc[1201, [f"hand_quat_{a}" for a in "xyzw"]] *= -1     # sign flip (same rotation)
ramp = np.r_[np.linspace(0, 0.30, 8)[1:], [0.30] * 3]      # ramp-hold-snap: rises at a normal
raw.loc[1000:1009, "hip_pos_y"] += ramp                     # speed, holds, drops in one frame
leg = np.r_[np.linspace(0, 1, 8)[1:], [1.0] * 4]           # the 925 s case: knee + foot rise together,
raw.loc[1100:1110, "leftLeg_pos_y"] += 0.13 * leg           # the stream stalls during the hold (repeats),
raw.loc[1100:1110, "leftFoot_pos_y"] += 0.24 * leg          # then both drop back in one sample
raw.loc[1108:1110, raw.columns[1:]] = raw.loc[1107, raw.columns[1:]].to_numpy()
for i in range(300, 1500, 37):                              # repeated packets
    raw.iloc[i, 1:] = raw.iloc[i - 1, 1:]
raw = raw.drop(index=range(1400, 1420))                     # 0.65 s dropout

fps = 30.0
vt = np.arange(0, raw["time"].iloc[-1], 1 / fps)
idx = np.clip(np.searchsorted(raw["time"].to_numpy(), vt), 0, len(raw) - 1)
pf = raw.iloc[idx].reset_index(drop=True)
pf.insert(0, "match_error_ms", (pf["time"].to_numpy() - vt) * 1000)
pf.insert(0, "video_time", vt)
pf.insert(0, "frame", np.arange(len(vt)))

with tempfile.TemporaryDirectory() as d:
    raw.to_csv(Path(d, "x_sync.csv"), index=False)
    pf.to_csv(Path(d, "x_perframe.csv"), index=False)
    rep = clean(Path(d))
    out = pd.read_csv(Path(d, "x_perframe_clean.csv"))

truth_f = 0.5 * np.sin(2 * np.pi * 2.0 * vt)
err = np.abs(out["hand_pos_x"].to_numpy() - truth_f)
ok = out["valid"].to_numpy() == 1
print(rep)
assert rep["before_v>8"] > 0 and rep["after_v>8"] == 0, "spikes not removed"
assert rep["after_rot>60"] == 0, "rotation flip not removed"
fixed = out["repaired"].to_numpy() > 0
assert err[ok & ~fixed].max() < 0.05, f"untouched frames off truth by {err[ok & ~fixed].max():.3f} m"
assert err[ok & fixed].max() < 0.15, f"repaired frames off truth by {err[ok & fixed].max():.3f} m"
peaks = ok & ~fixed & (np.abs(truth_f) > 0.48)
assert np.abs(out["hand_pos_x"].to_numpy()[peaks]).min() > 0.44, "real swing peaks flattened"
assert (~ok).sum() >= 3 and not ok[int(vt.searchsorted(raw["time"].iloc[1399]) + 5)], "dropout not marked"
hip_err = np.abs(out["hip_pos_y"].to_numpy() - 1.0)
assert hip_err.max() < 0.02, f"ramp-hold-snap glitch left: hip off by {hip_err.max():.3f} m"
shin = out["leftLeg_pos_y"].to_numpy() - out["leftFoot_pos_y"].to_numpy()
foot_err = np.abs(out["leftFoot_pos_y"].to_numpy() - 0.07)
assert foot_err.max() < 0.02, f"stalled ramp-hold-snap left on the foot: off by {foot_err.max():.3f} m"
assert np.abs(shin - 0.37).max() < 1e-6, f"shin length changed by {np.abs(shin - 0.37).max():.2e} m (must stay raw)"
seg = out["segment"].to_numpy()
assert sorted(set(seg[seg >= 0])) == [0, 1], f"0.65 s freeze must split into 2 sub-sessions, got {sorted(set(seg))}"
assert (seg[ok] >= 0).all() and (seg[~ok] == -1).all()
assert rep["static_joints"] == ["toeEnd"]
assert np.allclose(np.linalg.norm(out[[f"hand_quat_{a}" for a in "xyzw"]], axis=1), 1, atol=1e-6)
print("OK: spikes + ramp-hold-snap removed (hip max err %.1f cm); untouched" % (hip_err.max() * 100), end=" ")
print(" max err %.1f cm, repaired %.1f cm; dropout flagged"
      % (err[ok & ~fixed].max() * 100, err[ok & fixed].max() * 100))

# fast video (60 fps, more frames than Rokoko's ~30 Hz samples): Rokoko-driven pairing, one frame per sample, no
# interpolated in-between frames; the same repair / freeze split as above
vt2 = np.arange(0, raw["time"].iloc[-1], 1 / 60)
idx2 = np.clip(np.searchsorted(raw["time"].to_numpy(), vt2), 0, len(raw) - 1)
pf2 = raw.iloc[idx2].reset_index(drop=True)
pf2.insert(0, "match_error_ms", (pf2["time"].to_numpy() - vt2) * 1000)
pf2.insert(0, "video_time", vt2)
pf2.insert(0, "frame", np.arange(len(vt2)))
with tempfile.TemporaryDirectory() as d:
    raw.to_csv(Path(d, "x_sync.csv"), index=False)
    pf2.to_csv(Path(d, "x_perframe.csv"), index=False)
    clean(Path(d))
    out2 = pd.read_csv(Path(d, "x_perframe_clean.csv"))
ok2 = out2["valid"].to_numpy() == 1
n_samples = int((np.abs(np.diff(raw[[c for c in raw.columns if c.endswith("_pos_x")]].to_numpy(), axis=0)).max(1) > 0).sum())
assert 0.4 < ok2.mean() < 0.55, f"60 fps: {ok2.mean():.2f} of the frames kept, expected about half (one per Rokoko sample)"
assert ok2.sum() <= n_samples + 1, "60 fps: more kept frames than Rokoko samples"
shin2 = out2["leftLeg_pos_y"].to_numpy() - out2["leftFoot_pos_y"].to_numpy()
assert np.abs(shin2 - 0.37).max() < 1e-6, "60 fps: shin length changed"
print("OK: 60 fps video -> Rokoko-driven pairing, %d of %d frames kept (one per sample)" % (ok2.sum(), len(ok2)))
