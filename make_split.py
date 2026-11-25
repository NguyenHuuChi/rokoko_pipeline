#!/usr/bin/env python3
"""80/20 train/test split of the final joints, balanced by task and difficulty.

    python make_split.py [OUT_ROOT] [--test 0.2] [--seed 0]

Inputs (OUT_ROOT of settings.py):
  task_labels.csv              session,start_s,end_s,task,notes - activity per time span, labelled from the video
  <session>/camera/joints.csv    final joints (dead-signal frames already dropped by drop_dead.py)
  <session>/*_perframe_clean.csv  Rokoko per video frame (posture: hip height above the lower foot)

Task = video activity, except sitting vs standing, which comes from Rokoko (a head camera looking down cannot tell
them apart; Rokoko hip-above-foot height is bimodal: standing ~0.8 m, sitting < 0.6 m): table_* and sit_relax /
stand_idle become *_sit or *_stand by the chunk's median height. Small activities are grouped for balancing
(task_group), the fine name is kept (task).

Every labelled span is cut into chunks of ~CHUNK_S seconds. Difficulty per chunk, each scored as a percentile rank over
all chunks and averaged:
  motion   median body-joint speed (Rokoko, m/s)
  hands    fraction of hand-frames the hand stage did not detect in the video (interpolated / held / Rokoko offset / none)
  frame    fraction of body joints projected outside the image
-> easy / medium / hard (tertiles). Within each task_group chunks are sorted by difficulty score and cut into blocks
of 1/test (5) chunks; one random chunk per block goes to test, so every group is split ~80/20 along its whole
difficulty range. A group too small for a block (< 3 chunks) stays in train. Frames within GAP_S of a chunk in the other split are marked gap
(dropped from both), so near-identical neighbouring frames never sit on both sides.

need_to_be_reviewed (set aside before balancing): the chunk's sync is not confirmed by <session>/camera/sync_check.csv
(sync_check.py: Rokoko head vs ARKit camera rotation per 30 s window) - a window overlapping the chunk peaks
(corr >= MIN_CORR) at |lag| > MAX_LAG frames, or the session's head -> camera calibration residual is > MAX_CALIB deg
(sync_check.json; good takes 1-4), or the session has no sync check.

Writes OUT_ROOT/split/frames.csv (session, frame, split, task, task_group, difficulty, chunk), chunks.csv, summary.txt.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

CHUNK_S = 60.0
MIN_CORR, MAX_LAG, MAX_CALIB = 0.5, 2, 10.0
GAP_S = 1.0
SIT_H = 0.62
W, H = 1920, 1440
GROUP = {"exercise_arms_stand": "exercise", "exercise_arms_sit": "exercise", "exercise_legs": "exercise",
         "billiards": "games", "new:foosball": "games", "walk": "walk_carry", "carry": "walk_carry",
         "transition": "other", "crouch_pick": "other", "no_view": "other"}
BODY = ["hip", "spine", "chest", "neck", "head", "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
        "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand", "leftUpLeg", "leftLeg", "leftFoot",
        "rightUpLeg", "rightLeg", "rightFoot"]


def session_dir(root, s):
    return glob.glob(f"{root}/*/*/session_{s}_*")[0]


def chunks_of(root, labels):
    rows = []
    for s, g in labels.groupby("session"):
        sd = session_dir(root, s)
        j = pd.read_csv(f"{sd}/camera/joints.csv")
        rk = pd.read_csv(glob.glob(f"{sd}/*_perframe_clean.csv")[0],
                         usecols=lambda c: c in ("video_time",) or c.endswith(("_pos_x", "_pos_y", "_pos_z")))
        f = j["output_row_id"].to_numpy()
        t = j["time_s"].to_numpy()
        height = (rk["hip_pos_y"] - np.minimum(rk["leftFoot_pos_y"], rk["rightFoot_pos_y"])).to_numpy()[f]
        body = [b for b in BODY if f"{b}_pos_x" in rk]
        P = np.stack([rk[[f"{b}_pos_{a}" for a in "xyz"]].to_numpy() for b in body], 1)       # (T, J, 3)
        speed = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1) * 30][f]       # m/s, per video frame
        src = [c for c in j.columns if c.endswith("_source")]
        miss = (j[src] != "detected").to_numpy().mean(1)
        uv = [(f"{b}_u", f"{b}_v") for b in body if f"{b}_u" in j]
        U = np.stack([j[u] for u, _ in uv], 1)
        V = np.stack([j[v] for _, v in uv], 1)
        out = ~((U >= 0) & (U < W) & (V >= 0) & (V < H))                                      # NaN counts as out
        for _, r in g.iterrows():
            m = np.flatnonzero((t >= r.start_s) & (t < r.end_s))
            if not len(m):
                continue
            n = max(1, round((r.end_s - r.start_s) / CHUNK_S))
            edges = np.linspace(r.start_s, r.end_s, n + 1)
            for k in range(n):
                c = m[(t[m] >= edges[k]) & (t[m] < edges[k + 1])]
                if not len(c):
                    continue
                task = r.task
                if task.startswith("table_") or task in ("sit_relax", "stand_idle", "sit_cross_legged"):
                    sit = np.median(height[c]) < SIT_H
                    task = {"table_sit": "table", "table_stand": "table", "stand_idle": "idle",
                            "sit_relax": "idle"}.get(task, task)
                    if task != "sit_cross_legged":
                        task += "_sit" if sit else "_stand"
                rows.append(dict(session=s, chunk=f"{s}_{len(rows):04d}", t0=t[c[0]], t1=t[c[-1]],
                                 frames=f[c], n=len(c), task=task, video_task=r.task,
                                 task_group=GROUP.get(task, task), speed=float(np.median(speed[c])),
                                 hand_miss=float(miss[c].mean()), out_frame=float(out[c].mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=__import__("settings").OUT_ROOT)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    labels = pd.read_csv(f"{a.root}/task_labels.csv", dtype={"session": str})
    done = {s for s in labels.session.unique() if os.path.exists(f"{session_dir(a.root, s)}/camera/joints.csv")}
    missing = sorted(set(labels.session) - done)
    C = chunks_of(a.root, labels[labels.session.isin(done)])

    score = sum(C[k].rank(pct=True) for k in ("speed", "hand_miss", "out_frame")) / 3
    C["difficulty"] = pd.qcut(score, 3, labels=["easy", "medium", "hard"]).astype(str)
    rng = np.random.default_rng(a.seed)
    C["score"] = score
    C["split"] = "train"
    C["review"] = ""
    for s, g in C.groupby("session"):
        hd = f"{session_dir(a.root, s)}/hung"
        if not os.path.exists(f"{hd}/sync_check.csv"):
            C.loc[g.index, "review"] = "no sync check"
            continue
        calib = json.load(open(f"{hd}/sync_check.json"))["calib_residual_deg"]
        Wd = pd.read_csv(f"{hd}/sync_check.csv")
        Wd = Wd[(Wd["corr"] >= MIN_CORR) & (Wd.lag.abs() > MAX_LAG)]
        for i, r in g.iterrows():
            f0, f1 = r.frames[0], r.frames[-1]
            off = Wd[(Wd.f1 >= f0) & (Wd.f0 <= f1)]
            if calib > MAX_CALIB:
                C.at[i, "review"] = f"calibration {calib:.1f} deg"
            elif len(off):
                C.at[i, "review"] = f"sync off {off.lag.min():+d}..{off.lag.max():+d} frames"
    C.loc[C.review != "", "split"] = "need_to_be_reviewed"
    k = round(1 / a.test)                       # 5 for 20 %: one chunk of every k, difficulty-sorted, goes to test
    for _, g in C[C.split == "train"].groupby("task_group"):
        idx = g.sort_values("score").index.to_numpy()
        seen = got = 0
        for b in range(0, len(idx), k):
            blk = idx[b:b + k]
            seen += C.loc[blk, "n"].sum()
            if len(blk) * 2 < k:                # a short last block (< k/2 chunks) stays in train
                continue
            blk = rng.permutation(blk)          # ties -> random
            i = blk[np.argmin(np.abs(got + C.loc[blk, "n"].to_numpy() - a.test * seen))]   # keep test frames on target
            C.at[i, "split"] = "test"
            got += C.at[i, "n"]

    F = []
    for s, g in C.groupby("session"):
        g = g.sort_values("t0")
        for _, r in g.iterrows():
            F.append(pd.DataFrame(dict(session=s, frame=r.frames, split=r.split, task=r.task, task_group=r.task_group,
                                       difficulty=r.difficulty, chunk=r.chunk)))
    F = pd.concat(F, ignore_index=True).sort_values(["session", "frame"], ignore_index=True)
    # gap: frames within GAP_S of a frame of the same session in the other split
    g = int(GAP_S * 30)
    for s, idx in F.groupby("session").groups.items():
        fr = F.loc[idx, "frame"].to_numpy()
        sp = F.loc[idx, "split"].to_numpy()
        flip = np.flatnonzero((sp[1:] != sp[:-1]) & np.isin(sp[1:], ("train", "test")) & np.isin(sp[:-1], ("train", "test")))
        gap = np.zeros(len(fr), bool)
        for b in flip:
            gap |= np.abs(fr - fr[b + 1]) <= g
        F.loc[idx[gap], "split"] = "gap"

    os.makedirs(f"{a.root}/split", exist_ok=True)
    F.to_csv(f"{a.root}/split/frames.csv", index=False)
    C.drop(columns="frames").to_csv(f"{a.root}/split/chunks.csv", index=False)
    lines = [f"sessions {len(done)} (missing joints: {' '.join(missing) or 'none'})",
             f"chunks {len(C)}  frames {len(F)}  = {len(F) / 30 / 3600:.2f} h at 30 fps", "",
             str(F.split.value_counts().rename("frames")), "",
             "frames by task_group x split:", str(pd.crosstab(F.task_group, F.split, margins=True)), "",
             "test share by task_group:", str((F[F.split == "test"].groupby("task_group").size()
                                               / F[F.split.isin(["train", "test"])].groupby("task_group").size()).round(3)), "",
             "need_to_be_reviewed chunks:", C[C.review != ""][["chunk", "t0", "t1", "task", "review"]].to_string(index=False), "",
             "frames by difficulty x split:", str(pd.crosstab(F.difficulty, F.split, margins=True)), "",
             "frames by task x split:", str(pd.crosstab(F.task, F.split, margins=True))]
    open(f"{a.root}/split/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
