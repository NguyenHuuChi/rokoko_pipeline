#!/usr/bin/env python3
"""Accept a session folder in either layout (any folder name) and give the pipeline the layout it reads.

  export layout   (Project1B offline export: sync_manifest.json, frame_timestamps.csv, pose.csv, depth/,
                   recording/<first-frame time>.mp4)                              -> used as it is
  raw layout      (the phone's own session folder: video.mp4, video_standard_reference.mp4, metadata.json with the
                   clock anchors, calibration.json, mocap/video_frame_index.csv, mocap/samples/pose.csv,
                   mocap/samples/tracking_state.csv, mocap/depth/)               -> converted into <staging>/

Both need exactly one rokoko_recording_*.csv (the Rokoko take; its name holds its start time).
The conversion only reads the session and writes <staging>/ (big files are symbolic links).  It derives what the
export layout adds, with the export's own rules (checked on 01/10 10:36, where both layouts exist):
  wall clock     nearest clock anchor (metadata.json clock_sync) + monotonic difference
  video names    local wall-clock time of the first frame, YYYY-MM-DD-HH-MM-SS-mmm(.mp4 | _reference.mp4)
  depth frames   primary_frame_index = nearest primary frame on the monotonic clock, reference = same timestamp,
                 *_pts_us = capture_pts_us from the video's first frame
  time zone      from the session's own start name (metadata sensor_mcap_filename session_data_YYYYMMDD_HHMMSS)
                 against the recording-start anchor; else the computer's time zone; --tz-offset-min overrides
    .venv/bin/python session_adapter.py <session> <staging>      (run_all.py calls prepare() itself)
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = "rokoko-pipeline/adapted_session/v1"


def layout(session: Path) -> str:
    """'export', 'raw' or 'unknown'."""
    if (session / "sync_manifest.json").exists() or (session / "frame_timestamps.csv").exists():
        return "export"
    if (session / "metadata.json").exists() and (session / "mocap" / "video_frame_index.csv").exists():
        return "raw"
    return "unknown"


def check(session: Path) -> list[str]:
    """Problems that stop the run (empty = usable)."""
    if not session.is_dir():
        return [f"session folder not found: {session}"]
    errs = []
    n = len(list(session.glob("rokoko_recording_*.csv")))
    if n != 1:
        errs.append(f"need exactly 1 rokoko_recording_*.csv in {session}, found {n}")
    kind = layout(session)
    if kind == "unknown":
        errs.append(f"unknown session layout in {session}: expected sync_manifest.json / frame_timestamps.csv (export) "
                    "or metadata.json + mocap/video_frame_index.csv (phone session)")
    elif kind == "raw":
        meta = json.loads((session / "metadata.json").read_text())
        if not (meta.get("clock_sync") or {}).get("anchors"):
            errs.append("metadata.json has no clock_sync anchors: wall-clock time of the video is unknown")
        for k in ("primary_video_filename", "reference_video_filename"):
            f = meta.get(k)
            if k.startswith("primary") and (not f or not (session / f).is_file()):
                errs.append(f"primary video {f!r} (metadata.json {k}) not found")
        for rel in ("calibration.json", "mocap/samples/pose.csv", "mocap/samples/tracking_state.csv", "mocap/depth/depth_frames.csv"):
            if not (session / rel).exists():
                errs.append(f"{rel} missing")
    return errs


# --------------------------------------------------------------------------- wall clock
class Clock:
    def __init__(self, anchors, tz_offset_min):
        self.t = np.array([int(a["timestamp_ns"]) for a in anchors], np.int64)
        self.u = np.array([int(a["unix_time_ms"]) for a in anchors], np.int64)
        self.tz = timezone(timedelta(minutes=tz_offset_min))
        self.tz_offset_min = tz_offset_min

    def unix_us(self, mono):
        """Wall clock in integer microseconds: anchor + monotonic difference truncated to the microsecond (as the export)."""
        m = np.asarray(mono, np.int64)
        k = np.abs(m[:, None] - self.t[None]).argmin(1)                     # nearest anchor
        d = m - self.t[k]
        return self.u[k] * 1000 + np.sign(d) * (np.abs(d) // 1000)

    def unix_ms(self, mono):
        return self.unix_us(mono) / 1000.0

    def strings(self, unix_us):
        """utc_time, local_time (milliseconds truncated, as in the export)."""
        out = []
        for v in np.asarray(unix_us, np.int64):
            dt = datetime.fromtimestamp((int(v) // 1000) / 1000.0, timezone.utc)
            lt = dt.astimezone(self.tz)
            out.append((dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{(int(v) // 1000) % 1000:03d}Z", lt.isoformat(timespec="milliseconds")))
        return out

    def name(self, unix_us):
        v = int(unix_us)
        lt = datetime.fromtimestamp((v // 1000) / 1000.0, timezone.utc).astimezone(self.tz)
        return lt.strftime("%Y-%m-%d-%H-%M-%S-") + f"{(v // 1000) % 1000:03d}"


def time_zone(meta, anchors, override):
    if override is not None:
        return int(override), "--tz-offset-min"
    start = next((a for a in anchors if a.get("reason") == "recording_start"), anchors[0])
    m = re.search(r"(\d{8})_(\d{6})", str(meta.get("sensor_mcap_filename", "")))
    if m:
        local = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        utc = datetime.fromtimestamp(float(start["unix_time_ms"]) / 1000.0, timezone.utc).replace(tzinfo=None)
        off = round((local - utc).total_seconds() / 900.0) * 15
        if abs((local - utc).total_seconds() - 60 * off) <= 120:
            return int(off), f"metadata sensor_mcap_filename {m.group(0)} vs recording-start anchor"
    off = datetime.fromtimestamp(float(start["unix_time_ms"]) / 1000.0).astimezone().utcoffset()
    return int(off.total_seconds() // 60), "this computer's time zone"


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def _write(df, path, cols):
    df[cols].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


# --------------------------------------------------------------------------- conversion
def convert(session: Path, staging: Path, tz_offset_min=None, log=print) -> dict:
    meta = json.loads((session / "metadata.json").read_text())
    anchors = meta["clock_sync"]["anchors"]
    tz, tz_src = time_zone(meta, anchors, tz_offset_min)
    clk = Clock(anchors, tz)
    staging.mkdir(parents=True, exist_ok=False)
    rokoko = next(session.glob("rokoko_recording_*.csv"))
    _link(rokoko, staging / rokoko.name)
    # videos, named by the wall-clock time of their first frame
    vfi = pd.read_csv(session / "mocap" / "video_frame_index.csv")
    videos, frames = [], []
    for role, key, suffix in (("primary", "primary_video_filename", ""), ("reference", "reference_video_filename", "_reference")):
        src = meta.get(key)
        rows = vfi[vfi.role == role].sort_values("frame_index")
        if not src or not (session / src).is_file() or rows.empty:
            continue
        us = clk.unix_us(rows.log_time_ns.to_numpy()); w = us / 1000.0
        name = clk.name(us[0]) + suffix + Path(src).suffix
        _link(session / src, staging / "recording" / name)
        s = clk.strings(us)
        frames.append(pd.DataFrame({"video_file": name, "role": role, "frame_index": rows.frame_index.to_numpy(),
                                    "capture_pts_us": rows.capture_pts_us.to_numpy(), "monotonic_ns": rows.log_time_ns.to_numpy(),
                                    "unix_time_ms": w, "utc_time": [a for a, _ in s], "local_time": [b for _, b in s]}))
        videos.append({"role": role, "original_filename": src, "archive_path": f"recording/{name}", "named_by_first_frame": True,
                       "frame_count": int(len(rows)), "first_frame_index": int(rows.frame_index.iloc[0]),
                       "first_frame_monotonic_ns": int(rows.log_time_ns.iloc[0]), "first_frame_unix_time_ms": round(float(w[0]), 3),
                       "first_frame_local_time": s[0][1]})
    pd.concat(frames).to_csv(staging / "frame_timestamps.csv", index=False, float_format="%.3f")
    for f in ("calibration.json", "metadata.json", "recording_selection.json", "session_summary.txt", meta.get("sensor_mcap_filename", "")):
        if f and (session / f).is_file():
            _link(session / f, staging / "recording" / f)
    # pose + tracking state, with the wall-clock columns
    wall = lambda df: (lambda us, s: df.assign(unix_time_ms=us / 1000.0, utc_time=[a for a, _ in s], local_time=[b for _, b in s]))(
        clk.unix_us(df.monotonic_ns.to_numpy()), clk.strings(clk.unix_us(df.monotonic_ns.to_numpy())))
    ts = pd.read_csv(session / "mocap" / "samples" / "tracking_state.csv").sort_values("monotonic_ns")
    pose = pd.read_csv(session / "mocap" / "samples" / "pose.csv").sort_values("monotonic_ns")
    j = np.searchsorted(ts.monotonic_ns.to_numpy(), pose.monotonic_ns.to_numpy(), "right") - 1   # latest state at or before
    pose["tracking_state"] = np.where(j >= 0, ts.state.to_numpy()[np.clip(j, 0, None)], "")
    pose["tracking_reason"] = np.where(j >= 0, ts.reason.to_numpy()[np.clip(j, 0, None)], "")
    wc = ["monotonic_ns", "unix_time_ms", "utc_time", "local_time"]
    _write(wall(pose), staging / "pose.csv", wc + ["pose_time_ns", "source", "frame_id", "position_x", "position_y", "position_z",
                                                   "orientation_x", "orientation_y", "orientation_z", "orientation_w",
                                                   "tracking_state", "tracking_reason"])
    _write(wall(ts.rename(columns={})), staging / "tracking_state.csv", wc + ["state", "reason"])
    # depth: maps linked, depth_frames.csv completed with the video frame of each map
    dsrc = session / "mocap" / "depth"
    dfm = pd.read_csv(dsrc / "depth_frames.csv")
    if "primary_frame_index" not in dfm.columns:
        for role in ("primary", "reference"):
            rows = vfi[vfi.role == role].sort_values("frame_index")
            if rows.empty:
                dfm[f"{role}_frame_index"], dfm[f"{role}_pts_us"] = "", ""
                continue
            lt = rows.log_time_ns.to_numpy(np.int64)
            k = np.clip(np.searchsorted(lt, dfm.log_time_ns.to_numpy(np.int64)), 1, len(lt) - 1)
            k = np.where(np.abs(lt[k - 1] - dfm.log_time_ns.to_numpy(np.int64)) <= np.abs(lt[k] - dfm.log_time_ns.to_numpy(np.int64)), k - 1, k)
            dfm[f"{role}_frame_index"] = rows.frame_index.to_numpy()[k]
            dfm[f"{role}_pts_us"] = rows.capture_pts_us.to_numpy()[k] - rows.capture_pts_us.iloc[0]
    (staging / "depth").mkdir()
    dfm.to_csv(staging / "depth" / "depth_frames.csv", index=False)
    for p in dsrc.iterdir():
        if p.suffix == ".png" or p.name in ("camera_info.json", "video_frame_index.csv"):
            (staging / "depth" / p.name).symlink_to(p.resolve())
    # clock anchors + manifest
    pd.DataFrame(anchors)[["sequence", "reason", "timestamp_ns", "monotonic_before_ns", "unix_time_ms", "monotonic_after_ns",
                           "sampling_span_ns"]].to_csv(staging / "clock_anchors.csv", index=False)
    man = {"schema_version": SCHEMA, "converted_from": str(session.resolve()), "converted_at_utc": datetime.now(timezone.utc).isoformat(),
           "source_layout": "raw phone session (metadata.json + mocap/)", "clock_sync": meta["clock_sync"], "videos": videos,
           "frame_timestamps": {"csv": "frame_timestamps.csv", "wall_clock": "nearest clock anchor + monotonic difference",
                                "timezone_offset_minutes": tz, "timezone_source": tz_src},
           "depth": {"frames_csv": "depth/depth_frames.csv", "frames": int(len(dfm)),
                     "frame_join": "primary = nearest primary frame (monotonic clock), reference = same timestamp"},
           "pose": {"file": "pose.csv", "rows": int(len(pose)), "tracking_state": "latest state at or before each pose"},
           "rokoko_csv": rokoko.name}
    (staging / "sync_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    log(f"session converted (phone layout -> export layout): {staging}  | time zone UTC{tz / 60:+g} ({tz_src}) | "
        f"primary video {videos[0]['archive_path']} ({videos[0]['frame_count']} frames), depth {len(dfm)}, pose {len(pose)}")
    return man


def describe(session: Path, tz_offset_min=None) -> dict:
    """What the pipeline will use, without writing anything (run_all.py --check)."""
    session = Path(session).expanduser().resolve()
    out = {"layout": layout(session), "rokoko_csv": next(session.glob("rokoko_recording_*.csv")).name}
    if out["layout"] == "export":
        man = json.loads((session / "sync_manifest.json").read_text()) if (session / "sync_manifest.json").exists() else {}
        prim = [v for v in man.get("videos", []) if v.get("role") == "primary"]
        out["primary_video"] = prim[0]["archive_path"] if prim else "(single MP4 in the folder)"
        return out
    meta = json.loads((session / "metadata.json").read_text())
    anchors = meta["clock_sync"]["anchors"]
    tz, src = time_zone(meta, anchors, tz_offset_min)
    clk = Clock(anchors, tz)
    rows = pd.read_csv(session / "mocap" / "video_frame_index.csv")
    rows = rows[rows.role == "primary"].sort_values("frame_index")
    us = clk.unix_us(rows.log_time_ns.to_numpy()[[0, -1]])
    s = clk.strings(us)
    out.update(primary_video=meta.get("primary_video_filename"), primary_frames=int(len(rows)),
               first_frame_local_time=s[0][1], last_frame_local_time=s[1][1],
               video_name_for_sync=clk.name(us[0]) + Path(meta.get("primary_video_filename", ".mp4")).suffix,
               time_zone=f"UTC{tz / 60:+g} ({src})", clock_anchors=len(anchors))
    return out


def prepare(session: Path, staging: Path, tz_offset_min=None, log=print) -> Path:
    """The session folder to give the pipeline: the session itself (export layout) or its conversion in staging."""
    session = Path(session).expanduser().resolve()
    errs = check(session)
    if errs:
        raise ValueError("; ".join(errs))
    if layout(session) == "export":
        return session
    convert(session, Path(staging), tz_offset_min, log)
    return Path(staging)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    print(prepare(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None))
