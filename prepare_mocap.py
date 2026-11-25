#!/usr/bin/env python3
"""Pair iPhone session zips with Rokoko CSVs and stage them for fit_rokoko.py.

Raw layout: <raw>/<dd-mm-yyyy>/<pair>/{session_*.zip, rokoko_recording_*.csv}
Each zip holds video.mp4 + metadata.json (clock_sync: unix_ms <-> monotonic ns) +
mocap/video_frame_index.csv (per-frame capture pts, monotonic). That gives the exact
wall-clock start of video frame 0, which we write into the file name so fit_rokoko
reads it with ms precision. Each video is paired with the CSV it overlaps most.

    python prepare_mocap.py RAW OUT --scan        # print pairing only
    python prepare_mocap.py RAW OUT -j 6          # stage + run fit_rokoko into OUT
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

FIT = Path(__file__).with_name("fit_rokoko.py")
CLEAN = Path(__file__).with_name("clean_mocap.py")
EXTRAS = ("metadata.json", "calibration.json", "session_summary.txt")


def zip_member(zf: zipfile.ZipFile, suffix: str) -> str | None:
    hits = [n for n in zf.namelist() if ("/" + n).endswith(suffix) and not n.startswith("__MACOSX")]
    return hits[0] if hits else None


def primary_video(zf: zipfile.ZipFile) -> str:
    hits = [n for n in zf.namelist() if n.endswith(".mp4") and not n.startswith("__MACOSX")
            and "reference" not in n.rsplit("/", 1)[-1]]
    assert len(hits) == 1, hits
    return hits[0]


def primary_rows(zf: zipfile.ZipFile, member: str) -> list:
    with zf.open(member) as f:
        return [r for r in csv.DictReader(io.TextIOWrapper(f)) if r["role"] == "primary"]


def video_start(zp: Path) -> dict:
    """Wall-clock start/end (epoch s) of the primary video inside a session zip.

    project1b_take export: frame_timestamps.csv already holds unix_time_ms per frame.
    Legacy session export: first clock anchor (unix_ms <-> monotonic ns) + per-frame
    capture pts (same monotonic clock) from mocap/video_frame_index.csv.
    """
    with zipfile.ZipFile(zp) as zf:
        if (ft := zip_member(zf, "/frame_timestamps.csv")):
            rows = primary_rows(zf, ft)
            t0, t1 = float(rows[0]["unix_time_ms"]) / 1000, float(rows[-1]["unix_time_ms"]) / 1000
        else:
            a = json.loads(zf.read(zip_member(zf, "/metadata.json")))["clock_sync"]["anchors"][0]
            rows = primary_rows(zf, zip_member(zf, "/mocap/video_frame_index.csv"))
            t0 = a["unix_time_ms"] / 1000 + (int(rows[0]["capture_pts_us"]) * 1000 - a["timestamp_ns"]) / 1e9
            t1 = t0 + (int(rows[-1]["capture_pts_us"]) - int(rows[0]["capture_pts_us"])) / 1e6
        video = primary_video(zf)
    return {"zip": zp, "video": video, "start": t0, "end": t1, "frames": len(rows)}


def csv_span(cp: Path) -> dict:
    ts = datetime.strptime("_".join(cp.stem.split("_")[2:5]), "%Y%m%d_%H%M%S_%f").timestamp()
    with open(cp, "rb") as f:
        f.readline()
        t_first = float(f.readline().split(b",", 1)[0])
        f.seek(max(0, f.seek(0, 2) - 65536))
        t_last = float(f.read().strip().splitlines()[-1].split(b",", 1)[0])
    return {"csv": cp, "start": ts + t_first, "end": ts + t_last}


def scan(raw: Path) -> list:
    jobs = []
    for d in sorted(p for p in raw.glob("*/*") if p.is_dir()):
        zips, csvs = sorted(d.glob("*.zip")), sorted(d.glob("rokoko_recording_*.csv"))
        if not zips:
            continue
        cs = [csv_span(c) for c in csvs]
        used, seen, seen_start = set(), {}, {}
        for z in zips:
            try:
                v = video_start(z)
            except zipfile.BadZipFile as e:
                print(f"  [corrupt zip] {d.relative_to(raw)}/{z.name}: {e} -> skipped")
                continue
            except KeyError:                               # camera-only take: no frame timestamps / clock sync
                print(f"  [no timing] {d.relative_to(raw)}/{z.name}: no frame_timestamps or clock_sync -> skipped")
                continue
            key = "_".join(session_name(z).split("_")[:6])  # session_YYYYMMDD_HHMMSS_device_n_hash
            if key in seen:                                # same take exported in both layouts
                print(f"  [duplicate] {z.name} == {seen[key].name}, start diff "
                      f"{(v['start'] - seen_start[key]) * 1000:+.1f} ms -> skipped")
                continue
            seen[key], seen_start[key] = z, v["start"]
            ov = [(min(v["end"], c["end"]) - max(v["start"], c["start"]), c) for c in cs]
            best = max(ov, key=lambda o: o[0], default=(0, None))
            v["csv"] = best[1]["csv"] if best[0] > 0 else None
            v["overlap"] = best[0]
            v["rel"] = d.relative_to(raw)
            used.add(v["csv"])
            jobs.append(v)
        for c in cs:
            if c["csv"] not in used:
                print(f"  [unpaired csv] {d.relative_to(raw)}/{c['csv'].name} "
                      f"({c['end'] - c['start']:.0f}s)")
    return jobs


def session_name(zp: Path) -> str:
    return zp.stem.replace("project1b_take_", "")


def run_job(job: dict, out: Path, stage: Path, fit_args: list, ffmpeg_dir: str, clean_py: str) -> tuple:
    name = session_name(job["zip"])
    sdir = stage / job["rel"] / name
    odir = out / job["rel"] / name
    if (odir / "sync_report.json").exists():
        return name, "exists"
    sdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(job["start"]).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    mp4 = sdir / f"{stamp}.mp4"
    with zipfile.ZipFile(job["zip"]) as zf:
        if not mp4.exists():
            with zf.open(job["video"]) as src, open(mp4, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
        odir.mkdir(parents=True, exist_ok=True)
        for e in EXTRAS:
            if (m := zip_member(zf, "/" + e)):
                (odir / e).write_bytes(zf.read(m))
    link = sdir / job["csv"].name
    if not link.exists():
        link.symlink_to(job["csv"])
    log = odir / "fit_rokoko.log"
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, str(FIT), str(sdir), "--out-dir", str(odir), *fit_args],
                            stdout=lf, stderr=subprocess.STDOUT,
                            env={**os.environ, "PATH": f"{ffmpeg_dir}:{os.environ['PATH']}"}).returncode
    if rc == 0 and (odir / "sync_report.json").exists():
        shutil.rmtree(sdir)                                 # staged copy no longer needed
        if clean_py:                                        # spike filter -> *_perframe_clean.csv
            with open(odir / "clean_mocap.log", "w") as lf:
                rc = subprocess.run([clean_py, str(CLEAN), str(odir)], stdout=lf,
                                    stderr=subprocess.STDOUT).returncode
            if rc != 0:
                return name, f"synced, clean FAILED rc={rc}, see {odir / 'clean_mocap.log'}"
        return name, "ok"
    return name, f"FAILED rc={rc}, see {log}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--scan", action="store_true")
    p.add_argument("-j", "--jobs", type=int, default=6)
    p.add_argument("--ffmpeg-dir", default="", help="dir with ffmpeg/ffprobe >= 5 (fit_rokoko needs -fps_mode)")
    p.add_argument("--only", default="", help="comma-separated substrings of session names")
    p.add_argument("--clean-python", default="",
                   help="python with pandas+scipy to run clean_mocap.py after each sync ('' = skip)")
    p.add_argument("--fit-args", default="--measure --per-frame --plot --preset fast")
    a = p.parse_args()

    jobs = scan(a.raw)
    for j in jobs:
        print(f"{str(j['rel']):28s} {session_name(j['zip'])[:44]:44s} "
              f"{datetime.fromtimestamp(j['start']):%H:%M:%S.%f} {j['end'] - j['start']:7.1f}s  "
              f"-> {j['csv'].name if j['csv'] else 'NO CSV'}  overlap {j['overlap']:.1f}s")
    if a.scan:
        return
    todo = [j for j in jobs if j["csv"]
            and (not a.only or any(k in j["zip"].name for k in a.only.split(",")))]
    stage = a.out / ".staging"
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_job, j, a.out, stage, a.fit_args.split(), a.ffmpeg_dir, a.clean_python) for j in todo]
        for f in futs:
            print(*f.result(), flush=True)


if __name__ == "__main__":
    main()
