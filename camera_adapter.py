#!/usr/bin/env python3
"""Present one session of OUR sync (prepare_mocap.py + clean_mocap.py) in the input format of the hand / camera
stage, so his code runs unchanged on our frames.

    python camera_adapter.py SYNC_SESSION_DIR [--raw RAW_ROOT]

SYNC_SESSION_DIR = Mocap_sync_data/<date>/<pair>/<session>/ (has *_sync.mp4, *_perframe_clean.csv, sync_report.json).
Writes SYNC_SESSION_DIR/camera/:
  session_raw/   the session zip extracted WITHOUT videos and MCAP (it reads ARKit pose, frame timestamps, LiDAR
                 depth and calibration from it, the video from materialized/); the primary video name is a link to our
                 synced video (only its name is used: the camera stage's session converter checks that it exists)
  session/       the camera stage's export layout: session_raw itself (export takes) or session_raw converted with his own
                 session_adapter (phone takes).  iPhone 15 "Wide" takes get reference_camera = ultrawide_camera =
                 primary_camera and an identity ultrawide_to_reference (one camera; exact, not an approximation).
  materialized/  filtered_synced.mp4            -> our *_sync.mp4 (every frame kept)
                 filtered_synced_rokoko.csv     our *_perframe_clean.csv rows (repaired, frame i = row i), Rokoko format
                 filtered_synced_manifest.csv   row i: output time, ORIGINAL video frame (k0 + i), Rokoko row, match error,
                                                quality (ok / repaired / invalid from clean_mocap)
  pipeline_report.json   status PASS (the camera stage's hand stage refuses to start without it)
  kind.txt       glove (Rokoko finger joints present -> Rokoko gloves -> r9m) or noglove (r17c)
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import settings                                                              # noqa: E402
RAW = Path(settings.RAW_ROOT)
SKIP = (".mcap", ".mp4", ".mov")
FINGER = ("leftIndexProximal", "rightIndexProximal")


def find_zip(raw: Path, sd: Path) -> Path:
    key = "_".join(sd.name.split("_")[1:3])                     # session_<YYYYMMDD>_<HHMMSS>_...
    hits = [z for z in raw.glob(f"{sd.parent.relative_to(sd.parents[2])}/*{key}*.zip")]
    hits = hits or list(raw.glob(f"*/*/*{key}*.zip"))
    if not hits:
        raise FileNotFoundError(f"no raw zip for {sd.name}")
    return sorted(hits, key=lambda z: (not z.name.startswith("project1b"), z.name))[0]


def extract(zp: Path, dst: Path) -> Path:
    if not dst.exists():
        tmp = dst.with_name(dst.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zp) as zf:
            for m in zf.infolist():
                if m.is_dir() or m.filename.startswith("__MACOSX") or m.filename.lower().endswith(SKIP):
                    continue
                out = tmp / m.filename
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as s, open(out, "wb") as d:
                    shutil.copyfileobj(s, d, 16 << 20)
        tmp.rename(dst)
    roots = [p.parent for p in dst.rglob("sync_manifest.json")]
    roots += [p.parent for p in dst.rglob("metadata.json") if (p.parent / "mocap").exists()]
    return sorted(roots, key=lambda p: len(p.parts))[0]


def wide_patch(ses: Path):
    for p in (ses / "calibration.json", ses / "recording" / "calibration.json"):
        if not p.exists():
            continue
        c = json.loads(p.read_text())
        if "reference_camera" in c or "primary_camera" not in c:
            continue
        c["reference_camera"] = dict(c["primary_camera"])
        c["ultrawide_camera"] = dict(c["primary_camera"])
        c.setdefault("transforms", {})["ultrawide_to_reference"] = {
            "matrix_3x4_row_major": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], "note": "identity: Wide take, one camera"}
        if "depth_to_primary_transform" in c.get("depth", {}):
            c["depth"]["depth_to_reference_transform"] = c["depth"]["depth_to_primary_transform"]
        if p.is_symlink():
            p.unlink()
        p.write_text(json.dumps(c, indent=1))
        print(f"patched Wide-take calibration: {p}")


def build(sd: Path, raw: Path):
    out = sd / "camera"
    out.mkdir(exist_ok=True)
    sync_mp4 = next(sd.glob("*_sync.mp4"))
    rep = json.loads((sd / "sync_report.json").read_text())
    clean = pd.read_csv(next(sd.glob("*_perframe_clean.csv")))
    n = len(clean)

    # --- session in the camera stage's export layout
    root = extract(find_zip(raw, sd), out / "session_raw")
    if (root / "sync_manifest.json").exists():
        ses = root
    else:
        meta = json.loads((root / "metadata.json").read_text())
        for k in ("primary_video_filename",):                    # name only: converter checks it exists
            f = root / meta[k]
            if not f.exists():
                f.symlink_to(sync_mp4)
        if not list(root.glob("rokoko_recording_*.csv")):        # converter wants exactly one: our paired CSV
            src = next(raw.glob(f"*/*/{rep['csv'].replace('_raw', '')}"), None) or next(raw.glob(f"*/*/{rep['csv']}"))
            (root / src.name).symlink_to(src)
        ses = out / "session"
        if not (ses / "sync_manifest.json").exists():
            shutil.rmtree(ses, ignore_errors=True)
            sys.path.insert(0, str(HERE))
            import session_adapter
            session_adapter.prepare(root, ses)
    if ses == root and not (out / "session").exists():
        (out / "session").symlink_to(root)
    wide_patch(ses)

    # --- original frame of our first synced frame (k0): our cut starts video_trim_head_s after the first frame
    man = json.loads((ses / "sync_manifest.json").read_text())
    prim = next(v for v in man["videos"] if v.get("role") == "primary")
    ft = pd.read_csv(ses / "frame_timestamps.csv")
    ft = ft[(ft.role == "primary") & (ft.video_file == Path(prim["archive_path"]).name)].sort_values("frame_index")
    pts = (ft.capture_pts_us.to_numpy(float) - ft.capture_pts_us.iloc[0]) / 1e6
    head = rep["sync"]["video_trim_head_s"]
    k0 = int(np.argmin(np.abs(pts - head)))
    fidx = ft.frame_index.to_numpy(int)
    if k0 + n > len(fidx):
        raise RuntimeError(f"{sd.name}: {n} synced frames from original frame {k0} run past the {len(fidx)} source frames")

    # --- materialized 1:1 pair + manifest
    m = out / "materialized"
    m.mkdir(exist_ok=True)
    v = m / "filtered_synced.mp4"
    if v.is_symlink() or v.exists():
        v.unlink()
    v.symlink_to(sync_mp4)
    jc = [c for c in clean.columns if "_pos_" in c or "_quat_" in c]
    rok = pd.concat([clean[["video_time"]].rename(columns={"video_time": "time"}), clean[jc]], axis=1)
    rok.to_csv(m / "filtered_synced_rokoko.csv", index=False, float_format="%.7g")
    sync_csv = pd.read_csv(next(sd.glob("*_sync.csv")), usecols=[0]).iloc[:, 0].to_numpy(float)
    rows = np.clip(np.searchsorted(sync_csv, clean["time"].to_numpy(float) - 1e-6), 0, len(sync_csv) - 1)
    quality = np.where(clean["valid"] == 0, "invalid", np.where(clean["repaired"] > 0, "repaired", "ok"))
    pd.DataFrame({"output_row_id": np.arange(n), "output_frame_id": np.arange(n),
                  "output_time_s": clean["video_time"].round(6), "source_rokoko_row_id": rows,
                  "source_rokoko_time_s": clean["time"], "source_real_frame_id": fidx[k0:k0 + n],
                  "source_capture_pts_s": np.round(pts[k0:k0 + n], 6), "pair_delta_ms": clean["match_error_ms"],
                  "noise_quality": quality, "noise_flags": ""}).to_csv(m / "filtered_synced_manifest.csv", index=False)
    (out / "pipeline_report.json").write_text(json.dumps({
        "status": "PASS", "source": "final_code: our sync (prepare_mocap / fit_rokoko) + clean_mocap (repaired, none dropped)",
        "materialized": {"output_rows": n, "video": str(sync_mp4), "first_source_frame": int(fidx[k0]),
                         "rows_repaired": int((quality == "repaired").sum()), "rows_invalid": int((quality == "invalid").sum())}},
        indent=1))
    fingers = all(c in clean and clean[c].abs().max() > 0 for c in (f"{f}_pos_x" for f in FINGER))
    kind = "glove" if fingers else "noglove"
    (out / "kind.txt").write_text(kind + "\n")
    print(f"{sd.name}: {n} rows, original frames {fidx[k0]}..{fidx[k0 + n - 1]}, {kind}, session {ses}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session_dir", type=Path)
    p.add_argument("--raw", type=Path, default=RAW)
    a = p.parse_args()
    build(a.session_dir.resolve(), a.raw)


if __name__ == "__main__":
    main()
