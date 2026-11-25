#!/usr/bin/env python3
"""Raw app upload (5090 platform: raw/phone/<uuid>/ = video.mp4 + metadata.json + calibration.json + session_data_*.mcap)
-> the session zip layout prepare_mocap.py / camera_adapter.py read (the phone's own export: <session>/mocap/... files
rebuilt from the MCAP), plus the paired Rokoko CSV linked next to it.

    python mcap_session.py PHONE_DIR CSV OUT_PAIR_DIR        (~/envs/handseg: mcap, mcap-ros2-support, cv2, numpy)
    python mcap_session.py --check SESSION_ZIP               rebuild mocap/ from the zip's own MCAP, compare with its export

Writes OUT_PAIR_DIR/session_<YYYYMMDD_HHMMSS>_<device>_<hash>.zip (stored, not compressed) and
OUT_PAIR_DIR/rokoko_recording_<YYYYMMDD_HHMMSS_mmm>.csv -> CSV (the name the camera stage's session converter globs for; a
rokoko_manus_* CSV is a Rokoko CSV plus Manus hand columns).
"""
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

KEEP = ("calibration.json", "metadata.json", "recording_selection.json", "face_redaction.json", "session_summary.txt")
FRAMES = {"/video/frame_index": "primary", "/video/reference/frame_index": "reference"}
POSE = ("/camera/reference/pose", "/camera/pose")                       # ultrawide takes / Wide-only takes
IMU = {"/device/imu/gyroscope": "imu_gyroscope", "/device/imu/accelerometer": "imu_accelerometer"}
INFO = ("binning_x", "binning_y", "d", "distortion_model", "height", "k", "p", "r", "width")


def table(header, rows) -> bytes:
    s = io.StringIO()
    s.write(",".join(header) + "\n")                                     # header unquoted, strings quoted: as the app
    csv.writer(s, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n").writerows(rows)
    return s.getvalue().encode()


def image(m) -> np.ndarray:
    dt = np.uint16 if m.encoding == "16UC1" else np.uint8
    return np.frombuffer(bytes(m.data), dt).reshape(m.height, m.step // np.dtype(dt).itemsize)[:, :m.width]


def export(mcap: Path, put, calibration: dict):
    """put(relative path under the session dir, bytes) for every mocap/ file the app's export holds."""
    vfi, pose, ts, dfr = [], [], [], []
    imu = {k: [] for k in IMU.values()}
    info, n_info, n_conf = None, 0, 0
    with open(mcap, "rb") as f:
        for _, ch, m, d in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=[*FRAMES, *IMU, *POSE, "/camera/tracking_state", "/camera/depth",
                        "/camera/depth/confidence", "/camera/depth/camera_info"]):
            t, stamp = m.log_time, lambda: d.header.stamp.sec * 10**9 + d.header.stamp.nanosec
            if ch.topic in FRAMES:
                vfi.append((FRAMES[ch.topic], d.frame_index, t, m.publish_time,
                            d.capture_pts_value * 10**6 // d.capture_pts_timescale))
            elif ch.topic in POSE:
                p, q = d.pose.position, d.pose.orientation
                pose.append((t, stamp(), ch.topic, d.header.frame_id, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            elif ch.topic == "/camera/tracking_state":
                ts.append((t, d.state_str, d.reason_str))
            elif ch.topic in IMU:
                v = d.vector
                imu[IMU[ch.topic]].append((t, stamp(), "", v.x, v.y, v.z))
            elif ch.topic == "/camera/depth":
                put(f"mocap/depth/{len(dfr):06d}.png", cv2.imencode(".png", image(d))[1].tobytes())
                dfr.append((len(dfr), t, m.publish_time, stamp(), d.width, d.height))
            elif ch.topic == "/camera/depth/confidence":
                put(f"mocap/depth/{n_conf:06d}_confidence.png", cv2.imencode(".png", image(d))[1].tobytes())
                n_conf += 1
            else:
                n_info += 1
                if info is None:
                    info = {k: getattr(d, k) for k in INFO} | {"frame_id": d.header.frame_id, "header_stamp_ns": stamp(),
                                                                "log_time_ns": t, "topic": ch.topic}
    vfi_b = table(["role", "frame_index", "log_time_ns", "publish_time_ns", "capture_pts_us"], vfi)
    put("mocap/video_frame_index.csv", vfi_b)
    put("mocap/depth/video_frame_index.csv", vfi_b)
    put("mocap/depth/depth_frames.csv", table(["index", "log_time_ns", "publish_time_ns", "header_stamp_ns", "width", "height"], dfr))
    info |= {"message_count": n_info, "calibration": calibration,
             "frame_index_message_counts": {r: n for r in FRAMES.values() if (n := sum(v[0] == r for v in vfi))}}
    put("mocap/depth/camera_info.json", json.dumps(info, indent=2, sort_keys=True).encode())
    put("mocap/samples/pose.csv", table(["monotonic_ns", "pose_time_ns", "source", "frame_id", "position_x", "position_y",
                                         "position_z", "orientation_x", "orientation_y", "orientation_z", "orientation_w"], pose))
    put("mocap/samples/tracking_state.csv", table(["monotonic_ns", "state", "reason"], ts))
    for k, rows in imu.items():
        put(f"mocap/samples/{k}.csv", table(["monotonic_ns", "sensor_time_ns", "accuracy", "x", "y", "z"], rows))
    return len(vfi), len(pose), len(dfr)


def build(phone: Path, csv_path: Path, out: Path):
    meta = json.loads((phone / "metadata.json").read_text())
    mcap = phone / meta["sensor_mcap_filename"]
    name = "session_" + re.match(r"session_data_(.+)\.mcap", mcap.name).group(1)
    out.mkdir(parents=True, exist_ok=True)
    zp = out / f"{name}.zip"
    if zp.exists():
        return zp
    tmp = zp.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        put = lambda rel, b: zf.writestr(f"{name}/{rel}", b)
        zf.write(phone / meta["primary_video_filename"], f"{name}/{meta['primary_video_filename']}")
        for k in KEEP:
            if (phone / k).exists():
                zf.write(phone / k, f"{name}/{k}")
        n = export(mcap, put, json.loads((phone / "calibration.json").read_text()))
    tmp.rename(zp)
    stamp = re.search(r"(\d{8}_\d{6}_\d{3})", csv_path.name).group(1)
    link = out / f"rokoko_recording_{stamp}.csv"
    if not link.exists():
        link.symlink_to(csv_path.resolve())
    print(f"{zp}  video frame rows {n[0]}, pose {n[1]}, depth {n[2]}  csv {csv_path.name}")
    return zp


def check(zp: Path):
    """Rebuild mocap/ from the MCAP inside an app-exported zip and compare every file with the app's own export."""
    zf = zipfile.ZipFile(zp)
    root = next(n.split("/")[0] for n in zf.namelist() if n.endswith(".mcap") and "__MACOSX" not in n)
    mcap = Path(__import__("tempfile").gettempdir()) / f"{root}.mcap"
    if not mcap.exists():
        mcap.write_bytes(zf.read(next(n for n in zf.namelist() if n.startswith(root) and n.endswith(".mcap"))))
    calib = json.loads(zf.read(f"{root}/calibration.json"))
    bad, seen = [], 0

    def put(rel, b):
        nonlocal seen
        ref = zf.read(f"{root}/{rel}")
        seen += 1
        if rel.endswith(".png"):
            a, r = (cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_UNCHANGED) for x in (b, ref))
            ok = a.dtype == r.dtype and np.array_equal(a, r)
        elif rel.endswith(".json"):
            a, r = json.loads(b), json.loads(ref)
            ok = all(a[k] == r[k] for k in (*INFO, "header_stamp_ns", "message_count", "frame_index_message_counts"))
        else:
            ok = b.decode().split() == ref.decode().split() or same_csv(b, ref)
        if not ok:
            bad.append(rel)
    export(mcap, put, calib)
    print(f"{zp.name}: {seen} files compared, {len(bad)} differ {bad[:10]}")
    return not bad


def same_csv(a: bytes, b: bytes) -> bool:
    """Equal up to float formatting (python repr vs the app's printf) and row order (the app interleaves the
    primary / reference frame rows in arrival order; every reader filters by role and sorts)."""
    ra, rb = (list(csv.reader(io.StringIO(x.decode()))) for x in (a, b))
    if len(ra) != len(rb) or ra[0] != rb[0]:
        return False
    for x, y in zip(sorted(ra[1:]), sorted(rb[1:])):
        for u, v in zip(x, y):
            if u != v and not (u and v and np.isclose(float(u), float(v), rtol=1e-6, atol=1e-9)):
                return False
    return True


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        sys.exit(0 if check(Path(sys.argv[2])) else 1)
    build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
