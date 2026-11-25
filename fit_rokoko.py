#!/usr/bin/env python3
"""
fit_rokoko.py - Dong bo file MP4 quay tay voi file CSV mocap cua Rokoko.

Nguyen ly
---------
1. Moc thoi gian tuyet doi (wall clock) cua moi file suy ra tu TEN FILE:
     MP4 : "12-03-24-952.mp4"  -> 12:03:24.952   (mac dinh: thoi diem BAT DAU quay)
     CSV : "rokoko_recording_20260921_120339_876.csv" -> 2026-09-21 12:03:39.876
   Wall clock cua 1 dong CSV = moc CSV + gia tri cot `time` cua dong do.
   Script tu do (auto) xem ten MP4 la moc dau hay moc cuoi, va cot `time` cua CSV
   la tuyet doi hay tinh tu dong dau - bang cach chon gia thuyet cho phan giao lon nhat.

2. Dung PTS THAT cua tung frame video (ffprobe -show_packets), khong dung fps danh nghia.
   Video quay tay thuong bi rot frame (VFR): gia su "frame i o giay i/fps" co the
   lech vai giay o cuoi file.

3. Khoang giao nhau [T0, T1] = [max(start), min(end)] -> cat ca MP4 va CSV ve dung
   khoang do: MP4 cat dung bien frame, CSV lay dong co thoi diem gan nhat.

Chay
----
    python3 fit_rokoko.py <thu_muc>                    # 1 thu muc, hoac thu muc cha chua nhieu thu muc con
    python3 fit_rokoko.py <thu_muc> --dry-run          # chi phan tich, khong tao file (rat nhanh)
    python3 fit_rokoko.py <thu_muc> --video-mode copy  # cat nhanh, khong encode lai (bam keyframe)
    python3 fit_rokoko.py <thu_muc> --per-frame        # xuat them CSV: 1 dong <-> 1 frame video

Yeu cau: python3 (chi stdlib) + ffmpeg/ffprobe trong PATH.
"""

from __future__ import annotations

import argparse
import bisect
import json
import platform
import re
import math
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# ======================================================================================
# 1. Doc moc thoi gian tu ten file
# ======================================================================================

DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)")

TIME_PATTERNS = [
    re.compile(r"(?<![0-9A-Za-z])(\d{1,2})[-_.:](\d{2})[-_.:](\d{2})[-_.:,](\d{1,3})(?!\d)"),  # 12-03-24-952
    re.compile(r"(?<![0-9A-Za-z])(\d{2})(\d{2})(\d{2})[-_.:,](\d{1,3})(?!\d)"),                # 120324_952
    re.compile(r"(?<![0-9A-Za-z])(\d{2})(\d{2})(\d{2})(\d{3})(?!\d)"),                         # 120324952
    re.compile(r"(?<![0-9A-Za-z])(\d{1,2})[-_.:](\d{2})[-_.:](\d{2})(?!\d)"),                  # 12-03-24
    re.compile(r"(?<![0-9A-Za-z])(\d{2})(\d{2})(\d{2})(?!\d)"),                                # 120324
]


def parse_name_datetime(stem: str, fallback_date=None):
    """Doc ngay + gio:phut:giay.ms tu ten file -> (datetime | None, co_ngay_trong_ten)."""
    text = stem
    the_date = None
    m = DATE_RE.search(text)
    if m:
        the_date = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        text = text[: m.start()] + "#" * (m.end() - m.start()) + text[m.end():]

    for pat in TIME_PATTERNS:
        found = None
        for pos in range(len(text)):          # quet ca vi tri chong nhau, nen ten kieu
            tm = pat.match(text, pos)         # "...29_01_12-03-24-952" van ra 12:03:24.952
            if not tm:
                continue
            h, mi, s = int(tm.group(1)), int(tm.group(2)), int(tm.group(3))
            if h > 23 or mi > 59 or s > 59:
                continue
            ms = int(tm.group(4).ljust(3, "0")) if (tm.lastindex or 0) >= 4 else 0
            found = (h, mi, s, ms)            # lay cai CUOI CUNG: tien to co chu so khong lam lech
        if found:
            d = the_date or fallback_date
            if d is None:
                return None, False
            h, mi, s, ms = found
            return datetime(d.year, d.month, d.day, h, mi, s, ms * 1000), the_date is not None
    return None, False


# ======================================================================================
# 2. Doc video bang ffprobe (ke ca PTS tung frame)
# ======================================================================================

@dataclass
class VideoInfo:
    path: Path
    duration: float          # do dai theo container
    start_time: float
    nb_frames: int
    fps: float               # fps thuc te trung binh
    r_fps: float             # fps danh nghia
    width: int
    height: int
    bit_rate: int
    has_audio: bool
    pts: list = field(default_factory=list)   # PTS tuyet doi cua tung frame (giay, theo timeline file)
    vfr: bool = False
    creation_time: object = None              # datetime gio dia phuong, tu metadata (neu co)

    @property
    def t0(self) -> float:
        return self.pts[0] if self.pts else self.start_time

    @property
    def span(self) -> float:
        """Do dai thuc = tu frame dau den het frame cuoi."""
        if len(self.pts) >= 2:
            return (self.pts[-1] - self.pts[0]) + self.median_dt
        return self.duration

    @property
    def median_dt(self) -> float:
        if len(self.pts) >= 3:
            return statistics.median(b - a for a, b in zip(self.pts, self.pts[1:]))
        return 1.0 / self.fps if self.fps else 1 / 30


def ffprobe_video(path: Path, read_pts: bool = True) -> VideoInfo:
    data = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout)
    v = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError(f"Khong tim thay video stream trong {path.name}")

    def frac(val, default=0.0):
        try:
            num, den = val.split("/")
            return float(num) / float(den) if float(den) else default
        except Exception:
            return default

    duration = float(v.get("duration") or data["format"]["duration"])
    nb_frames = int(v.get("nb_frames") or 0)
    r_fps = frac(v.get("r_frame_rate", ""))
    avg_fps = frac(v.get("avg_frame_rate", ""))
    fps = (nb_frames / duration) if (nb_frames and duration) else (avg_fps or r_fps or 30.0)

    ctime = None
    for tags in (data["format"].get("tags", {}), v.get("tags", {})):
        raw = tags.get("creation_time")
        if not raw:
            continue
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:                       # tag cua QuickTime la gio UTC -> doi ve gio may
                ctime = datetime.strptime(raw, fmt).astimezone().replace(tzinfo=None)
                break
            except ValueError:
                continue
        if ctime:
            break

    info = VideoInfo(
        path=path, duration=duration, start_time=float(v.get("start_time") or 0.0),
        nb_frames=nb_frames or int(round(duration * fps)), fps=fps, r_fps=r_fps or fps,
        width=int(v.get("width") or 0), height=int(v.get("height") or 0),
        bit_rate=int(v.get("bit_rate") or data["format"].get("bit_rate") or 0),
        has_audio=any(s.get("codec_type") == "audio" for s in data["streams"]),
        creation_time=ctime,
    )

    if read_pts:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True).stdout
        pts = []
        for tok in out.split():
            try:
                pts.append(float(tok))
            except ValueError:      # "N/A" o vai packet
                pass
        pts.sort()
        if len(pts) >= 2:
            info.pts = pts
            info.nb_frames = len(pts)
            info.fps = (len(pts) - 1) / (pts[-1] - pts[0])
            gaps = [b - a for a, b in zip(pts, pts[1:])]
            info.vfr = (max(gaps) - min(gaps)) > 0.005
    return info


# ======================================================================================
# 3. Index CSV (stream tung dong, khong nap het vao RAM)
# ======================================================================================

@dataclass
class CsvIndex:
    path: Path
    header: bytes
    time_col: int
    times: list
    offsets: list
    lengths: list


def index_csv(path: Path, time_col_name: str | None = None) -> CsvIndex:
    with open(path, "rb") as f:
        header = f.readline()
        cols = [c.strip().strip('"').lower()
                for c in header.decode("utf-8", "replace").rstrip("\r\n").split(",")]
        if time_col_name:
            if time_col_name.lower() not in cols:
                raise RuntimeError(f"Khong co cot '{time_col_name}' trong {path.name}")
            tcol = cols.index(time_col_name.lower())
        else:
            tcol = next((cols.index(c) for c in ("time", "timestamp", "t") if c in cols), 0)

        times, offsets, lengths = [], [], []
        off = len(header)
        for line in f:
            n = len(line)
            if line.strip():
                try:
                    val = float(line.split(b",", 1)[0] if tcol == 0 else line.split(b",")[tcol])
                    times.append(val)
                    offsets.append(off)
                    lengths.append(n)
                except (ValueError, IndexError):
                    pass
            off += n
    if len(times) < 2:
        raise RuntimeError(f"CSV rong hoac khong doc duoc cot thoi gian: {path.name}")
    return CsvIndex(path, header, tcol, times, offsets, lengths)


def gap_stats(times: list, prefix: str = "row") -> dict:
    """Thong ke khoang cach giua cac mau -> biet sai so 'khop gan nhat' toi da."""
    if len(times) < 3:
        return {}
    d = sorted(b - a for a, b in zip(times, times[1:]))
    n = len(d)
    return {
        f"{prefix}_gap_median_ms": round(d[n // 2] * 1000, 2),
        f"{prefix}_gap_p99_ms": round(d[int(n * 0.99)] * 1000, 2),
        f"{prefix}_gap_max_ms": round(d[-1] * 1000, 2),
        f"{prefix}_gaps_over_50ms": sum(1 for x in d if x > 0.05),
    }


def find_dead_zones(cidx: CsvIndex, min_len: float = 10.0, sample_dt: float = 0.5) -> list:
    """Tim cac doan mocap DUNG HAN (suit mat ket noi / dung ghi nhung file van chay).
    27_04 chet o giay ~750: moi khop deu bang 0.000 trong 15 phut cuoi."""
    cols = [c.strip().strip('"') for c in
            cidx.header.decode("utf-8", "replace").rstrip("\r\n").split(",")]
    idx = [cols.index(c) for c in
           (f"{j}_pos_{a}" for j in ("chest", "rightHand", "leftHand", "head") for a in "xyz")
           if c in cols]
    if len(idx) < 6:
        return []
    step = max(1, int(sample_dt / max(1e-6, (cidx.times[-1] - cidx.times[0]) / len(cidx.times))))
    samples = []
    with open(cidx.path, "rb") as f:
        for i in range(0, len(cidx.times), step):
            f.seek(cidx.offsets[i])
            parts = f.read(cidx.lengths[i]).split(b",")
            try:
                samples.append((cidx.times[i], [float(parts[k]) for k in idx]))
            except (ValueError, IndexError):
                pass
    zones, run_start = [], None
    for (t1, a), (t2, b) in zip(samples, samples[1:]):
        moved = sum(abs(x - y) for x, y in zip(a, b))
        if moved < 1e-6:
            run_start = t1 if run_start is None else run_start
        else:
            if run_start is not None and t1 - run_start >= min_len:
                zones.append((round(run_start, 3), round(t1, 3)))
            run_start = None
    if run_start is not None and samples[-1][0] - run_start >= min_len:
        zones.append((round(run_start, 3), round(samples[-1][0], 3)))
    return zones


def nearest_index(sorted_vals: list, target: float) -> int:
    i = bisect.bisect_left(sorted_vals, target)
    if i <= 0:
        return 0
    if i >= len(sorted_vals):
        return len(sorted_vals) - 1
    return i if (sorted_vals[i] - target) < (target - sorted_vals[i - 1]) else i - 1


def video_time_candidates(mp4: Path, vinfo: VideoInfo, fallback_date, args) -> list:
    """Moi nguon thoi gian -> mot ung vien cho thoi diem BAT DAU video.

    Luu y 3 moc khac nhau, y nghia nguoc nhau:
      creation_time (metadata) = luc BAT DAU quay  -> dung thang, KHONG tru duration
      mtime (sua file lan cuoi) = luc GHI XONG     -> phai TRU duration
      birthtime                 = luc tao file; file da copy thi bang mtime -> bo qua
    """
    cands = []
    name_ts, _ = parse_name_datetime(mp4.stem, fallback_date=fallback_date)
    if name_ts:
        cands.append({"source": "name", "anchor": "start", "start": name_ts, "resolution_s": 0.001})
        cands.append({"source": "name", "anchor": "end", "resolution_s": 0.001,
                      "start": name_ts - timedelta(seconds=vinfo.span)})
    if vinfo.creation_time:
        cands.append({"source": "creation", "anchor": "start", "resolution_s": 1.0,
                      "start": vinfo.creation_time})
    mt = datetime.fromtimestamp(mp4.stat().st_mtime)
    cands.append({"source": "mtime", "anchor": "end", "resolution_s": 1.0,
                  "start": mt - timedelta(seconds=vinfo.span)})

    # loc theo lua chon cua nguoi dung
    if args.video_time_source != "auto":
        cands = [c for c in cands if c["source"] == args.video_time_source]
    if args.mp4_anchor in ("start", "end"):
        cands = [c for c in cands if c["source"] != "name" or c["anchor"] == args.mp4_anchor]
    elif args.mp4_anchor is None:
        cands = [c for c in cands if c["source"] != "name" or c["anchor"] == "start"]
    return cands


PRIORITY = {("name", "start"): 0, ("name", "end"): 1, ("creation", "start"): 2, ("mtime", "end"): 3}


def pick_video_time(cands: list, vinfo: VideoInfo, c0e: float, c1e: float) -> tuple:
    """Chon ung vien theo do tin cay, NHUNG doi chieu voi CSV: neu ung vien uu tien
    gan nhu khong giao voi CSV ma ung vien khac giao nhieu -> tu chuyen + canh bao."""
    for c in cands:
        v0 = c["start"].timestamp()
        c["overlap_s"] = round(overlap_seconds(v0, v0 + vinfo.span, c0e, c1e), 3)
    by_priority = sorted(cands, key=lambda c: PRIORITY.get((c["source"], c["anchor"]), 9))
    best_overlap = max(cands, key=lambda c: c["overlap_s"])
    chosen, note = by_priority[0], None
    if best_overlap["overlap_s"] > 0 and chosen["overlap_s"] < 0.25 * best_overlap["overlap_s"]:
        note = (f"moc tu '{chosen['source']}' gan nhu khong khop CSV "
                f"(giao {chosen['overlap_s']}s); da chuyen sang '{best_overlap['source']}' "
                f"(giao {best_overlap['overlap_s']}s). Kiem tra lai ten file!")
        chosen = best_overlap
    return chosen, note, cands


# ======================================================================================
# 4. Chon gia thuyet dong bo
# ======================================================================================

def overlap_seconds(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def choose_hypothesis(mp4_ts: datetime, v_span: float, csv_ts: datetime,
                      t_first: float, t_last: float, mp4_anchor: str, csv_base: str):
    results = []
    for anc in ("start", "end"):
        v0 = mp4_ts if anc == "start" else mp4_ts - timedelta(seconds=v_span)
        v1 = v0 + timedelta(seconds=v_span)
        for base in ("absolute", "relative"):
            shift = 0.0 if base == "absolute" else -t_first
            c0 = csv_ts + timedelta(seconds=t_first + shift)
            c1 = csv_ts + timedelta(seconds=t_last + shift)
            ov = overlap_seconds(v0.timestamp(), v1.timestamp(), c0.timestamp(), c1.timestamp())
            results.append({
                "mp4_anchor": anc, "csv_base": base,
                "video_start": v0.isoformat(sep=" "), "video_end": v1.isoformat(sep=" "),
                "csv_start": c0.isoformat(sep=" "), "csv_end": c1.isoformat(sep=" "),
                "overlap_s": round(ov, 3),
                "overlap_pct_of_video": round(100.0 * ov / v_span, 2) if v_span else 0.0,
            })
    cands = [r for r in results
             if (mp4_anchor == "auto" or r["mp4_anchor"] == mp4_anchor)
             and (csv_base == "auto" or r["csv_base"] == csv_base)]
    order = {("start", "absolute"): 0, ("start", "relative"): 1,
             ("end", "absolute"): 2, ("end", "relative"): 3}
    ranked = sorted(cands, key=lambda r: (-r["overlap_s"], order[(r["mp4_anchor"], r["csv_base"])]))
    best = ranked[0]
    best["auto_detected"] = len(cands) > 1
    # canh bao khi ung vien hang 2 gan bang hang 1 -> lua chon khong chac chan
    best["ambiguous"] = bool(len(ranked) > 1 and ranked[1]["overlap_s"] > 0.8 * best["overlap_s"]
                             and best["overlap_s"] > 0)
    best["runner_up"] = (f"{ranked[1]['mp4_anchor']}/{ranked[1]['csv_base']} "
                         f"({ranked[1]['overlap_s']}s)") if len(ranked) > 1 else None
    alt = max((r for r in results if (r["mp4_anchor"], r["csv_base"])
               != (best["mp4_anchor"], best["csv_base"])), key=lambda r: r["overlap_s"], default=None)
    if alt and (best["overlap_s"] <= 0 or alt["overlap_s"] > 1.5 * best["overlap_s"]):
        best["better_alternative"] = f"--mp4-anchor {alt['mp4_anchor']} --csv-base {alt['csv_base']}"
        best["better_alternative_overlap_s"] = alt["overlap_s"]
    return best, results


# ======================================================================================
# 4b. Do do tre THUC bang tuong quan chuyen dong (--measure)
# ======================================================================================
#
# Vi sao can: ten file va metadata chi noi len dong ho cua THIET BI quay, ma dong ho do
# lech dong ho may chay Rokoko (do duoc 8s o 29_01, 19.4s o 29_07). Cach duy nhat biet
# do lech that la doi chieu chinh noi dung: camera gan tren nguoi -> chuyen dong khung
# hinh chinh la chuyen dong cua dau -> so voi van toc goc khop 'head' trong CSV.

MEAS_HZ = 10.0
MEAS_W, MEAS_H = 64, 48


def video_motion_signal(vinfo: VideoInfo, win_start: float, win_len: float) -> list:
    """(thoi_diem_trong_video, do_khac_biet_so_voi_frame_truoc) cho tung frame."""
    n_px = MEAS_W * MEAS_H
    sel = [p for p in vinfo.pts if win_start <= p - vinfo.t0 < win_start + win_len] or []
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{win_start + vinfo.t0:.6f}", "-t", f"{win_len:.6f}",
         "-i", str(vinfo.path), "-vf", f"scale={MEAS_W}:{MEAS_H},format=gray",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-"], capture_output=True).stdout
    fr = [raw[i * n_px:(i + 1) * n_px] for i in range(len(raw) // n_px)]
    m = min(len(fr), len(sel))
    return [(sel[i] - vinfo.t0 - win_start,
             sum(abs(a - b) for a, b in zip(fr[i], fr[i - 1])) / n_px) for i in range(1, m)]


def csv_motion_signal(cidx: CsvIndex, t_from: float, t_to: float) -> list:
    """Van toc goc cua khop 'head' (rad/s). Khong co quaternion thi dung toc do 2 ban tay."""
    cols = [c.strip().strip('"') for c in
            cidx.header.decode("utf-8", "replace").rstrip("\r\n").split(",")]
    try:
        idx = [cols.index(f"head_quat_{a}") for a in "xyzw"]
        kind = "quat"
    except ValueError:
        try:
            idx = [cols.index(f"{j}_pos_{a}") for j in ("leftHand", "rightHand") for a in "xyz"]
            kind = "pos"
        except ValueError:
            return []
    i0 = bisect.bisect_left(cidx.times, t_from)
    i1 = bisect.bisect_right(cidx.times, t_to)
    rows = []
    with open(cidx.path, "rb") as f:
        for i in range(max(0, i0 - 1), min(len(cidx.times), i1 + 1)):
            f.seek(cidx.offsets[i])
            parts = f.read(cidx.lengths[i]).split(b",")
            try:
                rows.append((cidx.times[i], [float(parts[k]) for k in idx]))
            except (ValueError, IndexError):
                pass
    out = []
    for (t1, a), (t2, b) in zip(rows, rows[1:]):
        dt = t2 - t1
        if dt <= 0:
            continue
        if kind == "quat":
            dot = min(1.0, abs(sum(x * y for x, y in zip(a, b))))
            out.append((t2, 2 * math.acos(dot) / dt))
        else:
            out.append((t2, sum(abs(x - y) for x, y in zip(a, b)) / dt))
    return out


def _resample(series: list, n: int, hz: float, off: float = 0.0) -> list:
    """Noi suy tuyen tinh len luoi deu, de dich duoc theo buoc nho hon 1 mau."""
    out = [0.0] * n
    j = 0
    for i in range(n):
        t = off + i / hz
        while j + 1 < len(series) and series[j + 1][0] < t:
            j += 1
        if j + 1 < len(series):
            t1, v1 = series[j]
            t2, v2 = series[j + 1]
            w = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
            out[i] = v1 + (v2 - v1) * max(0.0, min(1.0, w))
        elif series:
            out[i] = series[-1][1]
    return out


def _detrend(x: list, w: int) -> list:
    cs = [0.0]
    for v in x:
        cs.append(cs[-1] + v)
    return [x[i] - (cs[min(len(x), i + w // 2)] - cs[max(0, i - w // 2)])
            / (min(len(x), i + w // 2) - max(0, i - w // 2)) for i in range(len(x))]


def _norm(x: list) -> list:
    m = sum(x) / len(x)
    sd = (sum((v - m) ** 2 for v in x) / len(x)) ** 0.5 or 1.0
    return [(v - m) / sd for v in x]


def correlate_lag(vid: list, mo: list, win_len: float, maxlag: float,
                  want_series: bool = False) -> tuple:
    """Tra ve (lag, r, do_sac_net_sigma[, du_lieu_ve]). lag > 0 = CSV cham hon video."""
    n = int((win_len - 2 * min(maxlag, win_len / 4)) * MEAS_HZ)
    if n < int(30 * MEAS_HZ) or not vid or not mo:
        return (None, 0.0, 0.0, None) if want_series else (None, 0.0, 0.0)
    a = _norm(_detrend(_resample(vid, n, MEAS_HZ), int(MEAS_HZ * 5)))

    def score(lag):
        b = _norm(_detrend(_resample(mo, n, MEAS_HZ, off=lag), int(MEAS_HZ * 5)))
        return sum(x * y for x, y in zip(a, b)) / n

    coarse = [(score(k / 10.0), k / 10.0) for k in range(int(-maxlag * 10), int(maxlag * 10) + 1)]
    r, lag = max(coarse)
    mean = sum(p[0] for p in coarse) / len(coarse)
    sd = (sum((p[0] - mean) ** 2 for p in coarse) / len(coarse)) ** 0.5 or 1.0
    fine = [(score(lag + k / 100.0), lag + k / 100.0) for k in range(-12, 13)]
    r2, lag2 = max(fine)
    sharp = (r2 - mean) / sd
    if not want_series:
        return lag2, r2, sharp
    k = min(n, int(40 * MEAS_HZ))          # 40 giay dau cua so, du de nhin ra nhip
    series = {"hz": MEAS_HZ, "lag": round(lag2, 3), "r": round(r2, 4),
              "video": [round(v, 3) for v in a[:k]],
              "csvNominal": [round(v, 3) for v in _norm(_detrend(
                  _resample(mo, n, MEAS_HZ, off=0.0), int(MEAS_HZ * 5)))[:k]],
              "csvAligned": [round(v, 3) for v in _norm(_detrend(
                  _resample(mo, n, MEAS_HZ, off=lag2), int(MEAS_HZ * 5)))[:k]],
              "profile": [[round(l, 2), round(rr, 4)] for rr, l in coarse]}
    return lag2, r2, sharp, series


def measure_offset(vinfo: VideoInfo, cidx: CsvIndex, v0e: float, csv_epoch0: float,
                   t0: float, t1: float, args) -> dict:
    """Do do lech that giua video va CSV o nhieu cua so, roi doi chieu ket qua."""
    overlap = t1 - t0
    wl = min(args.measure_window, max(120.0, overlap / args.measure_windows))
    res = {"windows": [], "window_len_s": round(wl, 1)}
    starts = []
    for k in range(args.measure_windows):
        # dat cua so trong 80% giua vung giao: dau/cuoi lan quay hay co doan lap dat, chinh camera
        frac = 0.1 + 0.8 * (k + 0.5) / args.measure_windows
        ws = (t0 - v0e) + frac * overlap - wl / 2
        starts.append(max(0.0, min(ws, vinfo.span - wl)))
    for ws in starts:
        vid = video_motion_signal(vinfo, ws, wl)
        t_from = (v0e + ws - csv_epoch0) - args.measure_maxlag - 5
        t_to = (v0e + ws + wl - csv_epoch0) + args.measure_maxlag + 5
        mo = [(t - (v0e + ws - csv_epoch0), v) for t, v in csv_motion_signal(cidx, t_from, t_to)]
        want = bool(args.plot)
        got = correlate_lag(vid, mo, wl, args.measure_maxlag, want_series=want)
        lag, r, sharp = got[0], got[1], got[2]
        if want and got[3] and r > res.get("_plot_r", -9):
            res["_plot"] = got[3]
            res["_plot"]["window_start_s"] = round(ws, 1)
            res["_plot_r"] = r
        res["windows"].append({"video_start_s": round(ws, 1), "lag_s": lag,
                               "r": round(r, 4), "sharpness_sigma": round(sharp, 1)})
        if args.verbose:
            print(f"      cua so {ws:7.1f}s: lag {lag:+.3f}s  r={r:.3f}  {sharp:.1f} sigma")
    # Dong thuan ben vung: bo cua so yeu, roi tim cum lon nhat trong nguong sai so.
    # Mot cua so roi vao doan xau (dang lap dat, dang nghi) khong duoc lam hong ca phep do.
    good = [w for w in res["windows"]
            if w["lag_s"] is not None and w["r"] >= 0.15 and w["sharpness_sigma"] >= 5.0]
    res["windows_used"] = len(good)
    res["windows_total"] = len(res["windows"])
    if not good:
        res.update(ok=False, lag_median_s=None,
                   reason="khong cua so nao dat nguong (r>=0.15 va >=5 sigma)")
        return res
    best_cluster = []
    for w in good:
        cl = [u for u in good if abs(u["lag_s"] - w["lag_s"]) <= args.measure_tol]
        if len(cl) > len(best_cluster):
            best_cluster = cl
    lags = sorted(u["lag_s"] for u in best_cluster)
    res["lag_median_s"] = round(lags[len(lags) // 2], 3)
    res["lag_spread_s"] = round(lags[-1] - lags[0], 3)
    res["agree_count"] = len(lags)
    res["r_min"] = min(u["r"] for u in best_cluster)
    res["sharpness_min"] = min(u["sharpness_sigma"] for u in best_cluster)
    res["outliers"] = [{"video_start_s": u["video_start_s"], "lag_s": u["lag_s"],
                        "r": u["r"], "sharpness_sigma": u["sharpness_sigma"]}
                       for u in res["windows"] if u not in best_cluster]
    res["ok"] = bool(len(lags) >= 2 and len(lags) * 2 >= len(res["windows"]))
    if not res["ok"]:
        res["reason"] = (f"khong dang tin: chi {len(lags)}/{len(res['windows'])} cua so dong thuan "
                         f"(can >=2 va >=mot nua). Thu tang --measure-windows hoac --measure-window.")
    return res


PLOT_HTML = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#f4f4f1; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834; --crit:#d03b3b;
  --line:rgba(11,11,11,.10);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --crit:#d03b3b;
  --line:rgba(255,255,255,.10);
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:28px 16px 56px}
h1{font-size:23px;line-height:1.25;margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--ink2);margin:0 0 22px;font-size:14px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 24px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:11px 15px;min-width:132px;flex:1 1 132px}
.stat .k{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.stat .v{font-size:21px;margin-top:3px;font-family:ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:16px 16px 10px;margin:0 0 16px}
.panel h2{font-size:14px;margin:0 0 2px;letter-spacing:-.005em}
.panel p{font-size:13px;color:var(--ink2);margin:0 0 12px}
.legend{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:0 0 8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.key{width:14px;height:2px;border-radius:1px;display:inline-block}
svg{display:block;width:100%;height:auto;overflow:visible}
.read{font-size:12px;color:var(--muted);min-height:17px;margin:2px 0 4px}
.note{font-size:13px;color:var(--ink2);border-left:2px solid var(--crit);
  padding:2px 0 2px 12px;margin:18px 0 0}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1>
<p class="sub">__SUB__</p>
<div class="stats">__STATS__</div>

<div class="panel">
  <h2>1 &middot; Căn theo mốc thời gian có sẵn</h2>
  <p>Hai đường không trùng nhịp: đỉnh của đường này rơi vào chỗ khác của đường kia.</p>
  <div class="legend">
    <span><i class="key" style="background:var(--s1)"></i>video &mdash; khác biệt giữa 2 khung hình</span>
    <span><i class="key" style="background:var(--s2)"></i>mocap &mdash; vận tốc góc khớp đầu</span>
  </div>
  <svg id="c1" viewBox="0 0 860 190"></svg>
  <div class="read" id="r1"></div>
</div>

<div class="panel">
  <h2>2 &middot; Sau khi dịch __LAG__ giây</h2>
  <p>Cùng dữ liệu, chỉ dịch trục thời gian của video. Các đỉnh chồng lên nhau.</p>
  <div class="legend">
    <span><i class="key" style="background:var(--s1)"></i>video</span>
    <span><i class="key" style="background:var(--s2)"></i>mocap (đã dịch)</span>
  </div>
  <svg id="c2" viewBox="0 0 860 190"></svg>
  <div class="read" id="r2"></div>
</div>

<div class="panel">
  <h2>3 &middot; Thử mọi độ trễ</h2>
  <p>Hệ số tương quan khi trượt hai đường qua nhau. Một đỉnh duy nhất, nhọn &mdash; nếu là nhiễu
     hoặc bắt nhầm nhịp lặp thì sẽ có nhiều đỉnh ngang nhau.</p>
  <svg id="c3" viewBox="0 0 860 210"></svg>
  <div class="read" id="r3"></div>
</div>
__NOTE__
</div>
<script>
const DATA=__DATA__;
const css=k=>getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const NS="http://www.w3.org/2000/svg";
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function axis(svg,W,H,P,xt,yt){
  for(const t of yt){svg.appendChild(el("line",{x1:P.l,x2:W-P.r,y1:t.y,y2:t.y,
    stroke:css("--grid"),"stroke-width":1}));
    svg.appendChild(el("text",{x:P.l-8,y:t.y+4,"text-anchor":"end","font-size":11,
      fill:css("--muted"),"font-family":"ui-monospace,Menlo,monospace"})).textContent=t.t;}
  svg.appendChild(el("line",{x1:P.l,x2:W-P.r,y1:H-P.b,y2:H-P.b,stroke:css("--axis"),"stroke-width":1}));
  for(const t of xt){svg.appendChild(el("text",{x:t.x,y:H-P.b+18,"text-anchor":"middle",
    "font-size":11,fill:css("--muted"),"font-family":"ui-monospace,Menlo,monospace"})).textContent=t.t;}
}
function curves(id,readId,a,b,label){
  const svg=document.getElementById(id);svg.textContent="";
  const W=860,H=190,P={l:34,r:12,t:10,b:26};
  const n=a.length,hz=DATA.hz,secs=n/hz;
  const all=a.concat(b),lo=Math.min(...all),hi=Math.max(...all);
  const X=i=>P.l+(W-P.l-P.r)*i/(n-1), Y=v=>P.t+(H-P.t-P.b)*(1-(v-lo)/(hi-lo));
  axis(svg,W,H,P,[...Array(5)].map((_,k)=>({x:X((n-1)*k/4),t:(secs*k/4).toFixed(0)+"s"})),
       [{y:Y(hi),t:"cao"},{y:Y((hi+lo)/2),t:""},{y:Y(lo),t:"thấp"}]);
  const path=(arr,c)=>{let d="";arr.forEach((v,i)=>d+=(i?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1));
    svg.appendChild(el("path",{d,fill:"none",stroke:c,"stroke-width":2,
      "stroke-linejoin":"round","stroke-linecap":"round","opacity":.92}));};
  path(b,css("--s2"));path(a,css("--s1"));
  const cur=el("line",{y1:P.t,y2:H-P.b,stroke:css("--axis"),"stroke-width":1,opacity:0});
  svg.appendChild(cur);
  const read=document.getElementById(readId);
  svg.addEventListener("pointermove",e=>{
    const r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*W;
    const i=Math.max(0,Math.min(n-1,Math.round((x-P.l)/(W-P.l-P.r)*(n-1))));
    cur.setAttribute("x1",X(i));cur.setAttribute("x2",X(i));cur.setAttribute("opacity",.55);
    read.textContent=`giây ${(i/hz).toFixed(1)} · video ${a[i].toFixed(2)} · mocap ${b[i].toFixed(2)}`;});
  svg.addEventListener("pointerleave",()=>{cur.setAttribute("opacity",0);read.textContent=label;});
  read.textContent=label;
}
function profile(){
  const svg=document.getElementById("c3");svg.textContent="";
  const p=DATA.profile,W=860,H=210,P={l:44,r:14,t:12,b:28};
  const xs=p.map(q=>q[0]),ys=p.map(q=>q[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs),lo=Math.min(...ys),hi=Math.max(...ys);
  const X=v=>P.l+(W-P.l-P.r)*(v-x0)/(x1-x0), Y=v=>P.t+(H-P.t-P.b)*(1-(v-lo)/(hi-lo));
  const ticks=[];for(let v=Math.ceil(lo*10)/10;v<=hi;v+=0.1)ticks.push({y:Y(v),t:v.toFixed(1)});
  const xt=[];for(let v=x0;v<=x1+.001;v+=(x1-x0)/6)xt.push({x:X(v),t:v.toFixed(0)+"s"});
  axis(svg,W,H,P,xt,ticks);
  if(lo<0&&hi>0)svg.appendChild(el("line",{x1:P.l,x2:W-P.r,y1:Y(0),y2:Y(0),
    stroke:css("--axis"),"stroke-width":1,"stroke-dasharray":"0"}));
  let d="";p.forEach((q,i)=>d+=(i?"L":"M")+X(q[0]).toFixed(1)+" "+Y(q[1]).toFixed(1));
  svg.appendChild(el("path",{d,fill:"none",stroke:css("--s1"),"stroke-width":2,
    "stroke-linejoin":"round"}));
  const best=p.reduce((m,q)=>q[1]>m[1]?q:m,p[0]);
  svg.appendChild(el("circle",{cx:X(best[0]),cy:Y(best[1]),r:5,fill:css("--s2"),
    stroke:css("--surface"),"stroke-width":2}));
  const above=Y(best[1])>P.t+20;                       // dinh sat mep tren thi dat nhan xuong duoi
  const tx=el("text",{x:X(best[0])+(above?0:10),y:Y(best[1])+(above?-12:16),
    "text-anchor":above?(X(best[0])>W-140?"end":"middle"):"start",
    "font-size":12,fill:css("--ink"),"font-family":"ui-monospace,Menlo,monospace"});
  tx.textContent=`${best[0]>0?"+":""}${best[0].toFixed(2)}s · r=${best[1].toFixed(3)}`;
  svg.appendChild(tx);
  const cur=el("line",{y1:P.t,y2:H-P.b,stroke:css("--axis"),"stroke-width":1,opacity:0});
  svg.appendChild(cur);
  const read=document.getElementById("r3");
  const base=`đỉnh tại ${best[0].toFixed(2)}s, r=${best[1].toFixed(3)} — nền nhiễu quanh r≈0`;
  read.textContent=base;
  svg.addEventListener("pointermove",e=>{
    const r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*W;
    let i=Math.round((x-P.l)/(W-P.l-P.r)*(p.length-1));i=Math.max(0,Math.min(p.length-1,i));
    cur.setAttribute("x1",X(p[i][0]));cur.setAttribute("x2",X(p[i][0]));cur.setAttribute("opacity",.55);
    read.textContent=`độ trễ ${p[i][0].toFixed(2)}s · r=${p[i][1].toFixed(3)}`;});
  svg.addEventListener("pointerleave",()=>{cur.setAttribute("opacity",0);read.textContent=base;});
}
curves("c1","r1",DATA.video,DATA.csvNominal,"lệch nhịp — di chuột để đọc giá trị");
curves("c2","r2",DATA.video,DATA.csvAligned,"trùng nhịp — di chuột để đọc giá trị");
profile();
</script></body></html>"""


def write_plot(path: Path, folder_name: str, series: dict, meas: dict, extra_note: str = "") -> None:
    """Xuat 1 file HTML tu chua, mo bang trinh duyet la thay ngay 3 bieu do."""
    lag = series["lag"]
    stats = [("Độ lệch đo được", f"{lag:+.3f}s"),
             ("Tương quan đỉnh", f"{series['r']:.3f}"),
             ("Cửa sổ đồng thuận", f"{meas.get('agree_count', '?')}/{meas.get('windows_total', '?')}"),
             ("Chênh giữa cửa sổ", f"{meas.get('lag_spread_s', 0):.3f}s")]
    html = (PLOT_HTML
            .replace("__TITLE__", f"Đo lệch — {folder_name}")
            .replace("__SUB__", f"Cửa sổ {series.get('window_start_s', 0):.0f}s–"
                                f"{series.get('window_start_s', 0) + len(series['video']) / series['hz']:.0f}s "
                                f"của video, lấy mẫu {series['hz']:.0f} Hz. "
                                f"Hai đường đều đã bỏ xu hướng chậm và chuẩn hoá về cùng thang.")
            .replace("__LAG__", f"{lag:+.3f}")
            .replace("__STATS__", "".join(
                f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>'
                for k, v in stats))
            .replace("__NOTE__", f'<p class="note">{extra_note}</p>' if extra_note else "")
            .replace("__DATA__", json.dumps(series, separators=(",", ":"), ensure_ascii=False)))
    path.write_text(html, encoding="utf-8")


# ======================================================================================
# 5. Cat video bang ffmpeg
# ======================================================================================

def pick_encoder(args) -> str:
    if args.codec != "auto":
        return args.codec
    if platform.system() == "Darwin":
        try:
            if "h264_videotoolbox" in subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout:
                return "h264_videotoolbox"
        except Exception:
            pass
    return "libx264"


def build_ffmpeg_cmd(vinfo: VideoInfo, dst: Path, ss: float, dur: float, nframes: int, args) -> list:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
           "-stats", "-stats_period", "5", "-y",
           "-ss", f"{ss:.6f}", "-i", str(vinfo.path)]
    if args.video_mode == "copy" or args.cfr:
        # copy: khong the bat dau giua GOP -> ffmpeg lui ve keyframe gan nhat.
        # cfr: so frame ra khac so frame nguon (chen frame o cho rot) -> gioi han theo thoi luong.
        cmd += ["-t", f"{dur:.6f}"]
    else:
        # -frames:v dam bao ra DUNG so frame (chinh xac hon -t khi video VFR)
        cmd += ["-frames:v", str(nframes)]
    cmd += ["-map", "0:v:0"]
    if vinfo.has_audio:
        cmd += ["-map", "0:a?"]
    if args.video_mode == "copy":
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        enc = pick_encoder(args)
        cmd += ["-c:v", enc]
        if enc.startswith("libx26"):
            cmd += ["-crf", str(args.crf), "-preset", args.preset]
        else:
            cmd += ["-b:v", str(args.bitrate or max(int(vinfo.bit_rate * 1.1), 4_000_000))]
        cmd += ["-pix_fmt", "yuv420p"]
        # giu nguyen nhip frame goc (video quay tay thuong VFR); --cfr de ep ve fps co dinh
        cmd += ["-fps_mode", "cfr", "-r", f"{vinfo.r_fps:.6f}"] if args.cfr else ["-fps_mode", "passthrough"]
        cmd += ["-c:a", "aac", "-b:a", "192k"] if vinfo.has_audio else ["-an"]
    cmd += ["-movflags", "+faststart", str(dst)]
    return cmd


# ======================================================================================
# 6. Xu ly 1 thu muc
# ======================================================================================

VIDEO_EXT = (".mp4", ".mov", ".m4v")
OUT_SUFFIX = ("_sync", "_fit", "_perframe")


def find_job_files(folder: Path):
    mp4s = sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXT and not p.stem.endswith(OUT_SUFFIX))
    csvs = sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".csv" and not p.stem.endswith(OUT_SUFFIX))
    return mp4s, csvs


def output_dir_for(folder: Path, args) -> Path:
    """--out-dir cho 1 thu muc = chinh no; cho nhieu thu muc = thu muc GOC, moi
    thu muc nguon duoc mot thu muc con cung ten."""
    if not args.out_dir:
        return folder / "synced"
    base = Path(args.out_dir).expanduser()
    return base / folder.name if getattr(args, "out_dir_is_root", False) else base


def process_folder(folder: Path, args) -> dict:
    report = {"folder": str(folder), "status": "ok"}
    mp4s, csvs = find_job_files(folder)
    if len(mp4s) != 1 or len(csvs) != 1:
        report.update(status="skipped",
                      reason=f"can dung 1 video + 1 csv (thay {len(mp4s)} video, {len(csvs)} csv)")
        return report
    mp4, csvf = mp4s[0], csvs[0]
    report["mp4"], report["csv"] = mp4.name, csvf.name

    # ---- moc thoi gian tu ten file
    csv_ts, _ = parse_name_datetime(csvf.stem)
    if csv_ts is None:
        report.update(status="error", reason=f"khong doc duoc moc thoi gian tu ten CSV: {csvf.name}")
        return report
    vinfo = ffprobe_video(mp4, read_pts=not args.no_pts)
    cidx = index_csv(csvf, args.time_col)
    t_first, t_last = cidx.times[0], cidx.times[-1]
    csv_span = t_last - t_first

    dead = find_dead_zones(cidx) if not args.no_dead_check else []

    cands = video_time_candidates(mp4, vinfo, csv_ts.date(), args)
    if not cands:
        report.update(status="error",
                      reason=f"khong xac dinh duoc moc thoi gian cua video {mp4.name} "
                             f"(nguon '{args.video_time_source}'). Doi ten file thanh dang "
                             f"HH-MM-SS-mmm.mp4 hoac dung --video-time-source khac.")
        return report
    # xu ly truong hop qua ngay (ten file khong mang ngay)
    for c in cands:
        if c["source"] == "name":
            dh = (c["start"] - csv_ts).total_seconds() / 3600.0
            c["start"] += timedelta(days=-1 if dh > 12 else (1 if dh < -12 else 0))

    def csv_interval(base):
        shift = 0.0 if base == "absolute" else -t_first
        e0 = (csv_ts + timedelta(seconds=shift)).timestamp()
        return e0, e0 + t_first, e0 + t_last

    if args.csv_base == "auto":     # chon base cho vung giao lon hon, theo ung vien uu tien nhat
        ref = sorted(cands, key=lambda c: PRIORITY.get((c["source"], c["anchor"]), 9))[0]
        r0 = ref["start"].timestamp()
        bases = [(b, *csv_interval(b)) for b in ("absolute", "relative")]
        csv_base_used, csv_epoch0, c0e, c1e = max(
            bases, key=lambda b: overlap_seconds(r0, r0 + vinfo.span, b[2], b[3]))
    else:
        csv_base_used = args.csv_base
        csv_epoch0, c0e, c1e = csv_interval(csv_base_used)

    pick, source_note, cands = pick_video_time(cands, vinfo, c0e, c1e)
    ts_source, ts_res, anchor = pick["source"], pick["resolution_s"], pick["anchor"]
    mp4_ts = pick["start"] if anchor == "start" else pick["start"] + timedelta(seconds=vinfo.span)
    video_start = pick["start"]
    v0e = video_start.timestamp()

    # ---- do do lech that tu noi dung (tuy chon, nhung la cach duy nhat dat do chinh xac ms)
    meas = None
    if args.measure:
        v1e_tmp = v0e + vinfo.span
        t0m, t1m = max(v0e, c0e), min(v1e_tmp, c1e)
        if t1m - t0m < 180:
            meas = {"ok": False, "reason": "vung giao qua ngan (<3 phut) de do"}
        else:
            print(f"    do do lech tu chuyen dong ({args.measure_windows} cua so)...")
            meas = measure_offset(vinfo, cidx, v0e, csv_epoch0, t0m, t1m, args)
        if meas.get("ok") or (meas.get("lag_median_s") is not None and args.measure_force):
            lag = meas["lag_median_s"]
            video_start += timedelta(seconds=lag)
            v0e = video_start.timestamp()
            meas["applied_s"] = lag
        else:
            meas["applied_s"] = None

    # wall clock (epoch giay) cua tung frame video va tung dong CSV
    if vinfo.pts:
        fw = [v0e + (p - vinfo.pts[0]) for p in vinfo.pts]
    else:
        fw = [v0e + i / vinfo.fps for i in range(vinfo.nb_frames)]
    rw = [csv_epoch0 + t for t in cidx.times]

    v1e = fw[-1] + vinfo.median_dt
    c0e, c1e = rw[0], rw[-1]
    if args.trim_dead and dead:
        c1e = min(c1e, csv_epoch0 + dead[0][0])
    T0, T1 = max(v0e, c0e), min(v1e, c1e)

    report["analysis"] = {
        "mp4_name_time": mp4_ts.isoformat(sep=" "),
        "video_time_source": ts_source,
        "video_time_resolution_s": ts_res,
        "csv_name_time": csv_ts.isoformat(sep=" "),
        "mp4_anchor_used": anchor,
        "csv_base_used": csv_base_used,
        "overlap_s": pick["overlap_s"],
        "video_time_warning": source_note,
        "video_time_all_candidates": [
            {"source": c["source"], "anchor": c["anchor"],
             "video_start": c["start"].isoformat(sep=" "),
             "overlap_with_csv_s": c["overlap_s"], "resolution_s": c["resolution_s"]}
            for c in sorted(cands, key=lambda c: PRIORITY.get((c["source"], c["anchor"]), 9))],
        "video": {
            "start": video_start.isoformat(sep=" "),
            "end": datetime.fromtimestamp(v1e).isoformat(sep=" "),
            "duration_s": round(vinfo.span, 3), "frames": vinfo.nb_frames,
            "fps_avg": round(vinfo.fps, 6), "fps_nominal": round(vinfo.r_fps, 6),
            "vfr": vinfo.vfr, "size": f"{vinfo.width}x{vinfo.height}",
            "pts_source": "ffprobe packets" if vinfo.pts else "fps danh nghia (uoc luong)",
            **gap_stats(vinfo.pts or [], "frame"),
        },
        "csv": {
            "start": datetime.fromtimestamp(c0e).isoformat(sep=" "),
            "end": datetime.fromtimestamp(c1e).isoformat(sep=" "),
            "span_s": round(csv_span, 3), "rows": len(cidx.times),
            "sample_rate_hz": round((len(cidx.times) - 1) / csv_span, 4) if csv_span else 0,
            **gap_stats(cidx.times, "row"),
            "dead_zones": [{"from_s": z[0], "to_s": z[1], "len_s": round(z[1] - z[0], 1)}
                           for z in dead],
        },
        "head_offset_csv_minus_video_s": round(c0e - v0e, 4),
        "tail_offset_csv_minus_video_s": round(c1e - v1e, 4),
        "clock_drift_ratio_video_over_csv": round(vinfo.span / csv_span, 8) if csv_span else None,
        "clock_drift_reliable": bool(abs(c0e - v0e) < 2.0 and abs(c1e - v1e) < 2.0),
        "measured_offset": meas,
    }

    if T1 - T0 <= 0:
        report.update(status="error",
                      reason="video va CSV khong giao nhau ve thoi gian (kiem tra lai ten file / --mp4-anchor)")
        return report

    # ---- chon frame dau/cuoi va dong CSV dau/cuoi gan T0, T1 nhat
    k0, k1 = nearest_index(fw, T0), nearest_index(fw, T1)
    if k1 < k0:
        k1 = k0
    T0s, T1s = fw[k0], fw[k1]                       # moc thuc te sau khi bam vao frame
    i0, i1 = nearest_index(rw, T0s), nearest_index(rw, T1s)
    if i1 < i0:
        i1 = i0

    guard = min(vinfo.median_dt, (fw[k0] - fw[k0 - 1]) if k0 > 0 else vinfo.median_dt) / 2.0
    ss = (vinfo.pts[k0] if vinfo.pts else k0 / vinfo.fps) - guard
    ss = max(ss, 0.0)
    last_pts = vinfo.pts[k1] if vinfo.pts else k1 / vinfo.fps
    dur = (last_pts + guard) - ss

    drift_reliable = report["analysis"]["clock_drift_reliable"]
    drift_k = 1.0
    if args.drift_correct and csv_span > 0:
        if drift_reliable:
            drift_k = vinfo.span / csv_span
        else:
            print("    [canh bao] 2 file khong phu cung mot khoang thuc -> bo qua --drift-correct")

    report["sync"] = {
        "cut_start_wall": datetime.fromtimestamp(T0s).isoformat(sep=" "),
        "cut_end_wall": datetime.fromtimestamp(T1s).isoformat(sep=" "),
        "video_trim_head_s": round(fw[k0] - v0e, 6),
        "video_trim_tail_s": round(v1e - (fw[k1] + vinfo.median_dt), 6),
        "video_out_frames": k1 - k0 + 1,
        "video_out_duration_s": round(dur, 6),
        "video_frame_snap_error_ms": round((T0s - T0) * 1000, 3),
        "csv_trim_head_s": round(rw[i0] - c0e, 6),
        "csv_trim_tail_s": round(c1e - rw[i1], 6),
        "csv_out_rows": i1 - i0 + 1,
        "csv_head_match_error_ms": round((rw[i0] - T0s) * 1000, 3),
        "csv_tail_match_error_ms": round((rw[i1] - T1s) * 1000, 3),
        "ffmpeg_ss": round(ss, 6), "ffmpeg_t": round(dur, 6),
        "drift_correction_applied": drift_k if (args.drift_correct and drift_reliable) else None,
    }

    if args.plot and meas and meas.get("_plot"):
        pdir = output_dir_for(folder, args)
        pdir.mkdir(parents=True, exist_ok=True)
        note = ""
        if report["analysis"]["csv"].get("dead_zones"):
            z = report["analysis"]["csv"]["dead_zones"][0]
            note = (f"Canh bao: mocap dung han tu giay {z['from_s']} den {z['to_s']} "
                    f"({z['len_s']}s). Doan do khong dung duoc, chay lai voi --trim-dead.")
        write_plot(pdir / "sync_plot.html", folder.name, meas["_plot"], meas, note)
        report["plot_output"] = str(pdir / "sync_plot.html")
        print(f"    -> bieu do: {pdir / 'sync_plot.html'}")
    if meas:
        meas.pop("_plot", None)
        meas.pop("_plot_r", None)

    if args.dry_run:
        report["status"] = "dry-run"
        return report

    outdir = output_dir_for(folder, args)
    outdir.mkdir(parents=True, exist_ok=True)

    # canh bao dung luong: file ra xap xi file vao (video + csv, them csv perframe neu bat)
    need = mp4.stat().st_size + csvf.stat().st_size * (2 if args.per_frame else 1)
    free = shutil.disk_usage(outdir).free
    if free < need * 1.1:
        print(f"    [canh bao] dia con {free / 2**30:.1f} GB, can khoang {need / 2**30:.1f} GB. "
              f"Dung --video-mode none hoac --out-dir tro sang o dia khac neu thieu.")
    out_mp4 = outdir / f"{mp4.stem}_sync{mp4.suffix}"
    out_csv = outdir / f"{csvf.stem}_sync.csv"

    # ---- cat video
    if args.video_mode == "none":
        report["video_output"] = None
    elif k0 == 0 and k1 == len(fw) - 1:
        # video da nam gon trong khoang giao -> khong can cat
        if args.hardlink:
            out_mp4.unlink(missing_ok=True)
            out_mp4.hardlink_to(mp4)
            action = "hardlink (khong can cat)"
        else:
            shutil.copy2(mp4, out_mp4)
            action = "copy nguyen ban (khong can cat)"
        report["video_output"] = {"file": str(out_mp4), "action": action}
    else:
        cmd = build_ffmpeg_cmd(vinfo, out_mp4, ss, dur, k1 - k0 + 1, args)
        print(f"    ffmpeg: bo {fw[k0] - v0e:.3f}s dau + {v1e - fw[k1] - vinfo.median_dt:.3f}s cuoi "
              f"-> {dur:.3f}s / {k1 - k0 + 1} frames ({args.video_mode})")
        if args.verbose:
            print("    " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        got = ffprobe_video(out_mp4, read_pts=False).nb_frames
        want = int(round(dur * vinfo.r_fps)) if args.cfr else (k1 - k0 + 1)
        report["video_output"] = {"file": str(out_mp4),
                                  "action": args.video_mode + (" (cfr)" if args.cfr else ""),
                                  "frames_expected": want, "frames_actual": got,
                                  "ffmpeg": " ".join(cmd)}
        if args.video_mode == "precise" and abs(got - want) > 1:
            print(f"    [canh bao] so frame ra {got}, ky vong {want}")

    # ---- cat CSV
    rebase = not args.keep_time
    with open(csvf, "rb") as src, open(out_csv, "wb") as dst:
        dst.write(cidx.header if cidx.header.endswith(b"\n") else cidx.header + b"\n")
        for i in range(i0, i1 + 1):
            src.seek(cidx.offsets[i])
            line = src.read(cidx.lengths[i])
            if rebase:
                new_t = (rw[i] - T0s) * drift_k
                if cidx.time_col == 0:
                    rest = line.split(b",", 1)[1] if b"," in line else b"\n"
                    line = f"{new_t:.6f}".encode() + b"," + rest
                else:
                    fields = line.rstrip(b"\r\n").split(b",")
                    fields[cidx.time_col] = f"{new_t:.6f}".encode()
                    line = b",".join(fields) + b"\n"
            if not line.endswith(b"\n"):
                line += b"\n"
            dst.write(line)
    report["csv_output"] = {"file": str(out_csv), "rows": i1 - i0 + 1,
                            "time_column": "rebase ve 0.0 tai frame dau" if rebase else "giu nguyen"}

    # ---- CSV 1 dong / 1 frame (tuy chon)
    if args.per_frame:
        out_pf = outdir / f"{csvf.stem}_perframe.csv"
        max_err, sum_abs = 0.0, 0.0
        with open(csvf, "rb") as src, open(out_pf, "wb") as dst:
            dst.write(b"frame,video_time,match_error_ms," + cidx.header.rstrip(b"\r\n") + b"\n")
            j = i0
            for n, k in enumerate(range(k0, k1 + 1)):
                target = fw[k]
                while j + 1 <= i1 and abs(rw[j + 1] - target) <= abs(rw[j] - target):
                    j += 1
                while j - 1 >= i0 and abs(rw[j - 1] - target) < abs(rw[j] - target):
                    j -= 1
                err = (rw[j] - target) * 1000.0
                max_err = max(max_err, abs(err))
                sum_abs += abs(err)
                src.seek(cidx.offsets[j])
                line = src.read(cidx.lengths[j]).rstrip(b"\r\n")
                dst.write(f"{n},{target - T0s:.6f},{err:.3f},".encode() + line + b"\n")
        report["per_frame_output"] = {
            "file": str(out_pf), "rows": k1 - k0 + 1,
            "max_match_error_ms": round(max_err, 3),
            "mean_match_error_ms": round(sum_abs / max(k1 - k0 + 1, 1), 3),
        }

    with open(outdir / "sync_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


# ======================================================================================
# 7. CLI
# ======================================================================================

def print_report(r: dict):
    print(f"\n=== {Path(r['folder']).name}")
    if r["status"] in ("skipped", "error"):
        print(f"    [{r['status'].upper()}] {r.get('reason')}")
        return
    a, v, c = r["analysis"], r["analysis"]["video"], r["analysis"]["csv"]
    print(f"    VIDEO {r['mp4']}")
    print(f"          {v['start']}  ->  {v['end']}   ({v['duration_s']}s, {v['frames']} frames, "
          f"{v['fps_avg']:.3f}fps{' VFR' if v['vfr'] else ''}, {v['size']})")
    if v.get("frame_gap_max_ms"):
        print(f"          frame gap: median {v['frame_gap_median_ms']}ms / max {v['frame_gap_max_ms']}ms "
              f"({v['frame_gaps_over_50ms']} cho rot frame)")
    print(f"    CSV   {r['csv']}")
    print(f"          {c['start']}  ->  {c['end']}   ({c['span_s']}s, {c['rows']} rows, {c['sample_rate_hz']}Hz)")
    if c.get("dead_zones"):
        for z in c["dead_zones"]:
            print(f"    [CANH BAO] mocap DUNG HAN tu giay {z['from_s']} den {z['to_s']} "
                  f"({z['len_s']}s) - du lieu doan nay vo dung, dung --trim-dead de cat bo")
    if c.get("row_gap_max_ms"):
        print(f"          row gap  : median {c['row_gap_median_ms']}ms / max {c['row_gap_max_ms']}ms "
              f"({c['row_gaps_over_50ms']} cho rot mau)")
    src_note = {"name": "ten file (ms)", "creation": "metadata creation_time (1 giay)",
                "mtime": "gio sua file (1 giay)"}.get(a.get("video_time_source"), a.get("video_time_source"))
    print(f"    Moc video: {a['mp4_name_time']}  (nguon: {src_note})")
    if a.get("video_time_resolution_s", 0) >= 1.0:
        print(f"    [chu y] nguon nay chi chinh xac toi 1 giay -> dong bo sai so ~±0.5s, khong phai ms. "
              f"Doi ten video thanh HH-MM-SS-mmm.mp4 neu biet gio bat dau chinh xac.")
    print(f"    Lech dau {a['head_offset_csv_minus_video_s']:+.3f}s (CSV - video), "
          f"lech cuoi {a['tail_offset_csv_minus_video_s']:+.3f}s  |  csv_base={a['csv_base_used']}")
    if a.get("video_time_warning"):
        print(f"    [CANH BAO] {a['video_time_warning']}")
    if len(a.get("video_time_all_candidates", [])) > 1:
        print("    Cac nguon moc thoi gian (giao voi CSV):")
        for c in a["video_time_all_candidates"]:
            mark = "*" if (c["source"] == a["video_time_source"]
                           and c["anchor"] == a["mp4_anchor_used"]) else " "
            kind = "bat dau" if c["anchor"] == "start" else "ket thuc - da tru duration"
            print(f"      {mark} {c['source']:8s} ({kind:28s}) -> {c['video_start']}  "
                  f"giao {c['overlap_with_csv_s']}s")
    m = a.get("measured_offset")
    if m:
        if m.get("lag_median_s") is not None:
            print(f"    DO TU CHUYEN DONG: lech {m['lag_median_s']:+.3f}s  "
                  f"({m['agree_count']}/{m['windows_total']} cua so dong thuan trong "
                  f"{m['lag_spread_s']}s, r>={m['r_min']}, >={m['sharpness_min']} sigma)")
            for o in m.get("outliers", []):
                print(f"       (bo cua so {o['video_start_s']}s: lag {o['lag_s']}s, "
                      f"r={o['r']}, {o['sharpness_sigma']} sigma)")
        if m.get("applied_s") is not None:
            print(f"    -> DA AP DUNG: moc video dich {m['applied_s']:+.3f}s so voi "
                  f"{a['video_time_source']}")
        else:
            print(f"    -> KHONG ap dung: {m.get('reason', 'khong dat nguong tin cay')}")
    s = r["sync"]
    print(f"    CAT  {s['cut_start_wall']}  ->  {s['cut_end_wall']}   ({s['video_out_duration_s']}s)")
    print(f"      video: -{s['video_trim_head_s']:.3f}s dau, -{s['video_trim_tail_s']:.3f}s cuoi "
          f"-> {s['video_out_frames']} frames (bam frame: {s['video_frame_snap_error_ms']:+.1f}ms)")
    print(f"      csv  : -{s['csv_trim_head_s']:.3f}s dau, -{s['csv_trim_tail_s']:.3f}s cuoi "
          f"-> {s['csv_out_rows']} rows (khop dau {s['csv_head_match_error_ms']:+.1f}ms, "
          f"cuoi {s['csv_tail_match_error_ms']:+.1f}ms)")
    if a.get("clock_drift_ratio_video_over_csv"):
        k = a["clock_drift_ratio_video_over_csv"]
        tag = "" if a.get("clock_drift_reliable") else "  [khong dang tin: 2 file phu 2 khoang khac nhau]"
        print(f"      drift: video/csv = {k:.6f} ({(k - 1) * 1000:+.3f} ms/giay){tag}")
    for key, label in (("video_output", "video"), ("csv_output", "csv"), ("per_frame_output", "perframe")):
        if r.get(key):
            extra = ""
            if key == "per_frame_output":
                extra = (f"  (sai so khop: TB {r[key]['mean_match_error_ms']}ms, "
                         f"max {r[key]['max_match_error_ms']}ms)")
            if key == "video_output" and "frames_actual" in r[key]:
                extra = f"  ({r[key]['frames_actual']} frames)"
            print(f"      -> [{label}] {r[key]['file']}{extra}")


def main():
    p = argparse.ArgumentParser(
        description="Fit MP4 voi CSV Rokoko theo wall-clock, cat 2 dau cho khop nhau.")
    p.add_argument("folder", help="Thu muc chua 1 video + 1 csv, hoac thu muc cha chua nhieu thu muc nhu vay")
    p.add_argument("--out-dir", default=None, help="Thu muc xuat (mac dinh <folder>/synced)")
    p.add_argument("--dry-run", action="store_true", help="Chi phan tich, khong tao file")
    p.add_argument("--video-mode", choices=["precise", "copy", "none"], default="precise",
                   help="precise=encode lai, cat dung frame (mac dinh) | copy=nhanh, bam keyframe | none=bo qua video")
    p.add_argument("--codec", default="auto", help="auto | h264_videotoolbox | libx264 | hevc_videotoolbox")
    p.add_argument("--crf", type=int, default=18, help="CRF cho libx264 (mac dinh 18)")
    p.add_argument("--preset", default="veryfast", help="preset libx264 (mac dinh veryfast)")
    p.add_argument("--bitrate", default=None, help="Bitrate cho encoder phan cung, vd 20M")
    p.add_argument("--cfr", action="store_true", help="Ep video ra fps co dinh (mac dinh giu nhip frame goc)")
    p.add_argument("--video-time-source", choices=["auto", "name", "creation", "mtime"], default="auto",
                   help="Lay moc thoi gian video tu dau: ten file (chinh xac ms) -> metadata "
                        "creation_time (chinh xac 1 giay) -> mtime (1 giay, coi la moc KET THUC). "
                        "auto = thu lan luot theo thu tu do.")
    p.add_argument("--mp4-anchor", choices=["auto", "start", "end"], default=None,
                   help="Moc video la BAT DAU (start) hay KET THUC (end). Mac dinh theo nguon: "
                        "ten file/creation_time -> start, mtime -> end. "
                        "'auto' chon theo vung giao lon nhat - co the doan sai khi 2 file "
                        "chi giao nhau mot phan, script se canh bao neu gia thuyet khac tot hon nhieu.")
    p.add_argument("--csv-base", choices=["auto", "absolute", "relative"], default="absolute",
                   help="Cot time cong thang vao moc ten CSV (absolute, mac dinh - dung voi Rokoko) "
                        "hay tru gia tri dong dau (relative). 'auto' co the doan sai khi CSV bat dau "
                        "tre trong phien va 2 file chi giao nhau mot phan.")
    p.add_argument("--time-col", default=None, help="Ten cot thoi gian trong CSV (mac dinh tu dong)")
    p.add_argument("--keep-time", action="store_true", help="Giu nguyen cot time, khong rebase ve 0")
    p.add_argument("--per-frame", action="store_true", help="Xuat them CSV: 1 dong ung voi 1 frame video")
    p.add_argument("--drift-correct", action="store_true", help="Hieu chinh lech nhip dong ho 2 thiet bi")
    p.add_argument("--hardlink", action="store_true",
                   help="Khi video khong can cat: tao hardlink thay vi copy (tiet kiem dia)")
    p.add_argument("--no-dead-check", action="store_true",
                   help="Bo qua viec do tim doan mocap dung han (suit mat ket noi)")
    p.add_argument("--trim-dead", action="store_true",
                   help="Cat bo luon phan sau doan mocap chet dau tien (ca video lan CSV)")
    p.add_argument("--measure", action="store_true",
                   help="Do do lech THUC giua video va CSV bang tuong quan chuyen dong "
                        "(camera gan tren nguoi <-> van toc goc khop head). Cham hon nhung la "
                        "cach duy nhat chinh xac toi ms khi dong ho 2 thiet bi lech nhau.")
    p.add_argument("--plot", action="store_true",
                   help="Xuat sync_plot.html: 2 duong cong truoc/sau khi can + do thi tuong quan. "
                        "Di kem --measure. Mo file bang trinh duyet la thay ngay.")
    p.add_argument("--measure-windows", type=int, default=3, help="So cua so do (mac dinh 3)")
    p.add_argument("--measure-window", type=float, default=500.0,
                   help="Do dai moi cua so do, giay (mac dinh 500)")
    p.add_argument("--measure-maxlag", type=float, default=60.0,
                   help="Do lech toi da can quet, giay (mac dinh 60)")
    p.add_argument("--measure-tol", type=float, default=0.25,
                   help="Cac cua so duoc phep lech nhau bao nhieu giay thi con tin (mac dinh 0.25)")
    p.add_argument("--measure-force", action="store_true",
                   help="Ap dung ket qua do ke ca khi khong dat nguong tin cay")
    p.add_argument("--no-pts", action="store_true", help="Khong doc PTS tung frame (nhanh hon, kem chinh xac voi VFR)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    for tool in ("ffprobe", "ffmpeg"):
        if shutil.which(tool) is None:
            sys.exit(f"Thieu {tool} trong PATH. Cai bang: brew install ffmpeg")

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Khong phai thu muc: {root}")

    mp4s, csvs = find_job_files(root)
    if len(mp4s) == 1 and len(csvs) == 1:
        folders = [root]
    else:
        folders = [d for d in sorted(root.iterdir())
                   if d.is_dir() and d.name != "synced"
                   and not d.name.startswith((".", "__"))] or [root]

    args.out_dir_is_root = bool(args.out_dir and len(folders) > 1)
    if args.out_dir_is_root:
        print(f"Xuat tat ca vao {Path(args.out_dir).expanduser()}/<ten thu muc nguon>/\n")

    reports = []
    for folder in folders:
        try:
            r = process_folder(folder, args)
        except subprocess.CalledProcessError as e:
            r = {"folder": str(folder), "status": "error", "reason": f"ffmpeg loi: {e}"}
        except Exception as e:
            r = {"folder": str(folder), "status": "error", "reason": f"{type(e).__name__}: {e}"}
        reports.append(r)
        print_report(r)

    ok = sum(1 for r in reports if r["status"] in ("ok", "dry-run"))
    print(f"\nXong: {ok}/{len(reports)} thu muc.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
