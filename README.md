# rokoko_pipeline

Body-pose labels from a Rokoko suit (CSV) and a head-mounted iPhone (video + ARKit), end to end:

```
raw Rokoko CSV + iPhone zip
  -> 1 sync -> 2 clean + wrong-pose filter -> 3 camera          run_final.sh   (CPU, all sessions)
  -> 4 detect -> 5 masks -> 6 hands -> 7 arm fit -> 8 filter    run_arm2d.sh   (GPU, one session)
  -> labels: arm2d/fit/<clip>.npz + <clip>_keep.npy
```

The labels are the output of the **whole** pipeline. The cleaned Rokoko alone (stage 2) is not the ground truth.

## 1. Quick start: only a Rokoko CSV (CPU, no video, no weights)

Cleans a recording and flags the wrong poses. Samples (5 with suit drift, 2 good):
https://huggingface.co/datasets/NguyenHuuChi/rokoko_raw_samples

```bash
git clone https://github.com/NguyenHuuChi/rokoko_pipeline.git && cd rokoko_pipeline
python -m venv ~/envs/pipeline && PY=~/envs/pipeline/bin/python
$PY -m pip install -r requirements/pipeline.txt huggingface_hub
~/envs/pipeline/bin/hf download NguyenHuuChi/rokoko_raw_samples --repo-type dataset --local-dir samples
bash prepare_samples.sh samples takes                 # one folder per recording, <name>_sync.csv inside
$PY clean_mocap.py takes --rokoko-only --force
```

Each `takes/<name>/` gets `<name>_perframe_clean.csv` and `segments.csv`. A frame with `valid = 0` must not be used.
`frames_implausible` in the printed summary = frames removed by the wrong-pose filter.

## 2. Full setup

Linux, NVIDIA GPU (CUDA 12) for stages 4-8, ffmpeg / ffprobe >= 5.

**Environments**: four, because their versions conflict. The Python version is on the first line of each file.

| Env | Install | Used for |
|---|---|---|
| pipeline | `pip install -r requirements/pipeline.txt` | sync, clean, camera |
| detect | `pip install -r requirements/detect.txt` | hand5 / RTMPose |
| handseg | `pip install -r requirements/handseg.txt`, then SAM 2 (and SAM 3 only for `SEG=sam3`), below | masks, fit, filter |
| wilor | `pip install -r requirements/wilor.txt`, then chumpy, below | WiLoR hands |

```bash
# in handseg
pip install git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4
pip install git+https://github.com/facebookresearch/sam3.git@7f473cfd983d5eb43040ff608f66bff0d7bdcb53    # also needed by SEG=sam2 (fit imports it)
# in wilor
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
```

**Paths**

```bash
cp .env.example .env        # then edit: RAW_ROOT, OUT_ROOT, the four pythons, FFMPEG_DIR
```

**Weights**

```bash
hf auth login               # account with access to the private weights repo
bash download_models.sh     # models/ + third_party/WiLoR (code + its weights)
bash download_fah.sh        # third_party/FAH (YOLO architecture) + models/fah/shared/detector.pt (~52 MB, for fasam/fah)
```

**MANO (by hand, its license forbids sharing it)**: WiLoR needs it. Register at https://mano.is.tue.mpg.de, download
"Models & Code", and copy `MANO_RIGHT.pkl` to `third_party/WiLoR/mano_data/`.
Without MANO, run with `WRIST=hand5` (no WiLoR).

**Check**

```bash
python check_setup.py       # lists everything that is missing
```

## 3. Raw data layout

```
RAW_ROOT/<dd-mm-yyyy>/<group>/session_<YYYYMMDD>_<HHMMSS>_*.zip         iPhone app export
RAW_ROOT/<dd-mm-yyyy>/<group>/rokoko_recording_<YYYYMMDD>_<HHMMSS>_<ms>.csv
```

The zip holds `video.mp4`, `metadata.json` (clock sync), `mocap/video_frame_index.csv`, ARKit poses. Video and CSV are
paired by wall clock, no manual pairing. App uploads as video + MCAP: convert first with `mcap_session.py`.
Check the pairing without writing anything: `python prepare_mocap.py RAW_ROOT OUT_ROOT --scan`.

## 4. Run

```bash
bash run_final.sh                                          # stages 1-3, every session
bash run_arm2d.sh SESSION_DIR                              # stages 4-8, default MODE=fasam (M3b)
bash run_arm2d.sh SESSION_DIR --mode fah                   # M3a: FAH YOLO-seg + RTMPose lock + gap-fill/SavGol -> WiLoR (no SAM 2.1)
bash run_arm2d.sh SESSION_DIR --mode hand5                 # legacy baseline: RTMPose + Hand5 -> SAM 2/3 -> WiLoR
JOB=<slurm id> bash run_arm2d.sh SESSION_DIR               # same, GPU steps inside that Slurm job
JOB=<slurm id> bash run_arm2d.sh SESSION_DIR 0 2400        # only the 240-frame clips starting at these frames
```

Use tmux. Expect roughly 6-7 GPU minutes per 240-frame clip (less for `MODE=fah`). Finished clip steps are skipped on a rerun.

### Hand detection / segmentation mode (`MODE` / `--mode`)

- **`fasam` (default, M3b):** Uses `RTMPose` + `FAH` (`shared/detector.pt`, YOLOv8-seg) to lock Left/Right hand masks on clean separated frames as 2-ID anchors (`L=1, R=2`), tracks them across the whole clip with base **SAM 2.1** (`sam2_clipmask.py --mode fasam`), smooths tight square crops with SavGol, and runs **WiLoR** (`rescale=2.0`) before Rokoko `arm2d_fit.py` / `arm2d_filter.py`.
- **`fah` (M3a):** Skips SAM 2.1 completely; runs `FAH` (`fah_detect.py --mode fah`) + `RTMPose` L/R Hungarian matching + short linear gap-fill (`1-3` frames) + SavGol smoothing -> **WiLoR** (`rescale=2.0`) -> `arm2d_fit.py` / `arm2d_filter.py`.
- **`hand5` (legacy):** Original repo pipeline (`RTMPose + Hand5` -> `SEG=sam2` or `SEG=sam3` -> `WiLoR`).

Options: `--mode` / `MODE=fasam` (default, M3b), `fah` (M3a), or `hand5`; `WRIST=wilor` (default) or `hand5`; `SEG=sam2` (default for `MODE=hand5`) or `sam3`; `LEN=240`.
Note: `MODE=fasam` and `MODE=fah` require `bash download_fah.sh` first (or `FAH_DET=/path/to/detector.pt`).

## 5. Output: `OUT_ROOT/<date>/<group>/<session>/`

| File | Content |
|---|---|
| `*_sync.mp4`, `*_sync.csv`, `sync_report.json` | synced video / Rokoko CSV, measured offset |
| `*_perframe.csv` | Rokoko row per video frame, raw |
| `*_perframe_clean.csv` | cleaned; row i = video frame i; `valid` (0 = do not use), `segment`, `repaired`, `gap_ms` |
| `segments.csv` | kept continuous pieces |
| `camera/rokoko_cam.npz` | joints in the camera frame + projection (`Ku`, `Ruw`, `tuw`) |
| `arm2d/clips/<start>/` | clip.mp4, detections, masks, `wilor_wrists.npz` |
| `arm2d/fit/<clip>.npz` | **labels**: every joint in the corrected camera frame (`seg_<joint>`) |
| `arm2d/fit/<clip>_keep.npy` | frames that passed the last filter; use only these |

Rokoko axes: y up, **left-handed**. Mirror x for a right-handed frame: position `(-x, y, z)`, quaternion `(x, -y, -z, w)`.

## 6. What each stage does

| Stage | Code | What it does |
|---|---|---|
| 1 sync | `prepare_mocap.py`, `fit_rokoko.py` | pairs video and CSV, measures the clock offset from motion, one Rokoko row per video frame |
| 2 clean | `clean_mocap.py` | cuts stream freezes > 0.5 s, repairs glitches and smooths joint directions (bone lengths never change), drops pieces < 3 s, removes wrong poses |
| 3 camera | `camera_adapter.py`, `rokoko_cam.py` | head -> camera geometry from motion |
| 4 detect | `hung5_detect.py`, `fah_detect.py` | RTMPose elbows/shoulders/wrists + FAH YOLOv8-seg (`fasam`/`fah`, default) or Hand5 (`hand5`) |
| 5 masks | `sam2_clipmask.py` (or `handseg_clipmask.py`) | `fasam`: 2-ID SAM 2.1 (`L=1, R=2`) on FAH anchors -> `masks.npz` + `fah_boxes.npz`; skipped in `fah`; `hand5`: wearer + hand masks |
| 6 hands | `wilor_wrists.py`, `hand_track.py` | WiLoR on `fah_boxes.npz` (`fasam`/`fah`, `rescale=2.0`) or `masks.npz` (`hand5`) -> `wilor_wrists.npz` |
| 7 arm fit | `arm2d_fit.py` | small camera correction (<= 3 deg, <= 5 cm), then arms bent to wrists and masks |
| 8 filter | `arm2d_filter.py` | drops stretches where the arms still miss the image |
| QA | `sync_check.py`, `make_split.py` | Rokoko head vs ARKit; train / test split |

### Wrong-pose filter (`implausible()` in `clean_mocap.py`)

Suit drift is slow, nothing jumps, so glitch detection misses it. On 10 s rolling medians a frame is removed when:

| Rule | Condition |
|---|---|
| twist | hip line vs shoulder line > 60 deg (extended while > 45 deg) |
| feet apart | hip below 0.55 m and feet > 1.0 m apart |
| thigh backward | hip below 0.55 m and a knee > 5 cm behind its hip joint |

Removed frames get `valid = 0` and cut the session. Nothing is invented or corrected. Hand-set thresholds; it misses
twist below 60 deg, drift shorter than ~5 s, wrong legs while standing, and swapped arms.

## 7. Known issues

- Stage 1-3 and the detector were re-tested after the code was made standalone; stages 5-8 have not been re-run since.
- Rokoko hip heading drifts away from the shoulders in some recordings (see the sample dataset); not corrected here.
- Self-check of the cleaner: `python test_clean_mocap.py`.

Third-party licenses: `licenses/NOTICE.md`. WiLoR (CC BY-NC-ND), MANO, and FAH `shared/detector.pt` (`WEIGHTS_LICENSE`) are non-commercial; Ultralytics YOLO weights use AGPL-3.0.
