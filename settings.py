"""Machine-specific paths: environment variable > .env next to this file > default.  See .env.example, check_setup.py."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ENV = {}
if (HERE / ".env").exists():
    for _l in (HERE / ".env").read_text().splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.split("=", 1)
            _ENV[_k.strip()] = os.path.expandvars(_v.strip().strip('"'))


def get(k, default):
    return os.environ.get(k) or _ENV.get(k) or str(default)


RAW_ROOT = get("RAW_ROOT", HERE / "data/raw")
OUT_ROOT = get("OUT_ROOT", HERE / "data/sync")
FFMPEG_DIR = get("FFMPEG_DIR", "")
FFMPEG = f"{FFMPEG_DIR}/ffmpeg" if FFMPEG_DIR else "ffmpeg"
MODELS = HERE / "models"
SAM2_FT = get("SAM2_FT", MODELS / "sam2/sam2bp_ft_det.pth")
SAM2_BASE = get("SAM2_BASE", MODELS / "sam2/sam2.1_hiera_base_plus.pt")
SAM2_LARGE = get("SAM2_LARGE", MODELS / "sam2/sam2.1_hiera_large.pt")
SAM3 = get("SAM3", MODELS / "sam3/sam3.pt")
WILOR = get("WILOR", HERE / "third_party/WiLoR")
FAH = get("FAH", HERE / "third_party/FAH")
FAH_DET = get("FAH_DET", MODELS / "fah/shared/detector.pt")
