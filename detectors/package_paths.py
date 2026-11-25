"""Roots for the detector modules of this folder (weights: final_code/models)."""
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE_ROOT / "models"
SMPLH_MODEL = None                                   # not used by the detectors (imported by name only)


def activate():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
