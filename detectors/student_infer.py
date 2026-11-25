"""Production crop helper; no standalone experiment entry point."""
import numpy as np
BOX_X=1.5

def grow_box(b, x):
    b = np.asarray(b, float)
    c, s = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) * x / 2
    return np.r_[c - s, c + s]
