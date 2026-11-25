"""Saved production Hand5 inference/camera shape helpers, no eval loop."""
import time
import cv2,numpy as np,torch
import hand_model_runtime as T
F_W,DEPTH,PALM=37500.0,35.0,.09

def square(box, rescale=2.0):
    b = np.asarray(box, float)
    return (b[:2] + b[2:]) / 2, max(rescale * float(np.max(b[2:] - b[:2])), 16.0)

def grow(kp, x=1.5):
    lo, hi = kp.min(0), kp.max(0)
    c, s = (lo + hi) / 2, (hi - lo) * x / 2
    return np.r_[c - s, c + s]

class Runner:
    def __init__(self, ckpt, dev):
        self.m = T.Student()
        self.m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.m.eval().to(dev); self.dev = dev; self.t = 0.0; self.n = 0

    def __call__(self, ims, cs, Ss, hands):
        X, As = [], []
        for im, c, S, h in zip(ims, cs, Ss, hands):
            A, _ = T.input_affine(c, S, 0.0, h == 0)
            M = np.c_[A, T.N / 2 - A @ c]
            x = cv2.warpAffine(im, M, (T.N, T.N), flags=cv2.INTER_LINEAR)
            X.append(((x[..., ::-1].astype(np.float32) - T.MEAN) / T.STD).transpose(2, 0, 1)); As.append(A)
        kp, sc, rel = [], [], []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for k0 in range(0, len(X), 64):
                sx, sy, r3 = self.m(torch.from_numpy(np.stack(X[k0:k0 + 64])).to(self.dev))
                p, _ = T.decode(sx.float().cpu(), sy.float().cpu())
                kp.append(p.numpy()); rel.append(r3.float().cpu().numpy())
                sc.append(torch.minimum(sx.max(-1).values, sy.max(-1).values).float().cpu().numpy())
        self.t += time.perf_counter() - t0; self.n += len(X)
        kp, sc, rel = np.concatenate(kp), np.concatenate(sc), np.concatenate(rel)
        out = np.stack([(k - T.N / 2) @ np.linalg.inv(A).T + c for k, A, c in zip(kp, As, cs)])
        rel = rel.copy(); rel[np.asarray(hands) == 0, :, 0] *= -1                  # un-mirror the 3D shape
        return out, sc, rel

def to_j3d(kp, rel, wh):
    ray = np.c_[(kp[:, 0] - wh[0] / 2) / F_W, (kp[:, 1] - wh[1] / 2) / F_W, np.ones(len(kp))]
    j0 = ray * DEPTH
    return j0[:, None] + rel * PALM
