"""Inference-only definitions extracted unchanged from trained Hand5 source."""
from package_paths import PACKAGE_ROOT,SOURCE_ROOT,MODEL_ROOT,SMPLH_MODEL
from pathlib import Path
import os,numpy as np,torch,torch.nn as nn,onnx,onnx2torch
from package_paths import MODEL_ROOT
ONNX=Path(os.environ.get('RTM_HAND_ARCH',MODEL_ROOT/'hand5/rtmpose_m_hand5_clip2max.onnx'))
MEAN=np.array([123.675,116.28,103.53],np.float32)
STD=np.array([58.395,57.12,57.375],np.float32)
N,BINS,SIG=256,512,5.66

def input_affine(c, S, rot, flip):
    """image px -> input px: u = A (x - c) + N/2."""
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    A = (N / S) * R.T
    if flip:
        A = np.diag([-1.0, 1.0]) @ A
    return A, R

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        g = onnx2torch.convert(onnx.load(str(ONNX)))
        feat = next(n for n in g.graph.nodes if n.op == "call_module" and isinstance(g.get_submodule(n.target), nn.Conv2d)
                    and g.get_submodule(n.target).out_channels == 21).args[0]
        out = next(n for n in g.graph.nodes if n.op == "output")
        out.args = ((out.args[0][0], out.args[0][1], feat),)
        g.graph.lint(); g.recompile()
        for name, b in [(n_, b_) for n_, b_ in g.named_buffers() if b_.is_floating_point() and b_.numel() > 1]:   # GAU scales etc. trainable too
            mod = g.get_submodule(name.rsplit(".", 1)[0]) if "." in name else g
            delattr(mod, name.rsplit(".", 1)[-1]); mod.register_parameter(name.rsplit(".", 1)[-1], nn.Parameter(b.clone()))
        self.g = g
        self.h3 = nn.Sequential(nn.Conv2d(768, 128, 1), nn.SiLU(), nn.Flatten(), nn.Linear(128 * 64, 512), nn.SiLU(), nn.Linear(512, 63))

    def forward(self, x):
        sx, sy, f = self.g(x)
        return sx, sy, self.h3(f).view(-1, 21, 3)

def decode(sx, sy):
    return torch.stack([sx.argmax(-1), sy.argmax(-1)], -1).float() / 2, torch.minimum(sx.max(-1).values, sy.max(-1).values)
