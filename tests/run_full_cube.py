"""Headline full-cube run, lossless, through the real bitstream.

Uses real Indian Pines (145x145x220) if the .mat is available locally, otherwise
a synthetic cube of the same size — so this runs from a fresh clone. With numba
installed it finishes in ~1 s each way; pure-Python takes a couple of minutes.
"""
import importlib.util
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location(
    "reference_codec", os.path.join(REPO, "src/ccsds/core/reference_codec.py"))
rc = importlib.util.module_from_spec(spec)
sys.modules["reference_codec"] = rc
spec.loader.exec_module(rc)

MAT = ("/home/joshua/Documents/phd_university/code/deepdynamichsicompression"
       "/data/indian_pines/mat/indian_pines.mat")
if os.path.exists(MAT):
    import scipy.io as sio
    img = np.transpose(sio.loadmat(MAT)["indian_pines"], (2, 0, 1)).astype(np.int64)
    label = "Indian Pines"
else:
    from synthetic_hsi import make_synthetic_hsi
    img = make_synthetic_hsi(num_bands=220, height=145, width=145)
    label = "synthetic"

Nz, Ny, Nx = img.shape
raw_bits = Nz * Ny * Nx * 16
codec = rc.Ccsds123(rc.CodecParams(num_bands=Nz, height=Ny, width=Nx,
                                   dynamic_range=16, num_prediction_bands=3, full=True))
t0 = time.time(); blob = codec.compress(img); t1 = time.time()
out = rc.Ccsds123.decompress_standalone(blob); t2 = time.time()
ok = np.array_equal(img, out)
print(f"FULL {label} {img.shape}  lossless={ok}  "
      f"ratio={raw_bits/(len(blob)*8):.3f}:1  bpppb={len(blob)*8/(Nz*Ny*Nx):.3f}  "
      f"size={len(blob)} bytes  enc={t1-t0:.1f}s dec={t2-t1:.1f}s  numba={codec.use_numba}")
assert ok, "FULL-CUBE LOSSLESS ROUND-TRIP FAILED"
print("FULL-CUBE LOSSLESS ROUND-TRIP OK")
