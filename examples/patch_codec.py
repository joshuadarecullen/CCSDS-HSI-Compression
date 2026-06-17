"""Compress a dataset of [Z, Y, X] patches with the CCSDS-123.0-B-2 codec.

Each patch becomes one self-contained bitstream. A single codec is reused across
patches, so numba's JIT compiles only once. Parallelism is across patches (they are
independent); within a patch the predictor is a causal recurrence and cannot batch.
"""
import os
import sys

import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.ccsds import Ccsds123, CodecParams          # running from the repo
except ImportError:
    from ccsds import Ccsds123, CodecParams              # pip-installed package


def _codec(z, y, x, dynamic_range, params):
    return Ccsds123(CodecParams(num_bands=z, height=y, width=x,
                                dynamic_range=dynamic_range, **params))


def compress_patches(patches, dynamic_range=16, **params):
    """patches: (N, Z, Y, X) array or an iterable of (Z, Y, X) integer patches.
    Returns list[bytes] -- one compressed blob per patch. Extra kwargs go to
    CodecParams, e.g. entropy_coder="hybrid", absolute_error_limit=4."""
    patches = list(patches)
    z, y, x = patches[0].shape
    codec = _codec(z, y, x, dynamic_range, params)          # built once; JIT warms on patch 0
    return [codec.compress(np.ascontiguousarray(p, np.int64)) for p in patches]


def decompress_patches(blobs):
    """Inverse of compress_patches. Returns an (N, Z, Y, X) int64 array. Each blob's
    CCSDS header is self-describing, so no parameters are needed here."""
    return np.stack([Ccsds123.decompress_standalone(b) for b in blobs])


def compression_ratio(patches, blobs, dynamic_range=16):
    """Overall ratio = total raw bits / total compressed bits."""
    raw = sum(int(np.asarray(p).size) * dynamic_range for p in patches)
    return raw / sum(len(b) * 8 for b in blobs)


# ---- optional: spread patches across CPU cores (each patch is independent) ----
_W = {}


def _init(z, y, x, dynamic_range, params):
    _W["codec"] = _codec(z, y, x, dynamic_range, params)


def _compress_one(patch):
    return _W["codec"].compress(np.ascontiguousarray(patch, np.int64))


def compress_patches_parallel(patches, workers=None, dynamic_range=16, **params):
    """Same result as compress_patches, spread across processes (re-JITs once per
    worker). Worth it only when there are many patches."""
    patches = list(patches)
    z, y, x = patches[0].shape
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(z, y, x, dynamic_range, params)) as ex:
        return list(ex.map(_compress_one, patches, chunksize=8))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1 << 16, (6, 8, 32, 32)).astype(np.int64)    # 6 patches of (8, 32, 32)
    blobs = compress_patches(data, entropy_coder="hybrid")
    assert np.array_equal(decompress_patches(blobs), data), "round-trip failed"
    assert compress_patches_parallel(data, workers=2, entropy_coder="hybrid") == blobs, \
        "parallel output differs from serial"
    print(f"patch_codec self-test OK: {len(data)} patches, "
          f"{compression_ratio(data, blobs):.3f}:1 lossless, parallel == serial")
