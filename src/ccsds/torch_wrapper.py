"""PyTorch wrapper for the CCSDS-123.0-B-2 reference codec.

`CCSDS123Module` is an eval-only nn.Module: forward() runs a [Z, Y, X] or
[B, Z, Y, X] integer tensor through the real bitstream and returns
{reconstruction, bitstreams, bits, bpppb} on the input's device/dtype.
Non-differentiable (integer arithmetic); `straight_through=True` passes
gradients through the reconstruction unchanged. Geometry comes from the input
shape, every other `CodecParams` field from the constructor or a per-call
override; codecs are cached per shape + settings. `num_workers > 1` spreads a
batch over worker processes. CPU only, and ~100x slower without numba.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .codec import CCSDS123


def _freeze(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(e) for e in v)
    return v


def _run_item(args):
    arr, kw, decode = args
    nz, ny, nx = arr.shape
    codec = CCSDS123(num_bands=nz, height=ny, width=nx, **kw)
    blob = codec.compress(arr)
    return blob, codec.decompress(blob) if decode else None


class CCSDS123Module(torch.nn.Module):

    def __init__(self, straight_through: bool = False, num_workers: int = 0,
                 **codec_kwargs) -> None:
        super().__init__()
        self.straight_through = straight_through
        self.num_workers = num_workers
        self.codec_kwargs = codec_kwargs
        self._codecs: Dict[tuple, CCSDS123] = {}
        self._pool = None

    def codec_for(self, shape: Tuple[int, int, int], **overrides) -> CCSDS123:
        """The cached CCSDS123 for a [Z, Y, X] shape (see its .params / .codec)."""
        kw = {**self.codec_kwargs, **overrides}
        key = (tuple(shape), tuple(sorted((k, _freeze(v)) for k, v in kw.items())))
        if key not in self._codecs:
            nz, ny, nx = shape
            self._codecs[key] = CCSDS123(num_bands=nz, height=ny, width=nx, **kw)
        return self._codecs[key]

    def _run_batch(self, arr, overrides, decode):
        """[(blob, reconstruction or None)] per batch item, in order."""
        if self.num_workers > 1 and arr.shape[0] > 1:
            import concurrent.futures as cf
            if self._pool is None:
                self._pool = cf.ProcessPoolExecutor(self.num_workers)
            kw = {**self.codec_kwargs, **overrides}
            return list(self._pool.map(
                _run_item, [(arr[i], kw, decode) for i in range(arr.shape[0])]))
        codec = self.codec_for(arr.shape[1:], **overrides)
        out = []
        for i in range(arr.shape[0]):
            blob = codec.compress(arr[i])
            out.append((blob, codec.decompress(blob) if decode else None))
        return out

    @staticmethod
    def _to_batched_int64(x: torch.Tensor) -> Tuple[np.ndarray, bool]:
        """[Z,Y,X] or [B,Z,Y,X] tensor -> ([B,Z,Y,X] int64 array, was_unbatched)."""
        if x.dim() == 3:
            xb, squeeze = x.unsqueeze(0), True
        elif x.dim() == 4:
            xb, squeeze = x, False
        else:
            raise ValueError(f"expected [Z, Y, X] or [B, Z, Y, X], got shape {tuple(x.shape)}")
        arr = xb.detach().cpu()
        if arr.is_floating_point():
            rounded = arr.round()
            if not torch.equal(rounded, arr):
                raise ValueError("non-integer sample values; quantize before compressing")
            arr = rounded
        return arr.numpy().astype(np.int64), squeeze

    def compress(self, x: torch.Tensor, **overrides) -> List[bytes]:
        """Compress to a list of self-contained decodable byte strings, one per batch item."""
        arr, _ = self._to_batched_int64(x)
        return [blob for blob, _ in self._run_batch(arr, overrides, decode=False)]

    @staticmethod
    def decompress(blobs, device=None, dtype=None) -> torch.Tensor:
        """Decode byte string(s) back to a tensor (params come from each CCSDS header)."""
        if isinstance(blobs, (bytes, bytearray)):
            out = torch.from_numpy(CCSDS123.decompress_bytes(bytes(blobs)))
        else:
            out = torch.stack([torch.from_numpy(CCSDS123.decompress_bytes(b)) for b in blobs])
        return out.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, **overrides) -> Dict[str, object]:
        with torch.no_grad():
            arr, squeeze = self._to_batched_int64(x)
            items = self._run_batch(arr, overrides, decode=True)
            blobs = [b for b, _ in items]
            rec = torch.stack([torch.from_numpy(r) for _, r in items]).to(
                device=x.device, dtype=x.dtype)
            bits = torch.tensor([len(b) * 8 for b in blobs], device=x.device)
            bpppb = bits.double() / arr[0].size
        if squeeze:
            rec = rec.squeeze(0)
        if self.straight_through and x.is_floating_point() and x.requires_grad:
            rec = x + (rec - x).detach()
        return {"reconstruction": rec, "bitstreams": blobs, "bits": bits, "bpppb": bpppb}

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
