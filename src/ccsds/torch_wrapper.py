"""PyTorch wrapper for the CCSDS-123.0-B-2 reference codec.

`CCSDS123Module` is an eval-only nn.Module: forward() runs a [Z, Y, X] or
[B, Z, Y, X] integer tensor through the real bitstream and returns
{reconstruction, bitstreams, bits, bpppb} on the input's device/dtype.
Non-differentiable (integer arithmetic); `straight_through=True` passes
gradients through the reconstruction unchanged. Geometry comes from the input
shape (codecs cached per shape); other kwargs are forwarded to `CCSDS123`.
CPU only, and ~100x slower without numba (check `Ccsds123.use_numba`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .codec import CCSDS123


class CCSDS123Module(torch.nn.Module):

    def __init__(self, straight_through: bool = False, **codec_kwargs) -> None:
        super().__init__()
        self.straight_through = straight_through
        self.codec_kwargs = codec_kwargs
        self._codecs: Dict[Tuple[int, int, int], CCSDS123] = {}

    def _codec_for(self, shape: Tuple[int, int, int]) -> CCSDS123:
        if shape not in self._codecs:
            nz, ny, nx = shape
            self._codecs[shape] = CCSDS123(num_bands=nz, height=ny, width=nx,
                                           **self.codec_kwargs)
        return self._codecs[shape]

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

    def compress(self, x: torch.Tensor) -> List[bytes]:
        """Compress to a list of self-contained decodable byte strings, one per batch item."""
        arr, _ = self._to_batched_int64(x)
        codec = self._codec_for(arr.shape[1:])
        return [codec.compress(arr[i]) for i in range(arr.shape[0])]

    @staticmethod
    def decompress(blobs, device=None, dtype=None) -> torch.Tensor:
        """Decode byte string(s) back to a tensor (params come from each CCSDS header)."""
        if isinstance(blobs, (bytes, bytearray)):
            out = torch.from_numpy(CCSDS123.decompress_bytes(bytes(blobs)))
        else:
            out = torch.stack([torch.from_numpy(CCSDS123.decompress_bytes(b)) for b in blobs])
        return out.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        with torch.no_grad():
            arr, squeeze = self._to_batched_int64(x)
            codec = self._codec_for(arr.shape[1:])
            blobs, recs = [], []
            for i in range(arr.shape[0]):
                blob = codec.compress(arr[i])
                blobs.append(blob)
                recs.append(torch.from_numpy(codec.decompress(blob)))
            rec = torch.stack(recs).to(device=x.device, dtype=x.dtype)
            bits = torch.tensor([len(b) * 8 for b in blobs], device=x.device)
            bpppb = bits.double() / arr[0].size
        if squeeze:
            rec = rec.squeeze(0)
        if self.straight_through and x.is_floating_point() and x.requires_grad:
            rec = x + (rec - x).detach()
        return {"reconstruction": rec, "bitstreams": blobs, "bits": bits, "bpppb": bpppb}
