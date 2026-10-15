"""
High-level front-end for the CCSDS-123.0-B-2 reference codec.

`CCSDS123` takes a numpy array or torch tensor [Z, Y, X], compresses it to a
self-contained decodable byte string, and decompresses it back. numpy-only;
torch is used only when a tensor is passed in or requested back.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .core.reference_codec import Ccsds123, CodecParams


def _to_numpy(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    try:
        import torch

        if isinstance(image, torch.Tensor):
            return image.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(image)


def _any_limit(*vals) -> bool:
    """True if any error-limit argument requests a nonzero limit."""
    for v in vals:
        if isinstance(v, np.ndarray):
            if v.any():
                return True
        elif isinstance(v, (list, tuple)):
            if any(np.any(np.asarray(e)) for e in v):
                return True
        elif v:
            return True
    return False


class CCSDS123:
    """Lossless / near-lossless CCSDS-123.0-B-2 codec.

    `lossless` defaults to None, meaning it is inferred from the error limits
    (no limits -> lossless). Passing lossless=True together with nonzero error
    limits is a contradiction and raises ValueError.
    """

    def __init__(
        self,
        num_bands: int,
        height: int,
        width: int,
        dynamic_range: int = 16,
        signed: bool = False,
        lossless: Optional[bool] = None,
        absolute_error_limit: int = 0,
        relative_error_limit: int = 0,
        num_prediction_bands: int = 3,
        full: bool = True,
        local_sum_type: str = "wide_neighbor",
        **kwargs,
    ) -> None:
        near = (_any_limit(absolute_error_limit, relative_error_limit)
                or kwargs.get("update_period_exp", -1) >= 0)
        if lossless is None:
            lossless = not near
        elif lossless and near:
            raise ValueError(
                "lossless=True conflicts with nonzero error limits or periodic "
                "error-limit updating; pass lossless=False (or omit lossless) "
                "to compress near-lossless")
        self.params = CodecParams(
            num_bands=num_bands,
            height=height,
            width=width,
            dynamic_range=dynamic_range,
            signed=signed,
            absolute_error_limit=absolute_error_limit,
            relative_error_limit=relative_error_limit,
            num_prediction_bands=num_prediction_bands,
            full=full,
            local_sum_type=local_sum_type,
            **kwargs,
        )
        self.codec = Ccsds123(self.params)

    @classmethod
    def from_image(cls, image, **kwargs) -> "CCSDS123":
        """Build a codec whose geometry matches a [Z, Y, X] image."""
        arr = _to_numpy(image)
        if arr.ndim != 3:
            raise ValueError(f"expected [Z, Y, X], got shape {arr.shape}")
        nz, ny, nx = arr.shape
        return cls(num_bands=nz, height=ny, width=nx, **kwargs)

    def compress(self, image, state=None) -> bytes:
        """Compress to a self-contained byte string. `state` (an optional dict) receives
        "reconstruction" and "representatives"; see encode()."""
        arr = _to_numpy(image)
        if np.issubdtype(arr.dtype, np.floating):
            r = np.rint(arr)
            if not np.array_equal(r, arr):
                raise ValueError("non-integer sample values; quantize before compressing")
            arr = r
        return self.codec.compress(arr, state=state)       # range-checked, then stored as plane_dtype

    def encode(self, image, as_torch: bool = False) -> dict:
        """Compress and return the decoder-side state from the same predictor pass.

        Returns {"bitstream": bytes, "reconstruction": s', "representatives": s''}, both
        [Z, Y, X] integer arrays (int32 for D <= 29, else int64). s' is what decompress() returns. s'' are the sample
        representatives (CCSDS 123.0-B-2 section 4.9): the values the predictor used as
        neighbours, pulled from the reconstruction toward the prediction by the per-band
        damping phi and offset psi at resolution theta. They equal s' when theta = 0,
        when phi = psi = 0, and in lossless mode. The encoder tracks both in its closed
        loop, so no decode is needed. as_torch=True returns the arrays as tensors.
        """
        state = {}
        blob = self.compress(image, state=state)
        out = {"bitstream": blob, **state}
        if as_torch:
            import torch

            out = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                   for k, v in out.items()}
        return out

    def decompress(self, blob: bytes, as_torch: bool = False,
                   return_representatives: bool = False):
        """Decode to the [Z, Y, X] reconstruction. With return_representatives=True
        returns (reconstruction, representatives); see encode() for what s'' is."""
        state = {} if return_representatives else None
        out = self.codec.decompress(blob, state=state)
        rep = state["representatives"] if return_representatives else None
        if as_torch:
            import torch

            out = torch.from_numpy(out)
            rep = torch.from_numpy(rep) if rep is not None else None
        return (out, rep) if return_representatives else out

    @staticmethod
    def decompress_bytes(blob: bytes, return_representatives: bool = False):
        """Decode a byte string without already holding a configured codec."""
        state = {} if return_representatives else None
        out = Ccsds123.decompress_standalone(blob, state=state)
        return (out, state["representatives"]) if return_representatives else out
