"""
High-level front-end for the CCSDS-123.0-B-2 reference codec.

`CCSDS123` takes a numpy array or torch tensor [Z, Y, X], compresses it to a
self-contained decodable byte string, and decompresses it back. numpy-only;
torch is used only when a tensor is passed in or requested back.
"""

from __future__ import annotations

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


class CCSDS123:
    """Lossless / near-lossless CCSDS-123.0-B-2 codec."""

    def __init__(
        self,
        num_bands: int,
        height: int,
        width: int,
        dynamic_range: int = 16,
        signed: bool = False,
        lossless: bool = True,
        absolute_error_limit: int = 0,
        relative_error_limit: int = 0,
        num_prediction_bands: int = 3,
        full: bool = True,
        local_sum_type: str = "wide_neighbor",
        **kwargs,
    ) -> None:
        if lossless:
            absolute_error_limit = 0
            relative_error_limit = 0
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

    def compress(self, image) -> bytes:
        return self.codec.compress(_to_numpy(image).astype(np.int64))

    def decompress(self, blob: bytes, as_torch: bool = False):
        out = self.codec.decompress(blob)
        if as_torch:
            import torch

            return torch.from_numpy(out)
        return out

    @staticmethod
    def decompress_bytes(blob: bytes) -> np.ndarray:
        """Decode a byte string without already holding a configured codec."""
        return Ccsds123.decompress_standalone(blob)
