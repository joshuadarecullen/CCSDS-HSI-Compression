"""CCSDS-123.0-B-2 codec.

`CCSDS123` is the high-level front-end (numpy or torch, [Z, Y, X]); `Ccsds123`
and `CodecParams` are the pure-integer reference codec underneath.
"""

__version__ = "2.0.0"

from .core.reference_codec import Ccsds123, CodecParams
from .codec import CCSDS123
from .metrics import (
    calculate_psnr, calculate_mssim, calculate_spectral_angle, quality_report)

__all__ = ["CCSDS123", "Ccsds123", "CodecParams", "CCSDS123Module",
           "calculate_psnr", "calculate_mssim", "calculate_spectral_angle", "quality_report",
           "__version__"]


def __getattr__(name):
    # lazy so that importing the package never requires torch
    if name == "CCSDS123Module":
        from .torch_wrapper import CCSDS123Module
        return CCSDS123Module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
