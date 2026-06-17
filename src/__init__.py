"""CCSDS-123.0-B-2 lossless / near-lossless hyperspectral image compression."""

__version__ = "2.0.0"

from .ccsds import CCSDS123, Ccsds123, CodecParams

__all__ = ["CCSDS123", "Ccsds123", "CodecParams"]
