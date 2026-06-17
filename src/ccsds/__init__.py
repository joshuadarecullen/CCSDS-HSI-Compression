"""CCSDS-123.0-B-2 codec.

`CCSDS123` is the high-level front-end (numpy or torch, [Z, Y, X]); `Ccsds123`
and `CodecParams` are the pure-integer reference codec underneath.
"""

from .core.reference_codec import Ccsds123, CodecParams
from .codec import CCSDS123

__all__ = ["CCSDS123", "Ccsds123", "CodecParams"]
