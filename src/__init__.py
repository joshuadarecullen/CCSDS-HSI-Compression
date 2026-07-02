"""Repo-root shim so `import src.ccsds` works without installing; the real
package (and its `__version__`) lives in src/ccsds/."""

from .ccsds import (
    __version__, CCSDS123, Ccsds123, CodecParams,
    calculate_psnr, calculate_mssim, calculate_spectral_angle, quality_report)

__all__ = ["CCSDS123", "Ccsds123", "CodecParams",
           "calculate_psnr", "calculate_mssim", "calculate_spectral_angle", "quality_report"]
