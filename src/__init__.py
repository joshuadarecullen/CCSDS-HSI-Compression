"""
CCSDS-123.0-B-2 Compressor Implementation

PyTorch implementation of the CCSDS-123.0-B-2 standard for low-complexity 
lossless and near-lossless multispectral and hyperspectral image compression.
"""

__version__ = "1.0.0"
__author__ = "CCSDS Implementation Team"

from .ccsds import CCSDS123Compressor, create_lossless_compressor, create_near_lossless_compressor
from .ccsds import decompress, calculate_psnr, calculate_mssim, calculate_spectral_angle

__all__ = [
    'CCSDS123Compressor',
    'create_lossless_compressor', 
    'create_near_lossless_compressor',
    'decompress',
    'calculate_psnr',
    'calculate_mssim', 
    'calculate_spectral_angle'
]