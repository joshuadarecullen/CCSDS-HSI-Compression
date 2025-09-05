"""
Core CCSDS-123.0-B-2 Implementation

This module contains the core implementation of the CCSDS-123.0-B-2 standard
including the main compressor, predictor, quantizer, entropy coder, and 
sample representative components.
"""

from .ccsds_compressor import (
    CCSDS123Compressor,
    create_lossless_compressor,
    create_near_lossless_compressor,
    create_block_adaptive_lossless_compressor,
    create_block_adaptive_near_lossless_compressor,
    decompress
)

from .metrics import (
    calculate_psnr,
    calculate_mssim,
    calculate_spectral_angle
)

from .predictor import SpectralPredictor, NarrowLocalSumPredictor
from .quantizer import UniformQuantizer, LosslessQuantizer, PeriodicErrorLimitUpdater
from .entropy_coder import HybridEntropyCoder, encode_image, BitWriter, BlockAdaptiveEntropyCoder
from .sample_representative import SampleRepresentativeCalculator, OptimizedSampleRepresentative

__all__ = [
    # Main compressor interface
    'CCSDS123Compressor',
    'create_lossless_compressor',
    'create_near_lossless_compressor',
    'create_block_adaptive_lossless_compressor',
    'create_block_adaptive_near_lossless_compressor',
    'decompress',
    
    # Quality assessment functions
    'calculate_psnr',
    'calculate_mssim',
    'calculate_spectral_angle',
    
    # Core components
    'SpectralPredictor',
    'NarrowLocalSumPredictor',
    'UniformQuantizer',
    'LosslessQuantizer',
    'PeriodicErrorLimitUpdater',
    'HybridEntropyCoder',
    'BlockAdaptiveEntropyCoder',
    'encode_image',
    'BitWriter',
    'SampleRepresentativeCalculator',
    'OptimizedSampleRepresentative'
]
