"""
Core CCSDS-123.0-B-2 Components

This module contains the core compression components:
- CCSDS123Compressor: Main compressor class
- SpectralPredictor: Adaptive linear predictor
- UniformQuantizer/LosslessQuantizer: Quantization
- SampleRepresentativeCalculator: Sample representative computation
"""

from .compressor import (
    CCSDS123Compressor,
    create_lossless_compressor,
    create_near_lossless_compressor,
    create_block_adaptive_lossless_compressor,
    create_block_adaptive_near_lossless_compressor,
    decompress
)

from .predictor import SpectralPredictor, NarrowLocalSumPredictor

from .quantizer import (
    UniformQuantizer,
    LosslessQuantizer,
    PeriodicErrorLimitUpdater
)

from .sample_representative import (
    SampleRepresentativeCalculator,
    OptimizedSampleRepresentative
)

__all__ = [
    'CCSDS123Compressor',
    'create_lossless_compressor',
    'create_near_lossless_compressor',
    'create_block_adaptive_lossless_compressor',
    'create_block_adaptive_near_lossless_compressor',
    'decompress',
    'SpectralPredictor',
    'NarrowLocalSumPredictor',
    'UniformQuantizer',
    'LosslessQuantizer',
    'PeriodicErrorLimitUpdater',
    'SampleRepresentativeCalculator',
    'OptimizedSampleRepresentative'
]
