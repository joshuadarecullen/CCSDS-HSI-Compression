"""
CCSDS-123.0-B-2 Compressor Implementation

PyTorch implementation of the CCSDS-123.0-B-2 standard for low-complexity
lossless and near-lossless multispectral and hyperspectral image compression.

Package Structure:
    - core/: Core compression components (compressor, predictor, quantizer)
    - entropy/: Entropy coding (hybrid, rice, block-adaptive)
    - io/: I/O components (header, encoding orders)
    - optimized/: Performance-optimized implementations
    - metrics/: Quality assessment metrics
"""

# Core components
from .core import (
    CCSDS123Compressor,
    create_lossless_compressor,
    create_near_lossless_compressor,
    create_block_adaptive_lossless_compressor,
    create_block_adaptive_near_lossless_compressor,
    decompress,
    SpectralPredictor,
    NarrowLocalSumPredictor,
    UniformQuantizer,
    LosslessQuantizer,
    PeriodicErrorLimitUpdater,
    SampleRepresentativeCalculator,
    OptimizedSampleRepresentative
)

# Entropy coding
from .entropy import (
    HybridEntropyCoder,
    encode_image,
    BitWriter,
    BlockAdaptiveEntropyCoder,
    CCSDS123HybridEntropyCoder,
    RiceCoder,
    CCSDS121BlockAdaptiveEntropyCoder,
    encode_image_rice
)

# I/O components
from .io import (
    CCSDS123Header,
    PredictorMode,
    EncodingOrder,
    SampleIterator
)

# Quality metrics
from .metrics import (
    calculate_psnr,
    calculate_mssim,
    calculate_spectral_angle
)

__all__ = [
    # Main compressor interface
    'CCSDS123Compressor',
    'create_lossless_compressor',
    'create_near_lossless_compressor',
    'create_block_adaptive_lossless_compressor',
    'create_block_adaptive_near_lossless_compressor',
    'decompress',

    # Predictor
    'SpectralPredictor',
    'NarrowLocalSumPredictor',

    # Quantizer
    'UniformQuantizer',
    'LosslessQuantizer',
    'PeriodicErrorLimitUpdater',

    # Sample representative
    'SampleRepresentativeCalculator',
    'OptimizedSampleRepresentative',

    # Entropy coding
    'HybridEntropyCoder',
    'encode_image',
    'BitWriter',
    'BlockAdaptiveEntropyCoder',
    'CCSDS123HybridEntropyCoder',
    'RiceCoder',
    'CCSDS121BlockAdaptiveEntropyCoder',
    'encode_image_rice',

    # I/O
    'CCSDS123Header',
    'PredictorMode',
    'EncodingOrder',
    'SampleIterator',

    # Quality metrics
    'calculate_psnr',
    'calculate_mssim',
    'calculate_spectral_angle'
]
