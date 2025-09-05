"""
Optimized CCSDS-123.0-B-2 Implementations

This module contains optimized versions of the CCSDS-123.0-B-2 components
designed for improved performance, including GPU acceleration, batch processing,
and vectorized operations.
"""

# Import optimized components when available
try:
    from .optimized_compressor import (
            OptimizedCCSDS123Compressor,
            create_optimized_near_lossless_compressor,
            create_optimized_lossless_compressor,
            create_optimized_block_adaptive_lossless_compressor,
            create_optimized_block_adaptive_near_lossless_compressor
     )
    from .batch_optimized_compressor import BatchOptimizedCCSDS123Compressor
    from .optimized_predictor import OptimizedSpectralPredictor
    from .optimized_quantizer import OptimizedUniformQuantizer
    from .optimized_entropy_coder import (
        OptimizedHybridEntropyCoder, 
        OptimizedBlockAdaptiveEntropyCoder,
        encode_image_block_adaptive_optimized
    )
    
    __all__ = [
        'BatchOptimizedCCSDS123Compressor',
        'OptimizedSpectralPredictor',
        'OptimizedUniformQuantizer', 
        'OptimizedHybridEntropyCoder',
        'OptimizedBlockAdaptiveEntropyCoder',
        'encode_image_block_adaptive_optimized',
        'OptimizedCCSDS123Compressor',
        'create_optimized_near_lossless_compressor',
        'create_optimized_lossless_compressor',
        'create_optimized_block_adaptive_lossless_compressor',
        'create_optimized_block_adaptive_near_lossless_compressor'
    ]
    
except ImportError as e:
    # Graceful fallback if optimized components are not available
    import warnings
    warnings.warn(f"Some optimized components could not be imported: {e}")
    __all__ = []
