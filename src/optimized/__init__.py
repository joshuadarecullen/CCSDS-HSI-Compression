"""
Optimized CCSDS-123.0-B-2 Implementations

This module contains optimized versions of the CCSDS-123.0-B-2 components
designed for improved performance, including GPU acceleration, batch processing,
and vectorized operations.
"""

# Import optimized components when available
try:
    from .optimized_compressor import OptimizedCCSDS123Compressor
    from .batch_optimized_compressor import BatchOptimizedCCSDS123
    from .optimized_predictor import OptimizedSpectralPredictor
    from .optimized_quantizer import OptimizedUniformQuantizer
    from .optimized_entropy_coder import OptimizedHybridEntropyCoder
    
    __all__ = [
        'OptimizedCCSDS123Compressor',
        'BatchOptimizedCCSDS123',
        'OptimizedSpectralPredictor',
        'OptimizedUniformQuantizer', 
        'OptimizedHybridEntropyCoder'
    ]
    
except ImportError as e:
    # Graceful fallback if optimized components are not available
    import warnings
    warnings.warn(f"Some optimized components could not be imported: {e}")
    __all__ = []