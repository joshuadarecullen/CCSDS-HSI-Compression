# CCSDS-123.0-B-2 Compressor Implementation

PyTorch implementation of the CCSDS-123.0-B-2 standard for low-complexity lossless and near-lossless multispectral and hyperspectral image compression.

## Overview

This implementation includes all major components of Issue 2 of the CCSDS-123.0-B standard:

- **Adaptive Linear Predictor**: Predicts sample values using nearby samples in 3D neighborhood
- **Uniform Quantizer**: Closed-loop scalar quantization with user-specified error bounds
- **Sample Representative Calculator**: Computes representatives used in prediction calculations
- **Hybrid Entropy Coder**: Combines GPO2 codes with 16 variable-to-variable length low-entropy codes

## Features

### New in Issue 2
- Near-lossless compression with absolute and/or relative error limits
- Sample representative parameters (φ, ψ, Θ) for improved compression performance
- Hybrid entropy coding for better low-entropy data compression
- Periodic error limit updating for adaptive rate control
- Support for up to 32-bit signed/unsigned integer samples
- Narrow local sums option for reduced prediction complexity

### Maintained from Issue 1
- Lossless compression capability
- Low computational complexity
- Single-pass compression and decompression
- Backwards compatibility with Issue 1

## Usage

### Basic Lossless Compression

```python
import torch
from src.ccsds import create_lossless_compressor, decompress

# Create test image [Z, Y, X]
image = torch.randn(10, 64, 64) * 100  # 10 bands, 64x64 pixels

# Create lossless compressor
compressor = create_lossless_compressor(num_bands=10, dynamic_range=16)

# Compress image - returns results dictionary with all compression outputs
results = compressor(image)

print(f"Compression ratio: {results['compression_ratio']:.2f}:1")
print(f"Compressed size: {results['compressed_size']} bits")
print(f"Original size: {results['original_size']} bits")

# Verify lossless compression
reconstructed = results['reconstructed_samples']
print(f"Lossless: max error = {torch.max(torch.abs(image - reconstructed))}")

# For full decompression from bitstream (if available)
if 'compressed_bitstream' in results:
    decompressed = decompress(results)
    print(f"Decompressed max error = {torch.max(torch.abs(image - decompressed))}")
```

### Near-Lossless Compression with Quality Assessment

```python
from src.ccsds import create_near_lossless_compressor, calculate_psnr, calculate_mssim, calculate_spectral_angle

# Set absolute error limits per band
error_limits = torch.tensor([2.0, 2.0, 4.0, 4.0, 8.0])  # Varying by band

# Create near-lossless compressor
compressor = create_near_lossless_compressor(
    num_bands=5,
    dynamic_range=16,
    absolute_error_limits=error_limits
)

# Compress image - returns comprehensive results
results = compressor(image[:5])  # Use first 5 bands
reconstructed = results['reconstructed_samples']

print(f"Compression ratio: {results['compression_ratio']:.2f}:1")
print(f"Max error: {torch.max(torch.abs(image[:5] - reconstructed))}")
print(f"Compression time: {results['compression_time']:.4f}s")

# Calculate quality metrics with callbacks
def quality_callback(metric_name, value, additional_data=None):
    print(f"{metric_name}: {value:.4f}")
    
psnr = calculate_psnr(image[:5], reconstructed, 16, 
                      callback=lambda p, m: quality_callback("PSNR (dB)", p))
mssim = calculate_mssim(image[:5], reconstructed,
                       callback=lambda m, s: quality_callback("MSSIM", m))
sam = calculate_spectral_angle(image[:5], reconstructed,
                              callback=lambda s, m: quality_callback("SAM (radians)", s))
```

### Advanced Configuration

```python
from src.ccsds import CCSDS123Compressor

# Create compressor with custom parameters
compressor = CCSDS123Compressor(
    num_bands=10,
    dynamic_range=16,
    use_narrow_local_sums=True,  # Reduced complexity
    lossless=False
)

# Configure compression parameters
compressor.set_compression_parameters(
    absolute_error_limits=torch.ones(10) * 3,
    sample_rep_phi=torch.ones(10) * 2,     # φ parameters
    sample_rep_psi=torch.ones(10) * 4,     # ψ parameters  
    sample_rep_theta=6.0,                   # Θ parameter
    periodic_error_update=True,
    update_interval=500
)

# Compress with configured parameters
results = compressor(image)
print(f"Advanced compression ratio: {results['compression_ratio']:.2f}:1")
print(f"Throughput: {results.get('throughput_samples_per_sec', 0):.0f} samples/sec")
```

## Command Line Interface

The package includes command-line tools for easy compression and benchmarking:

### Compression Tool

```bash
# Lossless compression
ccsds-compress input.npy output.pt --lossless

# Near-lossless compression
ccsds-compress input.npy output.pt --error-limit 2.0

# With verbose output
ccsds-compress input.npy output.pt --lossless --verbose
```

### Benchmarking Tool

```bash
# Run standard benchmark
ccsds-benchmark

# Custom benchmark parameters
ccsds-benchmark --bands 20 --height 128 --width 128 --iterations 10

# Test specific error limits
ccsds-benchmark --error-limits 1 2 4 8
```

## Project Structure

```
CCSDS-HSI-Compression/
├── src/
│   ├── ccsds/                         # Core CCSDS-123.0-B-2 implementation  
│   │   ├── __init__.py                # Package interface with main exports
│   │   ├── ccsds_compressor.py        # Main CCSDS123Compressor class
│   │   ├── predictor.py               # SpectralPredictor & NarrowLocalSumPredictor
│   │   ├── quantizer.py               # UniformQuantizer & LosslessQuantizer
│   │   ├── entropy_coder.py           # HybridEntropyCoder & encode_image
│   │   ├── sample_representative.py   # SampleRepresentativeCalculator
│   │   ├── cli.py                     # Command-line interface tools
│   │   └── metrics/                   # Quality assessment metrics
│   │       ├── __init__.py            # Metrics package interface
│   │       └── quality_metrics.py     # PSNR, MSSIM, SAM implementations
│   └── optimized/                     # Performance-optimized implementations
│       ├── __init__.py                # Optimized package interface
│       ├── optimized_compressor.py    # OptimizedCCSDS123Compressor
│       ├── batch_optimized_compressor.py  # BatchOptimizedCCSDS123Compressor
│       ├── optimized_entropy_coder.py     # OptimizedHybridEntropyCoder
│       └── ...                        # Other optimized components
├── tests/
│   ├── unit/                          # Unit tests for all components
│   │   ├── test_ccsds.py             # Main CCSDS compressor tests
│   │   ├── test_lossless.py          # Lossless compression tests
│   │   ├── test_optimized_basic.py   # Basic optimized compressor tests
│   │   ├── test_performance.py       # Performance comparison tests
│   │   └── test_*.py                 # Additional unit tests
│   ├── performance/                   # Performance benchmarks
│   │   ├── test_speed_comparison.py  # Comprehensive speed benchmarks
│   │   └── test_quick_speed.py       # Quick performance tests
│   └── utils.py                      # Test utilities
├── examples/                         # Usage examples (Jupyter notebooks)
├── environment.yaml                  # Conda environment specification
├── pyproject.toml                   # Modern Python packaging
└── README.md                        # This file
```

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone <repository-url>
cd ccsds

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,gpu,benchmarks]"
```

### Using pip (when published)

```bash
pip install ccsds-123-compressor
```

## Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/performance/   # Performance tests

# Run with coverage
pytest --cov=src tests/
```

Run individual test files:

```bash
python tests/unit/test_ccsds.py                # Main CCSDS compressor tests
python tests/unit/test_lossless.py             # Focused lossless compression tests
python tests/unit/test_optimized_basic.py      # Basic optimized compressor tests
python tests/performance/test_speed_comparison.py  # Comprehensive speed benchmarks
python tests/performance/test_quick_speed.py   # Quick performance tests
```

The test suite includes:
- **Core functionality**: Lossless and near-lossless compression verification
- **Component testing**: Individual predictor, quantizer, entropy coder tests
- **Quality metrics**: PSNR, MSSIM, and Spectral Angle Mapper validation
- **Optimization testing**: GPU-accelerated and batch-optimized compressor tests
- **Performance analysis**: Speed comparisons and throughput measurements
- **Device compatibility**: CPU/CUDA device handling and tensor management
- **Error handling**: Robust fallback mechanisms and device mismatch resolution
- **Standards compliance**: CCSDS-123.0-B-2 specification adherence

## Implementation Notes

### Predictor
- Uses adaptive linear prediction with 3D neighborhood
- Supports both full and narrow local sum modes
- Weight adaptation based on local image statistics
- Handles boundary conditions appropriately

### Quantizer
- Implements closed-loop scalar quantization
- Supports absolute error limits: `m_z(t) = a_z`
- Supports relative error limits: `m_z(t) = ⌊r_z|ŝ_z(t)|/2^D⌋`
- Combined mode: `m_z(t) = min(a_z, ⌊r_z|ŝ_z(t)|/2^D⌋)`
- Step size: `2*m_z(t) + 1`

### Sample Representatives
- Lies between quantizer bin center and prediction
- Controlled by user parameters φ_z, ψ_z, Θ
- Can improve compression performance and reconstruction quality
- Optimized vectorized computation

### Entropy Coder
- Hybrid approach with high/low entropy classification
- 16 variable-to-variable length low-entropy codes (Table 1 from paper)
- GPO2-equivalent codes for high-entropy samples
- Suffix-free codes with reverse-order decoding
- Escape symbol handling for unlikely values

### Standards Compliance
- Implements key algorithms from CCSDS-123.0-B-2 specification
- Maintains backwards compatibility with Issue 1
- Supports 32-bit dynamic range (vs 16-bit in Issue 1)
- Includes periodic error limit updating capability

## Current Implementation Status

### Core CCSDS-123.0-B-2 Features ✅
- **Complete compression pipeline**: Prediction → Quantization → Sample Representatives → Entropy Coding
- **Lossless compression**: Verified zero-error reconstruction with `create_lossless_compressor()`
- **Near-lossless compression**: User-specified error limits with `create_near_lossless_compressor()`
- **Adaptive linear predictor**: Full 3D spatial-spectral neighborhood prediction
- **Hybrid entropy coder**: 16 low-entropy codes + GPO2 high-entropy codes
- **Sample representatives**: Configurable φ, ψ, Θ parameters for improved compression
- **Quality metrics**: PSNR, MSSIM, and Spectral Angle Mapper with callback support

### Performance Optimizations ✅
- **GPU acceleration**: CUDA-optimized compressor with automatic device management
- **Vectorized operations**: Batch processing for improved throughput
- **Memory efficiency**: Streaming entropy coding for large images
- **Device compatibility**: Automatic CPU/CUDA tensor placement and error handling
- **Multiple optimization modes**: Full, causal, and streaming processing options

### Recent Improvements
- **Fixed import errors** in all test files with proper `src.ccsds` import paths
- **Resolved device mismatch issues** in optimized entropy coder with automatic device detection
- **Enhanced test coverage** with comprehensive lossless, optimized, and performance tests
- **Organized metrics module** separated quality assessment functions (PSNR, MSSIM, SAM) into dedicated `src.ccsds.metrics` package
- **Type annotations** throughout the codebase for better development experience
- **Mathematical documentation** explaining the theoretical foundation of each component

### Quality Metrics

The package includes comprehensive quality assessment metrics for evaluating compression performance. These metrics are organized in the `src.ccsds.metrics` module and can be imported directly from the main package:

```python
# Import metrics from main package (recommended)
from src.ccsds import calculate_psnr, calculate_mssim, calculate_spectral_angle

# Or import directly from metrics module
from src.ccsds.metrics import calculate_psnr, calculate_mssim, calculate_spectral_angle

# Calculate PSNR with callback
def psnr_callback(psnr_value, mse_value):
    print(f"PSNR: {psnr_value:.2f} dB (MSE: {mse_value:.6f})")
    
psnr = calculate_psnr(original, reconstructed, dynamic_range=16, callback=psnr_callback)

# Calculate MSSIM with per-band analysis
def mssim_callback(mssim_value, ssim_per_band):
    print(f"MSSIM: {mssim_value:.4f}")
    print(f"Per-band SSIM: {[f'{s:.3f}' for s in ssim_per_band]}")
    
mssim = calculate_mssim(original, reconstructed, callback=mssim_callback)

# Calculate Spectral Angle with spatial analysis
def sam_callback(mean_sam, sam_map):
    print(f"Mean SAM: {mean_sam:.6f} radians ({mean_sam*180/3.14159:.3f} degrees)")
    print(f"SAM std: {np.std(sam_map):.6f} radians")
    
sam = calculate_spectral_angle(original, reconstructed, callback=sam_callback)
```

## Limitations

This is a research implementation with some simplifications:

1. **Low-entropy code implementation**: While the exact code specifications from Table 1 are used, the actual encoding/decoding logic uses algorithmically generated codes rather than pre-computed lookup tables
2. **Weight adaptation**: Uses simplified weight update rules in the predictor rather than the full adaptive strategy from the standard
3. **Error limit updates**: Implements basic periodic updating (`PeriodicErrorLimitUpdater`) rather than sophisticated rate control algorithms
4. **Full bitstream decoding**: The `decompress()` function currently relies on intermediate data from compression results rather than full entropy decoding from bitstream bytes
5. **Supplementary information**: Reserved bits and supplementary information tables are not supported
6. **Exact standard compliance**: Some algorithmic details may differ from the exact CCSDS specification to prioritize clarity and educational value

## Performance

Typical compression results on synthetic test images:

- **Lossless**: 2-4:1 compression ratio
- **Near-lossless** (error=2): 3-6:1 compression ratio  
- **Near-lossless** (error=4): 4-8:1 compression ratio

Performance varies significantly based on image content, spectral correlation, and noise levels.

## References

1. *Lossless Multispectral & Hyperspectral Image Compression*, CCSDS 123.0-B-1, May 2012
2. *The New CCSDS Standard for Low-Complexity Lossless and Near-Lossless Multispectral and Hyperspectral Image Compression*, Kiely et al.
3. *Fast Lossless Compression of Multispectral Images*, Klimesh, 2005

## License

This implementation is for research and educational purposes. The CCSDS standard specifications are publicly available from the Consultative Committee for Space Data Systems.