# CCSDS-123.0-B-2 Speed Comparison Tests

This directory contains comprehensive speed comparison tests for all CCSDS-123.0-B-2 compression implementations, including the new GPU-optimized versions.

## Test Files Overview

### 1. `test_speed_comparison.py` - Comprehensive Benchmark Suite
**Purpose**: Complete performance analysis across all compressor variants and image sizes.

**Features**:
- Tests original sample-by-sample implementation
- Tests all CPU optimization modes (full, causal, streaming)
- Tests all GPU optimization modes (batch, streaming, mixed precision)
- Tests batch-optimized versions
- Multiple image sizes (small to extra-large)
- Statistical analysis with multiple runs
- Detailed CSV output
- Memory usage tracking for GPU tests

**Usage**:
```bash
python test_speed_comparison.py
```

**Expected Output**:
```
COMPREHENSIVE SPEED COMPARISON RESULTS
=====================================
Small (10×32×32):
Method                         Time (s)   Speedup    Throughput      Device   Ratio
-------------------------------------------------------------------------------
GPU + Mixed Precision          0.015      67.3x      672.0 MSamp/s   GPU      2.45:1
GPU Optimized                  0.018      56.1x      560.0 MSamp/s   GPU      2.45:1
CPU Full Vectorized            0.095      10.6x      106.3 MSamp/s   CPU      2.45:1
...
```

### 2. `test_quick_speed.py` - Development Speed Test
**Purpose**: Quick verification of main optimization modes for development.

**Features**:
- Single image size testing
- Core optimization modes only
- Faster execution
- Immediate feedback
- Good for iterative development

**Usage**:
```bash
python test_quick_speed.py
```

### 3. `test_minimal_speed.py` - Basic Functionality Test
**Purpose**: Minimal test for basic verification and CI/CD.

**Features**:
- Very small test images
- Basic CPU/GPU comparison
- Quick execution (<10 seconds)
- Error handling and fallback
- Perfect for automated testing

**Usage**:
```bash
python test_minimal_speed.py
```

### 4. `gpu_optimized_example.py` - GPU Feature Demonstration
**Purpose**: Demonstrate GPU-specific features and capabilities.

**Features**:
- GPU vs CPU benchmarking
- Mixed precision demonstration
- Memory efficiency testing
- Feature showcase with examples

## Compressor Variants Tested

### Original Implementation
- **File**: `ccsds_compressor.py`
- **Mode**: Sample-by-sample processing
- **Expected Performance**: Baseline (1x)
- **Complexity**: O(Z×Y×X × operations_per_sample)

### CPU Optimized Implementations
- **File**: `optimized_compressor.py`
- **Modes**: 
  - `full`: Maximum vectorization
  - `causal`: Standards-compliant optimization
  - `streaming`: Memory-efficient for large images
- **Expected Performance**: 10-100x speedup
- **Complexity**: O(Z×Y×X) total operations

### GPU Optimized Implementations (NEW)
- **File**: `optimized_compressor.py` (with GPU enhancements)
- **Modes**:
  - `GPU Batch Optimized`: Parallel band processing
  - `GPU Streaming`: Memory-efficient GPU processing
  - `GPU + Mixed Precision`: FP16 acceleration
- **Expected Performance**: 5-50x additional speedup over CPU
- **Requirements**: CUDA-compatible GPU

### Batch Optimized Implementation
- **File**: `batch_optimized_compressor.py`
- **Mode**: Process multiple images simultaneously
- **Expected Performance**: High throughput for batch scenarios
- **Use Case**: Batch processing multiple images

## Performance Expectations

Based on typical hardware configurations:

### CPU Performance (Intel i7/AMD Ryzen)
| Method | Image Size | Expected Throughput |
|--------|------------|-------------------|
| Original | 50×64×64 | ~5-10 KSamp/s |
| CPU Full Optimized | 50×64×64 | ~100-500 KSamp/s |
| CPU Causal Optimized | 50×64×64 | ~50-200 KSamp/s |

### GPU Performance (Modern CUDA GPU)
| Method | Image Size | Expected Throughput |
|--------|------------|-------------------|
| GPU Batch Optimized | 50×64×64 | ~1-5 MSamp/s |
| GPU + Mixed Precision | 50×64×64 | ~2-10 MSamp/s |
| GPU Streaming | 100×256×256 | ~5-20 MSamp/s |

### Memory Usage
- **CPU**: ~2-3x image size
- **GPU Full Mode**: ~3-4x image size
- **GPU Streaming**: Configurable (chunk-based)

## Running Speed Tests

### Prerequisites
```bash
pip install torch torchvision pandas numpy
```

For GPU tests:
```bash
# Ensure CUDA-compatible PyTorch is installed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Quick Start
1. **Basic functionality check**:
   ```bash
   python test_minimal_speed.py
   ```

2. **Development testing**:
   ```bash
   python test_quick_speed.py
   ```

3. **Full benchmark suite**:
   ```bash
   python test_speed_comparison.py
   ```

### Expected Speedup Factors

| Optimization | vs Original | vs CPU Optimized |
|--------------|-------------|------------------|
| CPU Full Vectorized | 10-100x | 1x (baseline) |
| CPU Causal | 10-30x | 0.3-0.8x |
| GPU Batch | 50-500x | 5-20x |
| GPU + Mixed Precision | 100-1000x | 10-50x |

## Interpreting Results

### Key Metrics
- **Time**: Compression time in seconds
- **Throughput**: Samples processed per second
- **Speedup**: Relative to slowest method
- **Compression Ratio**: Quality metric (higher = better compression)
- **Memory**: Peak GPU memory usage (GPU tests only)

### What to Look For
1. **Consistent compression ratios** across methods (validates correctness)
2. **Significant speedup** with GPU optimizations
3. **Memory efficiency** for large images
4. **Stable performance** across multiple runs

### Troubleshooting
- **GPU tests fail**: Check CUDA installation and compatibility
- **Very slow performance**: Check image size and available memory
- **Inconsistent ratios**: May indicate algorithm differences (expected for some modes)

## Customizing Tests

### Modify Test Parameters
```python
# In test files, adjust these parameters:
num_runs = 5          # Number of benchmark runs
warmup_runs = 2       # Number of warmup runs
image_sizes = [        # Test image dimensions
    (10, 32, 32),     # Small
    (50, 64, 64),     # Medium
    (100, 128, 128),  # Large
]
```

### Add Custom Configurations
```python
# Example: Add custom GPU batch size test
test_configs.append((
    "GPU Large Batch",
    lambda: create_optimized_lossless_compressor(
        num_bands=num_bands,
        optimization_mode='full',
        device='cuda',
        gpu_batch_size=32  # Custom batch size
    ),
    True  # Needs warmup
))
```

## Integration with Existing Code

These speed tests are designed to work alongside existing CCSDS implementations:

```python
# Use any compressor variant in your code
from optimized_compressor import create_optimized_lossless_compressor

# Auto-detect best available option
compressor = create_optimized_lossless_compressor(
    num_bands=50,
    device='auto',              # Auto GPU/CPU selection
    use_mixed_precision=True,   # Enable if supported
    optimization_mode='full'    # Fastest mode
)

# Compress your data
result = compressor(your_image_tensor)
```

The speed tests help you choose the optimal configuration for your specific hardware and use case.