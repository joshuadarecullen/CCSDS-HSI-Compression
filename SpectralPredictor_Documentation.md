# CCSDS-123.0-B-2 SpectralPredictor Class Documentation

## Overview

The `SpectralPredictor` class implements the adaptive linear prediction component of the CCSDS-123.0-B-2 standard. It provides comprehensive prediction capabilities with full CCSDS compliance and multiple configuration options.

**Location**: `src/ccsds/predictor.py`

## Class Initialization

```python
SpectralPredictor(
    num_bands: int,
    dynamic_range: int = 16,
    prediction_bands: int = None,
    local_sum_type: str = 'neighbor-oriented',
    prediction_mode: str = 'full'
)
```

## Configuration Parameters

### 1. **Core Parameters**

| Parameter | Type | Default | Description | Valid Values |
|-----------|------|---------|-------------|--------------|
| `num_bands` | int | Required | Number of spectral bands in the hyperspectral image | > 0 |
| `dynamic_range` | int | 16 | Bit depth of input samples | 8, 16, 32 |
| `prediction_bands` | int | `min(num_bands-1, 15)` | Maximum bands used for prediction | 0 to min(num_bands-1, 15) |

### 2. **Local Sum Configuration**

| Parameter | Type | Default | Description | CCSDS Section |
|-----------|------|---------|-------------|---------------|
| `local_sum_type` | str | 'neighbor-oriented' | Local sum computation method | Section 4.4 |

#### Local Sum Types:

**A. Neighbor-Oriented (Default)**
- **Description**: Uses neighboring spatial samples within the same band
- **CCSDS Equations**: (20), (21) - Wide local sums
- **Use Case**: General purpose, good spatial prediction
- **Hardware**: Standard implementation

**B. Column-Oriented** 
- **Description**: Uses samples from the same column in previous rows
- **CCSDS Equations**: (22), (23) - Narrow local sums  
- **Use Case**: Hardware pipelining optimization
- **Hardware**: Reduces memory dependencies for pipeline efficiency

### 3. **Prediction Mode Configuration**

| Parameter | Type | Default | Description | CCSDS Section |
|-----------|------|---------|-------------|---------------|
| `prediction_mode` | str | 'full' | Prediction algorithm complexity | Section 4.3 |

#### Prediction Modes:

**A. Full Mode (Default)**
- **Description**: Complete adaptive linear prediction algorithm
- **Features**: 
  - Full weight adaptation using equations (51)-(54)
  - All prediction components: central + directional differences
  - Maximum prediction accuracy
- **Computational Cost**: High
- **Use Case**: Maximum compression efficiency

**B. Reduced Mode**
- **Description**: Simplified prediction for faster processing
- **Features**:
  - Reduced weight adaptation
  - Fewer prediction components
  - Simplified local difference calculations
- **Computational Cost**: Medium
- **Use Case**: Real-time applications, hardware constraints

## Internal Configuration Options

### 4. **Weight Update Parameters**

The predictor uses several internal parameters that affect weight adaptation:

```python
# Weight scaling parameters (from CCSDS equations 51-54)
self.weight_exponent = 4      # ω scaling exponent
self.weight_resolution = 19   # Weight update resolution
self.weight_limit = 2**18     # Maximum weight magnitude
```

### 5. **Sample Representative Options**

```python
# Sample representative computation methods
def _compute_sample_representative(self, predicted_value, residual, max_error):
    """
    Computes s''_z(t) using equations (46)-(48)
    
    Options:
    - Lossless: s''_z(t) = s_z(t) (exact reconstruction)  
    - Near-lossless: s''_z(t) = clamp(predicted + residual, bounds)
    """
```

## Method Configuration Options

### 6. **Runtime Configuration Methods**

```python
# Set prediction mode dynamically
predictor.set_prediction_mode('full')    # or 'reduced'

# Configure local sum computation  
predictor.set_local_sum_type('neighbor-oriented')  # or 'column-oriented'
```

### 7. **Forward Method Options**

```python
predictions, residuals, sample_reps = predictor.forward(
    image,                    # [Z, Y, X] input image
    max_errors=None          # None for lossless, tensor for near-lossless
)
```

#### Max Errors Parameter:
- **None**: Lossless compression (default)
- **Tensor**: Near-lossless with per-sample error limits
- **Shape**: Same as image [Z, Y, X]

## Mathematical Implementation Details

### 8. **Local Sum Equations**

#### Neighbor-Oriented (Wide) Local Sums:
- **North Local Sum**: σ_N = s''_{z-1}(t-1) + s''_{z-1}(t) + s''_{z-1}(t+1)
- **West Local Sum**: σ_W = s''_z(t-1) + s''_z(t-2) + s''_z(t-3)  
- **Northwest Local Sum**: σ_{NW} = s''_{z-1}(t-1) + s''_{z-1}(t-2)

#### Column-Oriented (Narrow) Local Sums:
- **North Local Sum**: σ_N = s''_{z-1}(t)
- **West Local Sum**: σ_W = s''_z(t-1)
- **Northwest Local Sum**: σ_{NW} = s''_{z-1}(t-1)

### 9. **Central and Directional Local Differences**

```python
# Central local difference (equation 24)
d = σ_N + σ_W - σ_{NW}

# Directional local differences (equations 25-27) 
d_N = σ_N - σ_{NW}      # North difference
d_W = σ_W - σ_{NW}      # West difference  
d_{NW} = σ_{NW}         # Northwest difference
```

### 10. **Weight Update Mechanisms**

```python
# High-resolution weight update (equations 51-54)
def _update_weights(self, prediction_error, weight_vector):
    """
    Available update strategies:
    1. Standard CCSDS update with scaling exponent ω
    2. Clipped update to prevent overflow  
    3. Adaptive scaling based on prediction accuracy
    """
```

## Configuration Examples

### Example 1: Maximum Accuracy Configuration
```python
predictor = SpectralPredictor(
    num_bands=224,
    dynamic_range=16,
    prediction_bands=15,              # Use maximum bands
    local_sum_type='neighbor-oriented', # Full spatial context
    prediction_mode='full'            # Maximum accuracy
)
```

### Example 2: Hardware-Optimized Configuration  
```python
predictor = SpectralPredictor(
    num_bands=224,
    dynamic_range=16,
    prediction_bands=8,               # Reduced for speed
    local_sum_type='column-oriented', # Hardware pipelining
    prediction_mode='reduced'         # Faster processing
)
```

### Example 3: Near-Lossless Configuration
```python
predictor = SpectralPredictor(
    num_bands=224,
    dynamic_range=16,
    prediction_mode='full'
)

# Define per-sample error limits
max_errors = torch.ones(224, 512, 512) * 2  # Allow ±2 error

predictions, residuals, sample_reps = predictor.forward(image, max_errors)
```

### Example 4: Real-time Processing
```python  
predictor = SpectralPredictor(
    num_bands=128,
    dynamic_range=12,                 # Lower precision
    prediction_bands=4,               # Minimal prediction context
    local_sum_type='column-oriented', # Pipeline friendly
    prediction_mode='reduced'         # Fastest processing
)
```

## Advanced Configuration

### 11. **Internal Parameter Tuning**

For research or specialized applications, internal parameters can be modified:

```python
# After initialization, modify internal parameters
predictor.weight_exponent = 3       # Faster weight adaptation  
predictor.weight_resolution = 16    # Lower precision weights
predictor.rescale_interval = 32     # More frequent rescaling
```

### 12. **Debugging and Analysis Options**

```python
# Enable detailed analysis (not in standard)
predictor.enable_analysis_mode()

# Get detailed prediction statistics
stats = predictor.get_prediction_statistics()
# Returns: weight magnitudes, prediction errors, convergence metrics
```

## Performance vs Accuracy Trade-offs

| Configuration | Accuracy | Speed | Memory | Use Case |
|---------------|----------|-------|---------|----------|
| Full + Neighbor + 15 bands | Highest | Slowest | Highest | Archive quality |
| Full + Column + 8 bands | High | Medium | Medium | General purpose |  
| Reduced + Column + 4 bands | Medium | Fast | Low | Real-time |
| Reduced + Neighbor + 2 bands | Lower | Fastest | Lowest | Ultra low-latency |

## CCSDS-123.0-B-2 Compliance

### Standards Compliance Matrix:

| Feature | Implementation | CCSDS Section | Status |
|---------|---------------|---------------|---------|
| Local sum computation | ✅ Complete | 4.4, Equations 20-23 | Full compliance |
| Weight update mechanism | ✅ Complete | 4.3, Equations 51-54 | Full compliance |
| Sample representatives | ✅ Complete | 4.3, Equations 46-48 | Full compliance |
| Prediction modes | ✅ Complete | 4.3 | Full compliance |
| Directional differences | ✅ Complete | 4.3, Equations 25-27 | Full compliance |
| Issue 2 narrow local sums | ✅ Complete | Issue 2, Section 4.4 | Full compliance |

## Integration Notes

### With Quantizers:
- **LosslessQuantizer**: Use `max_errors=None`
- **NearLosslessQuantizer**: Provide appropriate `max_errors` tensor
- **UniformQuantizer**: Configure `dynamic_range` to match quantizer

### With Entropy Coders:
- Predictor output `residuals` feed directly into entropy coding
- `sample_representatives` used internally for causality
- No additional processing required

### With Optimized Components:
- `OptimizedSpectralPredictor` provides same interface
- Additional GPU acceleration and vectorization
- Same configuration parameters supported

---

**Note**: This documentation covers the complete configuration space of the SpectralPredictor class. For most applications, the default parameters provide excellent compression performance while maintaining full CCSDS-123.0-B-2 compliance.