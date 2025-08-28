#!/usr/bin/env python3
"""
Test individual components of CCSDS-123.0-B-2 implementation

Tests each component in isolation to verify basic functionality.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from predictor import SpectralPredictor
from quantizer import LosslessQuantizer, UniformQuantizer
from sample_representative import SampleRepresentativeCalculator


def test_predictor():
    """Test the predictor component"""
    print("Testing Predictor Component...")
    
    # Simple 2x2x2 test image
    image = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])
    
    predictor = SpectralPredictor(num_bands=2, dynamic_range=8)
    
    # Test single prediction
    pred = predictor.predict_sample(image, image, z=1, y=1, x=1)
    print(f"  Prediction for sample [1,1,1]: {pred.item():.2f}")
    
    # Test full prediction
    predictions, residuals = predictor(image)
    
    print(f"  Full predictions shape: {predictions.shape}")
    print(f"  Residuals range: [{residuals.min():.2f}, {residuals.max():.2f}]")
    
    if predictions.shape == image.shape:
        print("  ✓ Predictor test passed")
        return True
    else:
        print("  ✗ Predictor test failed")
        return False


def test_lossless_quantizer():
    """Test the lossless quantizer"""
    print("\nTesting Lossless Quantizer...")
    
    # Simple residuals
    residuals = torch.tensor([[[1.0, -2.0], [0.0, 3.0]]])
    predictions = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    
    quantizer = LosslessQuantizer(num_bands=1, dynamic_range=8)
    
    quant_residuals, mapped_indices, reconstructed = quantizer(residuals, predictions)
    
    print(f"  Original residuals: {residuals}")
    print(f"  Quantized residuals: {quant_residuals}")
    print(f"  Mapped indices: {mapped_indices}")
    
    # Check if residuals unchanged (lossless)
    if torch.allclose(residuals, quant_residuals):
        print("  ✓ Lossless quantizer test passed")
        return True
    else:
        print("  ✗ Lossless quantizer test failed")
        return False


def test_uniform_quantizer():
    """Test the uniform quantizer with error limits"""
    print("\nTesting Uniform Quantizer...")
    
    residuals = torch.tensor([[[1.5, -2.3], [0.7, 3.1]]])
    predictions = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    
    quantizer = UniformQuantizer(num_bands=1, dynamic_range=8)
    
    # Set error limit
    error_limits = torch.tensor([1.0])  # Allow 1 unit of error
    quantizer.set_error_limits(absolute_limits=error_limits)
    
    quant_residuals, mapped_indices, reconstructed = quantizer(residuals, predictions)
    
    print(f"  Original residuals: {residuals}")
    print(f"  Quantized residuals: {quant_residuals}")
    print(f"  Reconstructed: {reconstructed}")
    
    # Check error bounds
    original_samples = predictions + residuals
    max_error = torch.max(torch.abs(original_samples - reconstructed))
    
    print(f"  Max reconstruction error: {max_error.item():.2f} (limit: 1.0)")
    
    if max_error <= 1.0:
        print("  ✓ Uniform quantizer test passed")
        return True
    else:
        print("  ✗ Uniform quantizer test failed")
        return False


def test_sample_representative():
    """Test sample representative calculator"""
    print("\nTesting Sample Representative Calculator...")
    
    # Simple test data
    original = torch.tensor([[[10.0, 20.0]]])
    predicted = torch.tensor([[[9.0, 18.0]]])
    max_errors = torch.tensor([[[1, 2]]])
    
    calc = SampleRepresentativeCalculator(num_bands=1)
    
    representatives, bin_centers = calc(original, predicted, max_errors)
    
    print(f"  Original samples: {original}")
    print(f"  Predicted samples: {predicted}")
    print(f"  Representatives: {representatives}")
    print(f"  Bin centers: {bin_centers}")
    
    if representatives.shape == original.shape:
        print("  ✓ Sample representative test passed")
        return True
    else:
        print("  ✗ Sample representative test failed")
        return False


def run_component_tests():
    """Run all component tests"""
    print("CCSDS-123.0-B-2 Component Tests")
    print("===============================")
    
    try:
        test1 = test_predictor()
        test2 = test_lossless_quantizer()
        test3 = test_uniform_quantizer()
        test4 = test_sample_representative()
        
        all_passed = test1 and test2 and test3 and test4
        
        if all_passed:
            print("\n✓ All component tests passed!")
        else:
            print("\n✗ Some component tests failed")
            
        return all_passed
        
    except Exception as e:
        print(f"\nComponent tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_component_tests()
    sys.exit(0 if success else 1)