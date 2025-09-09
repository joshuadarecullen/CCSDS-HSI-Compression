#!/usr/bin/env python3
"""
Test script for CCSDS-123.0-B-2 compliant float handling
"""

import torch
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from ccsds.ccsds_compressor import CCSDS123Compressor


def test_float_quantization():
    """Test float to integer quantization compliance"""
    print("CCSDS-123.0-B-2 Float Quantization Test")
    print("=" * 50)
    
    compressor = CCSDS123Compressor(num_bands=2, dynamic_range=8, lossless=True)
    
    test_cases = [
        {
            'name': 'Positive floats',
            'data': torch.tensor([
                [[100.7, 200.3, 150.9]],
                [[75.2, 225.8, 180.1]]
            ]).float(),
            'expected_min': 75.0,
            'expected_max': 226.0
        },
        {
            'name': 'Mixed positive/negative floats',
            'data': torch.tensor([
                [[-50.7, 100.3, 200.9]],
                [[75.2, -10.8, 180.1]]
            ]).float(),
            'expected_min': 0.0,    # Negative values clamped to 0
            'expected_max': 201.0
        },
        {
            'name': 'Out-of-range values',
            'data': torch.tensor([
                [[300.0, -100.0, 1000.0]],
                [[50.5, 200.5, -200.0]]
            ]).float(),
            'expected_min': 0.0,    # Negative clamped to 0
            'expected_max': 255.0   # > 255 clamped to 255
        },
        {
            'name': 'Integer values (should pass through)',
            'data': torch.tensor([
                [[100, 200, 150]],
                [[75, 225, 180]]
            ], dtype=torch.long).float(),
            'expected_min': 75.0,
            'expected_max': 225.0
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        input_data = test_case['data']
        print(f"Input range: [{input_data.min():.1f}, {input_data.max():.1f}]")
        print(f"Input dtype: {input_data.dtype}")
        
        try:
            # Test the validation step (our float handling)
            validated = compressor._validate_input(input_data)
            
            print(f"Output range: [{validated.min():.1f}, {validated.max():.1f}]")
            print(f"Output dtype: {validated.dtype}")
            
            # Check if values are integers
            rounded = torch.round(validated)
            is_integer = torch.allclose(validated, rounded, atol=1e-6)
            
            # Check if values are in valid range [0, 255] for 8-bit
            in_range = (validated.min() >= 0 and validated.max() <= 255)
            
            # Check expected values
            expected_min = test_case['expected_min']
            expected_max = test_case['expected_max']
            correct_range = (abs(validated.min() - expected_min) < 0.1 and 
                           abs(validated.max() - expected_max) < 0.1)
            
            if is_integer and in_range and correct_range:
                print("✓ PASS - Proper quantization to integer values in valid range")
                passed += 1
            else:
                print(f"✗ FAIL - Issues detected:")
                if not is_integer:
                    print(f"  - Values not properly quantized to integers")
                if not in_range:
                    print(f"  - Values outside valid 8-bit range [0, 255]")
                if not correct_range:
                    print(f"  - Unexpected range, expected [{expected_min}, {expected_max}]")
                    
        except Exception as e:
            print(f"✗ FAIL - Exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Float handling is CCSDS-123.0-B-2 compliant.")
        print("\nFeatures verified:")
        print("  ✓ Float values properly rounded to integers")
        print("  ✓ Values clamped to valid D-bit range [0, 2^D-1]")
        print("  ✓ Negative values handled correctly (clamped to 0)")
        print("  ✓ Out-of-range values handled correctly")
        print("  ✓ Integer inputs preserved correctly")
    else:
        print("⚠️  Some tests failed. Float handling needs attention.")
        
    return passed == total


if __name__ == "__main__":
    success = test_float_quantization()
    sys.exit(0 if success else 1)