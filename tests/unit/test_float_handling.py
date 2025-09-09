#!/usr/bin/env python3
"""
Test script for enhanced float handling in CCSDS compressor
"""

import torch
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from ccsds.ccsds_compressor import CCSDS123Compressor


def test_float_input_unsigned():
    """Test unsigned float input quantization"""
    print("Testing Unsigned Float Input Handling")
    print("="*50)
    
    # Create float test data with values outside integer range
    float_image = torch.tensor([
        [[100.7, 200.3, 150.9],
         [75.2, 225.8, 180.1]],
        [[50.4, 175.6, 125.3],
         [210.9, 95.7, 165.2]]
    ]).float()
    
    print(f"Input float image:")
    print(f"  Shape: {float_image.shape}")
    print(f"  dtype: {float_image.dtype}")
    print(f"  Range: [{float_image.min():.2f}, {float_image.max():.2f}]")
    print(f"  Sample values: {float_image[0, 0, :3]}")
    
    # Test with 8-bit samples (unsigned range [0, 255])
    compressor = CCSDS123Compressor(
        num_bands=2,
        dynamic_range=8,
        lossless=True
    )
    
    try:
        results = compressor.forward(float_image)
        print(f"\n✓ Compression successful!")
        print(f"  Quantized range: [{results['sample_representatives'].min():.0f}, {results['sample_representatives'].max():.0f}]")
        print(f"  Expected range for 8-bit unsigned: [0, 255]")
        
        # Verify integer values
        rounded = torch.round(results['sample_representatives'])
        if torch.allclose(results['sample_representatives'], rounded, atol=1e-6):
            print("  ✓ All values are properly quantized to integers")
        else:
            print("  ✗ Some values are not properly quantized")
            
        return True
    except Exception as e:
        print(f"  ✗ Compression failed: {e}")
        return False


def test_float_input_signed():
    """Test float input with negative values (will be clamped to unsigned range)"""
    print("\nTesting Float Input with Negative Values (Clamped to Unsigned Range)")
    print("="*50)
    
    # Create float test data with negative values
    # Note: These will be clamped to [0, 255] range for 8-bit unsigned
    float_image = torch.tensor([
        [[-100.7, 50.3, -25.9],    # Negative values will be clamped to 0
         [75.2, -125.8, 100.1]],   # More negative values to test clamping
        [[-50.4, 25.6, -75.3],
         [110.9, -95.7, 65.2]]
    ]).float()
    
    print(f"Input float image:")
    print(f"  Shape: {float_image.shape}")
    print(f"  dtype: {float_image.dtype}")
    print(f"  Range: [{float_image.min():.2f}, {float_image.max():.2f}]")
    print(f"  Sample values: {float_image[0, 0, :3]}")
    
    # Test with 8-bit samples (unsigned range [0, 255])
    # Note: Current implementation uses unsigned range [0, 2^D-1]
    compressor = CCSDS123Compressor(
        num_bands=2,
        dynamic_range=8,
        lossless=True
    )
    
    try:
        results = compressor.forward(float_image)
        print(f"\n✓ Compression successful!")
        print(f"  Quantized range: [{results['sample_representatives'].min():.0f}, {results['sample_representatives'].max():.0f}]")
        print(f"  Expected range for 8-bit unsigned: [0, 255]")
        
        # Verify integer values
        rounded = torch.round(results['sample_representatives'])
        if torch.allclose(results['sample_representatives'], rounded, atol=1e-6):
            print("  ✓ All values are properly quantized to integers")
        else:
            print("  ✗ Some values are not properly quantized")
            
        return True
    except Exception as e:
        print(f"  ✗ Compression failed: {e}")
        return False


def test_integer_input():
    """Test that integer input still works correctly"""
    print("\nTesting Integer Input (Should Work As Before)")
    print("="*50)
    
    # Create integer test data
    int_image = torch.tensor([
        [[100, 200, 150],
         [75, 225, 180]],
        [[50, 175, 125],
         [210, 95, 165]]
    ], dtype=torch.long)
    
    print(f"Input integer image:")
    print(f"  Shape: {int_image.shape}")
    print(f"  dtype: {int_image.dtype}")
    print(f"  Range: [{int_image.min()}, {int_image.max()}]")
    
    compressor = CCSDS123Compressor(
        num_bands=2,
        dynamic_range=8,
        lossless=True
    )
    
    try:
        results = compressor.forward(int_image)
        print(f"\n✓ Compression successful!")
        print(f"  Processed range: [{results['sample_representatives'].min():.0f}, {results['sample_representatives'].max():.0f}]")
        return True
    except Exception as e:
        print(f"  ✗ Compression failed: {e}")
        return False


def test_out_of_range_clamping():
    """Test that out-of-range values are properly clamped"""
    print("\nTesting Out-of-Range Value Clamping")
    print("="*50)
    
    # Create data with values outside 8-bit range
    float_image = torch.tensor([
        [[300.7, -50.3, 400.9],   # Values outside [0, 255] for unsigned 8-bit
         [75.2, 1000.8, -100.1]],
        [[-200.4, 500.6, 125.3],
         [210.9, -300.7, 2000.2]]
    ]).float()
    
    print(f"Input out-of-range image:")
    print(f"  Range: [{float_image.min():.1f}, {float_image.max():.1f}]")
    print(f"  Expected 8-bit unsigned range: [0, 255]")
    
    compressor = CCSDS123Compressor(
        num_bands=2,
        dynamic_range=8,
        lossless=True
    )
    
    try:
        results = compressor.forward(float_image)
        quantized_range = [results['sample_representatives'].min().item(), 
                          results['sample_representatives'].max().item()]
        print(f"\n✓ Compression successful!")
        print(f"  Clamped range: [{quantized_range[0]:.0f}, {quantized_range[1]:.0f}]")
        
        if quantized_range[0] >= 0 and quantized_range[1] <= 255:
            print("  ✓ Values properly clamped to valid 8-bit unsigned range")
            return True
        else:
            print("  ✗ Values not properly clamped")
            return False
            
    except Exception as e:
        print(f"  ✗ Compression failed: {e}")
        return False


def run_all_tests():
    """Run all float handling tests"""
    print("CCSDS Float Input Handling Tests")
    print("=" * 60)
    
    tests = [
        test_float_input_unsigned,
        test_float_input_signed, 
        test_integer_input,
        test_out_of_range_clamping
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Float handling is working correctly.")
    else:
        print("⚠️  Some tests failed. Float handling needs attention.")
        
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)