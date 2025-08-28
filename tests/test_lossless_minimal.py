#!/usr/bin/env python3
"""
Minimal test for CCSDS-123.0-B-2 Lossless Compression

Very simple test with tiny images to verify basic functionality.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ccsds_compressor import create_lossless_compressor


def test_minimal_lossless():
    """Test minimal lossless compression functionality"""
    print("Testing Minimal Lossless Compression...")
    
    # Very small test image - 2 bands, 3x3 pixels
    image = torch.tensor([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0], 
         [7.0, 8.0, 9.0]],
        [[11.0, 12.0, 13.0],
         [14.0, 15.0, 16.0],
         [17.0, 18.0, 19.0]]
    ])
    
    print(f"  Input image shape: {image.shape}")
    print(f"  Input image:\n{image}")
    
    # Create lossless compressor
    compressor = create_lossless_compressor(num_bands=2, dynamic_range=8)
    
    # Compress
    results = compressor(image)
    
    # Check results
    reconstructed = results['reconstructed_samples']
    max_error = torch.max(torch.abs(image - reconstructed))
    
    print(f"  Reconstructed image:\n{reconstructed}")
    print(f"  Max error: {max_error.item()}")
    print(f"  Compression ratio: {results['compression_ratio']:.2f}:1")
    
    # Verify lossless property
    if max_error == 0:
        print("  ✓ Lossless compression verified")
        return True
    else:
        print(f"  ✗ Error: Expected 0 error, got {max_error}")
        return False


def test_single_pixel():
    """Test single pixel image"""
    print("\nTesting Single Pixel Image...")
    
    # Single pixel, single band
    image = torch.tensor([[[42.0]]])
    
    compressor = create_lossless_compressor(num_bands=1, dynamic_range=8)
    results = compressor(image)
    
    reconstructed = results['reconstructed_samples']
    max_error = torch.max(torch.abs(image - reconstructed))
    
    print(f"  Input: {image.item()}")
    print(f"  Output: {reconstructed.item()}")
    print(f"  Error: {max_error.item()}")
    
    if max_error == 0:
        print("  ✓ Single pixel test passed")
        return True
    else:
        print(f"  ✗ Single pixel test failed")
        return False


def run_minimal_tests():
    """Run minimal test suite"""
    print("CCSDS-123.0-B-2 Minimal Lossless Tests")
    print("======================================")
    
    try:
        test1_passed = test_minimal_lossless()
        test2_passed = test_single_pixel()
        
        if test1_passed and test2_passed:
            print("\n✓ All minimal tests passed!")
            return True
        else:
            print("\n✗ Some tests failed")
            return False
            
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_minimal_tests()
    sys.exit(0 if success else 1)