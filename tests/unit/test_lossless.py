#!/usr/bin/env python3
"""
Focused test for CCSDS-123.0-B-2 Lossless Compression

Tests only the lossless compression functionality with minimal overhead.
"""

import torch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ccsds import create_lossless_compressor


def generate_simple_test_image(num_bands=3, height=16, width=16, dynamic_range=12):
    """
    Generate simple synthetic test image for lossless compression testing
    
    Args:
        num_bands: Number of spectral bands
        height, width: Spatial dimensions  
        dynamic_range: Bit depth
        
    Returns:
        Test image tensor [Z, Y, X]
    """
    # Create simple patterns that are easy to predict
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    
    image = torch.zeros(num_bands, height, width)
    
    for z in range(num_bands):
        # Simple gradients and patterns
        base_pattern = (
            x_coords.float() * (z + 1) +
            y_coords.float() * (z + 2) + 
            50  # DC offset
        )
        
        # Add some spectral correlation
        if z > 0:
            base_pattern += image[z-1] * 0.3
        
        image[z] = base_pattern
    
    # Scale to dynamic range
    max_val = 2**(dynamic_range - 1) - 1
    min_val = -2**(dynamic_range - 1) if dynamic_range > 1 else 0
    
    # Normalize and scale
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (max_val - min_val) + min_val
    
    return image.round()


def test_lossless_basic():
    """Test basic lossless compression functionality"""
    print("Testing Basic Lossless Compression...")
    
    # Small test image
    image = generate_simple_test_image(num_bands=5, height=8, width=8, dynamic_range=10)
    
    # Create lossless compressor
    compressor = create_lossless_compressor(num_bands=5, dynamic_range=10)
    
    # Compress
    results = compressor(image)
    
    # Verify lossless property
    reconstructed = results['reconstructed_samples']
    max_error = torch.max(torch.abs(image - reconstructed))
    
    print(f"  Original image shape: {image.shape}")
    print(f"  Max reconstruction error: {max_error.item()}")
    print(f"  Compression ratio: {results['compression_ratio']:.2f}:1")
    print(f"  Bits per sample: {results['compressed_size'] / image.numel():.2f}")
    
    # Assert lossless property
    assert max_error == 0, f"Lossless compression should have zero error, got {max_error}"
    assert torch.allclose(image, reconstructed), "Images should be identical"
    
    print("  ✓ Basic lossless compression test passed")
    return results


def test_lossless_edge_cases():
    """Test lossless compression edge cases"""
    print("\nTesting Lossless Edge Cases...")
    
    # Test 1: Single band
    print("  Testing single band image...")
    image_1band = generate_simple_test_image(num_bands=1, height=4, width=4, dynamic_range=8)
    compressor_1band = create_lossless_compressor(num_bands=1, dynamic_range=8)
    results_1band = compressor_1band(image_1band)
    
    max_error_1band = torch.max(torch.abs(image_1band - results_1band['reconstructed_samples']))
    assert max_error_1band == 0, "Single band should be lossless"
    print(f"    Single band compression ratio: {results_1band['compression_ratio']:.2f}:1")
    
    # Test 2: Constant image
    print("  Testing constant image...")
    image_const = torch.ones(2, 4, 4) * 100
    compressor_const = create_lossless_compressor(num_bands=2, dynamic_range=12)
    results_const = compressor_const(image_const)
    
    max_error_const = torch.max(torch.abs(image_const - results_const['reconstructed_samples']))
    assert max_error_const == 0, "Constant image should be lossless"
    print(f"    Constant image compression ratio: {results_const['compression_ratio']:.2f}:1")
    
    # Test 3: Random noise (worst case)
    print("  Testing random noise image...")
    torch.manual_seed(42)  # For reproducibility
    image_noise = torch.randint(0, 256, (2, 6, 6)).float()
    compressor_noise = create_lossless_compressor(num_bands=2, dynamic_range=12)
    results_noise = compressor_noise(image_noise)
    
    max_error_noise = torch.max(torch.abs(image_noise - results_noise['reconstructed_samples']))
    assert max_error_noise == 0, "Noisy image should still be lossless"
    print(f"    Noisy image compression ratio: {results_noise['compression_ratio']:.2f}:1")
    
    print("  ✓ Edge cases test passed")


def test_lossless_different_sizes():
    """Test lossless compression with different image sizes"""
    print("\nTesting Different Image Sizes...")
    
    sizes = [
        (2, 4, 4),    # Very small
        (3, 8, 8),    # Small  
        (5, 12, 12),  # Medium
    ]
    
    for num_bands, height, width in sizes:
        print(f"  Testing {num_bands} bands, {height}x{width} pixels...")
        
        image = generate_simple_test_image(num_bands, height, width, dynamic_range=12)
        compressor = create_lossless_compressor(num_bands, dynamic_range=12)
        results = compressor(image)
        
        # Verify lossless
        reconstructed = results['reconstructed_samples']
        max_error = torch.max(torch.abs(image - reconstructed))
        
        assert max_error == 0, f"Size {height}x{width} should be lossless"
        
        print(f"    Compression ratio: {results['compression_ratio']:.2f}:1")
        print(f"    Bits per sample: {results['compressed_size'] / image.numel():.2f}")
    
    print("  ✓ Different sizes test passed")


def test_lossless_dynamic_ranges():
    """Test lossless compression with different dynamic ranges"""
    print("\nTesting Different Dynamic Ranges...")
    
    dynamic_ranges = [8, 12, 16]
    
    for dr in dynamic_ranges:
        print(f"  Testing {dr}-bit dynamic range...")
        
        # Generate image appropriate for this dynamic range
        image = generate_simple_test_image(num_bands=3, height=8, width=8, dynamic_range=dr)
        compressor = create_lossless_compressor(num_bands=3, dynamic_range=dr)
        results = compressor(image)
        
        # Verify lossless
        reconstructed = results['reconstructed_samples']
        max_error = torch.max(torch.abs(image - reconstructed))
        
        assert max_error == 0, f"{dr}-bit should be lossless"
        
        print(f"    {dr}-bit compression ratio: {results['compression_ratio']:.2f}:1")
    
    print("  ✓ Dynamic range test passed")


def test_lossless_predictor_modes():
    """Test lossless compression with different predictor modes"""
    print("\nTesting Different Predictor Modes...")
    
    image = generate_simple_test_image(num_bands=4, height=8, width=8, dynamic_range=12)
    
    # Test normal predictor
    print("  Testing normal predictor...")
    compressor_normal = create_lossless_compressor(
        num_bands=4, 
        dynamic_range=12, 
        use_narrow_local_sums=False
    )
    results_normal = compressor_normal(image)
    
    max_error_normal = torch.max(torch.abs(image - results_normal['reconstructed_samples']))
    assert max_error_normal == 0, "Normal predictor should be lossless"
    print(f"    Normal predictor compression ratio: {results_normal['compression_ratio']:.2f}:1")
    
    # Test narrow local sums predictor
    print("  Testing narrow local sums predictor...")
    compressor_narrow = create_lossless_compressor(
        num_bands=4, 
        dynamic_range=12, 
        use_narrow_local_sums=True
    )
    results_narrow = compressor_narrow(image)
    
    max_error_narrow = torch.max(torch.abs(image - results_narrow['reconstructed_samples']))
    assert max_error_narrow == 0, "Narrow local sums should be lossless"
    print(f"    Narrow local sums compression ratio: {results_narrow['compression_ratio']:.2f}:1")
    
    print("  ✓ Predictor modes test passed")


def run_lossless_tests():
    """Run all lossless compression tests"""
    print("CCSDS-123.0-B-2 Lossless Compression Tests")
    print("==========================================")
    
    try:
        # Run all tests
        basic_results = test_lossless_basic()
        test_lossless_edge_cases()
        test_lossless_different_sizes()
        test_lossless_dynamic_ranges()
        test_lossless_predictor_modes()
        
        print("\n" + "="*50)
        print("All lossless compression tests passed! ✓")
        print("="*50)
        
        # Print summary
        if basic_results:
            print(f"\nSample Results (3 bands, 8x8 image):")
            print(f"  Compression ratio: {basic_results['compression_ratio']:.2f}:1")
            print(f"  Bits per sample: {basic_results['compressed_size'] / 192:.2f}")  # 3*8*8 = 192 samples
            print(f"  Original size: {basic_results['original_size']} bits")
            print(f"  Compressed size: {basic_results['compressed_size']} bits")
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_lossless_test():
    """Run all lossless compression tests"""
    print("CCSDS-123.0-B-2 Lossless Compression Tests")
    print("==========================================")
    
    try:
        # Run all tests
        basic_results = test_lossless_basic()
        # test_lossless_different_sizes()
        
        print("\n" + "="*50)
        print("Lossless compression tests passed! ✓")
        print("="*50)
        
        # Print summary
        if basic_results:
            print(f"\nSample Results (3 bands, 8x8 image):")
            print(f"  Compression ratio: {basic_results['compression_ratio']:.2f}:1")
            print(f"  Bits per sample: {basic_results['compressed_size'] / 192:.2f}")  # 3*8*8 = 192 samples
            print(f"  Original size: {basic_results['original_size']} bits")
            print(f"  Compressed size: {basic_results['compressed_size']} bits")
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_lossless_tests()
    # success = run_lossless_test()
    sys.exit(0 if success else 1)
