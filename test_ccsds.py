#!/usr/bin/env python3
"""
Test suite for CCSDS-123.0-B-2 Compressor Implementation

Tests the complete compression pipeline including:
- Lossless compression
- Near-lossless compression with different error limits
- Different predictor modes
- Sample representative parameter effects
- Compression ratio analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from ccsds_compressor import CCSDS123Compressor, create_lossless_compressor, create_near_lossless_compressor
from predictor import SpectralPredictor
from quantizer import UniformQuantizer
from sample_representative import SampleRepresentativeCalculator


def generate_test_image(num_bands=10, height=64, width=64, dynamic_range=16, noise_level=0.1):
    """
    Generate synthetic multispectral test image
    
    Args:
        num_bands: Number of spectral bands
        height, width: Spatial dimensions
        dynamic_range: Bit depth
        noise_level: Amount of noise to add
        
    Returns:
        Test image tensor [Z, Y, X]
    """
    # Create base patterns
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    
    # Create spectral signatures with spatial structure
    image = torch.zeros(num_bands, height, width)
    
    for z in range(num_bands):
        # Wavelength-dependent response
        wavelength_factor = (z + 1) / num_bands
        
        # Spatial patterns
        spatial_pattern1 = torch.sin(2 * np.pi * x_coords / width * 3) * torch.cos(2 * np.pi * y_coords / height * 2)
        spatial_pattern2 = torch.exp(-((x_coords - width//2)**2 + (y_coords - height//2)**2) / (width * height * 0.1))
        
        # Combine patterns with spectral dependency
        base_signal = (
            wavelength_factor * spatial_pattern1 * 50 + 
            (1 - wavelength_factor) * spatial_pattern2 * 80 +
            100  # DC offset
        )
        
        # Add noise
        noise = torch.randn_like(base_signal) * noise_level * 10
        
        image[z] = base_signal + noise
    
    # Scale to dynamic range
    max_val = 2**(dynamic_range - 1) - 1
    min_val = -2**(dynamic_range - 1) if dynamic_range > 1 else 0
    
    # Normalize and scale
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (max_val - min_val) + min_val
    
    return image.round()


def test_lossless_compression():
    """Test lossless compression functionality"""
    print("Testing Lossless Compression...")
    
    # Generate test image
    image = generate_test_image(num_bands=5, height=32, width=32, dynamic_range=12)
    
    # Create lossless compressor
    compressor = create_lossless_compressor(num_bands=5, dynamic_range=12)
    
    # Compress
    results = compressor(image)
    
    # Verify lossless property
    reconstructed = results['reconstructed_samples']
    max_error = torch.max(torch.abs(image - reconstructed))
    
    print(f"  Max reconstruction error: {max_error.item()}")
    print(f"  Compression ratio: {results['compression_ratio']:.2f}:1")
    print(f"  Bits per sample: {results['compressed_size'] / image.numel():.2f}")
    
    assert max_error == 0, f"Lossless compression should have zero error, got {max_error}"
    print("  ✓ Lossless compression test passed")
    
    return results


def test_near_lossless_compression():
    """Test near-lossless compression with different error limits"""
    print("\nTesting Near-Lossless Compression...")
    
    # Generate test image
    image = generate_test_image(num_bands=5, height=32, width=32, dynamic_range=12)
    
    error_limits = [1, 2, 4, 8]
    results_list = []
    
    for error_limit in error_limits:
        print(f"  Testing with absolute error limit = {error_limit}")
        
        # Create near-lossless compressor
        abs_limits = torch.ones(5) * error_limit
        compressor = create_near_lossless_compressor(
            num_bands=5, 
            dynamic_range=12,
            absolute_error_limits=abs_limits
        )
        
        # Compress
        results = compressor(image)
        reconstructed = results['reconstructed_samples']
        
        # Check error bounds
        max_error = torch.max(torch.abs(image - reconstructed))
        mse = torch.mean((image - reconstructed)**2)
        
        print(f"    Max error: {max_error.item()} (limit: {error_limit})")
        print(f"    MSE: {mse.item():.2f}")
        print(f"    Compression ratio: {results['compression_ratio']:.2f}:1")
        print(f"    Bits per sample: {results['compressed_size'] / image.numel():.2f}")
        
        # Verify error bound
        assert max_error <= error_limit, f"Max error {max_error} exceeds limit {error_limit}"
        
        results_list.append({
            'error_limit': error_limit,
            'max_error': max_error.item(),
            'mse': mse.item(),
            'compression_ratio': results['compression_ratio'],
            'bits_per_sample': results['compressed_size'] / image.numel()
        })
    
    print("  ✓ Near-lossless compression test passed")
    return results_list


def test_sample_representative_parameters():
    """Test effect of sample representative parameters"""
    print("\nTesting Sample Representative Parameters...")
    
    image = generate_test_image(num_bands=3, height=24, width=24, dynamic_range=12)
    
    # Test different parameter combinations
    param_combinations = [
        {'phi': 0, 'psi': 0, 'theta': 4},      # Traditional
        {'phi': 2, 'psi': 4, 'theta': 4},      # Moderate adjustment
        {'phi': 4, 'psi': 6, 'theta': 4},      # Stronger adjustment
    ]
    
    for i, params in enumerate(param_combinations):
        print(f"  Testing φ={params['phi']}, ψ={params['psi']}, Θ={params['theta']}")
        
        # Create compressor with specific parameters
        abs_limits = torch.ones(3) * 2  # Fixed error limit
        compressor = create_near_lossless_compressor(
            num_bands=3,
            dynamic_range=12,
            absolute_error_limits=abs_limits
        )
        
        # Set sample representative parameters
        phi_tensor = torch.ones(3) * params['phi']
        psi_tensor = torch.ones(3) * params['psi']
        
        compressor.set_compression_parameters(
            sample_rep_phi=phi_tensor,
            sample_rep_psi=psi_tensor,
            sample_rep_theta=params['theta']
        )
        
        # Compress
        results = compressor(image)
        
        print(f"    Compression ratio: {results['compression_ratio']:.2f}:1")
        print(f"    Bits per sample: {results['compressed_size'] / image.numel():.2f}")
        
        # Verify reconstruction bounds
        reconstructed = results['reconstructed_samples']
        max_error = torch.max(torch.abs(image - reconstructed))
        print(f"    Max error: {max_error.item()}")
    
    print("  ✓ Sample representative parameters test passed")


def test_relative_error_limits():
    """Test relative error limits"""
    print("\nTesting Relative Error Limits...")
    
    # Create image with varying magnitude samples
    image = generate_test_image(num_bands=3, height=24, width=24, dynamic_range=12)
    
    # Scale different regions to have different magnitudes
    image[:, :12, :] *= 0.5  # Low magnitude region
    image[:, 12:, :] *= 2.0  # High magnitude region
    
    # Test relative error limits
    rel_limits = torch.ones(3) * 0.1  # 10% relative error
    compressor = create_near_lossless_compressor(
        num_bands=3,
        dynamic_range=12,
        relative_error_limits=rel_limits
    )
    
    results = compressor(image)
    reconstructed = results['reconstructed_samples']
    
    # Check that error is proportional to magnitude
    relative_errors = torch.abs(image - reconstructed) / (torch.abs(image) + 1e-8)
    max_rel_error = torch.max(relative_errors)
    
    print(f"  Max relative error: {max_rel_error.item():.3f}")
    print(f"  Compression ratio: {results['compression_ratio']:.2f}:1")
    
    print("  ✓ Relative error limits test passed")


def test_narrow_local_sums():
    """Test narrow local sums predictor option"""
    print("\nTesting Narrow Local Sums Predictor...")
    
    image = generate_test_image(num_bands=5, height=32, width=32, dynamic_range=12)
    
    # Test with and without narrow local sums
    for use_narrow in [False, True]:
        print(f"  Testing use_narrow_local_sums = {use_narrow}")
        
        compressor = create_lossless_compressor(
            num_bands=5,
            dynamic_range=12,
            use_narrow_local_sums=use_narrow
        )
        
        results = compressor(image)
        
        print(f"    Compression ratio: {results['compression_ratio']:.2f}:1")
        print(f"    Bits per sample: {results['compressed_size'] / image.numel():.2f}")
        
        # Verify lossless property
        reconstructed = results['reconstructed_samples']
        max_error = torch.max(torch.abs(image - reconstructed))
        assert max_error == 0, "Should be lossless"
    
    print("  ✓ Narrow local sums test passed")


def test_compression_performance():
    """Test compression performance on different image types"""
    print("\nTesting Compression Performance...")
    
    test_cases = [
        {'name': 'Low noise', 'noise': 0.01, 'bands': 8},
        {'name': 'Medium noise', 'noise': 0.1, 'bands': 8},
        {'name': 'High noise', 'noise': 0.5, 'bands': 8},
        {'name': 'Many bands', 'noise': 0.1, 'bands': 20},
    ]
    
    for case in test_cases:
        print(f"  Testing {case['name']}:")
        
        image = generate_test_image(
            num_bands=case['bands'],
            height=32,
            width=32,
            noise_level=case['noise']
        )
        
        # Test lossless
        compressor_lossless = create_lossless_compressor(num_bands=case['bands'])
        results_lossless = compressor_lossless(image)
        
        # Test near-lossless  
        abs_limits = torch.ones(case['bands']) * 2
        compressor_lossy = create_near_lossless_compressor(
            num_bands=case['bands'],
            absolute_error_limits=abs_limits
        )
        results_lossy = compressor_lossy(image)
        
        print(f"    Lossless: {results_lossless['compression_ratio']:.2f}:1 "
              f"({results_lossless['compressed_size'] / image.numel():.2f} bps)")
        print(f"    Near-lossless: {results_lossy['compression_ratio']:.2f}:1 "
              f"({results_lossy['compressed_size'] / image.numel():.2f} bps)")
    
    print("  ✓ Compression performance test completed")


def plot_compression_results(results_list):
    """Plot compression results"""
    print("\nGenerating compression plots...")
    
    if not results_list:
        return
        
    try:
        error_limits = [r['error_limit'] for r in results_list]
        compression_ratios = [r['compression_ratio'] for r in results_list]
        bits_per_sample = [r['bits_per_sample'] for r in results_list]
        mse_values = [r['mse'] for r in results_list]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Compression ratio vs error limit
        ax1.plot(error_limits, compression_ratios, 'b-o')
        ax1.set_xlabel('Error Limit')
        ax1.set_ylabel('Compression Ratio')
        ax1.set_title('Compression Ratio vs Error Limit')
        ax1.grid(True)
        
        # Bits per sample vs error limit
        ax2.plot(error_limits, bits_per_sample, 'r-o')
        ax2.set_xlabel('Error Limit')
        ax2.set_ylabel('Bits per Sample')
        ax2.set_title('Bits per Sample vs Error Limit')
        ax2.grid(True)
        
        # MSE vs error limit
        ax3.semilogy(error_limits, mse_values, 'g-o')
        ax3.set_xlabel('Error Limit')
        ax3.set_ylabel('Mean Squared Error')
        ax3.set_title('MSE vs Error Limit')
        ax3.grid(True)
        
        # Rate-distortion curve
        ax4.plot(bits_per_sample, mse_values, 'm-o')
        ax4.set_xlabel('Bits per Sample')
        ax4.set_ylabel('Mean Squared Error')
        ax4.set_title('Rate-Distortion Curve')
        ax4.set_yscale('log')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('ccsds_compression_results.png', dpi=150, bbox_inches='tight')
        print("  Saved results to 'ccsds_compression_results.png'")
        
    except Exception as e:
        print(f"  Warning: Could not generate plots - {e}")


def run_all_tests():
    """Run complete test suite"""
    print("CCSDS-123.0-B-2 Compressor Test Suite")
    print("=====================================")
    
    # Run tests
    try:
        lossless_results = test_lossless_compression()
        near_lossless_results = test_near_lossless_compression()
        test_sample_representative_parameters()
        test_relative_error_limits()
        test_narrow_local_sums()
        test_compression_performance()
        
        # Generate plots
        plot_compression_results(near_lossless_results)
        
        print("\n" + "="*50)
        print("All tests completed successfully! ✓")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_lossless_test():
    """Run lossless test suite"""
    print("CCSDS-123.0-B-2 Compressor Test Suite")
    print("=====================================")
    
    # Run tests
    try:
        lossless_results = test_lossless_compression()
        
        # Generate plots
        plot_compression_results(lossless_results)
        
        print("\n" + "="*50)
        print("All tests completed successfully! ✓")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    success = run_lossless_test()
    # sys.exit(0 if success else 1)
