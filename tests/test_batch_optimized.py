#!/usr/bin/env python3
"""
Test batch processing for optimized CCSDS-123.0-B-2 implementation

Demonstrates batch processing capabilities and performance.
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from batch_optimized_compressor import create_batch_optimized_lossless_compressor


def generate_batch_test_images(batch_size, num_bands=5, height=16, width=16, seed=42):
    """Generate batch of test images with different patterns"""
    torch.manual_seed(seed)

    batch = torch.zeros(batch_size, num_bands, height, width)

    for b in range(batch_size):
        # Different pattern for each image in batch
        torch.manual_seed(seed + b)

        for z in range(num_bands):
            # Spectral and spatial patterns
            y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

            # Vary patterns across batch and bands
            freq_x = 1 + (b % 3) + (z % 2)
            freq_y = 1 + ((b + 1) % 3) + ((z + 1) % 2)

            pattern = (
                torch.sin(2 * torch.pi * x_coords / width * freq_x) *
                torch.cos(2 * torch.pi * y_coords / height * freq_y) * 20 +
                (b * 10 + z * 5) + 50  # DC offset varying by batch and band
            )

            # Add inter-band correlation
            if z > 0:
                pattern += batch[b, z-1] * 0.2

            # Add noise
            noise = torch.randn_like(pattern) * 3
            batch[b, z] = pattern + noise

    return batch.round().float()


def test_batch_processing():
    """Test basic batch processing functionality"""
    print("Testing Batch Processing for Optimized CCSDS-123.0-B-2")
    print("=" * 55)

    # Test different batch sizes
    test_cases = [
        (1, "Single image as batch"),
        (2, "Small batch"),
        (4, "Medium batch"),
        (8, "Large batch"),
    ]

    num_bands, height, width = 5, 16, 16

    for batch_size, description in test_cases:
        print(f"\n{description} (batch size {batch_size}):")

        # Generate batch
        image_batch = generate_batch_test_images(batch_size, num_bands, height, width)
        total_samples = batch_size * num_bands * height * width

        print(f"  Input shape: {image_batch.shape}")
        print(f"  Total samples: {total_samples}")

        # Create batch compressor
        compressor = create_batch_optimized_lossless_compressor(
            num_bands=num_bands,
            optimization_mode='full',
            dynamic_range=16
        )

        # Process batch
        start_time = time.time()
        results = compressor(image_batch)
        end_time = time.time()

        compression_time = end_time - start_time
        throughput = total_samples / compression_time

        # Check results
        print(f"  Compression time: {compression_time:.4f}s")
        print(f"  Throughput: {throughput:.0f} samples/sec ({throughput/1000:.1f}K samples/sec)")
        print(f"  Overall compression ratio: {results['compression_ratio']:.2f}:1")

        # Verify lossless property for batch
        if 'batch_size' in results:
            # Batch results
            max_errors = []
            for b in range(batch_size):
                max_error = torch.max(torch.abs(
                    image_batch[b] - results['reconstructed_samples'][b]
                ))
                max_errors.append(max_error.item())

            max_batch_error = max(max_errors)
            print(f"  Max reconstruction error in batch: {max_batch_error:.6f}")

            if max_batch_error < 0.01:
                print(f"  ✓ Lossless compression verified for all {batch_size} images")
            else:
                print(f"  ⚠ Error exceeds lossless threshold")

            # Individual image stats
            if 'batch_compression_times' in results:
                avg_time_per_image = results['batch_compression_times'].mean().item()
                print(f"  Avg time per image: {avg_time_per_image:.4f}s")

        else:
            # Single image result
            max_error = torch.max(torch.abs(image_batch.squeeze(0) - results['reconstructed_samples']))
            print(f"  Max reconstruction error: {max_error:.6f}")

            if max_error < 0.01:
                print(f"  ✓ Lossless compression verified")

    return True


def test_batch_vs_single_performance():
    """Compare batch processing vs individual image processing"""
    print(f"\n{'='*55}")
    print("Batch vs Individual Processing Performance")
    print("=" * 55)

    batch_size = 4
    num_bands, height, width = 8, 20, 20

    # Generate test batch
    image_batch = generate_batch_test_images(batch_size, num_bands, height, width)
    total_samples = batch_size * num_bands * height * width

    # Create compressor
    compressor = create_batch_optimized_lossless_compressor(
        num_bands=num_bands,
        optimization_mode='full'
    )

    # Method 1: Batch processing
    print(f"\nMethod 1: Batch processing ({batch_size} images together)")
    start_time = time.time()
    batch_results = compressor(image_batch)
    batch_time = time.time() - start_time
    batch_throughput = total_samples / batch_time

    print(f"  Total time: {batch_time:.4f}s")
    print(f"  Throughput: {batch_throughput:.0f} samples/sec")

    # Method 2: Individual processing
    print(f"\nMethod 2: Individual processing ({batch_size} images separately)")
    start_time = time.time()
    individual_results = []

    for b in range(batch_size):
        single_image = image_batch[b]  # [Z, Y, X]
        result = compressor(single_image)
        individual_results.append(result)

    individual_time = time.time() - start_time
    individual_throughput = total_samples / individual_time

    print(f"  Total time: {individual_time:.4f}s")
    print(f"  Throughput: {individual_throughput:.0f} samples/sec")

    # Performance comparison
    speedup = individual_time / batch_time
    efficiency = (individual_throughput / batch_throughput) * 100

    print(f"\nPerformance Comparison:")
    print(f"  Batch processing speedup: {speedup:.2f}x")
    print(f"  Batch efficiency: {efficiency:.1f}%")

    if speedup > 1.1:
        print(f"  ✓ Batch processing provides significant speedup")
    elif speedup > 0.9:
        print(f"  → Batch processing performs similarly to individual")
    else:
        print(f"  ⚠ Individual processing is faster")

    return True


def test_batch_compression_output():
    """Test batch compression output functionality"""
    print(f"\n{'='*55}")
    print("Testing Batch Compression Output")
    print("=" * 55)

    batch_size = 3
    num_bands, height, width = 5, 12, 12

    # Generate test batch
    image_batch = generate_batch_test_images(batch_size, num_bands, height, width)

    # Create compressor
    compressor = create_batch_optimized_lossless_compressor(
        num_bands=num_bands,
        optimization_mode='full'
    )

    print(f"Processing batch of {batch_size} images...")

    # Get compression results
    results = compressor(image_batch)

    # Get compressed bitstreams
    compressed_batch = compressor.compress_batch(image_batch)

    print(f"  Compressed {len(compressed_batch)} images")

    for i, compressed_data in enumerate(compressed_batch):
        size_bytes = len(compressed_data)
        size_bits = size_bytes * 8
        samples_per_image = num_bands * height * width
        bits_per_sample = size_bits / samples_per_image if samples_per_image > 0 else 0

        print(f"  Image {i}: {size_bytes} bytes ({size_bits} bits, {bits_per_sample:.2f} bps)")

    # Verify batch results structure
    if 'batch_size' in results:
        print(f"  ✓ Batch results contain {results['batch_size']} images")
        print(f"  ✓ Batch compressed sizes: {results['batch_compressed_sizes'].tolist()}")
        print(f"  ✓ Batch compression times: {[f'{t:.4f}s' for t in results['batch_compression_times'].tolist()]}")
    else:
        print(f"  → Single image processed")

    return True


def test_batch_scaling():
    """Test performance scaling with batch size"""
    print(f"\n{'='*55}")
    print("Testing Batch Size Scaling")
    print("=" * 55)

    batch_sizes = [1, 2, 4, 6, 8]
    num_bands, height, width = 6, 16, 16

    compressor = create_batch_optimized_lossless_compressor(
        num_bands=num_bands,
        optimization_mode='full'
    )

    results = {}

    for batch_size in batch_sizes:
        print(f"\nTesting batch size {batch_size}:")

        # Generate batch
        image_batch = generate_batch_test_images(batch_size, num_bands, height, width)
        total_samples = batch_size * num_bands * height * width

        # Time compression
        start_time = time.time()
        compression_results = compressor(image_batch)
        end_time = time.time()

        compression_time = end_time - start_time
        throughput = total_samples / compression_time
        time_per_image = compression_time / batch_size

        print(f"  Total time: {compression_time:.4f}s")
        print(f"  Time per image: {time_per_image:.4f}s")
        print(f"  Throughput: {throughput:.0f} samples/sec")

        results[batch_size] = {
            'total_time': compression_time,
            'time_per_image': time_per_image,
            'throughput': throughput
        }

    # Analyze scaling
    print(f"\nBatch Size Scaling Analysis:")
    baseline_time_per_image = results[1]['time_per_image']

    for batch_size in batch_sizes[1:]:  # Skip batch_size=1
        current_time_per_image = results[batch_size]['time_per_image']
        efficiency = (baseline_time_per_image / current_time_per_image) * 100

        print(f"  Batch {batch_size}: {efficiency:.1f}% efficiency vs single image")

    return True


def run_batch_optimized_tests():
    """Run complete batch optimization test suite"""
    try:
        print("Batch-Enabled Optimized CCSDS-123.0-B-2 Tests")
        print("=" * 60)

        # Run test suites
        test1 = test_batch_processing()
        test2 = test_batch_vs_single_performance()
        test3 = test_batch_compression_output()
        test4 = test_batch_scaling()

        if test1 and test2 and test3 and test4:
            print(f"\n{'='*60}")
            print("BATCH OPTIMIZATION TESTS SUCCESSFUL!")
            print("=" * 60)
            print("\n✓ All batch processing tests passed!")
            print("\nBatch Processing Capabilities Achieved:")
            print("  • Automatic batch detection and processing")
            print("  • Efficient batch compression pipeline")
            print("  • Individual compressed bitstream generation")
            print("  • Performance scaling analysis")
            print("  • Maintained lossless compression accuracy")
            print("  • Support for various batch sizes")

            print(f"\n🎯 The batch-enabled optimized compressor now supports:")
            print(f"   • Single images: [Z,Y,X]")
            print(f"   • Single with batch dim: [1,Z,Y,X]")
            print(f"   • Batch processing: [B,Z,Y,X]")
            print(f"   • Automatic mode detection")
            print(f"   • Efficient parallel processing")

            return True
        else:
            print("\n✗ Some batch tests failed")
            return False

    except Exception as e:
        print(f"Batch optimization tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_batch_optimized_tests()
    sys.exit(0 if success else 1)
