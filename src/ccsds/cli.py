#!/usr/bin/env python3
"""
Command Line Interface for CCSDS-123.0-B-2 Compressor

Provides command-line tools for compression, decompression, and benchmarking.
"""

import argparse
import torch
import numpy as np
import time
from pathlib import Path
from typing import Optional

from .ccsds_compressor import (
    create_lossless_compressor, 
    create_near_lossless_compressor,
    decompress,
    calculate_psnr,
    calculate_mssim,
    calculate_spectral_angle
)


def compress_cli():
    """Command-line interface for compression"""
    parser = argparse.ArgumentParser(description="CCSDS-123.0-B-2 Image Compressor")
    parser.add_argument("input", help="Input image file (.npy, .pt)")
    parser.add_argument("output", help="Output compressed file")
    parser.add_argument("--lossless", action="store_true", help="Use lossless compression")
    parser.add_argument("--error-limit", type=float, default=2.0, 
                        help="Absolute error limit for near-lossless compression")
    parser.add_argument("--num-bands", type=int, help="Number of spectral bands")
    parser.add_argument("--dynamic-range", type=int, default=16, 
                        help="Dynamic range in bits")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load input image
    input_path = Path(args.input)
    if input_path.suffix == '.npy':
        image = torch.from_numpy(np.load(input_path)).float()
    elif input_path.suffix == '.pt':
        image = torch.load(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")
    
    if args.verbose:
        print(f"Loaded image: {image.shape}")
        print(f"Data range: [{image.min():.2f}, {image.max():.2f}]")
    
    # Determine number of bands
    num_bands = args.num_bands or image.shape[0]
    
    # Create compressor
    if args.lossless:
        compressor = create_lossless_compressor(
            num_bands=num_bands,
            dynamic_range=args.dynamic_range
        )
        if args.verbose:
            print("Using lossless compression")
    else:
        error_limits = torch.ones(num_bands) * args.error_limit
        compressor = create_near_lossless_compressor(
            num_bands=num_bands,
            dynamic_range=args.dynamic_range,
            absolute_error_limits=error_limits
        )
        if args.verbose:
            print(f"Using near-lossless compression with error limit: {args.error_limit}")
    
    # Compress
    start_time = time.time()
    compressed_data = compressor.compress(image)
    compression_time = time.time() - start_time
    
    # Save compressed data
    torch.save(compressed_data, args.output)
    
    # Print statistics
    stats = compressed_data['compression_statistics']
    if args.verbose:
        print(f"Compression time: {compression_time:.2f} seconds")
        print(f"Original size: {stats['original_size_bits']} bits")
        print(f"Compressed size: {stats['compressed_size_bits']} bits")
        print(f"Compression ratio: {stats['compression_ratio']:.2f}:1")
        print(f"Bits per sample: {stats['bits_per_sample']:.2f}")
        
        if not args.lossless:
            print(f"PSNR: {stats['psnr_db']:.2f} dB")
            print(f"Max error: {stats['max_absolute_error']:.2f}")
    else:
        print(f"Compressed {args.input} -> {args.output}")
        print(f"Ratio: {stats['compression_ratio']:.2f}:1, Time: {compression_time:.2f}s")


def benchmark_cli():
    """Command-line interface for benchmarking"""
    parser = argparse.ArgumentParser(description="CCSDS-123.0-B-2 Benchmark Tool")
    parser.add_argument("--bands", type=int, default=10, help="Number of spectral bands")
    parser.add_argument("--height", type=int, default=64, help="Image height")
    parser.add_argument("--width", type=int, default=64, help="Image width")
    parser.add_argument("--dynamic-range", type=int, default=16, help="Dynamic range in bits")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--error-limits", type=float, nargs='+', default=[1, 2, 4], 
                        help="Error limits to test")
    
    args = parser.parse_args()
    
    # Generate test image
    print(f"Generating test image: {args.bands}x{args.height}x{args.width}")
    
    # Synthetic hyperspectral image with spectral correlation
    image = torch.zeros(args.bands, args.height, args.width)
    for z in range(args.bands):
        # Create spatially correlated data
        spatial = torch.randn(args.height, args.width) * 50
        # Add spectral correlation
        spectral_base = torch.sin(torch.tensor(z * 0.3)) * 100
        noise = torch.randn(args.height, args.width) * 10
        image[z] = spatial + spectral_base + noise
    
    # Clamp to valid range
    max_val = 2**(args.dynamic_range - 1) - 1
    min_val = -max_val if args.dynamic_range > 1 else 0
    image = torch.clamp(image, min_val, max_val)
    
    print(f"Image range: [{image.min():.2f}, {image.max():.2f}]")
    print()
    
    # Test lossless compression
    print("=== Lossless Compression ===")
    compressor = create_lossless_compressor(args.bands, args.dynamic_range)
    
    times = []
    for i in range(args.iterations):
        start_time = time.time()
        results = compressor.compress(image)
        times.append(time.time() - start_time)
    
    stats = results['compression_statistics']
    print(f"Compression ratio: {stats['compression_ratio']:.2f}:1")
    print(f"Bits per sample: {stats['bits_per_sample']:.2f}")
    print(f"Average time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print()
    
    # Test near-lossless compression
    print("=== Near-Lossless Compression ===")
    for error_limit in args.error_limits:
        print(f"Error limit: {error_limit}")
        
        error_limits = torch.ones(args.bands) * error_limit
        compressor = create_near_lossless_compressor(
            args.bands, args.dynamic_range,
            absolute_error_limits=error_limits
        )
        
        times = []
        for i in range(args.iterations):
            start_time = time.time()
            results = compressor.compress(image)
            times.append(time.time() - start_time)
        
        stats = results['compression_statistics']
        reconstructed = results['intermediate_data']['reconstructed_samples']
        
        # Calculate quality metrics
        psnr = calculate_psnr(image, reconstructed, args.dynamic_range)
        mssim = calculate_mssim(image, reconstructed)
        sam = calculate_spectral_angle(image, reconstructed)
        
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}:1")
        print(f"  Bits per sample: {stats['bits_per_sample']:.2f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  MSSIM: {mssim:.4f}")
        print(f"  SAM: {sam:.6f} rad ({sam*180/np.pi:.3f}°)")
        print(f"  Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_cli()
    else:
        compress_cli()