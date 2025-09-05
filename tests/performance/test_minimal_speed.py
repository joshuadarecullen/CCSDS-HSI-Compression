#!/usr/bin/env python3
"""
Minimal Speed Test - Basic verification of GPU optimizations

Tests just the core optimization modes with very small images for quick verification.
"""

import torch
import time
import numpy as np

def test_basic_functionality():
    """Test basic functionality of all compressor variants"""
    
    print("Minimal CCSDS Speed Test")
    print("=" * 30)
    
    # Very small test image for quick testing
    num_bands, height, width = 10, 16, 16
    test_image = torch.randn(num_bands, height, width) * 100
    
    print(f"Test image: {num_bands}×{height}×{width}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    results = {}
    
    # Test 1: CPU Optimized
    print("1. Testing CPU Optimized...")
    try:
        from optimized_compressor import create_optimized_lossless_compressor
        
        compressor = create_optimized_lossless_compressor(
            num_bands=num_bands,
            optimization_mode='full',
            device='cpu'
        )
        
        start_time = time.time()
        result = compressor(test_image)
        cpu_time = time.time() - start_time
        
        print(f"   Time: {cpu_time:.3f}s")
        print(f"   Throughput: {(num_bands*height*width)/cpu_time/1e3:.1f} KSamp/s")
        print(f"   Compression Ratio: {result['compression_ratio']:.2f}:1")
        
        results['CPU Optimized'] = cpu_time
        
    except Exception as e:
        print(f"   FAILED: {e}")
    
    print()
    
    # Test 2: GPU Optimized (if available)
    if torch.cuda.is_available():
        print("2. Testing GPU Optimized...")
        try:
            gpu_compressor = create_optimized_lossless_compressor(
                num_bands=num_bands,
                optimization_mode='full',
                device='cuda'
            )
            
            # Warmup
            _ = gpu_compressor(test_image)
            torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            result = gpu_compressor(test_image)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"   Time: {gpu_time:.3f}s")
            print(f"   Throughput: {(num_bands*height*width)/gpu_time/1e3:.1f} KSamp/s")
            print(f"   Compression Ratio: {result['compression_ratio']:.2f}:1")
            
            results['GPU Optimized'] = gpu_time
            
            if 'CPU Optimized' in results:
                speedup = results['CPU Optimized'] / gpu_time
                print(f"   GPU Speedup: {speedup:.1f}x")
            
        except Exception as e:
            print(f"   FAILED: {e}")
        
        print()
        
        # Test 3: GPU + Mixed Precision
        print("3. Testing GPU + Mixed Precision...")
        try:
            mixed_compressor = create_optimized_lossless_compressor(
                num_bands=num_bands,
                optimization_mode='full',
                device='cuda',
                use_mixed_precision=True
            )
            
            # Warmup
            _ = mixed_compressor(test_image)
            torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            result = mixed_compressor(test_image)
            torch.cuda.synchronize()
            mixed_time = time.time() - start_time
            
            print(f"   Time: {mixed_time:.3f}s")
            print(f"   Throughput: {(num_bands*height*width)/mixed_time/1e3:.1f} KSamp/s")
            print(f"   Compression Ratio: {result['compression_ratio']:.2f}:1")
            
            results['GPU + Mixed Precision'] = mixed_time
            
            if 'GPU Optimized' in results:
                mixed_speedup = results['GPU Optimized'] / mixed_time
                print(f"   Mixed Precision Speedup: {mixed_speedup:.1f}x")
            
        except Exception as e:
            print(f"   FAILED: {e}")
    
    else:
        print("2. GPU not available - skipping GPU tests")
    
    print()
    
    # Summary
    print("=" * 30)
    print("SUMMARY")
    print("=" * 30)
    
    if results:
        fastest = min(results.items(), key=lambda x: x[1])
        print(f"Fastest: {fastest[0]} ({fastest[1]:.3f}s)")
        
        for name, time_val in sorted(results.items(), key=lambda x: x[1]):
            speedup = fastest[1] / time_val if time_val > 0 else 1.0
            print(f"{name}: {time_val:.3f}s ({speedup:.1f}x)")
    else:
        print("No successful tests!")
    
    print("\nGPU optimization test complete!")


if __name__ == "__main__":
    test_basic_functionality()