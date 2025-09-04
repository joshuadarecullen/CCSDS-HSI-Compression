#!/usr/bin/env python3
"""
GPU-Optimized CCSDS-123.0-B-2 Compression Example

This example demonstrates how to use the GPU-optimized version of the
CCSDS compressor for maximum performance on CUDA-enabled systems.
"""

import torch
import time
import numpy as np
from optimized_compressor import create_optimized_lossless_compressor

def benchmark_gpu_vs_cpu():
    """Compare GPU vs CPU performance"""
    
    print("=== GPU vs CPU Performance Comparison ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test parameters
    num_bands = 50
    height, width = 128, 128
    num_runs = 3
    
    # Create test image
    print(f"\nGenerating test image: {num_bands}×{height}×{width}")
    test_image = torch.randn(num_bands, height, width) * 100
    
    # CPU Benchmark
    print("\n--- CPU Benchmark ---")
    cpu_compressor = create_optimized_lossless_compressor(
        num_bands=num_bands,
        optimization_mode='full',
        device='cpu'
    )
    
    cpu_times = []
    for run in range(num_runs):
        start_time = time.time()
        cpu_result = cpu_compressor(test_image)
        end_time = time.time()
        cpu_times.append(end_time - start_time)
        print(f"Run {run+1}: {end_time - start_time:.3f}s")
    
    cpu_avg_time = np.mean(cpu_times)
    cpu_throughput = (num_bands * height * width) / cpu_avg_time
    
    # GPU Benchmark (if available)
    if torch.cuda.is_available():
        print("\n--- GPU Benchmark ---")
        gpu_compressor = create_optimized_lossless_compressor(
            num_bands=num_bands,
            optimization_mode='full',
            device='cuda',
            use_mixed_precision=True,
            gpu_batch_size=8
        )
        
        # Warmup
        print("GPU warmup...")
        _ = gpu_compressor(test_image)
        torch.cuda.synchronize()
        
        gpu_times = []
        for run in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            gpu_result = gpu_compressor(test_image)
            torch.cuda.synchronize()
            end_time = time.time()
            gpu_times.append(end_time - start_time)
            print(f"Run {run+1}: {end_time - start_time:.3f}s")
        
        gpu_avg_time = np.mean(gpu_times)
        gpu_throughput = (num_bands * height * width) / gpu_avg_time
        speedup = cpu_avg_time / gpu_avg_time
        
        print("\n=== Results ===")
        print(f"CPU Average Time:    {cpu_avg_time:.3f}s")
        print(f"GPU Average Time:    {gpu_avg_time:.3f}s")
        print(f"Speedup:            {speedup:.1f}x")
        print(f"CPU Throughput:     {cpu_throughput/1e6:.1f} Msamples/sec")
        print(f"GPU Throughput:     {gpu_throughput/1e6:.1f} Msamples/sec")
        
        # Verify results are similar
        cpu_ratio = cpu_result['compression_ratio']
        gpu_ratio = gpu_result['compression_ratio']
        print(f"CPU Compression Ratio: {cpu_ratio:.2f}:1")
        print(f"GPU Compression Ratio: {gpu_ratio:.2f}:1")
        print(f"Ratio Difference: {abs(cpu_ratio - gpu_ratio):.4f}")
    else:
        print("\nGPU not available - CPU results only")
        print(f"CPU Average Time:    {cpu_avg_time:.3f}s")
        print(f"CPU Throughput:     {cpu_throughput/1e6:.1f} Msamples/sec")

def demonstrate_gpu_features():
    """Demonstrate GPU-specific features"""
    
    if not torch.cuda.is_available():
        print("GPU not available - skipping GPU features demo")
        return
    
    print("\n=== GPU Features Demonstration ===")
    
    # Large image test
    num_bands = 100
    height, width = 256, 256
    
    print(f"Testing large image: {num_bands}×{height}×{width}")
    large_image = torch.randn(num_bands, height, width) * 100
    
    # Test different GPU modes
    modes = [
        ('full', 'GPU Batch Optimized'),
        ('streaming', 'GPU Streaming'),
        ('causal', 'GPU Causal (standards compliant)')
    ]
    
    for mode, description in modes:
        print(f"\n--- {description} ---")
        
        compressor = create_optimized_lossless_compressor(
            num_bands=num_bands,
            optimization_mode=mode,
            device='cuda',
            use_mixed_precision=True,
            gpu_batch_size=16  # Larger batch for big image
        )
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        result = compressor(large_image)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"Time: {end_time - start_time:.3f}s")
        print(f"Throughput: {result['throughput_samples_per_sec']/1e6:.1f} Msamples/sec")
        print(f"Peak GPU Memory: {peak_memory:.1f} GB")
        print(f"Compression Ratio: {result['compression_ratio']:.2f}:1")

def memory_efficiency_test():
    """Test memory efficiency with different batch sizes"""
    
    if not torch.cuda.is_available():
        print("GPU not available - skipping memory test")
        return
    
    print("\n=== Memory Efficiency Test ===")
    
    num_bands = 64
    height, width = 128, 128
    test_image = torch.randn(num_bands, height, width) * 100
    
    batch_sizes = [4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        compressor = create_optimized_lossless_compressor(
            num_bands=num_bands,
            optimization_mode='full',
            device='cuda',
            gpu_batch_size=batch_size
        )
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        start_time = time.time()
        result = compressor(test_image)
        torch.cuda.synchronize()
        end_time = time.time()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
        
        print(f"Time: {end_time - start_time:.3f}s")
        print(f"Peak Memory: {peak_memory:.0f} MB")
        print(f"Throughput: {result['throughput_samples_per_sec']/1e6:.1f} Msamples/sec")

if __name__ == "__main__":
    print("CCSDS-123.0-B-2 GPU Optimization Demo")
    print("=" * 50)
    
    # Run benchmarks
    benchmark_gpu_vs_cpu()
    
    # Demonstrate GPU features
    demonstrate_gpu_features()
    
    # Test memory efficiency
    memory_efficiency_test()
    
    print("\n" + "=" * 50)
    print("GPU optimization demo complete!")