#!/usr/bin/env python3
"""
Quick Speed Test for CCSDS-123.0-B-2 Implementations

A lightweight version of the comprehensive speed test for quick development testing.
Tests only the main optimization modes with a single image size.
"""

import torch
import time
import numpy as np
from typing import Dict, List

# Import compressors
try:
    from ccsds_compressor import CCSDS123Compressor
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    print("Warning: Original CCSDS compressor not available")

from optimized_compressor import create_optimized_lossless_compressor

try:
    from batch_optimized_compressor import BatchOptimizedCCSDS123Compressor
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False
    print("Warning: Batch optimized compressor not available")


def quick_benchmark(num_runs: int = 3) -> List[Dict]:
    """Run quick benchmark with core optimization modes"""
    
    print("Quick CCSDS-123.0-B-2 Speed Test")
    print("=" * 40)
    
    # Test parameters
    num_bands = 50
    height, width = 64, 64
    
    print(f"Test image: {num_bands}×{height}×{width} ({num_bands*height*width:,} samples)")
    print(f"Runs per test: {num_runs}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    # Generate test image
    torch.manual_seed(42)
    test_image = torch.randn(num_bands, height, width) * 100
    
    results = []
    
    # Test configurations: (name, create_func, warmup_needed)
    test_configs = []
    
    # 1. Original (if available)
    if ORIGINAL_AVAILABLE:
        test_configs.append((
            "Original Sample-by-Sample",
            lambda: CCSDS123Compressor(num_bands=num_bands),
            True
        ))
    
    # 2. CPU Optimized - Full
    test_configs.append((
        "CPU Optimized (Full)",
        lambda: create_optimized_lossless_compressor(
            num_bands=num_bands,
            optimization_mode='full',
            device='cpu'
        ),
        False
    ))
    
    # 3. CPU Optimized - Causal
    test_configs.append((
        "CPU Optimized (Causal)",
        lambda: create_optimized_lossless_compressor(
            num_bands=num_bands,
            optimization_mode='causal',
            device='cpu'
        ),
        False
    ))
    
    # 4. GPU Tests (if available)
    if torch.cuda.is_available():
        # GPU Full
        test_configs.append((
            "GPU Optimized",
            lambda: create_optimized_lossless_compressor(
                num_bands=num_bands,
                optimization_mode='full',
                device='cuda'
            ),
            True  # GPU needs warmup
        ))
        
        # GPU + Mixed Precision
        test_configs.append((
            "GPU + Mixed Precision",
            lambda: create_optimized_lossless_compressor(
                num_bands=num_bands,
                optimization_mode='full',
                device='cuda',
                use_mixed_precision=True
            ),
            True
        ))
        
        # GPU Streaming
        test_configs.append((
            "GPU Streaming",
            lambda: create_optimized_lossless_compressor(
                num_bands=num_bands,
                optimization_mode='streaming',
                device='cuda'
            ),
            True
        ))
    
    # 5. Batch Optimized (if available)
    if BATCH_AVAILABLE:
        test_configs.append((
            "Batch Optimized",
            lambda: BatchOptimizedCCSDS123Compressor(
                num_bands=num_bands,
                batch_size=8
            ),
            False
        ))
    
    # Run tests
    baseline_time = None
    
    for name, create_compressor, needs_warmup in test_configs:
        print(f"Testing {name}...")
        
        try:
            # Create compressor
            compressor = create_compressor()
            
            # Prepare input
            if "Batch" in name:
                # Batch compressor needs batched input
                input_image = test_image.unsqueeze(0)  # Add batch dimension
            else:
                input_image = test_image
            
            # Warmup
            if needs_warmup:
                try:
                    _ = compressor(input_image)
                    if torch.cuda.is_available() and 'GPU' in name:
                        torch.cuda.synchronize()
                except Exception as e:
                    print(f"  Warmup failed: {e}")
                    continue
            
            # Benchmark runs
            times = []
            compression_ratios = []
            
            for run in range(num_runs):
                # Clear GPU cache if needed
                if torch.cuda.is_available() and 'GPU' in name:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                start_time = time.time()
                result = compressor(input_image)
                
                if torch.cuda.is_available() and 'GPU' in name:
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                # Extract compression ratio
                if isinstance(result, dict) and 'compression_ratio' in result:
                    compression_ratios.append(result['compression_ratio'])
                else:
                    compression_ratios.append(2.0)  # Default estimate
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = (num_bands * height * width) / avg_time
            avg_ratio = np.mean(compression_ratios) if compression_ratios else 0
            
            # Set baseline
            if baseline_time is None:
                baseline_time = avg_time
            
            speedup = baseline_time / avg_time if baseline_time else 1.0
            
            result_dict = {
                'name': name,
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput_samples_per_sec': throughput,
                'throughput_msamples_per_sec': throughput / 1e6,
                'speedup': speedup,
                'compression_ratio': avg_ratio,
                'success': True
            }
            
            results.append(result_dict)
            
            # Print immediate result
            print(f"  Time: {avg_time:.3f}±{std_time:.3f}s")
            print(f"  Throughput: {throughput/1e6:.1f} MSamp/s")
            print(f"  Speedup: {speedup:.1f}x")
            print(f"  Compression: {avg_ratio:.2f}:1")
            print()
            
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'name': name,
                'success': False,
                'error': str(e)
            })
            print()
    
    return results


def print_summary(results: List[Dict]):
    """Print summary table"""
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("No successful tests to summarize!")
        return
    
    print("=" * 70)
    print("QUICK SPEED TEST SUMMARY")
    print("=" * 70)
    
    # Sort by throughput (fastest first)
    successful_results.sort(key=lambda x: x['throughput_msamples_per_sec'], reverse=True)
    
    print(f"{'Method':<25} {'Time (s)':<10} {'Speedup':<10} {'Throughput':<15} {'Ratio':<10}")
    print("-" * 70)
    
    for result in successful_results:
        name = result['name'][:23]
        time_str = f"{result['avg_time']:.3f}"
        speedup_str = f"{result['speedup']:.1f}x"
        throughput_str = f"{result['throughput_msamples_per_sec']:.1f} MSamp/s"
        ratio_str = f"{result['compression_ratio']:.2f}:1"
        
        print(f"{name:<25} {time_str:<10} {speedup_str:<10} {throughput_str:<15} {ratio_str:<10}")
    
    print("-" * 70)
    
    # Highlight best performers
    fastest = successful_results[0]
    print(f"Fastest Method: {fastest['name']} ({fastest['throughput_msamples_per_sec']:.1f} MSamp/s)")
    
    # GPU vs CPU comparison
    gpu_results = [r for r in successful_results if 'GPU' in r['name']]
    cpu_results = [r for r in successful_results if 'GPU' not in r['name'] and 'Batch' not in r['name']]
    
    if gpu_results and cpu_results:
        best_gpu = max(gpu_results, key=lambda x: x['throughput_msamples_per_sec'])
        best_cpu = max(cpu_results, key=lambda x: x['throughput_msamples_per_sec'])
        
        gpu_vs_cpu_speedup = best_gpu['throughput_msamples_per_sec'] / best_cpu['throughput_msamples_per_sec']
        print(f"Best GPU vs Best CPU: {gpu_vs_cpu_speedup:.1f}x speedup")


def main():
    """Run quick speed test"""
    
    try:
        results = quick_benchmark()
        print_summary(results)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()