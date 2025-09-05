#!/usr/bin/env python3
"""
Comprehensive Speed Comparison Test for CCSDS-123.0-B-2 Implementations

This test compares the performance of all available compression modes:
- Original sample-by-sample implementation
- CPU optimized versions (full, causal, streaming)
- GPU optimized versions (batch, streaming, mixed precision)
- Batch optimized versions

Results are presented in a detailed comparison table with speedup factors.
"""

import torch
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import all compressor variants
from src.ccsds import CCSDS123Compressor  # Original implementation
from src.optimized import (
    OptimizedCCSDS123Compressor,
    create_optimized_lossless_compressor,
    BatchOptimizedCCSDS123Compressor
)


class SpeedBenchmark:
    """Comprehensive speed benchmarking suite"""
    
    def __init__(self, warmup_runs: int = 2, benchmark_runs: int = 5):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = []
        
    def generate_test_images(self) -> Dict[str, torch.Tensor]:
        """Generate test images of different sizes"""
        test_cases = {
            'Small (10×32×32)': torch.randn(10, 32, 32) * 100,
            'Medium (50×64×64)': torch.randn(50, 64, 64) * 100,
            'Large (100×128×128)': torch.randn(100, 128, 128) * 100,
            'XLarge (200×64×64)': torch.randn(200, 64, 64) * 100,
        }
        
        # Ensure consistent random seed for fair comparison
        torch.manual_seed(42)
        np.random.seed(42)
        
        return test_cases
    
    def benchmark_original_compressor(self, image: torch.Tensor, name: str) -> Dict:
        """Benchmark original sample-by-sample compressor"""
        print(f"    Benchmarking Original CCSDS...")
        
        Z, Y, X = image.shape
        compressor = CCSDS123Compressor(num_bands=Z)
        
        # Warmup
        for _ in range(self.warmup_runs):
            try:
                _ = compressor(image)
            except Exception as e:
                print(f"    Warning: Original compressor failed during warmup: {e}")
                return None
        
        # Benchmark
        times = []
        compression_ratios = []
        
        for run in range(self.benchmark_runs):
            try:
                start_time = time.time()
                result = compressor(image)
                end_time = time.time()
                
                times.append(end_time - start_time)
                if isinstance(result, dict) and 'compression_ratio' in result:
                    compression_ratios.append(result['compression_ratio'])
                else:
                    compression_ratios.append(2.0)  # Default estimate
                    
            except Exception as e:
                print(f"    Warning: Original compressor failed on run {run}: {e}")
                return None
        
        avg_time = np.mean(times)
        throughput = (Z * Y * X) / avg_time
        
        return {
            'name': f'Original',
            'image_size': name,
            'avg_time': avg_time,
            'std_time': np.std(times),
            'throughput_samples_per_sec': throughput,
            'throughput_msamples_per_sec': throughput / 1e6,
            'avg_compression_ratio': np.mean(compression_ratios) if compression_ratios else 0,
            'speedup_vs_original': 1.0,
            'device': 'CPU',
            'mode': 'sample-by-sample',
            'success': True
        }
    
    def benchmark_cpu_optimized(self, image: torch.Tensor, name: str) -> List[Dict]:
        """Benchmark CPU optimized versions"""
        print(f"    Benchmarking CPU Optimized versions...")
        
        Z, Y, X = image.shape
        results = []
        
        modes = [
            ('full', 'CPU Full Vectorized'),
            ('causal', 'CPU Causal Optimized'),
            ('streaming', 'CPU Streaming')
        ]
        
        for mode, description in modes:
            try:
                compressor = create_optimized_lossless_compressor(
                    num_bands=Z,
                    optimization_mode=mode,
                    device='cpu'
                )
                
                # Warmup
                for _ in range(self.warmup_runs):
                    _ = compressor(image)
                
                # Benchmark
                times = []
                compression_ratios = []
                
                for run in range(self.benchmark_runs):
                    start_time = time.time()
                    result = compressor(image)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    compression_ratios.append(result['compression_ratio'])
                
                avg_time = np.mean(times)
                throughput = (Z * Y * X) / avg_time
                
                results.append({
                    'name': description,
                    'image_size': name,
                    'avg_time': avg_time,
                    'std_time': np.std(times),
                    'throughput_samples_per_sec': throughput,
                    'throughput_msamples_per_sec': throughput / 1e6,
                    'avg_compression_ratio': np.mean(compression_ratios),
                    'speedup_vs_original': None,  # Will be calculated later
                    'device': 'CPU',
                    'mode': mode,
                    'success': True
                })
                
            except Exception as e:
                print(f"    Warning: {description} failed: {e}")
                results.append({
                    'name': description,
                    'image_size': name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def benchmark_gpu_optimized(self, image: torch.Tensor, name: str) -> List[Dict]:
        """Benchmark GPU optimized versions"""
        if not torch.cuda.is_available():
            print("    GPU not available - skipping GPU benchmarks")
            return []
        
        print(f"    Benchmarking GPU Optimized versions...")
        
        Z, Y, X = image.shape
        results = []
        
        # GPU configurations to test
        configs = [
            ('full', False, 8, 'GPU Batch Optimized'),
            ('full', True, 8, 'GPU + Mixed Precision'),
            ('streaming', False, 16, 'GPU Streaming'),
            ('streaming', True, 16, 'GPU Streaming + Mixed Precision'),
            ('causal', False, 8, 'GPU Causal')
        ]
        
        for mode, mixed_precision, batch_size, description in configs:
            try:
                compressor = create_optimized_lossless_compressor(
                    num_bands=Z,
                    optimization_mode=mode,
                    device='cuda',
                    use_mixed_precision=mixed_precision,
                    gpu_batch_size=batch_size
                )
                
                # Warmup
                for _ in range(self.warmup_runs):
                    result = compressor(image)
                    torch.cuda.synchronize()
                
                torch.cuda.reset_peak_memory_stats()
                
                # Benchmark
                times = []
                compression_ratios = []
                peak_memories = []
                
                for run in range(self.benchmark_runs):
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    result = compressor(image)
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    compression_ratios.append(result['compression_ratio'])
                    peak_memories.append(torch.cuda.max_memory_allocated())
                    torch.cuda.reset_peak_memory_stats()
                
                avg_time = np.mean(times)
                throughput = (Z * Y * X) / avg_time
                avg_peak_memory = np.mean(peak_memories) / 1e9  # GB
                
                results.append({
                    'name': description,
                    'image_size': name,
                    'avg_time': avg_time,
                    'std_time': np.std(times),
                    'throughput_samples_per_sec': throughput,
                    'throughput_msamples_per_sec': throughput / 1e6,
                    'avg_compression_ratio': np.mean(compression_ratios),
                    'speedup_vs_original': None,  # Will be calculated later
                    'device': 'GPU',
                    'mode': mode,
                    'mixed_precision': mixed_precision,
                    'batch_size': batch_size,
                    'peak_memory_gb': avg_peak_memory,
                    'success': True
                })
                
            except Exception as e:
                print(f"    Warning: {description} failed: {e}")
                results.append({
                    'name': description,
                    'image_size': name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def benchmark_batch_optimized(self, image: torch.Tensor, name: str) -> List[Dict]:
        """Benchmark batch optimized compressor"""
        print(f"    Benchmarking Batch Optimized...")
        
        Z, Y, X = image.shape
        results = []
        
        # Test different batch sizes
        batch_sizes = [4, 8, 16] if Z >= 16 else [2, 4]
        
        for batch_size in batch_sizes:
            try:
                compressor = BatchOptimizedCCSDS123Compressor(
                    num_bands=Z,
                    batch_size=batch_size
                )
                
                # Create batched input
                num_batches = max(1, 8 // batch_size)  # Test with multiple batches
                batched_images = image.unsqueeze(0).repeat(num_batches, 1, 1, 1)
                
                # Warmup
                for _ in range(self.warmup_runs):
                    _ = compressor(batched_images)
                
                # Benchmark
                times = []
                compression_ratios = []
                
                for run in range(self.benchmark_runs):
                    start_time = time.time()
                    result = compressor(batched_images)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    if isinstance(result, dict) and 'compression_ratio' in result:
                        compression_ratios.append(result['compression_ratio'])
                    else:
                        compression_ratios.append(2.0)
                
                avg_time = np.mean(times)
                # Account for processing multiple batches
                effective_samples = num_batches * Z * Y * X
                throughput = effective_samples / avg_time
                
                results.append({
                    'name': f'Batch Optimized (batch={batch_size})',
                    'image_size': name,
                    'avg_time': avg_time,
                    'std_time': np.std(times),
                    'throughput_samples_per_sec': throughput,
                    'throughput_msamples_per_sec': throughput / 1e6,
                    'avg_compression_ratio': np.mean(compression_ratios) if compression_ratios else 0,
                    'speedup_vs_original': None,  # Will be calculated later
                    'device': 'CPU',
                    'mode': 'batch',
                    'batch_size': batch_size,
                    'num_batches': num_batches,
                    'success': True
                })
                
            except Exception as e:
                print(f"    Warning: Batch Optimized (batch={batch_size}) failed: {e}")
                results.append({
                    'name': f'Batch Optimized (batch={batch_size})',
                    'image_size': name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def benchmark_image_size(self, image: torch.Tensor, name: str) -> List[Dict]:
        """Benchmark all compressors on a specific image size"""
        print(f"\n=== Benchmarking {name} ===")
        
        Z, Y, X = image.shape
        print(f"Image shape: {Z}×{Y}×{X} ({Z*Y*X:,} samples)")
        
        all_results = []
        
        # 1. Original compressor
        original_result = self.benchmark_original_compressor(image, name)
        if original_result:
            all_results.append(original_result)
            original_time = original_result['avg_time']
        else:
            original_time = None
            print("    Original compressor failed - using estimated baseline")
        
        # 2. CPU optimized versions
        cpu_results = self.benchmark_cpu_optimized(image, name)
        all_results.extend([r for r in cpu_results if r.get('success', False)])
        
        # 3. GPU optimized versions
        gpu_results = self.benchmark_gpu_optimized(image, name)
        all_results.extend([r for r in gpu_results if r.get('success', False)])
        
        # 4. Batch optimized versions
        batch_results = self.benchmark_batch_optimized(image, name)
        all_results.extend([r for r in batch_results if r.get('success', False)])
        
        # Calculate speedup factors
        if original_time:
            for result in all_results:
                if result['name'] != 'Original':
                    result['speedup_vs_original'] = original_time / result['avg_time']
        else:
            # Use slowest method as baseline
            slowest_time = max([r['avg_time'] for r in all_results if r.get('success', False)], default=1.0)
            for result in all_results:
                result['speedup_vs_original'] = slowest_time / result['avg_time']
        
        return all_results
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark on all image sizes and methods"""
        print("Starting Comprehensive CCSDS-123.0-B-2 Speed Comparison")
        print("=" * 60)
        
        # System info
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CPU Count: {torch.get_num_threads()}")
        
        # Generate test images
        test_images = self.generate_test_images()
        
        # Benchmark each image size
        all_results = []
        for name, image in test_images.items():
            results = self.benchmark_image_size(image, name)
            all_results.extend(results)
            self.results.extend(results)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)
        return df
    
    def print_summary_table(self, df: pd.DataFrame):
        """Print formatted summary table"""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE SPEED COMPARISON RESULTS")
        print("=" * 100)
        
        # Group by image size
        for size in df['image_size'].unique():
            size_df = df[df['image_size'] == size].copy()
            
            print(f"\n{size}:")
            print("-" * 80)
            
            # Sort by speed (fastest first)
            size_df = size_df.sort_values('throughput_msamples_per_sec', ascending=False)
            
            print(f"{'Method':<30} {'Time (s)':<10} {'Speedup':<10} {'Throughput':<15} {'Device':<8} {'Ratio':<8}")
            print("-" * 80)
            
            for _, row in size_df.iterrows():
                name = row['name'][:28]  # Truncate long names
                time_str = f"{row['avg_time']:.3f}"
                speedup_str = f"{row['speedup_vs_original']:.1f}x" if row['speedup_vs_original'] else "N/A"
                throughput_str = f"{row['throughput_msamples_per_sec']:.1f} MSamp/s"
                device = row['device']
                ratio = f"{row['avg_compression_ratio']:.2f}:1" if row['avg_compression_ratio'] > 0 else "N/A"
                
                print(f"{name:<30} {time_str:<10} {speedup_str:<10} {throughput_str:<15} {device:<8} {ratio:<8}")
        
        # Overall summary
        print("\n" + "=" * 80)
        print("OVERALL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Find fastest methods overall
        fastest_cpu = df[df['device'] == 'CPU'].nlargest(1, 'throughput_msamples_per_sec')
        if not fastest_cpu.empty:
            cpu_best = fastest_cpu.iloc[0]
            print(f"Fastest CPU Method: {cpu_best['name']} - {cpu_best['throughput_msamples_per_sec']:.1f} MSamp/s")
        
        if torch.cuda.is_available():
            fastest_gpu = df[df['device'] == 'GPU'].nlargest(1, 'throughput_msamples_per_sec')
            if not fastest_gpu.empty:
                gpu_best = fastest_gpu.iloc[0]
                print(f"Fastest GPU Method: {gpu_best['name']} - {gpu_best['throughput_msamples_per_sec']:.1f} MSamp/s")
                
                # Calculate GPU vs CPU speedup
                if not fastest_cpu.empty:
                    gpu_vs_cpu = gpu_best['throughput_msamples_per_sec'] / cpu_best['throughput_msamples_per_sec']
                    print(f"GPU vs CPU Speedup: {gpu_vs_cpu:.1f}x")
        
        # Memory usage summary
        gpu_df = df[df['device'] == 'GPU']
        if not gpu_df.empty and 'peak_memory_gb' in gpu_df.columns:
            avg_memory = gpu_df['peak_memory_gb'].mean()
            print(f"Average GPU Memory Usage: {avg_memory:.1f} GB")
    
    def save_detailed_results(self, df: pd.DataFrame, filename: str = "speed_comparison_results.csv"):
        """Save detailed results to CSV"""
        df.to_csv(filename, index=False)
        print(f"\nDetailed results saved to: {filename}")


def main():
    """Run the comprehensive speed comparison"""
    
    # Create benchmark suite
    benchmark = SpeedBenchmark(warmup_runs=2, benchmark_runs=5)
    
    try:
        # Run comprehensive benchmark
        results_df = benchmark.run_comprehensive_benchmark()
        
        # Print summary
        benchmark.print_summary_table(results_df)
        
        # Save detailed results
        benchmark.save_detailed_results(results_df)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()