import src
import torch

from src import create_lossless_compressor
from src.optimized.optimized_compressor import create_optimized_lossless_compressor

# Create test image [Z, Y, X]
image = torch.randn(50, 64, 64) * 100  # 50 bands, 64x64 pixels

# Create optimized compressor
compressor = create_optimized_lossless_compressor(
    num_bands=50, 
    optimization_mode='full'  # Fastest mode
)

# Compress (much faster than original)
results = compressor(image)
print(f"Result keys: {results.keys()}")
print(f"Compression time: {results['compression_time']:.3f}s")
print(f"Throughput: {results['throughput_samples_per_sec']:.0f} samples/sec")
print(f"Compression ratio: {results['compression_ratio']:.2f}:1")
