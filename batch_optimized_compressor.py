import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
import time

from optimized_compressor import OptimizedCCSDS123Compressor
from optimized_entropy_coder import encode_image_optimized, encode_image_streaming


class BatchOptimizedCCSDS123Compressor(OptimizedCCSDS123Compressor):
    """
    Batch-enabled Optimized CCSDS-123.0-B-2 Compressor

    Extends the optimized compressor to handle batches of images efficiently.
    Supports both single images and batches with automatic detection.
    """

    def _validate_input_batch(self, image: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Validate input and detect batch processing"""
        is_batch = False

        if image.dim() == 3:
            # Single image [Z, Y, X]
            Z, Y, X = image.shape
            if Z != self.num_bands:
                raise ValueError(f"Expected {self.num_bands} bands, got {Z}")

        elif image.dim() == 4 and image.shape[0] == 1:
            # Single image with batch dim [1, Z, Y, X]
            image = image.squeeze(0)
            Z, Y, X = image.shape
            if Z != self.num_bands:
                raise ValueError(f"Expected {self.num_bands} bands, got {Z}")

        elif image.dim() == 4:
            # Batch of images [B, Z, Y, X]
            B, Z, Y, X = image.shape
            is_batch = True
            if Z != self.num_bands:
                raise ValueError(f"Expected {self.num_bands} bands, got {Z}")

        else:
            raise ValueError(f"Expected 3D [Z,Y,X], 4D [1,Z,Y,X], or 4D [B,Z,Y,X] tensor, got {image.shape}")

        return image.float(), is_batch

    def forward_batch(self, image_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process batch of images

        Args:
            image_batch: [B, Z, Y, X] batch of images

        Returns:
            Batched results dictionary with [B, ...] tensors
        """
        B, Z, Y, X = image_batch.shape

        # Initialize batch results
        batch_predictions = torch.zeros_like(image_batch)
        batch_residuals = torch.zeros_like(image_batch)
        batch_quantized_residuals = torch.zeros_like(image_batch)
        batch_mapped_indices = torch.zeros(B, Z, Y, X, dtype=torch.long, device=image_batch.device)
        batch_sample_reps = torch.zeros_like(image_batch)
        batch_reconstructed = torch.zeros_like(image_batch)

        batch_compressed_sizes = torch.zeros(B)
        batch_compression_times = torch.zeros(B)
        batch_throughputs = torch.zeros(B)
        batch_entropy_stats = []

        total_start_time = time.time()

        # Process each image in batch
        for b in range(B):
            single_image = image_batch[b]  # [Z, Y, X]

            # Process single image
            start_time = time.time()

            if self.optimization_mode == 'full':
                results = self.forward_full_vectorized_single(single_image)
            elif self.optimization_mode == 'causal':
                results = self.forward_causal_optimized_single(single_image)
            elif self.optimization_mode == 'streaming':
                results = self.forward_streaming_single(single_image)
            else:
                raise ValueError(f"Unknown optimization mode: {self.optimization_mode}")

            end_time = time.time()

            # Store results
            batch_predictions[b] = results['predictions']
            batch_residuals[b] = results['residuals']
            batch_quantized_residuals[b] = results['quantized_residuals']
            batch_mapped_indices[b] = results['mapped_indices']
            batch_sample_reps[b] = results['sample_representatives']
            batch_reconstructed[b] = results['reconstructed_samples']

            batch_compressed_sizes[b] = results['compressed_size']
            batch_compression_times[b] = end_time - start_time
            batch_throughputs[b] = results['throughput_samples_per_sec']
            batch_entropy_stats.append(results.get('entropy_stats', {}))

        total_time = time.time() - total_start_time
        total_samples = B * Z * Y * X

        return {
            'predictions': batch_predictions,
            'residuals': batch_residuals,
            'quantized_residuals': batch_quantized_residuals,
            'mapped_indices': batch_mapped_indices,
            'sample_representatives': batch_sample_reps,
            'reconstructed_samples': batch_reconstructed,
            'compressed_size': batch_compressed_sizes.sum().item(),
            'original_size': B * Z * Y * X * self.dynamic_range,
            'compression_ratio': (B * Z * Y * X * self.dynamic_range) / max(batch_compressed_sizes.sum().item(), 1),
            'compression_time': total_time,
            'throughput_samples_per_sec': total_samples / total_time,
            'batch_size': B,
            'batch_compressed_sizes': batch_compressed_sizes,
            'batch_compression_times': batch_compression_times,
            'batch_throughputs': batch_throughputs,
            'batch_entropy_stats': batch_entropy_stats
        }

    def forward_full_vectorized_single(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process single image with full vectorization"""
        # Temporarily override _validate_input to handle single images
        original_validate = self._validate_input
        self._validate_input = lambda x: x.float()

        try:
            result = self.forward_full_vectorized(image)
            return result
        finally:
            self._validate_input = original_validate

    def forward_causal_optimized_single(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process single image with causal optimization"""
        original_validate = self._validate_input
        self._validate_input = lambda x: x.float()

        try:
            result = self.forward_causal_optimized(image)
            return result
        finally:
            self._validate_input = original_validate

    def forward_streaming_single(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process single image with streaming"""
        original_validate = self._validate_input
        self._validate_input = lambda x: x.float()

        try:
            result = self.forward_streaming(image)
            return result
        finally:
            self._validate_input = original_validate

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Main forward pass with automatic batch detection

        Args:
            image: [Z, Y, X] single image or [B, Z, Y, X] batch

        Returns:
            Results dictionary (batched if input was batched)
        """
        validated_image, is_batch = self._validate_input_batch(image)

        if is_batch:
            return self.forward_batch(validated_image)
        else:
            # Use parent class method for single images
            return super().forward(validated_image)

    def compress_batch(self, image_batch: torch.Tensor) -> List[bytes]:
        """
        Compress batch of images and return list of compressed bitstreams

        Args:
            image_batch: [B, Z, Y, X] batch of images

        Returns:
            List of compressed bitstreams (one per image)
        """
        results = self.forward(image_batch)

        if 'batch_size' not in results:
            # Single image case
            try:
                compressed_data, _ = encode_image_optimized(results['mapped_indices'], self.num_bands)
                return [compressed_data]
            except Exception as e:
                print(f"Warning: Batch entropy coding failed ({e})")
                return [b'']

        # Batch case
        B = results['batch_size']
        compressed_batch = []

        for b in range(B):
            try:
                single_mapped = results['mapped_indices'][b]  # [Z, Y, X]
                compressed_data, _ = encode_image_optimized(single_mapped, self.num_bands)
                compressed_batch.append(compressed_data)
            except Exception as e:
                print(f"Warning: Image {b} entropy coding failed ({e})")
                compressed_batch.append(b'')

        return compressed_batch

    def benchmark_batch(self, batch_sizes: List[int], image_size: Tuple[int, int, int], num_runs: int = 3) -> Dict:
        """
        Benchmark batch processing performance

        Args:
            batch_sizes: List of batch sizes to test
            image_size: (num_bands, height, width) for each image
            num_runs: Number of runs per batch size

        Returns:
            Benchmark results dictionary
        """
        num_bands, height, width = image_size
        results = {}

        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")

            # Generate test batch
            test_batch = torch.randn(batch_size, num_bands, height, width) * 100
            total_samples = batch_size * num_bands * height * width

            # Benchmark multiple runs
            times = []
            throughputs = []

            for run in range(num_runs):
                start_time = time.time()
                batch_results = self.forward(test_batch)
                end_time = time.time()

                compression_time = end_time - start_time
                throughput = total_samples / compression_time

                times.append(compression_time)
                throughputs.append(throughput)

            # Store results
            results[f"batch_{batch_size}"] = {
                'batch_size': batch_size,
                'total_samples': total_samples,
                'avg_time_seconds': np.mean(times),
                'std_time_seconds': np.std(times),
                'avg_throughput_samples_per_sec': np.mean(throughputs),
                'avg_throughput_megasamples_per_sec': np.mean(throughputs) / 1e6,
                'samples_per_sec_per_image': np.mean(throughputs) / batch_size,
            }

        return results


def create_batch_optimized_lossless_compressor(
    num_bands: int,
    optimization_mode: str = 'full',
    **kwargs
) -> BatchOptimizedCCSDS123Compressor:
    """
    Create batch-enabled optimized lossless CCSDS-123.0-B-2 compressor

    Args:
        num_bands: Number of spectral bands
        optimization_mode: 'full', 'causal', or 'streaming'
        **kwargs: Additional compressor parameters

    Returns:
        Configured batch-enabled optimized lossless compressor
    """
    return BatchOptimizedCCSDS123Compressor(
        num_bands=num_bands,
        lossless=True,
        optimization_mode=optimization_mode,
        **kwargs
    )


def create_batch_optimized_near_lossless_compressor(
    num_bands: int,
    absolute_error_limits: Optional[torch.Tensor] = None,
    relative_error_limits: Optional[torch.Tensor] = None,
    optimization_mode: str = 'full',
    **kwargs
) -> BatchOptimizedCCSDS123Compressor:
    """
    Create batch-enabled optimized near-lossless CCSDS-123.0-B-2 compressor

    Args:
        num_bands: Number of spectral bands
        absolute_error_limits: [num_bands] absolute error limits per band
        relative_error_limits: [num_bands] relative error limits per band
        optimization_mode: 'full', 'causal', or 'streaming'
        **kwargs: Additional compressor parameters

    Returns:
        Configured batch-enabled optimized near-lossless compressor
    """
    compressor = BatchOptimizedCCSDS123Compressor(
        num_bands=num_bands,
        lossless=False,
        optimization_mode=optimization_mode,
        **kwargs
    )

    # Set error limits
    if absolute_error_limits is None and relative_error_limits is None:
        absolute_error_limits = torch.ones(num_bands) * 2

    compressor.set_compression_parameters(
        absolute_error_limits=absolute_error_limits,
        relative_error_limits=relative_error_limits
    )

    return compressor
