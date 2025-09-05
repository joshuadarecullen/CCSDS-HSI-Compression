import torch
import torch.nn as nn
import numpy as np


class OptimizedUniformQuantizer(nn.Module):
    """
    Optimized CCSDS-123.0-B-2 Uniform Quantizer

    Vectorized implementation that processes entire bands/images at once
    instead of sample-by-sample processing.
    """

    def __init__(self, num_bands, dynamic_range=16):
        super().__init__()
        self.num_bands = num_bands
        self.dynamic_range = dynamic_range

        # Error limit parameters (vectorized for all bands)
        self.register_buffer('absolute_error_limits', torch.zeros(num_bands))
        self.register_buffer('relative_error_limits', torch.zeros(num_bands))

        self.error_limit_mode = 'absolute'

    def set_error_limits(self, absolute_limits=None, relative_limits=None):
        """Set error limits for all bands"""
        if absolute_limits is not None:
            self.absolute_error_limits = absolute_limits.clone()
            self.error_limit_mode = 'absolute'

        if relative_limits is not None:
            self.relative_error_limits = relative_limits.clone()
            if absolute_limits is not None:
                self.error_limit_mode = 'both'
            else:
                self.error_limit_mode = 'relative'

    def compute_max_errors_vectorized(self, predicted_values):
        """
        Compute maximum allowed error for all samples using vectorized operations

        Args:
            predicted_values: [Z, Y, X] tensor of predicted sample values

        Returns:
            max_errors: [Z, Y, X] tensor of maximum allowed errors
        """
        Z, Y, X = predicted_values.shape

        if self.error_limit_mode == 'absolute':
            # m_z(t) = a_z for all pixels in band z
            # Broadcast absolute limits: [Z] -> [Z, 1, 1] -> [Z, Y, X]
            max_errors = self.absolute_error_limits.view(Z, 1, 1).expand(Z, Y, X)

        elif self.error_limit_mode == 'relative':
            # m_z(t) = floor(r_z * |s_hat_z(t)| / 2^D)
            abs_pred = torch.abs(predicted_values)  # [Z, Y, X]
            # Broadcast relative limits: [Z] -> [Z, 1, 1] -> [Z, Y, X]
            relative_limits_expanded = self.relative_error_limits.view(Z, 1, 1).expand(Z, Y, X)
            max_errors = torch.floor(
                relative_limits_expanded * abs_pred / (2 ** self.dynamic_range)
            )

        else:  # both
            # m_z(t) = min(a_z, floor(r_z * |s_hat_z(t)| / 2^D))
            abs_pred = torch.abs(predicted_values)

            # Absolute component
            absolute_errors = self.absolute_error_limits.view(Z, 1, 1).expand(Z, Y, X)

            # Relative component
            relative_limits_expanded = self.relative_error_limits.view(Z, 1, 1).expand(Z, Y, X)
            relative_errors = torch.floor(
                relative_limits_expanded * abs_pred / (2 ** self.dynamic_range)
            )

            max_errors = torch.minimum(absolute_errors, relative_errors)

        return max_errors.long()

    def quantize_residuals_vectorized(self, residuals, predicted_values):
        """
        Quantize prediction residuals using vectorized operations

        Args:
            residuals: [Z, Y, X] tensor of prediction residuals
            predicted_values: [Z, Y, X] tensor of predicted values

        Returns:
            quantized_residuals: [Z, Y, X] tensor of quantized residuals
            quantizer_indices: [Z, Y, X] tensor of quantizer bin indices
        """
        # Compute maximum allowed error for all samples
        max_errors = self.compute_max_errors_vectorized(predicted_values)

        # Quantizer step size is 2*m_z(t) + 1 for all samples
        step_sizes = 2 * max_errors.float() + 1

        # Uniform quantization centered at predicted value
        # Handle case where step_size = 1 (lossless: max_error = 0)
        quantizer_indices = torch.where(
            step_sizes > 1,
            torch.round(residuals / step_sizes),
            residuals.round()  # Lossless case
        )

        quantized_residuals = torch.where(
            step_sizes > 1,
            quantizer_indices * step_sizes,
            residuals  # Lossless case
        )

        return quantized_residuals, quantizer_indices.long()

    def compute_reconstructed_samples_vectorized(self, quantized_residuals, predicted_values):
        """
        Compute reconstructed sample values using vectorized operations

        Args:
            quantized_residuals: [Z, Y, X] tensor of quantized residuals
            predicted_values: [Z, Y, X] tensor of predicted values

        Returns:
            reconstructed_samples: [Z, Y, X] tensor of reconstructed samples
        """
        # Reconstructed sample = predicted + quantized_residual  
        reconstructed = predicted_values + quantized_residuals

        # Clamp to valid dynamic range
        max_val = 2**(self.dynamic_range - 1) - 1
        min_val = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0

        return torch.clamp(reconstructed, min_val, max_val)

    def map_quantizer_indices_vectorized(self, quantizer_indices):
        """
        Map signed quantizer indices to unsigned integers using vectorized operations

        Args:
            quantizer_indices: [Z, Y, X] tensor of signed quantizer indices

        Returns:
            mapped_indices: [Z, Y, X] tensor of unsigned mapped indices
        """
        # Vectorized mapping: negative -> odd, non-negative -> even
        mapped = torch.where(
            quantizer_indices < 0,
            2 * torch.abs(quantizer_indices) - 1,  # Negative -> odd
            2 * quantizer_indices  # Non-negative -> even
        )

        return mapped.long()

    def unmap_quantizer_indices_vectorized(self, mapped_indices):
        """
        Inverse mapping from unsigned to signed quantizer indices

        Args:
            mapped_indices: [Z, Y, X] tensor of unsigned mapped indices

        Returns:
            quantizer_indices: [Z, Y, X] tensor of signed quantizer indices
        """
        # Vectorized inverse mapping
        quantizer_indices = torch.where(
            mapped_indices % 2 == 1,  # Odd -> negative
            -(mapped_indices + 1) // 2,
            mapped_indices // 2  # Even -> non-negative
        )

        return quantizer_indices.long()

    def forward_optimized(self, residuals, predicted_values):
        """
        Optimized quantization of all residuals at once

        Args:
            residuals: [Z, Y, X] tensor of prediction residuals
            predicted_values: [Z, Y, X] tensor of predicted values

        Returns:
            quantized_residuals: [Z, Y, X] quantized residuals
            mapped_indices: [Z, Y, X] mapped quantizer indices for encoding
            reconstructed_samples: [Z, Y, X] reconstructed sample values
        """
        # Vectorized quantization
        quantized_residuals, quantizer_indices = self.quantize_residuals_vectorized(
            residuals, predicted_values
        )

        # Vectorized mapping
        mapped_indices = self.map_quantizer_indices_vectorized(quantizer_indices)

        # Vectorized reconstruction
        reconstructed_samples = self.compute_reconstructed_samples_vectorized(
            quantized_residuals, predicted_values
        )

        return quantized_residuals, mapped_indices, reconstructed_samples

    def forward(self, residuals, predicted_values):
        """Use optimized vectorized version"""
        return self.forward_optimized(residuals, predicted_values)


class OptimizedLosslessQuantizer(OptimizedUniformQuantizer):
    """
    Optimized lossless quantizer (vectorized, all error limits = 0)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set all error limits to 0 for lossless compression
        self.absolute_error_limits.fill_(0.0)
        self.relative_error_limits.fill_(0.0)

    def forward_lossless_optimized(self, residuals, predicted_values):
        """
        Optimized lossless quantization (identity operation on residuals)

        Args:
            residuals: [Z, Y, X] tensor of prediction residuals
            predicted_values: [Z, Y, X] tensor of predicted values

        Returns:
            quantized_residuals: [Z, Y, X] unchanged residuals
            mapped_indices: [Z, Y, X] mapped indices for entropy coding
            reconstructed_samples: [Z, Y, X] perfect reconstruction
        """
        # Lossless: quantized residuals = original residuals (rounded to integers)
        quantized_residuals = residuals.round()
        quantizer_indices = quantized_residuals.long()

        # Map indices for entropy coding
        mapped_indices = self.map_quantizer_indices_vectorized(quantizer_indices)

        # Perfect reconstruction
        reconstructed_samples = predicted_values + quantized_residuals

        # Clamp to valid range
        max_val = 2**(self.dynamic_range - 1) - 1
        min_val = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0
        reconstructed_samples = torch.clamp(reconstructed_samples, min_val, max_val)

        return quantized_residuals, mapped_indices, reconstructed_samples

    def forward(self, residuals, predicted_values):
        """Use lossless optimized version"""
        return self.forward_lossless_optimized(residuals, predicted_values)


class BatchQuantizer:
    """
    Utility class for processing multiple images in batches
    """

    def __init__(self, quantizer):
        self.quantizer = quantizer

    def quantize_batch(self, residuals_batch, predictions_batch):
        """
        Process a batch of images

        Args:
            residuals_batch: [B, Z, Y, X] batch of residual images
            predictions_batch: [B, Z, Y, X] batch of prediction images

        Returns:
            quantized_batch: [B, Z, Y, X] quantized residuals
            mapped_batch: [B, Z, Y, X] mapped indices
            reconstructed_batch: [B, Z, Y, X] reconstructed samples
        """
        B = residuals_batch.shape[0]

        quantized_batch = []
        mapped_batch = []
        reconstructed_batch = []

        for b in range(B):
            quant_res, mapped_idx, reconstructed = self.quantizer(
                residuals_batch[b], predictions_batch[b]
            )
            quantized_batch.append(quant_res)
            mapped_batch.append(mapped_idx)
            reconstructed_batch.append(reconstructed)

        return (
            torch.stack(quantized_batch),
            torch.stack(mapped_batch),
            torch.stack(reconstructed_batch)
        )


class StreamingQuantizer:
    """
    Memory-efficient quantizer for very large images

    Processes images in chunks to reduce memory usage while
    maintaining vectorization benefits.
    """

    def __init__(self, quantizer, chunk_size=(8, 64, 64)):
        self.quantizer = quantizer
        self.chunk_size = chunk_size  # (Z_chunk, Y_chunk, X_chunk)

    def quantize_streaming(self, residuals, predicted_values):
        """
        Process large image in chunks

        Args:
            residuals: [Z, Y, X] very large residual image
            predicted_values: [Z, Y, X] very large prediction image

        Returns:
            Generator yielding (quantized_chunk, mapped_chunk, reconstructed_chunk)
        """
        Z, Y, X = residuals.shape
        Z_chunk, Y_chunk, X_chunk = self.chunk_size

        for z_start in range(0, Z, Z_chunk):
            for y_start in range(0, Y, Y_chunk):
                for x_start in range(0, X, X_chunk):
                    # Define chunk boundaries
                    z_end = min(z_start + Z_chunk, Z)
                    y_end = min(y_start + Y_chunk, Y)
                    x_end = min(x_start + X_chunk, X)

                    # Extract chunks
                    residual_chunk = residuals[z_start:z_end, y_start:y_end, x_start:x_end]
                    prediction_chunk = predicted_values[z_start:z_end, y_start:y_end, x_start:x_end]

                    # Process chunk
                    quant_chunk, mapped_chunk, reconstructed_chunk = self.quantizer(
                        residual_chunk, prediction_chunk
                    )

                    yield (
                        (z_start, y_start, x_start),  # Position
                        (quant_chunk, mapped_chunk, reconstructed_chunk)  # Results
                    )
