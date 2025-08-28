import torch
import torch.nn as nn
import numpy as np


class UniformQuantizer(nn.Module):
    """
    CCSDS-123.0-B-2 Uniform Quantizer with closed-loop quantization
    
    Provides near-lossless compression capability with user-specified error bounds.
    Supports both absolute and relative error limits.
    """
    
    def __init__(self, num_bands, dynamic_range=16):
        super().__init__()
        self.num_bands = num_bands
        self.dynamic_range = dynamic_range
        
        # Error limit parameters (can be updated periodically)
        self.register_buffer('absolute_error_limits', torch.zeros(num_bands))
        self.register_buffer('relative_error_limits', torch.zeros(num_bands))
        
        # Track whether we're using absolute, relative, or both error limits
        self.error_limit_mode = 'absolute'  # 'absolute', 'relative', 'both'
        
    def set_error_limits(self, absolute_limits=None, relative_limits=None):
        """
        Set error limits for each band
        
        Args:
            absolute_limits: [num_bands] tensor of absolute error limits
            relative_limits: [num_bands] tensor of relative error limits  
        """
        if absolute_limits is not None:
            self.absolute_error_limits = absolute_limits.clone()
            self.error_limit_mode = 'absolute'
            
        if relative_limits is not None:
            self.relative_error_limits = relative_limits.clone()
            if absolute_limits is not None:
                self.error_limit_mode = 'both'
            else:
                self.error_limit_mode = 'relative'
    
    def compute_max_error(self, predicted_values, z):
        """
        Compute maximum allowed error for each sample based on error limits
        
        Args:
            predicted_values: [Y, X] tensor of predicted sample values for band z
            z: Band index
            
        Returns:
            max_errors: [Y, X] tensor of maximum allowed errors
        """
        if self.error_limit_mode == 'absolute':
            # m_z(t) = a_z
            max_errors = self.absolute_error_limits[z].expand_as(predicted_values)
            
        elif self.error_limit_mode == 'relative':
            # m_z(t) = floor(r_z * |s_hat_z(t)| / 2^D)
            abs_pred = torch.abs(predicted_values)
            max_errors = torch.floor(
                self.relative_error_limits[z] * abs_pred / (2 ** self.dynamic_range)
            )
            
        else:  # both
            # m_z(t) = min(a_z, floor(r_z * |s_hat_z(t)| / 2^D))
            abs_pred = torch.abs(predicted_values)
            relative_errors = torch.floor(
                self.relative_error_limits[z] * abs_pred / (2 ** self.dynamic_range)
            )
            absolute_errors = self.absolute_error_limits[z].expand_as(predicted_values)
            max_errors = torch.minimum(absolute_errors, relative_errors)
            
        return max_errors.long()
    
    def quantize_residuals(self, residuals, predicted_values, z):
        """
        Quantize prediction residuals using uniform quantizer
        
        Args:
            residuals: [Y, X] tensor of prediction residuals for band z
            predicted_values: [Y, X] tensor of predicted values for band z
            z: Band index
            
        Returns:
            quantized_residuals: [Y, X] tensor of quantized residuals
            quantizer_indices: [Y, X] tensor of quantizer bin indices
        """
        # Compute maximum allowed error for each sample
        max_errors = self.compute_max_error(predicted_values, z)
        
        # Quantizer step size is 2*m_z(t) + 1
        step_sizes = 2 * max_errors + 1
        
        # Uniform quantization centered at predicted value
        # q = round((residual) / step_size) * step_size
        quantizer_indices = torch.round(residuals / step_sizes.float())
        quantized_residuals = quantizer_indices * step_sizes.float()
        
        return quantized_residuals, quantizer_indices.long()
    
    def compute_reconstructed_samples(self, quantized_residuals, predicted_values):
        """
        Compute reconstructed sample values from quantized residuals
        
        Args:
            quantized_residuals: [Y, X] tensor of quantized residuals
            predicted_values: [Y, X] tensor of predicted values
            
        Returns:
            reconstructed_samples: [Y, X] tensor of reconstructed samples
        """
        # Reconstructed sample = predicted + quantized_residual  
        reconstructed = predicted_values + quantized_residuals
        
        # Clamp to valid dynamic range
        max_val = 2**(self.dynamic_range - 1) - 1
        min_val = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0
        
        return torch.clamp(reconstructed, min_val, max_val)
    
    def map_quantizer_indices(self, quantizer_indices):
        """
        Map signed quantizer indices to unsigned integers (similar to Issue 1)
        
        This mapping is invertible so the decompressor can reconstruct indices.
        
        Args:
            quantizer_indices: [Y, X] tensor of signed quantizer indices
            
        Returns:
            mapped_indices: [Y, X] tensor of unsigned mapped indices
        """
        # Simple mapping: negative values become odd, non-negative become even
        # -1 -> 1, -2 -> 3, -3 -> 5, ...
        #  0 -> 0,  1 -> 2,  2 -> 4, ...
        
        mapped = torch.where(
            quantizer_indices < 0,
            2 * torch.abs(quantizer_indices) - 1,  # Negative -> odd
            2 * quantizer_indices  # Non-negative -> even
        )
        
        return mapped.long()
    
    def unmap_quantizer_indices(self, mapped_indices):
        """
        Inverse mapping from unsigned to signed quantizer indices
        
        Args:
            mapped_indices: [Y, X] tensor of unsigned mapped indices
            
        Returns:
            quantizer_indices: [Y, X] tensor of signed quantizer indices
        """
        # Inverse of the mapping above
        quantizer_indices = torch.where(
            mapped_indices % 2 == 1,  # Odd -> negative
            -(mapped_indices + 1) // 2,
            mapped_indices // 2  # Even -> non-negative
        )
        
        return quantizer_indices.long()
    
    def forward(self, residuals, predicted_values):
        """
        Quantize prediction residuals for all bands
        
        Args:
            residuals: [Z, Y, X] tensor of prediction residuals
            predicted_values: [Z, Y, X] tensor of predicted values
            
        Returns:
            quantized_residuals: [Z, Y, X] quantized residuals
            mapped_indices: [Z, Y, X] mapped quantizer indices for encoding
            reconstructed_samples: [Z, Y, X] reconstructed sample values
        """
        Z, Y, X = residuals.shape
        quantized_residuals = torch.zeros_like(residuals)
        mapped_indices = torch.zeros_like(residuals, dtype=torch.long)
        reconstructed_samples = torch.zeros_like(residuals)
        
        for z in range(Z):
            # Quantize residuals for this band
            quant_res, quant_idx = self.quantize_residuals(
                residuals[z], predicted_values[z], z
            )
            quantized_residuals[z] = quant_res
            
            # Map indices for entropy coding
            mapped_idx = self.map_quantizer_indices(quant_idx)
            mapped_indices[z] = mapped_idx
            
            # Compute reconstructed samples
            reconstructed = self.compute_reconstructed_samples(
                quant_res, predicted_values[z]
            )
            reconstructed_samples[z] = reconstructed
            
        return quantized_residuals, mapped_indices, reconstructed_samples


class LosslessQuantizer(UniformQuantizer):
    """
    Lossless quantizer (special case where all error limits are 0)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set all error limits to 0 for lossless compression
        self.absolute_error_limits.fill_(0.0)
        self.relative_error_limits.fill_(0.0)
    
    def quantize_residuals(self, residuals, predicted_values, z):
        """
        For lossless compression, residuals are not quantized
        """
        # No quantization - residuals pass through unchanged
        quantizer_indices = residuals.round().long()
        quantized_residuals = quantizer_indices.float()
        
        return quantized_residuals, quantizer_indices


class PeriodicErrorLimitUpdater:
    """
    Utility class for periodic update of error limits during compression
    
    This allows adaptive adjustment of fidelity parameters to meet
    downlink rate constraints or provide higher fidelity in regions of interest.
    """
    
    def __init__(self, update_interval=1000):
        self.update_interval = update_interval
        self.update_count = 0
        
    def should_update(self, sample_count):
        """
        Check if error limits should be updated at this sample
        """
        return sample_count % self.update_interval == 0
    
    def update_error_limits(self, quantizer, new_absolute_limits=None, new_relative_limits=None):
        """
        Update error limits in the quantizer
        """
        quantizer.set_error_limits(new_absolute_limits, new_relative_limits)
        self.update_count += 1