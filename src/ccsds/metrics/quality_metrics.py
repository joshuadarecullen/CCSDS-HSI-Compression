"""
Quality Metrics for Image Compression Assessment

This module implements standard quality metrics used to evaluate
the performance of lossy and near-lossless image compression:

- PSNR (Peak Signal-to-Noise Ratio)
- MSSIM (Mean Structural Similarity Index Measure)  
- SAM (Spectral Angle Mapper)

All functions support callback functions for custom result handling.
"""

import torch
import numpy as np
import math
from typing import Optional, Callable


def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor,
                   dynamic_range: int, callback: Optional[Callable] = None) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between original and reconstructed images

    Mathematical Formula:
    PSNR = 20 * log10(MAX_VAL) - 10 * log10(MSE)
    Where MSE = (1/N) * Σ(original - reconstructed)²
    MAX_VAL = 2^(dynamic_range-1) - 1 for signed integers

    Args:
        original: Original image tensor [Z, Y, X]
        reconstructed: Reconstructed image tensor [Z, Y, X]
        dynamic_range: Bit depth of samples
        callback: Optional callback function called with (psnr_value, mse_value)

    Returns:
        PSNR value in decibels (dB)
    """
    mse = torch.mean((original - reconstructed) ** 2).item()

    if mse == 0:
        psnr = float('inf')
    else:
        max_val = 2 ** (dynamic_range - 1) - 1
        psnr = 20 * math.log10(max_val) - 10 * math.log10(mse)

    if callback is not None:
        callback(psnr, mse)

    return psnr


def calculate_mssim(original: torch.Tensor, reconstructed: torch.Tensor,
                    window_size: int = 11, callback: Optional[Callable] = None) -> float:
    """
    Calculate Mean Structural Similarity Index Measure (MSSIM) between images

    Mathematical Formula for each band:
    SSIM(x,y) = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / ((μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂))

    Where:
    - μₓ, μᵧ are local means
    - σₓ², σᵧ² are local variances
    - σₓᵧ is local covariance
    - c₁, c₂ are stability constants

    MSSIM is the mean SSIM across all spatial locations and spectral bands.

    Args:
        original: Original image tensor [Z, Y, X]
        reconstructed: Reconstructed image tensor [Z, Y, X]
        window_size: Size of sliding window for local statistics
        callback: Optional callback function called with (mssim_value, ssim_per_band)

    Returns:
        MSSIM value in range [0, 1], where 1 indicates perfect similarity
    """
    # Simplified MSSIM computation (in practice would use proper sliding window)
    Z, Y, X = original.shape

    # Stability constants
    k1, k2 = 0.01, 0.03
    dynamic_range = torch.max(torch.max(original), torch.max(reconstructed)) - \
                   torch.min(torch.min(original), torch.min(reconstructed))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    ssim_values = []

    for z in range(Z):
        orig_band = original[z]
        recon_band = reconstructed[z]

        # Compute local statistics (simplified - using global statistics)
        mu_x = torch.mean(orig_band)
        mu_y = torch.mean(recon_band)

        sigma_x_sq = torch.var(orig_band)
        sigma_y_sq = torch.var(recon_band)
        sigma_xy = torch.mean((orig_band - mu_x) * (recon_band - mu_y))

        # SSIM calculation
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)

        ssim_band = numerator / (denominator + 1e-8)  # Add small epsilon for stability
        ssim_values.append(ssim_band.item())

    mssim = float(np.mean(ssim_values))
    
    if callback is not None:
        callback(mssim, ssim_values)

    return mssim


def calculate_spectral_angle(original: torch.Tensor, reconstructed: torch.Tensor,
                           callback: Optional[Callable] = None) -> float:
    """
    Calculate Spectral Angle Mapper (SAM) between original and reconstructed spectra

    Mathematical Formula:
    SAM = arccos(Σ(x_i * y_i) / (||x|| * ||y||))

    Where x and y are spectral vectors at each spatial location,
    ||x|| is the Euclidean norm, and the result is in radians.

    The mean SAM across all spatial locations is returned.

    Args:
        original: Original image tensor [Z, Y, X]
        reconstructed: Reconstructed image tensor [Z, Y, X]
        callback: Optional callback function called with (mean_sam, sam_map)

    Returns:
        Mean spectral angle in radians (lower values indicate better similarity)
    """
    Z, Y, X = original.shape
    sam_values = []

    for y in range(Y):
        for x in range(X):
            # Extract spectral vectors
            orig_spectrum = original[:, y, x]
            recon_spectrum = reconstructed[:, y, x]

            # Compute norms
            norm_orig = torch.norm(orig_spectrum)
            norm_recon = torch.norm(recon_spectrum)

            if norm_orig > 1e-8 and norm_recon > 1e-8:
                # Compute dot product
                dot_product = torch.sum(orig_spectrum * recon_spectrum)

                # Compute cosine of angle
                cos_angle = dot_product / (norm_orig * norm_recon)

                # Clamp to valid range for arccos
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

                # Compute spectral angle
                sam = torch.acos(cos_angle).item()
                sam_values.append(sam)
            else:
                # Handle zero vectors
                sam_values.append(0.0)

    mean_sam = float(np.mean(sam_values)) if sam_values else 0.0
    
    # Create sam_map for callback
    sam_map = np.array(sam_values).reshape(Y, X) if sam_values else np.zeros((Y, X))
    
    if callback is not None:
        callback(mean_sam, sam_map)

    return mean_sam