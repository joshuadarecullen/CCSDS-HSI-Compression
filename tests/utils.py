"""
Shared test utilities for CCSDS-HSI-Compression tests
"""
import torch
import numpy as np
from typing import Tuple, Optional


def generate_test_image(
    num_bands: int = 10, 
    height: int = 64, 
    width: int = 64, 
    dynamic_range: int = 16, 
    noise_level: float = 0.1,
    seed: Optional[int] = 42
) -> torch.Tensor:
    """
    Generate a synthetic hyperspectral test image with specified characteristics.
    
    Args:
        num_bands: Number of spectral bands
        height: Image height in pixels
        width: Image width in pixels
        dynamic_range: Bit depth for the image (2^dynamic_range max value)
        noise_level: Amount of random noise to add (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        torch.Tensor: Generated hyperspectral image of shape (num_bands, height, width)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    max_value = (2 ** dynamic_range) - 1
    
    # Generate base spectral curves
    wavelengths = torch.linspace(400, 2500, num_bands)
    
    # Create spatial patterns
    x, y = torch.meshgrid(
        torch.linspace(0, 2 * np.pi, width),
        torch.linspace(0, 2 * np.pi, height),
        indexing='xy'
    )
    
    # Initialize image
    image = torch.zeros(num_bands, height, width)
    
    # Add different materials with varying spectral signatures
    for band_idx in range(num_bands):
        # Base pattern with spatial variation
        spatial_pattern = (
            0.3 * torch.sin(x + band_idx * 0.1) + 
            0.3 * torch.cos(y + band_idx * 0.15) +
            0.4
        )
        
        # Spectral variation based on wavelength
        spectral_intensity = 0.5 + 0.5 * torch.sin(wavelengths[band_idx] / 200)
        
        # Combine spatial and spectral components
        image[band_idx] = spatial_pattern * spectral_intensity
    
    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(image) * noise_level
        image = image + noise
    
    # Normalize to dynamic range
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * max_value).round().clamp(0, max_value)
    
    return image.float()


def generate_simple_test_image(
    num_bands: int = 5,
    height: int = 16,
    width: int = 16,
    dynamic_range: int = 16,
    seed: Optional[int] = 42
) -> torch.Tensor:
    """
    Generate a simple test image for basic tests.
    
    Args:
        num_bands: Number of spectral bands
        height: Image height in pixels
        width: Image width in pixels  
        seed: Random seed for reproducibility
    
    Returns:
        torch.Tensor: Generated test image
    """
    return generate_test_image(
        num_bands=num_bands,
        height=height,
        width=width,
        dynamic_range=dynamic_range,
        noise_level=0.05,
        seed=seed
    )
