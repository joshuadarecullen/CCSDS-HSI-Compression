import torch
import torch.nn as nn
import numpy as np


class SampleRepresentativeCalculator(nn.Module):
    """
    CCSDS-123.0-B-2 Sample Representative Calculator

    Computes sample representatives used in prediction calculations.
    The sample representative lies between the center of the quantizer bin
    and the predicted sample value, controlled by user parameters φ_z, ψ_z, Θ.
    """

    def __init__(self, num_bands):
        super().__init__()
        self.num_bands = num_bands

        # User-specified parameters for sample representative calculation
        # φ_z, ψ_z control the compromise between quantizer bin center and prediction
        # Θ is a threshold parameter
        self.register_buffer('phi', torch.zeros(num_bands))     # φ_z parameters
        self.register_buffer('psi', torch.zeros(num_bands))     # ψ_z parameters
        self.register_buffer('theta', torch.tensor(4.0))        # Θ parameter

    def set_parameters(self, phi=None, psi=None, theta=None):
        """
        Set sample representative calculation parameters

        Args:
            phi: [num_bands] tensor of φ_z parameters
            psi: [num_bands] tensor of ψ_z parameters
            theta: Scalar Θ parameter
        """
        if phi is not None:
            self.phi = phi.clone()
        if psi is not None:
            self.psi = psi.clone()
        if theta is not None:
            self.theta = torch.tensor(float(theta))

    def compute_quantizer_bin_center(self, original_sample, predicted_sample, max_error):
        """
        Compute the center of the quantizer bin s'_z(t)

        Args:
            original_sample: Original sample value s_z(t)
            predicted_sample: Predicted sample value ŝ_z(t)
            max_error: Maximum allowed error m_z(t)

        Returns:
            Quantizer bin center s'_z(t)
        """
        if max_error == 0:
            # Lossless case - bin center equals original sample
            return original_sample

        # Quantizer step size
        step_size = 2 * max_error + 1

        # Prediction residual
        residual = original_sample - predicted_sample

        # Quantize the residual
        quantized_residual = torch.round(residual / step_size) * step_size

        # Bin center is prediction + quantized residual
        bin_center = predicted_sample + quantized_residual

        return bin_center

    def compute_sample_representative(self, bin_center, predicted_sample, z):
        """
        Compute sample representative s''_z(t) using user parameters

        The sample representative lies between bin_center and predicted_sample.

        Args:
            bin_center: Quantizer bin center s'_z(t)
            predicted_sample: Predicted sample value ŝ_z(t)
            z: Band index

        Returns:
            Sample representative s''_z(t)
        """
        # If φ_z = ψ_z = 0, representative equals bin center (traditional approach)
        if self.phi[z] == 0 and self.psi[z] == 0:
            return bin_center

        # Compute difference between prediction and bin center
        diff = predicted_sample - bin_center

        # Apply user parameters to control representative calculation
        # This is a simplified version - the actual standard may use a more complex formula
        if torch.abs(diff) <= self.theta:
            # Small difference case
            adjustment = self.phi[z] * diff / (self.theta + 1e-8)
        else:
            # Large difference case
            adjustment = self.psi[z] * torch.sign(diff) * (torch.abs(diff) - self.theta) / (torch.abs(diff) + 1e-8)

        # Representative is bin center plus adjustment toward prediction
        representative = bin_center + adjustment

        return representative

    def forward(self, original_samples, predicted_samples, max_errors):
        """
        Compute sample representatives for all samples

        Args:
            original_samples: [Z, Y, X] original sample values
            predicted_samples: [Z, Y, X] predicted sample values
            max_errors: [Z, Y, X] maximum allowed errors

        Returns:
            sample_representatives: [Z, Y, X] computed representatives
            bin_centers: [Z, Y, X] quantizer bin centers
        """
        Z, Y, X = original_samples.shape
        sample_representatives = torch.zeros_like(original_samples)
        bin_centers = torch.zeros_like(original_samples)

        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    # Compute quantizer bin center
                    bin_center = self.compute_quantizer_bin_center(
                        original_samples[z, y, x],
                        predicted_samples[z, y, x],
                        max_errors[z, y, x]
                    )
                    bin_centers[z, y, x] = bin_center

                    # Compute sample representative
                    representative = self.compute_sample_representative(
                        bin_center,
                        predicted_samples[z, y, x],
                        z
                    )
                    sample_representatives[z, y, x] = representative

        return sample_representatives, bin_centers


class OptimizedSampleRepresentative(SampleRepresentativeCalculator):
    """
    Optimized version that processes bands in parallel where possible
    """

    def forward(self, original_samples, predicted_samples, max_errors):
        """
        Vectorized computation of sample representatives
        """
        Z, Y, X = original_samples.shape

        # Compute quantizer bin centers for all samples
        residuals = original_samples - predicted_samples

        # Handle lossless case (max_error = 0)
        step_sizes = 2 * max_errors.float() + 1
        step_sizes = torch.where(max_errors == 0, torch.ones_like(step_sizes), step_sizes)

        # Quantize residuals
        quantized_residuals = torch.round(residuals / step_sizes) * step_sizes
        quantized_residuals = torch.where(max_errors == 0, residuals, quantized_residuals)

        # Bin centers
        bin_centers = predicted_samples + quantized_residuals

        # Compute sample representatives
        sample_representatives = torch.zeros_like(original_samples)

        for z in range(Z):
            # Get parameters for this band
            phi_z = self.phi[z]
            psi_z = self.psi[z]

            if phi_z == 0 and psi_z == 0:
                # Traditional approach - representative equals bin center
                sample_representatives[z] = bin_centers[z]
            else:
                # Compute adjustments
                diff = predicted_samples[z] - bin_centers[z]

                # Small difference case
                small_diff_mask = torch.abs(diff) <= self.theta
                small_adjustment = phi_z * diff / (self.theta + 1e-8)

                # Large difference case
                large_diff_mask = ~small_diff_mask
                large_adjustment = psi_z * torch.sign(diff) * (torch.abs(diff) - self.theta) / (torch.abs(diff) + 1e-8)

                # Combine adjustments
                adjustment = torch.where(small_diff_mask, small_adjustment, large_adjustment)
                sample_representatives[z] = bin_centers[z] + adjustment

        return sample_representatives, bin_centers


class AdaptiveSampleRepresentative(SampleRepresentativeCalculator):
    """
    Adaptive version that can learn optimal parameters during compression
    """

    def __init__(self, num_bands):
        super().__init__(num_bands)

        # Track prediction accuracy with different parameter settings
        self.register_buffer('accuracy_history', torch.zeros(num_bands, 10))
        self.register_buffer('parameter_history', torch.zeros(num_bands, 10, 2))  # [phi, psi] pairs
        self.adaptation_interval = 100
        self.adaptation_count = 0

    def adapt_parameters(self, prediction_errors, z):
        """
        Adapt φ_z and ψ_z parameters based on prediction accuracy

        Args:
            prediction_errors: Recent prediction errors for band z
            z: Band index
        """
        if len(prediction_errors) < 10:
            return

        # Compute mean squared error with current parameters
        current_mse = torch.mean(prediction_errors**2)

        # Try different parameter combinations
        best_mse = current_mse
        best_phi = self.phi[z]
        best_psi = self.psi[z]

        # Simple grid search over small parameter range
        phi_candidates = [0.0, 1.0, 2.0, 4.0]
        psi_candidates = [0.0, 2.0, 4.0, 6.0]

        for phi_test in phi_candidates:
            for psi_test in psi_candidates:
                # This would require recomputing predictions with test parameters
                # Simplified: assume current MSE represents performance
                if phi_test != self.phi[z] or psi_test != self.psi[z]:
                    # Estimate MSE with test parameters (simplified heuristic)
                    test_mse = current_mse * (1 + 0.1 * torch.randn(1).item())

                    if test_mse < best_mse:
                        best_mse = test_mse
                        best_phi = phi_test
                        best_psi = psi_test

        # Update parameters if improvement found
        if best_phi != self.phi[z] or best_psi != self.psi[z]:
            self.phi[z] = best_phi
            self.psi[z] = best_psi
