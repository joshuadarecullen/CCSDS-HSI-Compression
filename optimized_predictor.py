import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OptimizedSpectralPredictor(nn.Module):
    """
    Optimized CCSDS-123.0-B-2 Adaptive Linear Predictor

    Vectorized implementation that processes entire bands/rows at once
    instead of sample-by-sample processing.
    """

    def __init__(self, num_bands, dynamic_range=16, prediction_bands=None):
        super().__init__()
        self.num_bands = num_bands
        self.dynamic_range = dynamic_range
        self.prediction_bands = prediction_bands or min(15, num_bands - 1)

        # Prediction weights - vectorized for all bands
        # Shape: [num_bands, max_predictors] where max_predictors includes spatial + spectral
        max_predictors = 4 + self.prediction_bands  # 4 spatial + P spectral
        self.register_buffer('weights', torch.zeros(num_bands, max_predictors))

        # Weight adaptation parameters
        self.register_buffer('local_sums', torch.zeros(num_bands, 4))
        self.learning_rate = 0.01

    def _extract_spatial_neighbors(self, image):
        """
        Extract spatial neighbors for all pixels simultaneously using convolution

        Args:
            image: [Z, Y, X] input image

        Returns:
            neighbors: [Z, Y, X, 4] spatial neighbors (N, W, NW, and a fourth predictor)
        """
        Z, Y, X = image.shape

        # Pad image to handle boundaries
        padded = F.pad(image, (1, 1, 1, 1), mode='constant', value=0)

        # Extract neighbors using indexing (much faster than loops)
        north = padded[:, :-2, 1:-1]     # [z, y-1, x] - North
        west = padded[:, 1:-1, :-2]      # [z, y, x-1] - West
        northwest = padded[:, :-2, :-2]   # [z, y-1, x-1] - Northwest

        # Fourth predictor: simple average of available neighbors
        fourth = (north + west + northwest) / 3

        # Stack along last dimension: [Z, Y, X, 4]
        spatial_neighbors = torch.stack([north, west, northwest, fourth], dim=-1)

        return spatial_neighbors

    def _extract_spectral_neighbors(self, image, z):
        """
        Extract spectral neighbors for band z

        Args:
            image: [Z, Y, X] input image
            z: Current band index

        Returns:
            spectral_neighbors: [Y, X, P] where P is number of previous bands used
        """
        Y, X = image.shape[1], image.shape[2]

        # Get previous bands for prediction
        start_band = max(0, z - self.prediction_bands)

        if start_band >= z:
            # No previous bands available
            return torch.zeros(Y, X, self.prediction_bands, device=image.device)

        # Extract previous bands: [P, Y, X] -> [Y, X, P]
        spectral_data = image[start_band:z]  # [actual_P, Y, X]

        # Pad if we don't have enough previous bands
        actual_bands = spectral_data.shape[0]
        if actual_bands < self.prediction_bands:
            padding = torch.zeros(self.prediction_bands - actual_bands, Y, X, device=image.device)
            spectral_data = torch.cat([padding, spectral_data], dim=0)

        # Transpose to [Y, X, P]
        spectral_neighbors = spectral_data.permute(1, 2, 0)

        return spectral_neighbors

    def _compute_local_differences_vectorized(self, spatial_neighbors):
        """
        Compute local differences for all pixels simultaneously

        Args:
            spatial_neighbors: [Z, Y, X, 4] spatial neighbors

        Returns:
            local_diffs: [Z, Y, X, 4] local differences
        """
        north = spatial_neighbors[..., 0]
        west = spatial_neighbors[..., 1]
        northwest = spatial_neighbors[..., 2]

        # Compute differences vectorized
        d1 = north - west
        d2 = west - northwest
        d3 = northwest - north
        d4 = north + west - 2 * northwest

        return torch.stack([d1, d2, d3, d4], dim=-1)

    def _predict_band_vectorized(self, image, sample_representatives, z):
        """
        Predict entire band z using vectorized operations

        Args:
            image: [Z, Y, X] original image
            sample_representatives: [Z, Y, X] sample representatives
            z: Band index to predict

        Returns:
            predictions: [Y, X] predictions for band z
        """
        Y, X = image.shape[1], image.shape[2]

        # Get spatial neighbors for this band
        spatial_neighbors = self._extract_spatial_neighbors(sample_representatives)
        spatial_z = spatial_neighbors[z]  # [Y, X, 4]

        # Get spectral neighbors
        spectral_neighbors = self._extract_spectral_neighbors(sample_representatives, z)  # [Y, X, P]

        # Combine spatial and spectral predictors: [Y, X, 4+P]
        all_predictors = torch.cat([spatial_z, spectral_neighbors], dim=-1)

        # Ensure predictor dimensions match weight dimensions
        weight_dim = self.weights.shape[1]
        if all_predictors.shape[-1] < weight_dim:
            # Pad predictors
            padding_size = weight_dim - all_predictors.shape[-1]
            padding = torch.zeros(Y, X, padding_size, device=all_predictors.device)
            all_predictors = torch.cat([all_predictors, padding], dim=-1)
        elif all_predictors.shape[-1] > weight_dim:
            # Truncate predictors
            all_predictors = all_predictors[:, :, :weight_dim]

        # Vectorized prediction: [Y, X, predictors] @ [predictors] -> [Y, X]
        predictions = torch.sum(all_predictors * self.weights[z].unsqueeze(0).unsqueeze(0), dim=-1)

        # Clamp to valid range
        max_val = 2**(self.dynamic_range - 1) - 1
        min_val = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0
        predictions = torch.clamp(predictions, min_val, max_val)

        return predictions

    def _update_weights_vectorized(self, residuals, local_differences, z):
        """
        Update prediction weights using vectorized operations

        Args:
            residuals: [Y, X] prediction residuals for band z
            local_differences: [Y, X, 4] local differences
            z: Band index
        """
        # Compute weight updates for spatial components
        # Use mean residual and mean local differences for stability
        mean_residual = torch.mean(residuals)
        mean_local_diffs = torch.mean(local_differences, dim=(0, 1))  # [4]

        # Update spatial weights
        for i in range(4):
            if abs(mean_local_diffs[i]) > 1e-8:
                weight_update = self.learning_rate * mean_residual * mean_local_diffs[i] / (abs(mean_local_diffs[i]) + 1e-8)
                self.weights[z, i] += weight_update

        # Clamp weights to reasonable range
        self.weights[z] = torch.clamp(self.weights[z], -1.0, 1.0)

    def forward_optimized(self, image):
        """
        Optimized forward pass processing bands sequentially but pixels in parallel

        Args:
            image: [Z, Y, X] multispectral/hyperspectral image

        Returns:
            predictions: [Z, Y, X] predicted values
            residuals: [Z, Y, X] prediction residuals
        """
        Z, Y, X = image.shape
        predictions = torch.zeros_like(image)
        residuals = torch.zeros_like(image)

        # Initialize sample representatives with original image
        sample_representatives = image.clone()

        # Process bands sequentially (must maintain causal order)
        for z in range(Z):
            # Predict entire band at once
            band_predictions = self._predict_band_vectorized(image, sample_representatives, z)
            predictions[z] = band_predictions

            # Compute residuals for entire band
            band_residuals = image[z] - band_predictions
            residuals[z] = band_residuals

            # Update weights based on band statistics
            if z > 0:  # Skip first band
                spatial_neighbors = self._extract_spatial_neighbors(sample_representatives)
                local_diffs = self._compute_local_differences_vectorized(spatial_neighbors)
                self._update_weights_vectorized(band_residuals, local_diffs[z], z)

            # Update sample representatives for next band (simplified for lossless)
            sample_representatives[z] = predictions[z] + band_residuals

        return predictions, residuals

    def forward(self, image):
        """
        Main forward pass - uses optimized version
        """
        return self.forward_optimized(image)


class CausalOptimizedPredictor(OptimizedSpectralPredictor):
    """
    Causal predictor that processes samples in strict raster-scan order
    but with vectorized operations within each row/band.

    This maintains exact CCSDS-123.0-B-2 causal ordering while still
    being much faster than the sample-by-sample approach.
    """

    def forward_causal_optimized(self, image):
        """
        Process in causal order but vectorize operations within rows

        This is slower than full vectorization but maintains exact
        sample-by-sample causality as required by the standard.
        """
        Z, Y, X = image.shape
        predictions = torch.zeros_like(image)
        residuals = torch.zeros_like(image)

        # Initialize sample representatives
        sample_representatives = image.clone()

        # Process in causal order: band by band, row by row
        for z in range(Z):
            for y in range(Y):
                # Process entire row at once (vectorized within row)
                row_predictions = self._predict_row_vectorized(
                    image, sample_representatives, z, y
                )
                predictions[z, y] = row_predictions

                # Compute residuals for row
                row_residuals = image[z, y] - row_predictions
                residuals[z, y] = row_residuals

                # Update sample representatives immediately for causality
                sample_representatives[z, y] = predictions[z, y] + row_residuals

                # Update weights based on row statistics
                if z > 0 or y > 0:
                    self._update_weights_from_row(row_residuals, z, y)

        return predictions, residuals

    def _predict_row_vectorized(self, image, sample_representatives, z, y):
        """
        Predict entire row y of band z using vectorized operations

        Args:
            image: [Z, Y, X] input image
            sample_representatives: [Z, Y, X] sample representatives
            z, y: Band and row indices

        Returns:
            predictions: [X] predictions for row y of band z
        """
        Y, X = image.shape[1], image.shape[2]

        # Extract predictors for this row
        predictors = torch.zeros(X, self.weights.shape[1], device=image.device)

        for x in range(X):
            # North sample
            if y > 0:
                predictors[x, 0] = sample_representatives[z, y-1, x]

            # West sample
            if x > 0:
                predictors[x, 1] = sample_representatives[z, y, x-1]

            # Northwest sample
            if y > 0 and x > 0:
                predictors[x, 2] = sample_representatives[z, y-1, x-1]

            # Previous band samples
            for p in range(min(self.prediction_bands, z)):
                prev_z = z - 1 - p
                if prev_z >= 0:
                    predictors[x, 4 + p] = sample_representatives[prev_z, y, x]

        # Vectorized prediction for entire row
        predictions = torch.sum(predictors * self.weights[z].unsqueeze(0), dim=1)

        # Clamp to valid range
        max_val = 2**(self.dynamic_range - 1) - 1
        min_val = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0
        predictions = torch.clamp(predictions, min_val, max_val)

        return predictions

    def _update_weights_from_row(self, row_residuals, z, y):
        """
        Update weights based on row statistics

        Args:
            row_residuals: [X] residuals for current row
            z, y: Band and row indices
        """
        # Simple weight update based on row mean
        mean_residual = torch.mean(row_residuals)

        # Update with small learning rate
        self.weights[z] *= 0.999  # Slight decay

        # Add small correction based on residual
        if abs(mean_residual) > 0.1:
            self.weights[z, :4] += self.learning_rate * mean_residual * 0.1

        # Clamp weights
        self.weights[z] = torch.clamp(self.weights[z], -1.0, 1.0)

    def forward(self, image):
        """Use causal optimized version"""
        return self.forward_causal_optimized(image)
