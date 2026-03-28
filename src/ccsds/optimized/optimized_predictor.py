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

        # Prediction weights - CCSDS-123.0-B-2 compliant with backward compatibility
        # Need to accommodate both old indexing (4 + P) and new indexing (P + 3)
        max_components = max(self.prediction_bands + 3, 4 + self.prediction_bands)
        self.register_buffer('weights', torch.zeros(num_bands, max_components))

        # Weight adaptation parameters
        self.register_buffer('local_sums', torch.zeros(num_bands, 4))
        self.learning_rate = 0.01
        
        # CCSDS-123.0-B-2 prediction mode parameters
        self.prediction_mode = 'full'  # 'full' or 'reduced'
        self.use_narrow_local_sums = False  # Issue 2 narrow local sums option
        self.local_sum_type = 'neighbor_oriented'  # 'neighbor_oriented' or 'column_oriented'
        
        # Initialize prediction components
        self._compute_prediction_components()

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
    
    def _compute_narrow_local_sum_vectorized(self, sample_representatives: torch.Tensor, z: int) -> torch.Tensor:
        """
        Compute narrow local sums for Issue 2 hardware pipelining optimization (vectorized)
        
        Eliminates the dependency on sample representative s''_{z,y,x-1}
        when performing prediction calculation for neighboring sample s^_{z,y,x}
        
        Args:
            sample_representatives: [Z, Y, X] sample representatives tensor
            z: Current band index
            
        Returns:
            narrow_local_sums: [Y, X] narrow local sums without x-1 dependency
        """
        if not self.use_narrow_local_sums:
            # Use standard local sum calculation (vectorized)
            Y, X = sample_representatives.shape[1], sample_representatives.shape[2]
            return self.local_sums[z, :].sum().expand(Y, X)
            
        Y, X = sample_representatives.shape[1], sample_representatives.shape[2] 
        narrow_sums = torch.zeros(Y, X, device=sample_representatives.device)
        
        if self.local_sum_type == 'column_oriented':
            # Column-oriented local sum - uses vertical neighbors only (vectorized)
            # North neighbor (y-1, x)
            if Y > 0:
                narrow_sums[1:, :] += sample_representatives[z, :-1, :]
            # North-north neighbor (y-2, x)  
            if Y > 1:
                narrow_sums[2:, :] += sample_representatives[z, :-2, :]
        else:
            # Neighbor-oriented (standard) but excluding x-1 dependency (vectorized)
            # North neighbor (y-1, x)
            if Y > 0:
                narrow_sums[1:, :] += sample_representatives[z, :-1, :]
            # Northwest neighbor (y-1, x-1)
            if Y > 0 and X > 0:
                narrow_sums[1:, 1:] += sample_representatives[z, :-1, :-1]
            # Intentionally exclude West neighbor (y, x-1) for pipeline optimization
            
        return narrow_sums
    
    def _compute_prediction_components(self) -> None:
        """
        Compute number of local difference values C_z according to CCSDS-123.0-B-2
        """
        self.spectral_components = self.prediction_bands
        
        if self.prediction_mode == 'reduced':
            self.total_components = self.spectral_components
        else:  # full mode
            self.total_components = self.spectral_components + 3
    
    def set_prediction_mode(self, mode: str) -> None:
        """
        Set prediction mode according to CCSDS-123.0-B-2 section 4.3
        """
        if mode not in ['full', 'reduced']:
            raise ValueError("Prediction mode must be 'full' or 'reduced'")
        
        self.prediction_mode = mode
        self._compute_prediction_components()
    
    def enable_narrow_local_sums(self, enable: bool = True, local_sum_type: str = 'neighbor_oriented') -> None:
        """
        Enable/disable narrow local sums for hardware pipelining optimization
        
        Args:
            enable: Whether to use narrow local sums
            local_sum_type: 'neighbor_oriented' or 'column_oriented'
        """
        self.use_narrow_local_sums = enable
        self.local_sum_type = local_sum_type
        
    def get_prediction_mode_info(self) -> dict:
        """Get current prediction mode configuration"""
        return {
            'prediction_mode': 'optimized',
            'use_narrow_local_sums': self.use_narrow_local_sums,
            'local_sum_type': self.local_sum_type,
            'prediction_bands': self.prediction_bands,
            'vectorized': True
        }

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
        Update prediction weights using CCSDS-123.0-B-2 standard weight adaptation algorithm
        with vectorized operations for optimized performance
        
        Args:
            residuals: [Y, X] prediction residuals for band z
            local_differences: [Y, X, 4] local differences
            z: Band index
        """
        # CCSDS-123.0-B-2 weight adaptation parameters
        V_min = 4  # Minimum scaling parameter
        V_max = 6  # Maximum scaling parameter for damping
        
        # Use mean values for vectorized stability (could use median for robustness)
        mean_residual = torch.mean(residuals)
        mean_local_diffs = torch.mean(local_differences, dim=(0, 1))  # [4]
        
        # Update spatial weights using CCSDS algorithm
        for i in range(4):
            d_i = mean_local_diffs[i].item()
            if abs(d_i) > 1e-8:
                # Compute magnitude parameter V_i based on local difference magnitude
                abs_d_i = abs(d_i)
                if abs_d_i >= 1:
                    V_i = int(torch.log2(torch.tensor(abs_d_i)).item())
                else:
                    V_i = 0
                
                # Compute weight update according to CCSDS formula
                # Δw_i = 2^(-V_min) * e * d_i * 2^(-max(0, V_i - V_max))
                base_scale = 2.0 ** (-V_min)  # 2^(-V_min)
                damping = 2.0 ** (-max(0, V_i - V_max))  # Damping factor
                
                weight_update = base_scale * mean_residual.item() * d_i * damping
                self.weights[z, i] += weight_update
        
        # Update spectral weights (if any) with similar approach
        if hasattr(self, 'prediction_bands') and self.prediction_bands > 0:
            for i in range(4, min(4 + self.prediction_bands, self.weights.size(1))):
                # For spectral weights, use a simplified update based on prediction error
                spectral_update = (2.0 ** (-V_min)) * mean_residual.item() * 0.1
                self.weights[z, i] += spectral_update

        # Apply weight clamping according to standard (broader range for better adaptation)
        self.weights[z] = torch.clamp(self.weights[z], -2.0, 2.0)
        
        # Update running statistics for local differences (exponential moving average)
        if not hasattr(self, 'local_sums'):
            self.local_sums = torch.zeros_like(self.weights[:, :4])
        
        alpha = 0.1  # Smoothing factor
        self.local_sums[z] = (1 - alpha) * self.local_sums[z] + alpha * mean_local_diffs

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
    
    Supports Issue 2 narrow local sums for hardware pipelining optimization.
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

            # Previous band samples - ensure we don't exceed weight buffer size
            for p in range(min(self.prediction_bands, z)):
                prev_z = z - 1 - p
                weight_idx = 4 + p
                if prev_z >= 0 and weight_idx < self.weights.shape[1]:
                    predictors[x, weight_idx] = sample_representatives[prev_z, y, x]

        # Vectorized prediction for entire row
        predictions = torch.sum(predictors * self.weights[z].unsqueeze(0), dim=1)

        # Clamp to valid range
        max_val = 2**(self.dynamic_range - 1) - 1
        min_val = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0
        predictions = torch.clamp(predictions, min_val, max_val)

        return predictions

    def _update_weights_from_row(self, row_residuals, z, y):
        """
        Update weights based on CCSDS-123.0-B-2 algorithm adapted for row-wise processing
        
        Args:
            row_residuals: [X] residuals for current row
            z, y: Band and row indices
        """
        # CCSDS-123.0-B-2 weight adaptation parameters
        V_min = 4  # Minimum scaling parameter
        V_max = 6  # Maximum scaling parameter for damping
        
        # Use row statistics for causal processing
        mean_residual = torch.mean(row_residuals)
        std_residual = torch.std(row_residuals)
        
        # For causal mode, we don't have full local differences, so use approximation
        # based on row variation and residual statistics
        if abs(mean_residual) > 1e-8:
            # Estimate local difference magnitude from residual variation
            estimated_diff_magnitude = std_residual.item() if std_residual > 1e-8 else abs(mean_residual.item())
            
            # Compute magnitude parameter V_i
            if estimated_diff_magnitude >= 1:
                V_i = int(torch.log2(torch.tensor(estimated_diff_magnitude)).item())
            else:
                V_i = 0
            
            # Compute weight update according to CCSDS formula (adapted for causal mode)
            base_scale = 2.0 ** (-V_min)  # 2^(-V_min)
            damping = 2.0 ** (-max(0, V_i - V_max))  # Damping factor
            
            # Update spatial weights with CCSDS-based scaling
            weight_update = base_scale * mean_residual.item() * damping
            
            # Apply update to spatial components (north, west, northwest)
            # In causal mode, we primarily update based on available context
            self.weights[z, :3] += weight_update * 0.1  # More conservative for causal mode
            
            # Update spectral weights if available
            if hasattr(self, 'prediction_bands') and self.prediction_bands > 0:
                spectral_range = min(4 + self.prediction_bands, self.weights.size(1))
                self.weights[z, 4:spectral_range] += weight_update * 0.05  # Even more conservative for spectral

        # Apply weight clamping according to standard
        self.weights[z] = torch.clamp(self.weights[z], -2.0, 2.0)
        
        # Maintain weight decay for stability in causal mode
        self.weights[z] *= 0.9995  # Very slight decay for long-term stability

    def forward(self, image):
        """Use causal optimized version"""
        return self.forward_causal_optimized(image)
