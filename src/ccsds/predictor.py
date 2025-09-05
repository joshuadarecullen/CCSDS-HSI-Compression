import torch
import torch.nn as nn
from typing import Any, Tuple, Optional


class SpectralPredictor(nn.Module):
    """
    CCSDS-123.0-B-2 Adaptive Linear Predictor

    This predictor uses adaptive linear prediction to predict the value of each
    image sample based on nearby samples in a 3D neighborhood.
    """

    def __init__(self, num_bands: int, dynamic_range: int = 16, prediction_bands: Optional[int] = None) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.dynamic_range = dynamic_range
        self.prediction_bands = prediction_bands or min(15, num_bands - 1)

        # Prediction weights - these adapt based on image statistics
        self.register_buffer('weights', torch.zeros(num_bands, self.prediction_bands + 4))

        # Local sums for weight adaptation
        self.register_buffer('local_sums', torch.zeros(num_bands, 4))

        # Prediction mode parameters
        self.prediction_mode = 'full'  # 'full' or 'reduced'

    def _get_neighborhood_samples(self, image: torch.Tensor, z: int, y: int, x: int) -> torch.Tensor:
        """
        Extract samples from 3D neighborhood for prediction

        Args:
            image: [Z, Y, X] tensor - multispectral/hyperspectral image
            z, y, x: Current sample coordinates

        Returns:
            neighborhood samples for prediction
        """
        samples = []

        # North sample (same band, y-1, x)
        if y > 0:
            samples.append(image[z, y-1, x])
        else:
            samples.append(torch.tensor(0.0, device=image.device))

        # West sample (same band, y, x-1)
        if x > 0:
            samples.append(image[z, y, x-1])
        else:
            samples.append(torch.tensor(0.0, device=image.device))

        # Northwest sample (same band, y-1, x-1)
        if y > 0 and x > 0:
            samples.append(image[z, y-1, x-1])
        else:
            samples.append(torch.tensor(0.0, device=image.device))

        # Previous band samples at same spatial location
        for prev_z in range(max(0, z - self.prediction_bands), z):
            samples.append(image[prev_z, y, x])

        return torch.stack(samples)

    def _compute_local_differences(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute local differences used in weight adaptation
        """
        if len(samples) < 4:
            return torch.zeros(4, device=samples.device)

        # Extract spatial neighbors
        north = samples[0] if len(samples) > 0 else 0
        west = samples[1] if len(samples) > 1 else 0
        northwest = samples[2] if len(samples) > 2 else 0

        # Compute differences
        d1 = north - west
        d2 = west - northwest
        d3 = northwest - north
        d4 = north + west - 2 * northwest

        return torch.stack([d1, d2, d3, d4])

    def _update_weights(self, prediction_error: torch.Tensor, local_differences: torch.Tensor, z: int) -> None:
        """
        Adapt prediction weights based on prediction error and local statistics
        """
        # Simple weight update rule - in practice this would be more sophisticated
        learning_rate = 0.01

        if len(local_differences) >= 4:
            # Update weights for spatial components
            for i in range(4):
                if local_differences[i] != 0:
                    self.weights[z, i] += learning_rate * prediction_error * local_differences[i] / (abs(local_differences[i]) + 1e-8)

        # Clamp weights to reasonable range
        self.weights[z] = torch.clamp(self.weights[z], -1.0, 1.0)

    def predict_sample(self, image: torch.Tensor, sample_representatives: torch.Tensor, z: int, y: int, x: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict a single sample value

        Args:
            image: Original image (for initialization)
            sample_representatives: Quantized sample values used for prediction
            z, y, x: Sample coordinates

        Returns:
            Predicted sample value
        """
        # Get neighborhood samples (using sample representatives when available)
        neighborhood = self._get_neighborhood_samples(sample_representatives, z, y, x)

        if len(neighborhood) == 0:
            return torch.tensor(0.0, device=image.device)

        # Pad neighborhood to expected size
        if len(neighborhood) < len(self.weights[z]):
            padding_size = len(self.weights[z]) - len(neighborhood)
            padding = torch.zeros(padding_size, device=neighborhood.device)
            neighborhood = torch.cat([neighborhood, padding])
        elif len(neighborhood) > len(self.weights[z]):
            neighborhood = neighborhood[:len(self.weights[z])]

        # Linear prediction
        prediction = torch.sum(self.weights[z] * neighborhood)

        # Clamp to valid range
        max_val = 2**(self.dynamic_range - 1) - 1
        min_val = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0

        return torch.clamp(prediction, min_val, max_val)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict all samples in the image

        Args:
            image: [Z, Y, X] multispectral/hyperspectral image

        Returns:
            predictions: [Z, Y, X] predicted values
            residuals: [Z, Y, X] prediction residuals
        """
        Z, Y, X = image.shape
        predictions = torch.zeros_like(image)
        residuals = torch.zeros_like(image)

        # Initialize sample representatives with original values
        sample_representatives = image.clone()

        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    # Predict sample
                    pred = self.predict_sample(image, sample_representatives, z, y, x)
                    predictions[z, y, x] = pred

                    # Compute residual
                    residual = image[z, y, x] - pred
                    residuals[z, y, x] = residual

                    # Update weights based on prediction error
                    if z > 0 or y > 0 or x > 0:  # Skip first sample
                        neighborhood = self._get_neighborhood_samples(sample_representatives, z, y, x)
                        local_diffs = self._compute_local_differences(neighborhood)
                        self._update_weights(residual, local_diffs, z)

        return predictions, residuals


class NarrowLocalSumPredictor(SpectralPredictor):
    """
    Variant of predictor using narrow local sums for reduced complexity
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_narrow_sums = True

    def _get_neighborhood_samples(self, image: torch.Tensor, z: int, y: int, x: int) -> torch.Tensor:
        """
        Modified to use narrow local sums when enabled
        """
        if not self.use_narrow_sums:
            return super()._get_neighborhood_samples(image, z, y, x)

        samples = []

        # Use reduced neighborhood for narrow local sums
        # North sample
        if y > 0:
            samples.append(image[z, y-1, x])
        else:
            samples.append(torch.tensor(0.0, device=image.device))

        # Previous band samples (reduced set)
        for prev_z in range(max(0, z - min(3, self.prediction_bands)), z):
            samples.append(image[prev_z, y, x])

        return torch.stack(samples) if samples else torch.tensor([], device=image.device)
