"""
CCSDS-123.0-B-2 Sample Encoding Orders

Implements Band-Interleaved (BI) and Band-Sequential (BSQ) encoding orders
as specified in Section 4.1 of the CCSDS-123.0-B-2 standard.

The encoding order determines the sequence in which samples are processed
during compression, affecting prediction accuracy and compression efficiency.
"""

import torch
from typing import Iterator, Tuple, List
from enum import Enum


class EncodingOrder(Enum):
    """Sample encoding order types"""
    BAND_INTERLEAVED = 0  # BI: Process all bands of a pixel before moving to next pixel
    BAND_SEQUENTIAL = 1   # BSQ: Process entire band before moving to next band


class SampleIterator:
    """
    Iterator for processing samples in different encoding orders

    This class provides a unified interface for iterating over image samples
    in both Band-Interleaved (BI) and Band-Sequential (BSQ) orders while
    maintaining proper spatial and spectral neighborhood relationships.
    """

    def __init__(self, image: torch.Tensor, encoding_order: EncodingOrder = EncodingOrder.BAND_INTERLEAVED):
        """
        Initialize sample iterator

        Args:
            image: Input image tensor [Z, Y, X] or [1, Z, Y, X]
            encoding_order: Encoding order (BI or BSQ)
        """
        # Ensure image is in [Z, Y, X] format
        if len(image.shape) == 4:
            self.image = image.squeeze(0)  # Remove batch dimension
        else:
            self.image = image

        self.Z, self.Y, self.X = self.image.shape
        self.encoding_order = encoding_order
        self.position = 0
        self.total_samples = self.Z * self.Y * self.X

    def __iter__(self) -> Iterator[Tuple[int, int, int, torch.Tensor]]:
        """
        Iterate over samples in the specified encoding order

        Yields:
            z, y, x, sample_value: Band index, row, column, and sample value
        """
        if self.encoding_order == EncodingOrder.BAND_INTERLEAVED:
            yield from self._iterate_band_interleaved()
        else:  # BSQ
            yield from self._iterate_band_sequential()

    def _iterate_band_interleaved(self) -> Iterator[Tuple[int, int, int, torch.Tensor]]:
        """
        Iterate in Band-Interleaved (BI) order

        For each spatial location (y, x), process all spectral bands z before
        moving to the next spatial location. This order can improve spectral
        prediction but may reduce spatial prediction efficiency.

        Order: (0,0,0), (1,0,0), (2,0,0), ..., (Z-1,0,0), (0,0,1), (1,0,1), ...
        """
        for y in range(self.Y):
            for x in range(self.X):
                for z in range(self.Z):
                    yield z, y, x, self.image[z, y, x]

    def _iterate_band_sequential(self) -> Iterator[Tuple[int, int, int, torch.Tensor]]:
        """
        Iterate in Band-Sequential (BSQ) order

        Process entire bands sequentially: all samples of band 0, then all
        samples of band 1, etc. This order can improve spatial prediction
        but may reduce spectral prediction efficiency.

        Order: (0,0,0), (0,0,1), (0,0,2), ..., (0,Y-1,X-1), (1,0,0), ...
        """
        for z in range(self.Z):
            for y in range(self.Y):
                for x in range(self.X):
                    yield z, y, x, self.image[z, y, x]

    def get_sample_position(self, z: int, y: int, x: int) -> int:
        """
        Get the linear position of a sample in the encoding order

        Args:
            z, y, x: Band, row, column indices

        Returns:
            position: Linear position in encoding order
        """
        if self.encoding_order == EncodingOrder.BAND_INTERLEAVED:
            return y * self.X * self.Z + x * self.Z + z
        else:  # BSQ
            return z * self.Y * self.X + y * self.X + x

    def get_sample_coordinates(self, position: int) -> Tuple[int, int, int]:
        """
        Get sample coordinates from linear position in encoding order

        Args:
            position: Linear position in encoding order

        Returns:
            z, y, x: Band, row, column indices
        """
        if self.encoding_order == EncodingOrder.BAND_INTERLEAVED:
            # BI: position = y * X * Z + x * Z + z
            remaining = position
            y = remaining // (self.X * self.Z)
            remaining = remaining % (self.X * self.Z)
            x = remaining // self.Z
            z = remaining % self.Z
        else:  # BSQ
            # BSQ: position = z * Y * X + y * X + x
            remaining = position
            z = remaining // (self.Y * self.X)
            remaining = remaining % (self.Y * self.X)
            y = remaining // self.X
            x = remaining % self.X

        return z, y, x

    def get_spatial_neighbors(self, z: int, y: int, x: int,
                            neighborhood_size: int = 3) -> List[Tuple[int, int, int]]:
        """
        Get spatial neighbors for prediction based on encoding order

        Args:
            z, y, x: Current sample coordinates
            neighborhood_size: Size of spatial neighborhood (3 = 3x3)

        Returns:
            neighbors: List of (z, y, x) coordinates of available neighbors
        """
        neighbors = []
        current_pos = self.get_sample_position(z, y, x)

        # For spatial prediction, we look at neighbors in the same band
        # that have been processed before the current sample
        half_size = neighborhood_size // 2

        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                if dy == 0 and dx == 0:
                    continue  # Skip current sample

                ny, nx = y + dy, x + dx

                # Check bounds
                if 0 <= ny < self.Y and 0 <= nx < self.X:
                    neighbor_pos = self.get_sample_position(z, ny, nx)

                    # Only include neighbors that come before current sample
                    if neighbor_pos < current_pos:
                        neighbors.append((z, ny, nx))

        return neighbors

    def get_spectral_neighbors(self, z: int, y: int, x: int) -> List[Tuple[int, int, int]]:
        """
        Get spectral neighbors for prediction based on encoding order

        Args:
            z, y, x: Current sample coordinates

        Returns:
            neighbors: List of (z, y, x) coordinates of available spectral neighbors
        """
        neighbors = []
        current_pos = self.get_sample_position(z, y, x)

        # For spectral prediction, we look at previous bands at the same location
        # that have been processed before the current sample
        for nz in range(self.Z):
            if nz != z:  # Skip current band
                neighbor_pos = self.get_sample_position(nz, y, x)

                # Only include neighbors that come before current sample
                if neighbor_pos < current_pos:
                    neighbors.append((nz, y, x))

        return neighbors


class EncodingOrderOptimizer:
    """
    Utilities for optimizing encoding order selection based on image characteristics
    """

    @staticmethod
    def analyze_image_structure(image: torch.Tensor) -> dict:
        """
        Analyze image structure to recommend optimal encoding order

        Args:
            image: Input image tensor [Z, Y, X]

        Returns:
            analysis: Dictionary with structure analysis and recommendations
        """
        Z, Y, X = image.shape

        # Compute spatial correlation
        spatial_corr = 0.0
        if Y > 1 and X > 1:
            for z in range(Z):
                band = image[z]
                # Horizontal correlation
                h_corr = torch.corrcoef(torch.stack([
                    band[:, :-1].flatten(),
                    band[:, 1:].flatten()
                ]))[0, 1]
                # Vertical correlation
                v_corr = torch.corrcoef(torch.stack([
                    band[:-1, :].flatten(),
                    band[1:, :].flatten()
                ]))[0, 1]
                spatial_corr += (h_corr + v_corr) / 2
            spatial_corr /= Z

        # Compute spectral correlation
        spectral_corr = 0.0
        if Z > 1:
            pixel_spectra = image.reshape(Z, -1).T  # [pixels, bands]
            if pixel_spectra.shape[0] > 1:
                corr_matrix = torch.corrcoef(pixel_spectra.T)
                # Average off-diagonal correlations
                mask = ~torch.eye(Z, dtype=torch.bool)
                spectral_corr = corr_matrix[mask].mean()

        # Recommend encoding order
        if spectral_corr > spatial_corr:
            recommended_order = EncodingOrder.BAND_INTERLEAVED
            reason = "High spectral correlation favors band-interleaved order"
        else:
            recommended_order = EncodingOrder.BAND_SEQUENTIAL
            reason = "High spatial correlation favors band-sequential order"

        return {
            'spatial_correlation': float(spatial_corr),
            'spectral_correlation': float(spectral_corr),
            'recommended_order': recommended_order,
            'reason': reason,
            'image_shape': (Z, Y, X)
        }

    @staticmethod
    def estimate_compression_benefit(image: torch.Tensor,
                                   order1: EncodingOrder,
                                   order2: EncodingOrder) -> dict:
        """
        Estimate relative compression benefit of two encoding orders

        Args:
            image: Input image tensor [Z, Y, X]
            order1, order2: Encoding orders to compare

        Returns:
            comparison: Dictionary with estimated benefits
        """
        # Simple heuristic based on prediction error variance
        def estimate_prediction_error(encoding_order):
            iterator = SampleIterator(image, encoding_order)
            errors = []

            for z, y, x, sample_value in iterator:
                # Simple prediction using available neighbors
                neighbors = iterator.get_spatial_neighbors(z, y, x, 3)
                neighbors.extend(iterator.get_spectral_neighbors(z, y, x))

                if neighbors:
                    # Average of available neighbors as prediction
                    neighbor_values = [image[nz, ny, nx] for nz, ny, nx in neighbors[:4]]
                    prediction = torch.stack(neighbor_values).mean()
                    error = abs(sample_value - prediction)
                    errors.append(float(error))
                else:
                    errors.append(0.0)  # No prediction error for first sample

            return torch.tensor(errors).var().item() if errors else 0.0

        var1 = estimate_prediction_error(order1)
        var2 = estimate_prediction_error(order2)

        if var1 > 0 and var2 > 0:
            benefit_ratio = var1 / var2 if var2 != 0 else 1.0
        else:
            benefit_ratio = 1.0

        return {
            'order1': order1,
            'order2': order2,
            'order1_prediction_variance': var1,
            'order2_prediction_variance': var2,
            'benefit_ratio': benefit_ratio,
            'recommended': order1 if var1 < var2 else order2
        }
