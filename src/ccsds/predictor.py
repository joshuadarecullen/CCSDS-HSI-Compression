import torch
import torch.nn as nn
from typing import Any, Optional


class SpectralPredictor(nn.Module):
    """
    CCSDS-123.0-B-2 Adaptive Linear Predictor

    This predictor uses adaptive linear prediction to predict the value of each
    image sample based on nearby samples in a 3D neighborhood.
    """

    def __init__(self,
                 num_bands: int,
                 dynamic_range: int = 16,
                 prediction_bands: Optional[int] = None,
                 prediction_mode: str = 'full',
                 local_sum_type: str = 'neighbor_oriented',
                 use_narrow_local_sums: bool = False,
                 weight_exponent: int = 4.,
                 weight_resolution: int = 19.,
                 weight_limit: int = 2**18.,
                 R: int = 32.,
                 damping: int = 0.,  # 0<=x<2^{theta}-1
                 offset: int = 0.,  # 0<=x<2^{theta}-1
                 theta: int = 2.  # 0<=x<=4
                 ) -> None:

        super().__init__()
        self.num_bands = num_bands
        self.dynamic_range = dynamic_range
        self.prediction_bands = prediction_bands or min(15, num_bands - 1)

        # Initialize prediction weights according to CCSDS standard
        # For full mode: P* spectral + 3 directional = total components
        # For reduced mode: P* spectral components only
        max_components = self.prediction_bands + 3  # Account for directional components
        self.register_buffer('weights', torch.zeros(num_bands, max_components))

        # Local sums for weight adaptation
        self.register_buffer('local_sums', torch.zeros(num_bands, 4))

        # Prediction mode parameters according to CCSDS-123.0-B-2 section 4.3
        self.prediction_mode = prediction_mode  # 'full' or 'reduced'
        self.use_narrow_local_sums = use_narrow_local_sums # Issue 2 narrow local sums option
        self.local_sum_type = local_sum_type  # 'neighbor_oriented' or 'column_oriented'

        # Number of local difference values used for prediction (C_z)
        self.total_components: int = None
        self._compute_prediction_components()

        # Weight scaling parameters according to CCSDS-123.0-B-2 equations (51)-(54)
        self.weight_exponent = weight_exponent  # Ω - weight scaling exponent (typically 4)
        self.weight_resolution = weight_resolution  # Resolution of weight updates  
        self.weight_limit = weight_limit  # Maximum weight magnitude limit

        # Define sample range parameters
        self.s_min = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0
        self.s_max = 2**(self.dynamic_range - 1) - 1
        self.s_mid = 2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0

        self.damping = 2**(theta)-1 if not damping else damping
        self.offset = offset  # 0 is lossless
        self.theta = theta

        # R has to be in range max(32, D+Ω+2) <= R <=64
        # Increasing R reduces the chance of an overflow in the calc of high-res
        # predicted sample value
        self.R = R if R >= max(32, dynamic_range+weight_exponent+2) and R <=64 else 64

    def _get_neighborhood_samples(self,
                                  image: torch.Tensor,
                                  z: int, y: int, x: int) -> torch.Tensor:
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

    def _compute_local_sum(self,
                           sample_representatives: torch.Tensor,
                           z: int,
                           y: int,
                           x: int,
                           wide: bool = True) -> torch.Tensor:
        """
        Compute local sum σ_{z,y,x} according to CCSDS-123.0-B-2 equations (20)-(23)

        Args:
            sample_representatives: Sample representatives tensor
            z, y, x: Current sample coordinates
            wide: Whether to use wide or narrow local sums

        Returns:
            local_sum: Computed according to standard equations
        """
        Z, Y, X = sample_representatives.shape

        if self.local_sum_type == 'column_oriented':
            # Column-oriented local sums - equations (22)-(23)
            if wide:
                # Wide column-oriented: σ = 4*s''_{z,y-1,x} when y>0, else 4*s_{z,y,x-1}
                if y > 0:
                    return 4 * sample_representatives[z, y-1, x]
                else:
                    if x > 0:
                        return 4 * sample_representatives[z, y, x-1]
                    else:
                        return torch.tensor(0.0, device=sample_representatives.device)
            else:
                # Narrow column-oriented: same as wide but with smid when both y=0,x=0,z=0
                if y > 0:
                    return 4 * sample_representatives[z, y-1, x]
                else:
                    if x > 0 and z > 0:
                        return 4 * sample_representatives[z, y, x-1]
                    elif y == 0 and x == 0 and z == 0:
                        # Use smid (mid-range value) for first sample
                        # mid_val = 2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0
                        return 4 * self.s_mid
                    else:
                        return 4 * sample_representatives[z, y, x-1] if x > 0 else torch.tensor(0.0, device=sample_representatives.device)
        else:
            # Neighbor-oriented local sums - equations (20)-(21)
            if wide:
                # Wide neighbor-oriented: full neighborhood
                if y > 0 and 0 < x < X - 1:
                    # Full neighborhood available
                    return (sample_representatives[z, y-1, x-1] +
                           sample_representatives[z, y-1, x+1] +
                           sample_representatives[z, y, x-1] +
                           sample_representatives[z, y-1, x])
                elif y == 0 and x > 0:
                    return 4 * sample_representatives[z, y, x-1]
                elif y > 0 and x == 0:
                    return 2 * (sample_representatives[z, y-1, x] + sample_representatives[z, y-1, x+1])
                elif y > 0 and x == X - 1:
                    return 2 * (sample_representatives[z, y, x-1] + sample_representatives[z, y-1, x-1])
                else:
                    # for position 0, 0
                    return torch.tensor(0.0, device=sample_representatives.device)
            else:
                # Narrow neighbor-oriented: excludes x-1 dependency
                if y > 0 and 0 < x < X - 1:
                    # 2*north + 2*s''_{z,y-1,x-1} + s''_{z,y-1,x+1}
                    return (2 * sample_representatives[z, y-1, x] +
                           2 * sample_representatives[z, y-1, x-1] +
                           sample_representatives[z, y-1, x+1])
                elif y == 0 and x > 0 and z > 0:
                    return 4 * sample_representatives[z, y, x-1]
                elif y > 0 and x == 0:
                    return 2 * (sample_representatives[z, y-1, x] + sample_representatives[z, y-1, x+1])
                elif y > 0 and x == X - 1:
                    return 2 * (sample_representatives[z, y-1, x-1] + sample_representatives[z, y-1, x])
                elif y == 0 and x > 0 and z == 0:
                    # mid_val = 2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0
                    return 4 * self.s_mid
                else:
                    # for pixel 0, 0
                    return torch.tensor(0.0, device=sample_representatives.device)

    def _update_weights(self,
                        prediction_error: torch.Tensor,
                        local_differences: torch.Tensor,
                        z: int) -> None:
        """
        Adapt prediction weights based on CCSDS-123.0-B-2 standard weight adaptation algorithm

        The standard specifies a more sophisticated weight update mechanism:
        1. Use scaled prediction error and local differences
        2. Apply proper damping based on local difference magnitudes
        3. Update weights using: Δw_i = 2^(-V_min) * e * d_i * 2^(-max(0, V_i - V_max))

        Where:
        - e is the prediction error
        - d_i is the local difference for component i
        - V_i is the magnitude parameter for local difference i
        - V_min, V_max are damping parameters
        """
        # CCSDS-123.0-B-2 weight adaptation parameters
        V_min = 4  # Minimum scaling parameter
        V_max = 6  # Maximum scaling parameter for damping

        if len(local_differences) >= 4:
            # Update weights for spatial components using CCSDS algorithm
            for i in range(4):
                d_i = local_differences[i].item()
                if abs(d_i) > 0:
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

                    weight_update = base_scale * prediction_error.item() * d_i * damping
                    self.weights[z, i] += weight_update

            # Update spectral weights (if any) with similar approach
            if self.prediction_bands > 0:
                for i in range(4, min(4 + self.prediction_bands, self.weights.size(1))):
                    # For spectral weights, use a simplified update based on prediction error
                    # This could be enhanced with spectral local differences
                    spectral_update = (2.0 ** (-V_min)) * prediction_error.item() * 0.1  # Small spectral factor
                    self.weights[z, i] += spectral_update

        # Apply weight clamping according to standard (typically broader range than simple implementation)
        # CCSDS standard allows for larger weight range for better adaptation
        self.weights[z] = torch.clamp(self.weights[z], -2.0, 2.0)

        # Update local sums for next iteration (running average of local differences)
        if len(local_differences) >= 4:
            # Use exponential moving average to track local sum statistics
            alpha = 0.1  # Smoothing factor
            for i in range(4):
                self.local_sums[z, i] = (1 - alpha) * self.local_sums[z, i] + alpha * local_differences[i]

    def _compute_narrow_local_sum(self, sample_representatives: torch.Tensor, z: int, y: int, x: int) -> torch.Tensor:
        """
        Compute narrow local sum for Issue 2 hardware pipelining optimization

        This eliminates the dependency on sample representative s''_{z,y,x-1}
        when performing prediction calculation for neighboring sample s^_{z,y,x}

        Args:
            sample_representatives: Sample representatives tensor
            z, y, x: Current sample coordinates

        Returns:
            narrow_local_sum: Computed local sum without x-1 dependency
        """
        if not self.use_narrow_local_sums:
            # Use standard local sum calculation
            return self.local_sums[z, :].sum()

        # Narrow local sum excludes the x-1 neighbor to break pipeline dependency
        narrow_sum = torch.tensor(0.0, device=sample_representatives.device)

        if self.local_sum_type == 'column_oriented':
            # Column-oriented local sum - uses vertical neighbors only
            if y > 0:  # North neighbor
                narrow_sum += sample_representatives[z, y-1, x]
            if y > 1:  # North-north neighbor
                narrow_sum += sample_representatives[z, y-2, x]
        else:
            # Neighbor-oriented (standard) but excluding x-1 dependency
            if y > 0:  # North neighbor
                narrow_sum += sample_representatives[z, y-1, x]
            if y > 0 and x > 0:  # Northwest neighbor
                narrow_sum += sample_representatives[z, y-1, x-1]
            # Intentionally exclude West neighbor (z, y, x-1) for pipeline optimization

        return narrow_sum

    def enable_narrow_local_sums(self, enable: bool = True, local_sum_type: str = 'neighbor_oriented') -> None:
        """
        Enable/disable narrow local sums for hardware pipelining optimization

        Args:
            enable: Whether to use narrow local sums
            local_sum_type: 'neighbor_oriented' or 'column_oriented'
        """
        self.use_narrow_local_sums = enable
        self.local_sum_type = local_sum_type

    def _compute_prediction_components(self) -> None:
        """
        Compute number of local difference values C_z according to equation (19)
        C_z = P*_z for reduced mode, P*_z + 3 for full mode
        """
        # P*_z = min{z, P} - number of preceding bands to use
        self.spectral_components = self.prediction_bands

        if self.prediction_mode == 'reduced':
            self.total_components = self.spectral_components
        else:  # full mode
            self.total_components = self.spectral_components + 3  # Add directional components

    def set_prediction_mode(self, mode: str) -> None:
        """
        Set prediction mode according to CCSDS-123.0-B-2 section 4.3

        Args:
            mode: 'full' or 'reduced'
        """
        if mode not in ['full', 'reduced']:
            raise ValueError("Prediction mode must be 'full' or 'reduced'")

        self.prediction_mode = mode
        self._compute_prediction_components()

    def get_prediction_mode_info(self) -> dict:
        """Get current prediction mode configuration"""
        return {
            'prediction_mode': self.prediction_mode,
            'use_narrow_local_sums': self.use_narrow_local_sums,
            'local_sum_type': self.local_sum_type,
            'prediction_bands': self.prediction_bands,
            'spectral_components': getattr(self, 'spectral_components', self.prediction_bands),
            'total_components': getattr(self, 'total_components', self.prediction_bands + 3)
        }

    def _compute_central_local_difference(self,
                                          sample_representatives: torch.Tensor,
                                          local_sum: torch.Tensor,
                                          z: int,
                                          y: int,
                                          x: int) -> torch.Tensor:
        """
        Compute central local difference d_{z,y,x} according to equation (24)
        d_{z,y,x} = 4*s''_{z,y,x} - σ_{z,y,x}
        """
        if y == 0 and x == 0:
            return torch.tensor(0.0, device=sample_representatives.device)

        return 4 * sample_representatives[z, y, x] - local_sum

    def _compute_directional_local_differences(self,
                                               sample_representatives: torch.Tensor,
                                               local_sum: torch.Tensor,
                                               z: int,
                                               y: int,
                                               x: int) -> torch.Tensor:
        """
        Compute directional local differences according to equations (25)-(27)

        Returns:
            [d^N, d^W, d^NW] tensor
        """
        Z, Y, X = sample_representatives.shape

        if y == 0 and x == 0:
            return torch.zeros(3, device=sample_representatives.device)

        # North directional difference - equation (25)
        if y > 0:
            d_N = 4 * sample_representatives[z, y-1, x] - local_sum
        else:
            d_N = torch.tensor(0.0, device=sample_representatives.device)

        # West directional difference - equation (26)
        if x > 0 and y > 0:
            d_W = 4 * sample_representatives[z, y, x-1] - local_sum
        elif x == 0 and y > 0:
            d_W = 4 * sample_representatives[z, y-1, x] - local_sum
        else:
            d_W = torch.tensor(0.0, device=sample_representatives.device)

        # Northwest directional difference - equation (27)
        if x > 0 and y > 0:
            d_NW = 4 * sample_representatives[z, y-1, x-1] - local_sum
        elif x == 0 and y > 0:
            d_NW = 4 * sample_representatives[z, y-1, x] - local_sum
        else:
            d_NW = torch.tensor(0.0, device=sample_representatives.device)

        return torch.stack([d_N, d_W, d_NW])

    def predict_sample(self,
                       image: torch.Tensor,
                       sample_representatives: torch.Tensor,
                       z: int,
                       y: int,
                       x: int) -> torch.Tensor:
        """
        Predict a single sample value according to CCSDS-123.0-B-2 section 4.7

        Args:
            image: Original image (for initialization)
            sample_representatives: Sample representatives s''_{z,y,x} used for prediction
            z, y, x: Sample coordinates

        Returns:
            Predicted sample value \hat{s}_z(t)
        """
        # Compute local sum according to CCSDS standard
        local_sum = self._compute_local_sum(sample_representatives,
                                            z,
                                            y,
                                            x,
                                            wide=not self.use_narrow_local_sums)

        # Compute P*_z - number of preceding spectral bands to use
        P_star_z = min(self.prediction_bands, z)

        # Build local difference vector U_z(t) according to equations (28)-(29)
        local_differences = []

        # Add central local differences from preceding P*_z bands
        for i in range(P_star_z):
            band_idx = z - 1 - i  # Previous bands: z-1, z-2, ..., z-P*_z

            prev_local_sum = self._compute_local_sum(sample_representatives,
                                                     band_idx,
                                                     y,
                                                     x,
                                                     wide=not self.use_narrow_local_sums)

            prev_central_diff = self._compute_central_local_difference(sample_representatives,
                                                                       prev_local_sum,
                                                                       band_idx,
                                                                       y,
                                                                       x)
            local_differences.append(prev_central_diff)

        # if x == 2 and y == 2 and z > 0:
        print(f'\nz:{z}, x:{x}, y:{y}')
        print(f'Prediction bands: {P_star_z}')
        print(local_differences)
        print(f'Local Sum: {local_sum}')

        # For full prediction mode, add directional differences from current band
        if self.prediction_mode == 'full' and (y > 0 or x > 0):
            directional_diffs = self._compute_directional_local_differences(sample_representatives, local_sum, z, y, x)
            local_differences.extend(directional_diffs)

        # Compute predicted central local difference \hat{d}_z(t) = W_z^T(t) * U_z(t)
        predicted_local_diff = torch.tensor(0.0, device=image.device)

        # For z=0 under reduced mode, \hat{d}_z(t) = 0 (equation 36)
        if z == 0 and self.prediction_mode == 'reduced':
            predicted_local_diff = torch.tensor(0.0, device=image.device)
        else:
            for i, diff in enumerate(local_differences):
                if i < self.weights.size(1):
                    predicted_local_diff += self.weights[z, i] * diff

        print(f'predicted local diff {predicted_local_diff.item()}')
        # Apply clipping to valid sample range
        high_res_prediction = self._prediction_calculation(local_sum,
                                                           predicted_local_diff,
                                                           x=y,
                                                           y=y,
                                                           z=z,
                                                           sample_representatives=sample_representatives)
        print(f'Modular {high_res_prediction}')

        # Convert from high-resolution to regular prediction by scaling down
        # The high-resolution prediction needs to be scaled by 2^(-Ω) to get regular prediction
        prediction = high_res_prediction / (2**self.weight_exponent)
        print(f'High res prediction {prediction}')
        print(f'Prediction {torch.clamp(prediction, self.s_min, self.s_max)}')

        # Final clipping to ensure within valid range
        return torch.clamp(prediction, self.s_min, self.s_max)

    def _prediction_calculation(self,
                                local_sum: torch.Tensor,
                                predicted_local_diff: torch.Tensor,
                                x: int,
                                y: int,
                                z: int,
                                sample_representatives: torch.Tensor = None) -> torch.Tensor:
        """
        Compute predicted sample \hat{s}_z(t) using CCSDS-123.0-B-2
        Equations (37)–(39).

        Args:
            local_sum: σ_z(t), local sum
            predicted_local_diff: \hat{d}_z(t), predicted central local difference
            t: raster index in current band
            z: spectral band index
            P: number of previous bands used
            prev_band_sample: s_{z-1}(t), needed when t=0 and z>0

        Returns:
            Predicted sample \hat{s}_z(t)
        """
        t = y*sample_representatives.shape[-1] + x

        # ------------------------------------------------------------
        # Eq. (37): High-resolution predicted sample
        # ------------------------------------------------------------
        term1 = predicted_local_diff
        term2 = (2**self.weight_exponent) * (local_sum - 4*self.s_mid)
        term3 = (2**(self.weight_exponent + 1)) * self.s_mid
        term4 = 2**(self.weight_exponent + 1)

        # unclipped_sum = term1 + term2 + term3 + term4

        # Modular arithmetic mod_R
        # modular_result = ((unclipped_sum - self.s_min) % self.R) + self.s_min
        # print(f'Mod results b4 clamp {modular_result}')
        modular_result = ((term1 + term2) % self.R) + term3 + term4
        print(f'Mod results b4 clamp {modular_result}')

        # Clip to extended range
        s_tilde = torch.clamp(
            modular_result,
            2**(self.weight_exponent + 2) * self.s_min,
            2**(self.weight_exponent + 2) * self.s_max + 2**(self.weight_exponent + 1)
        )

        # ------------------------------------------------------------
        # Eq. (38): Double-resolution predicted sample
        # ------------------------------------------------------------
        if t > 0:
            s_double = torch.floor_divide(s_tilde, 2**(self.weight_exponent + 1))
        else:
            if self.prediction_bands > 0 and z > 0 and t == 0:
                s_double = 2 * sample_representatives[z, x-1, y]
            elif t == 0 and (self.prediction_bands == 0 or z == 0):
                s_double = 2 * self.s_mid

        # ------------------------------------------------------------
        # Eq. (39): Final predicted sample
        # ------------------------------------------------------------
        s_hat = torch.floor_divide(s_double, 2)

        return s_hat

    def _modular_arithmetic(self, local_sum: torch.Tensor,
                            predicted_local_diff: torch.Tensor) -> torch.Tensor:
        # Compute high-resolution predicted sample value using CCSDS-123.0-B-2 equation (37)
        # \tilde{s}_z(t) = clip[mod_R[\hat{d}_z(t) + 2^\Omega * (\sigma_z(t) - 4*s_mid) + 2^{\Omega+1}*s_mid + 2^{\Omega+1}], {s_min, s_max}]

        print(f'local sum: {local_sum}')
        print(f'after weights: {predicted_local_diff}')

        # if x==0 and y==0 and self.predicted_local_diff
        # Compute terms of equation (37)
        term1 = predicted_local_diff  # \hat{d}_z(t)
        term2 = (2**self.weight_exponent) * (local_sum - 4*self.s_mid)  # 2^\Omega * (\sigma_z(t) - 4*s_mid)
        term3 = (2**(self.weight_exponent + 1)) * self.s_mid  # 2^{\Omega+1}*s_mid
        term4 = 2**(self.weight_exponent + 1)  # 2^{\Omega+1}

        # Sum all terms
        unclipped_sum = term1 + term2 + term3 + term4

        # Apply modular arithmetic mod_R
        # mod_R(x) = ((x - s_min) % R) + s_min for range [s_min, s_max]
        modular_result = ((unclipped_sum - self.s_min) % self.R) + self.s_min

        return torch.clamp(modular_result, self.s_min, self.s_max)

    def _compute_sample_representative(self,
                                       original_sample: torch.Tensor,
                                       predicted_sample: torch.Tensor,
                                       quantizer_index: torch.Tensor,
                                       max_error: torch.Tensor,
                                       damping: int = 0,
                                       offset: int = 0,
                                       theta: int = 0) -> torch.Tensor:
        """
        Compute sample representative s''_z(t) according to CCSDS-123.0-B-2
        Equations (46)–(48).

        Args:
            original_sample: Original sample value s_z(t)
            predicted_sample: Predicted sample value \hat{s}_z(t)
            quantizer_index: Quantizer index q_z(t)
            max_error: Maximum error m_z(t)
            damping: Damping parameter φ_z
            offset: Offset parameter ψ_z
            theta: Resolution parameter Θ (0 ≤ Θ ≤ 4)

        Returns:
            Sample representative s''_z(t)
        """

        # --- Equation (48): quantizer bin center (clipped)
        quantizer_bin_center = predicted_sample + quantizer_index * (2 * max_error + 1)
        s_prime = torch.clamp(quantizer_bin_center, self.s_min, self.s_max)

        # Special case: t = 0 → representative is just the original sample
        if torch.numel(original_sample) == 1 and original_sample.item() == 0:
            return original_sample

        # --- Equation (47): double-resolution representative
        # sign of quantizer index
        sign_q = torch.sign(quantizer_index)

        numerator = (
            4 * (2**self.theta - self.damping) *
            (s_prime * (2**self.theta) - sign_q * max_error * self.offset * (2**(2**self.theta - self.theta)))
            + self.damping * predicted_sample - self.damping * (2**(self.theta + 1))
        )
        double_res = torch.floor_divide(numerator, 2**(self.theta + 1))

        # --- Equation (46): single-resolution representative
        s_double_prime = torch.floor_divide(double_res + 1, 2)

        return s_double_prime

    def forward(self, image: torch.Tensor, max_errors: torch.Tensor = None) -> torch.Tensor:
        """
        Predict all samples in the image using proper sample representatives

        Args:
            image: [Z, Y, X] multispectral/hyperspectral image
            max_errors: [Z, Y, X] maximum error values m_z(t) (None for lossless)

        Returns:
            predictions: [Z, Y, X] predicted values
            residuals: [Z, Y, X] prediction residuals
            sample_representatives: [Z, Y, X] sample representatives s''_z(t)
        """
        Z, Y, X = image.shape
        predictions = torch.zeros_like(image)
        residuals = torch.zeros_like(image)

        # Initialize sample representatives - these are computed during compression
        sample_representatives = torch.zeros_like(image)

        # Default to lossless compression if no max_errors provided
        if max_errors is None:
            max_errors = torch.zeros_like(image)

        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    # For the first sample, initialize sample representative with original value
                    if z == 0 and y == 0 and x == 0:
                        sample_representatives[z, y, x] = image[z, y, x]

                    # Predict sample using current sample representatives
                    pred = self.predict_sample(image, sample_representatives, z, y, x)
                    predictions[z, y, x] = pred

                    # Compute prediction residual
                    residual = image[z, y, x] - pred
                    residuals[z, y, x] = residual

                    # Quantize the residual (simplified - assuming lossless for now)
                    max_error = max_errors[z, y, x].int().item() if max_errors is not None else 0
                    if max_error == 0:
                        quantizer_index = residual  # Lossless: q_z(t) = \Delta_z(t)
                    else:
                        # Near-lossless quantization (simplified)
                        step_size = 2 * max_error + 1
                        quantizer_index = torch.round(residual / step_size)

                    # Compute sample representative for this sample
                    sample_rep = self._compute_sample_representative(
                        image[z, y, x], pred, quantizer_index, max_error
                    )
                    sample_representatives[z, y, x] = sample_rep

                    # Update weights based on prediction error according to CCSDS standard
                    if z > 0 or y > 0 or x > 0:  # Skip first sample
                        # Build the same local difference vector used in prediction
                        P_star_z = min(self.prediction_bands, z)
                        local_differences = []

                        # Add central local differences from preceding P*_z bands
                        for i in range(P_star_z):
                            band_idx = z - 1 - i
                            prev_local_sum = self._compute_local_sum(sample_representatives,
                                                                     band_idx,
                                                                     y,
                                                                     x,
                                                                     wide=not self.use_narrow_local_sums)
                            prev_central_diff = self._compute_central_local_difference(sample_representatives, prev_local_sum, band_idx, y, x)
                            local_differences.append(prev_central_diff)

                        # For full mode, add directional differences from current band
                        if self.prediction_mode == 'full':
                            current_local_sum = self._compute_local_sum(sample_representatives, z, y, x, wide=not self.use_narrow_local_sums)
                            directional_diffs = self._compute_directional_local_differences(sample_representatives, current_local_sum, z, y, x)
                            local_differences.extend(directional_diffs)

                        if local_differences:
                            all_diffs = torch.stack(local_differences)
                            self._update_weights(residual, all_diffs, z)

        return predictions, residuals, sample_representatives


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
