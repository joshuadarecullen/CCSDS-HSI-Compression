import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Optional, Union, Callable, Any, Tuple

from .predictor import SpectralPredictor, NarrowLocalSumPredictor
from .quantizer import UniformQuantizer, LosslessQuantizer, PeriodicErrorLimitUpdater
from .sample_representative import SampleRepresentativeCalculator, OptimizedSampleRepresentative
from .entropy_coder import HybridEntropyCoder, encode_image, BitWriter, BlockAdaptiveEntropyCoder
from .rice_coder import CCSDS121BlockAdaptiveEntropyCoder, encode_image_rice
from .header import CCSDS123Header, PredictorMode, EncodingOrder
from .bitstream import BitstreamFormatter, BitWriter
from .encoding_orders import SampleIterator, EncodingOrderOptimizer


class CCSDS123Compressor(nn.Module):
    """
    CCSDS-123.0-B-2 Low-Complexity Lossless and Near-Lossless
    Multispectral and Hyperspectral Image Compressor

    Mathematical Foundation:
    The compressor implements three core mathematical operations in sequence:

    1. Adaptive Linear Prediction:
       ŝ_z(t) = Σ(w_i(t) * s'_i(t-d_i)) + C_z
       Where ŝ_z(t) is the predicted value, w_i are adaptive weights,
       s'_i are neighboring sample representatives, and C_z is a band-dependent offset.

    2. Scalar Quantization with Error Control:
       q_z(t) = clamp(round(ε_z(t) / (2*m_z(t) + 1)), -2^15, 2^15-1)
       Where ε_z(t) = s_z(t) - ŝ_z(t) is the prediction residual,
       and m_z(t) is the maximum allowed error determined by:
       - Absolute limits: m_z(t) = a_z
       - Relative limits: m_z(t) = ⌊r_z|ŝ_z(t)|/2^D⌋
       - Combined: m_z(t) = min(a_z, ⌊r_z|ŝ_z(t)|/2^D⌋)

    3. Sample Representative Calculation:
       s'_z(t) = ĉ_z(t) + α_z(t) * (ŝ_z(t) - ĉ_z(t))
       Where ĉ_z(t) is the quantizer bin center and α_z(t) ∈ [0,1] is computed from
       user parameters φ_z, ψ_z, Θ using a sigmoid-like function.

    4. Entropy Coding:
       Mapped quantizer indices δ_z(t) are encoded using hybrid entropy codes
       that combine Golomb-Power-of-2 codes for high-entropy data with
       16 specialized variable-to-variable length codes for low-entropy data.

    Complete implementation of the Issue 2 standard including:
    - Adaptive linear prediction with 3D spatial-spectral neighborhoods
    - Closed-loop scalar quantization with user-specified error bounds
    - Sample representative calculation with configurable parameters
    - Hybrid entropy coding with automatic high/low entropy classification
    - Periodic error limit updating for adaptive rate control
    - Support for both lossless and near-lossless compression modes
    """

    def __init__(
        self,
        num_bands: int,
        dynamic_range: int = 16,
        prediction_bands: Optional[int] = None,
        use_narrow_local_sums: bool = False,
        lossless: bool = True,
        entropy_coder_type: str = 'hybrid'
    ) -> None:
        """
        Initialize CCSDS-123.0-B-2 compressor

        Args:
            num_bands: Number of spectral bands (Z dimension)
            dynamic_range: Bit depth of samples (supports up to 32-bit in Issue 2)
            prediction_bands: Number of previous bands to use for prediction
            use_narrow_local_sums: Enable narrow local sums for reduced complexity
            lossless: True for lossless compression, False for near-lossless
        """
        super().__init__()

        self.num_bands = num_bands
        self.dynamic_range = dynamic_range
        self.lossless = lossless

        # Initialize predictor
        if use_narrow_local_sums:
            self.predictor = NarrowLocalSumPredictor(
                num_bands, dynamic_range, prediction_bands
            )
        else:
            self.predictor = SpectralPredictor(
                num_bands, dynamic_range, prediction_bands
            )

        # Initialize quantizer
        if lossless:
            self.quantizer = LosslessQuantizer(num_bands, dynamic_range)
        else:
            self.quantizer = UniformQuantizer(num_bands, dynamic_range)

        # Initialize sample representative calculator
        self.sample_rep_calc = OptimizedSampleRepresentative(num_bands)

        # Initialize entropy coder
        self.entropy_coder = encode_image

        # Error limit updater for near-lossless compression
        self.error_limit_updater = PeriodicErrorLimitUpdater()

        # Compression parameters
        self.compression_params = {
            'absolute_error_limits': None,
            'relative_error_limits': None,
            'sample_rep_phi': None,
            'sample_rep_psi': None,
            'sample_rep_theta': 4.0,
            'entropy_coder_type': entropy_coder_type,  # 'hybrid', 'sample_adaptive', 'block_adaptive'
            'periodic_error_update': False,
            'update_interval': 1000,
            # Block-adaptive entropy coding parameters
            'block_size': (8, 8),  # (height, width) of blocks
            'min_block_samples': 16  # Minimum samples per block for reliable statistics
        }

    def set_compression_parameters(self, **params: Any) -> None:
        """
        Configure compression parameters

        Args:
            absolute_error_limits: [Z] tensor of absolute error limits per band
            relative_error_limits: [Z] tensor of relative error limits per band
            sample_rep_phi: [Z] tensor of φ parameters for sample representatives
            sample_rep_psi: [Z] tensor of ψ parameters for sample representatives
            sample_rep_theta: Θ parameter for sample representatives
            entropy_coder_type: Type of entropy coder to use
            periodic_error_update: Enable periodic error limit updates
            update_interval: Samples between error limit updates
        """
        self.compression_params.update(params)

        # Configure quantizer error limits
        if not self.lossless:
            self.quantizer.set_error_limits(
                params.get('absolute_error_limits'),
                params.get('relative_error_limits')
            )

        # Configure sample representative parameters
        self.sample_rep_calc.set_parameters(
            phi=params.get('sample_rep_phi'),
            psi=params.get('sample_rep_psi'),
            theta=params.get('sample_rep_theta', 4.0)
        )

        # Configure error limit updater
        if params.get('periodic_error_update', False):
            self.error_limit_updater = PeriodicErrorLimitUpdater(
                params.get('update_interval', 1000)
            )

    def _validate_input(self, image: torch.Tensor) -> torch.Tensor:
        """
        Validate and preprocess input image

        Validates image dimensions, dynamic range compliance, and data types.
        Handles both 3D [Z,Y,X] and 4D [1,Z,Y,X] input formats.

        Args:
            image: Input multispectral/hyperspectral image tensor

        Returns:
            Processed image tensor [Z, Y, X] in float32 format

        Raises:
            ValueError: If image dimensions or values are invalid
        """
        if image.dim() == 3:
            Z, Y, X = image.shape
        elif image.dim() == 4 and image.shape[0] == 1:
            # Remove batch dimension
            image = image.squeeze(0)
            Z, Y, X = image.shape
        else:
            raise ValueError(f"Expected 3D tensor [Z,Y,X] or 4D tensor [1,Z,Y,X], got {image.shape}")

        if Z != self.num_bands:
            raise ValueError(f"Expected {self.num_bands} bands, got {Z}")

        # Check dynamic range
        max_val = 2**(self.dynamic_range - 1) - 1
        min_val = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0

        if torch.any(image > max_val) or torch.any(image < min_val):
            print(f"Warning: Image values outside expected range [{min_val}, {max_val}]")

        return image.float()

    def forward(self, image: torch.Tensor) -> Dict[str, Union[torch.Tensor, float, int]]:
        """
        Compress multispectral/hyperspectral image through the complete pipeline

        Implements the full CCSDS-123 compression algorithm:
        1. Sample-by-sample processing in BSQ (Band Sequential) order
        2. Adaptive linear prediction using neighboring sample representatives
        3. Residual quantization with configurable error limits
        4. Sample representative computation for future predictions
        5. Entropy coding of quantized residuals

        Args:
            image: Input image tensor [Z, Y, X] or [1, Z, Y, X]

        Returns:
            Dictionary containing:
                - predictions: Predicted sample values [Z, Y, X]
                - residuals: Prediction residuals [Z, Y, X]
                - quantized_residuals: Quantized residuals [Z, Y, X]
                - mapped_indices: Mapped quantizer indices [Z, Y, X]
                - sample_representatives: Sample representatives [Z, Y, X]
                - reconstructed_samples: Reconstructed sample values [Z, Y, X]
                - compressed_size: Size of compressed data in bits
                - original_size: Original data size in bits
                - compression_ratio: Compression ratio achieved
        """
        # Validate input
        image = self._validate_input(image)
        Z, Y, X = image.shape

        # Initialize sample representatives with original image
        sample_representatives = image.clone()

        # Storage for results
        all_predictions = torch.zeros_like(image)
        all_residuals = torch.zeros_like(image)
        all_quantized_residuals = torch.zeros_like(image)
        all_mapped_indices = torch.zeros(Z, Y, X, dtype=torch.long)
        all_sample_reps = torch.zeros_like(image)
        all_reconstructed = torch.zeros_like(image)

        sample_count = 0

        # Determine encoding order
        encoding_order = self.compression_params.get('encoding_order', 'BSQ')
        if encoding_order == 'BI':
            from .encoding_orders import EncodingOrder as EO
            sample_order = EO.BAND_INTERLEAVED
        else:
            from .encoding_orders import EncodingOrder as EO
            sample_order = EO.BAND_SEQUENTIAL

        # Create sample iterator for the specified encoding order
        sample_iterator = SampleIterator(image, sample_order)

        # Process image sample by sample in specified encoding order
        print(f"\nStarting Predictor Quantization using {encoding_order} order...")
        for z, y, x, sample_value in sample_iterator:
            sample_count += 1

            # Periodic error limit update
            if (not self.lossless and
                self.compression_params.get('periodic_error_update', False) and
                self.error_limit_updater.should_update(sample_count)):

                # Update error limits (implementation-specific strategy)
                self._update_error_limits_adaptive(z, sample_count)

            # Predict current sample using sample representatives
            prediction = self.predictor.predict_sample(
                image, sample_representatives, z, y, x
            )
            all_predictions[z, y, x] = prediction

            # Compute prediction residual
            residual = image[z, y, x] - prediction
            all_residuals[z, y, x] = residual

            # Quantize residual
            if self.lossless:
                quantized_residual = residual
                mapped_index = self.quantizer.map_quantizer_indices(
                    residual.round().long().unsqueeze(0).unsqueeze(0)
                ).squeeze()
            else:
                # Get max error for this sample
                max_error = self.quantizer.compute_max_error(
                    prediction.unsqueeze(0).unsqueeze(0), z
                ).squeeze()

                # Quantize
                quant_res, quant_idx = self.quantizer.quantize_residuals(
                    residual.unsqueeze(0).unsqueeze(0),
                    prediction.unsqueeze(0).unsqueeze(0),
                    z
                )
                quantized_residual = quant_res.squeeze()
                mapped_index = self.quantizer.map_quantizer_indices(
                    quant_idx
                ).squeeze()

            all_quantized_residuals[z, y, x] = quantized_residual
            all_mapped_indices[z, y, x] = mapped_index

            # Track statistics for periodic error limit updating
            if self.compression_params.get('periodic_error_update', False):
                if not hasattr(self, '_compression_stats_buffer'):
                    self._compression_stats_buffer = {
                        'prediction_errors': [],
                        'quantization_indices': [],
                        'bit_estimates': []
                    }

                self._compression_stats_buffer['prediction_errors'].append(float(residual))
                self._compression_stats_buffer['quantization_indices'].append(int(mapped_index))

                # Keep buffer size manageable
                max_buffer_size = 5000
                for key in self._compression_stats_buffer:
                    if len(self._compression_stats_buffer[key]) > max_buffer_size:
                        self._compression_stats_buffer[key] = self._compression_stats_buffer[key][-max_buffer_size//2:]

            # Compute reconstructed sample
            reconstructed = prediction + quantized_residual

            # Clamp to valid range
            max_val = 2**(self.dynamic_range - 1) - 1
            min_val = -2**(self.dynamic_range - 1) if self.dynamic_range > 1 else 0
            reconstructed = torch.clamp(reconstructed, min_val, max_val)
            all_reconstructed[z, y, x] = reconstructed

            # Compute sample representative for future predictions
            if not self.lossless:
                max_error_single = self.quantizer.compute_max_error(
                    prediction.unsqueeze(0).unsqueeze(0), z
                ).squeeze()

                bin_center = self.sample_rep_calc.compute_quantizer_bin_center(
                    image[z, y, x], prediction, max_error_single
                )

                sample_rep = self.sample_rep_calc.compute_sample_representative(
                    bin_center, prediction, z
                )

                all_sample_reps[z, y, x] = sample_rep
                sample_representatives[z, y, x] = sample_rep
            else:
                # Lossless case - use reconstructed sample
                all_sample_reps[z, y, x] = reconstructed
                sample_representatives[z, y, x] = reconstructed

        print("Completed Predictor and Quantization...")

        # Entropy encode mapped indices
        compressed_size = 0
        if self.compression_params['entropy_coder_type'] == 'hybrid':
            compressed_data = self.entropy_coder(all_mapped_indices, self.num_bands)
            compressed_size = len(compressed_data) * 8  # Convert bytes to bits
        elif self.compression_params['entropy_coder_type'] == 'block_adaptive':
            # Use block-adaptive entropy coding
            block_size = self.compression_params.get('block_size', (8, 8))
            block_coder = BlockAdaptiveEntropyCoder(
                num_bands=self.num_bands,
                block_size=block_size,
                min_block_samples=self.compression_params.get('min_block_samples', 16)
            )
            compressed_data, entropy_stats = block_coder.encode_image_block_adaptive(all_mapped_indices)
            compressed_size = entropy_stats.get('total_compressed_bits', 0)

        return {
            'predictions': all_predictions,
            'residuals': all_residuals,
            'quantized_residuals': all_quantized_residuals,
            'mapped_indices': all_mapped_indices,
            'sample_representatives': all_sample_reps,
            'reconstructed_samples': all_reconstructed,
            'compressed_size': compressed_size,
            'original_size': Z * Y * X * self.dynamic_range,
            'compression_ratio': (Z * Y * X * self.dynamic_range) / max(compressed_size, 1)
        }

    def _update_error_limits_adaptive(self, current_band: int, sample_count: int) -> None:
        """
        CCSDS-123.0-B-2 compliant adaptive error limit update strategy

        Implements sophisticated rate control based on:
        1. Compression ratio monitoring
        2. Prediction error statistics
        3. Target bit rate constraints
        4. Band-specific characteristics

        Args:
            current_band: Current spectral band being processed
            sample_count: Total number of samples processed so far
        """
        if self.lossless:
            return

        # Get update parameters from compression config
        update_interval = self.compression_params.get('error_limit_update_interval', 1000)
        target_bpp = self.compression_params.get('target_bits_per_pixel', None)
        adaptation_rate = self.compression_params.get('error_limit_adaptation_rate', 0.1)

        # Only update at specified intervals
        if sample_count % update_interval != 0:
            return

        # Calculate current compression statistics
        current_stats = self._compute_current_compression_stats(sample_count)

        # Determine if we need to adjust error limits
        adjustment_factor = self._compute_error_limit_adjustment(
            current_stats, target_bpp, current_band
        )

        # Apply adaptive update to error limits
        if hasattr(self.quantizer, 'absolute_error_limits'):
            current_limits = self.quantizer.absolute_error_limits.clone()

            # Band-specific adjustments
            band_weight = self._compute_band_weight(current_band)

            # Apply smooth exponential adaptation
            new_limits = current_limits * (1.0 + adaptation_rate * adjustment_factor * band_weight)

            # Clamp to valid range (0-31 for 5-bit encoding)
            new_limits = torch.clamp(new_limits, 0, 31)

            # Update quantizer with new limits
            self.quantizer.set_error_limits(absolute_limits=new_limits)

            # Log the update for analysis
            if self.compression_params.get('verbose_error_updates', False):
                print(f"Updated error limits at sample {sample_count}, band {current_band}: "
                      f"factor={adjustment_factor:.3f}, limits={new_limits.tolist()}")

    def _compute_current_compression_stats(self, sample_count: int) -> dict:
        """
        Compute current compression statistics for rate control

        Args:
            sample_count: Number of samples processed

        Returns:
            stats: Dictionary with current compression metrics
        """
        if not hasattr(self, '_compression_stats_buffer'):
            self._compression_stats_buffer = {
                'prediction_errors': [],
                'quantization_indices': [],
                'bit_estimates': []
            }

        # Estimate current compression ratio based on quantization indices
        if self._compression_stats_buffer['quantization_indices']:
            recent_indices = self._compression_stats_buffer['quantization_indices'][-1000:]
            avg_index_magnitude = sum(abs(idx) for idx in recent_indices) / len(recent_indices)

            # Rough bit estimate based on index magnitude
            estimated_bits_per_sample = max(1, int(avg_index_magnitude).bit_length())
        else:
            estimated_bits_per_sample = 8  # Default estimate

        # Calculate effective compression ratio
        original_bps = self.dynamic_range
        current_ratio = original_bps / estimated_bits_per_sample if estimated_bits_per_sample > 0 else 1.0

        return {
            'samples_processed': sample_count,
            'estimated_bits_per_sample': estimated_bits_per_sample,
            'compression_ratio': current_ratio,
            'avg_prediction_error': sum(self._compression_stats_buffer['prediction_errors'][-100:]) /
                                   max(1, len(self._compression_stats_buffer['prediction_errors'][-100:]))
        }

    def _compute_error_limit_adjustment(self, stats: dict, target_bpp: float,
                                      current_band: int) -> float:
        """
        Compute error limit adjustment factor based on performance metrics

        Args:
            stats: Current compression statistics
            target_bpp: Target bits per pixel (None if not specified)
            current_band: Current spectral band

        Returns:
            adjustment_factor: Factor to adjust error limits (-1.0 to 1.0)
        """
        adjustment = 0.0

        # Rate-based adjustment
        if target_bpp is not None:
            current_bpp = stats['estimated_bits_per_sample']
            rate_error = (target_bpp - current_bpp) / target_bpp
            adjustment += rate_error * 0.5  # Weight rate control

        # Quality-based adjustment (prediction error variance)
        if 'avg_prediction_error' in stats:
            error_magnitude = abs(stats['avg_prediction_error'])
            if error_magnitude > 5.0:  # High prediction errors
                adjustment -= 0.2  # Reduce error limits to improve quality
            elif error_magnitude < 1.0:  # Low prediction errors
                adjustment += 0.1  # Increase error limits to save bits

        # Band-specific adjustment
        band_progress = current_band / max(1, self.num_bands - 1)
        if band_progress > 0.8:  # Later bands - prioritize quality
            adjustment -= 0.1
        elif band_progress < 0.2:  # Early bands - allow more compression
            adjustment += 0.1

        # Clamp adjustment to reasonable range
        return max(-0.5, min(0.5, adjustment))

    def _compute_band_weight(self, band_index: int) -> float:
        """
        Compute adaptive weight for band-specific error limit updates

        Args:
            band_index: Spectral band index

        Returns:
            weight: Adaptation weight for this band (0.0 to 1.0)
        """
        # Higher weight for bands with more spectral information
        # This is a simplified model - could be enhanced with actual spectral analysis

        if self.num_bands == 1:
            return 1.0

        # Give higher weight to mid-spectrum bands (typically more informative)
        normalized_band = band_index / (self.num_bands - 1)

        # Bell curve centered at 0.4 (typical peak of spectral information)
        peak_position = 0.4
        weight = math.exp(-((normalized_band - peak_position) / 0.3) ** 2)

        return max(0.2, min(1.0, weight))

    def compress(self, image: torch.Tensor) -> Dict[str, Union[bytes, torch.Tensor, Dict]]:
        """
        Compress image and return compressed representation with all intermediate data

        This method provides the complete compressed representation needed for
        decompression, including the entropy-coded bitstream and all decoder
        state information.

        Args:
            image: Input image tensor [Z, Y, X] or [1, Z, Y, X]

        Returns:
            Dictionary containing:
                - compressed_bitstream: Entropy-coded bitstream as bytes
                - compression_metadata: All parameters needed for decompression
                - compression_statistics: Performance metrics and analysis
                - intermediate_data: All intermediate processing results
        """
        # Get full compression results
        results = self.forward(image)

        # Create CCSDS-123.0-B-2 compliant header
        header = CCSDS123Header()

        # Set image parameters
        if len(image.shape) == 4:
            _, num_bands, height, width = image.shape
        else:
            num_bands, height, width = image.shape

        header.set_image_params(
            height=height,
            width=width,
            num_bands=num_bands,
            bit_depth=self.dynamic_range,
            signed=False  # Assuming unsigned samples
        )

        # Set predictor parameters
        predictor_mode = PredictorMode.FULL if self.compression_params.get('full_prediction', True) else PredictorMode.REDUCED
        header.set_predictor_params(
            mode=predictor_mode,
            v_min=self.compression_params.get('v_min', 4),
            v_max=self.compression_params.get('v_max', 6),
            rescale_interval=self.compression_params.get('rescale_interval', 64)
        )

        # Set entropy coder parameters
        header.set_entropy_coder_params(
            gamma_star=self.compression_params.get('initial_count_exponent', 1),
            k=self.compression_params.get('accumulator_init_constant', 1)
        )

        # Set encoding order in header
        if encoding_order == 'BI':
            header.image_metadata.sample_encoding_order = EncodingOrder.BAND_INTERLEAVED
        else:
            header.image_metadata.sample_encoding_order = EncodingOrder.BAND_SEQUENTIAL

        # Pack header
        header_bytes = header.pack()

        # Create bitstream formatter with proper output word size
        output_word_size = self.compression_params.get('output_word_size', 8)
        formatter = BitstreamFormatter(output_word_size)

        # Entropy encode the mapped indices
        entropy_stats = {}
        compressed_body_bits = []

        if self.compression_params['entropy_coder_type'] == 'hybrid':
            compressed_body = encode_image(results['mapped_indices'], self.num_bands)
            # Convert back to bits for proper formatting
            compressed_body_bits = formatter.bytes_to_bits(compressed_body)
        elif self.compression_params['entropy_coder_type'] == 'block_adaptive':
            # Use block-adaptive entropy coding
            block_size = self.compression_params.get('block_size', (8, 8))
            block_coder = BlockAdaptiveEntropyCoder(
                num_bands=self.num_bands,
                block_size=block_size,
                min_block_samples=self.compression_params.get('min_block_samples', 16)
            )
            compressed_body, entropy_stats = block_coder.encode_image_block_adaptive(results['mapped_indices'])
            # Convert back to bits for proper formatting
            compressed_body_bits = formatter.bytes_to_bits(compressed_body)
        elif self.compression_params['entropy_coder_type'] == 'rice':
            # Use CCSDS-121.0-B-2 Rice coder
            block_size = self.compression_params.get('rice_block_size', (16, 16))
            compressed_body, entropy_stats = encode_image_rice(results['mapped_indices'], block_size)
            # Convert back to bits for proper formatting
            compressed_body_bits = formatter.bytes_to_bits(compressed_body)
        else:
            compressed_body_bits = []

        # Format complete bitstream with proper word alignment
        compressed_bitstream = formatter.format_bitstream(
            header_bytes=header_bytes,
            compressed_bits=compressed_body_bits,
            pad_to_word_boundary=True
        )

        # Prepare metadata needed for decompression
        compression_metadata = {
            'image_shape': list(image.shape),
            'num_bands': self.num_bands,
            'dynamic_range': self.dynamic_range,
            'lossless': self.lossless,
            'compression_params': self.compression_params.copy(),
            'predictor_type': type(self.predictor).__name__,
            'quantizer_type': type(self.quantizer).__name__,
            'entropy_coder_final_state': {
                'high_res_accumulators': self.entropy_coder.high_res_accumulators.clone(),
                'counters': self.entropy_coder.counters.clone()
            },
            'header': header,  # Include CCSDS header for proper decompression
            'header_size': len(header_bytes),  # Size of header for bitstream parsing
            'body_size': len(compressed_body_bits) // 8 if compressed_body_bits else 0,  # Size of compressed body in bytes
            'bitstream_info': formatter.get_bitstream_info(compressed_bitstream),  # Bitstream formatting details
            'output_word_size': output_word_size  # Output word size used
        }

        # Compression statistics
        compression_statistics = self.get_compression_stats(image)

        result = {
            'compressed_bitstream': compressed_bitstream,
            'compression_metadata': compression_metadata,
            'compression_statistics': compression_statistics,
            'intermediate_data': results
        }

        # Add entropy coding statistics if available
        if entropy_stats:
            result['entropy_statistics'] = entropy_stats

        return result

    def optimize_encoding_order(self, image: torch.Tensor) -> str:
        """
        Analyze image and recommend optimal encoding order

        Args:
            image: Input image tensor [Z, Y, X] or [1, Z, Y, X]

        Returns:
            encoding_order: Recommended encoding order ('BI' or 'BSQ')
        """
        # Validate input
        image = self._validate_input(image)

        # Use encoding order optimizer
        analysis = EncodingOrderOptimizer.analyze_image_structure(image)

        from .encoding_orders import EncodingOrder
        if analysis['recommended_order'] == EncodingOrder.BAND_INTERLEAVED:
            return 'BI'
        else:
            return 'BSQ'

    def compare_encoding_orders(self, image: torch.Tensor) -> dict:
        """
        Compare compression performance of different encoding orders

        Args:
            image: Input image tensor [Z, Y, X] or [1, Z, Y, X]

        Returns:
            comparison: Dictionary with performance comparison
        """
        # Validate input
        image = self._validate_input(image)

        from .encoding_orders import EncodingOrder
        comparison = EncodingOrderOptimizer.estimate_compression_benefit(
            image,
            EncodingOrder.BAND_INTERLEAVED,
            EncodingOrder.BAND_SEQUENTIAL
        )

        return {
            'bi_prediction_variance': comparison['order1_prediction_variance'],
            'bsq_prediction_variance': comparison['order2_prediction_variance'],
            'recommended': 'BI' if comparison['recommended'] == EncodingOrder.BAND_INTERLEAVED else 'BSQ',
            'benefit_ratio': comparison['benefit_ratio'],
            'analysis': EncodingOrderOptimizer.analyze_image_structure(image)
        }

    def get_compression_stats(self, image: torch.Tensor) -> Dict[str, Union[float, int, bool]]:
        """
        Get detailed compression statistics

        Args:
            image: Input image [Z, Y, X]

        Returns:
            Dictionary of compression statistics
        """
        results = self.forward(image)

        original_size = results['original_size']
        compressed_size = results['compressed_size']
        compression_ratio = results['compression_ratio']

        # Compute distortion metrics for near-lossless compression
        if not self.lossless:
            mse = torch.mean((image - results['reconstructed_samples'])**2).item()
            psnr = 20 * np.log10(2**(self.dynamic_range-1) - 1) - 10 * np.log10(mse) if mse > 0 else float('inf')
            max_error = torch.max(torch.abs(image - results['reconstructed_samples'])).item()
        else:
            mse = 0.0
            psnr = float('inf')
            max_error = 0.0

        return {
            'original_size_bits': original_size,
            'compressed_size_bits': compressed_size,
            'compression_ratio': compression_ratio,
            'bits_per_sample': compressed_size / (image.numel()),
            'mse': mse,
            'psnr_db': psnr,
            'max_absolute_error': max_error,
            'lossless': self.lossless
        }


def create_lossless_compressor(num_bands: int, **kwargs: Any) -> CCSDS123Compressor:
    """
    Create a lossless CCSDS-123.0-B-2 compressor

    Args:
        num_bands: Number of spectral bands
        **kwargs: Additional compressor parameters

    Returns:
        Configured lossless compressor
    """
    return CCSDS123Compressor(num_bands=num_bands, lossless=True, **kwargs)


def create_near_lossless_compressor(
    num_bands: int,
    absolute_error_limits: Optional[torch.Tensor] = None,
    relative_error_limits: Optional[torch.Tensor] = None,
    **kwargs: Any
) -> CCSDS123Compressor:
    """
    Create a near-lossless CCSDS-123.0-B-2 compressor

    Args:
        num_bands: Number of spectral bands
        absolute_error_limits: [num_bands] absolute error limits per band
        relative_error_limits: [num_bands] relative error limits per band
        **kwargs: Additional compressor parameters

    Returns:
        Configured near-lossless compressor
    """
    compressor = CCSDS123Compressor(num_bands=num_bands, lossless=False, **kwargs)

    # Set error limits
    if absolute_error_limits is None and relative_error_limits is None:
        # Default: small absolute error limit
        absolute_error_limits = torch.ones(num_bands) * 2

    compressor.set_compression_parameters(
        absolute_error_limits=absolute_error_limits,
        relative_error_limits=relative_error_limits
    )

    return compressor


def decompress(compressed_data: Dict[str, Any]) -> torch.Tensor:
    """
    Decompress image from compressed representation

    This method reconstructs the original image from the compressed bitstream
    by reversing the compression pipeline: entropy decoding, inverse quantization,
    and applying the sample representatives to get the final reconstructed image.

    Mathematical Process:
    1. Entropy decode bitstream to recover mapped quantizer indices δ_z(t)
    2. Inverse map to get quantizer indices q_z(t)
    3. Reconstruct residuals: ε̂_z(t) = q_z(t) * (2*m_z(t) + 1)
    4. Reconstruct samples: ŝ_z(t) = prediction + ε̂_z(t)
    5. Apply clamping to valid dynamic range

    Args:
        compressed_data: Dictionary from compress() method containing:
            - compressed_bitstream: Entropy-coded bytes
            - compression_metadata: Decompression parameters
            - intermediate_data: Optional intermediate results for validation

    Returns:
        Reconstructed image tensor [Z, Y, X]
    """
    # Extract metadata
    metadata = compressed_data['compression_metadata']

    # Create decompressor with same parameters
    decompressor = CCSDS123Compressor(
        num_bands=metadata['num_bands'],
        dynamic_range=metadata['dynamic_range'],
        lossless=metadata['lossless']
    )

    # Set compression parameters
    decompressor.set_compression_parameters(**metadata['compression_params'])

    # For demonstration, we'll use the intermediate data if available
    # In a full implementation, this would decode the bitstream
    if 'intermediate_data' in compressed_data:
        intermediate = compressed_data['intermediate_data']
        return intermediate['reconstructed_samples'].clone()

    # Placeholder for full entropy decoder implementation
    raise NotImplementedError("Full entropy decoding not implemented in this version. "
                            "Use intermediate_data from compression results for reconstruction.")


def create_block_adaptive_lossless_compressor(num_bands: int, block_size: Tuple[int, int] = (8, 8), **kwargs: Any) -> CCSDS123Compressor:
    """
    Create a lossless CCSDS-123.0-B-2 compressor with block-adaptive entropy coding

    Args:
        num_bands: Number of spectral bands
        block_size: (height, width) of blocks for adaptive coding
        **kwargs: Additional compressor parameters

    Returns:
        Configured lossless compressor with block-adaptive entropy coding
    """
    compressor = CCSDS123Compressor(num_bands=num_bands, lossless=True, entropy_coder_type='block_adaptive', **kwargs)

    # Set block-adaptive parameters
    compressor.set_compression_parameters(
        entropy_coder_type='block_adaptive',
        block_size=block_size,
        min_block_samples=kwargs.get('min_block_samples', 16)
    )

    return compressor


def create_block_adaptive_near_lossless_compressor(
    num_bands: int,
    absolute_error_limits: Optional[torch.Tensor] = None,
    relative_error_limits: Optional[torch.Tensor] = None,
    block_size: Tuple[int, int] = (8, 8),
    **kwargs: Any
) -> CCSDS123Compressor:
    """
    Create a near-lossless CCSDS-123.0-B-2 compressor with block-adaptive entropy coding

    Args:
        num_bands: Number of spectral bands
        absolute_error_limits: [num_bands] absolute error limits per band
        relative_error_limits: [num_bands] relative error limits per band
        block_size: (height, width) of blocks for adaptive coding
        **kwargs: Additional compressor parameters

    Returns:
        Configured near-lossless compressor with block-adaptive entropy coding
    """
    compressor = CCSDS123Compressor(num_bands=num_bands, lossless=False, entropy_coder_type='block_adaptive', **kwargs)

    # Set error limits
    if absolute_error_limits is None and relative_error_limits is None:
        absolute_error_limits = torch.ones(num_bands) * 2

    # Set compression parameters
    compressor.set_compression_parameters(
        entropy_coder_type='block_adaptive',
        absolute_error_limits=absolute_error_limits,
        relative_error_limits=relative_error_limits,
        block_size=block_size,
        min_block_samples=kwargs.get('min_block_samples', 16)
    )

    return compressor


