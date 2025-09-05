import torch
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Union


class OptimizedHybridEntropyCoder:
    """
    Optimized CCSDS-123.0-B-2 Hybrid Entropy Coder

    Enhanced version with vectorized operations and better performance
    for the optimized compressor pipeline.

    Features:
    - Vectorized code selection statistics
    - Batch processing of mapped indices
    - Memory-efficient streaming encoding
    - Optimized low-entropy code handling
    """

    def __init__(self, num_bands: int, rescale_interval: int = 64, device: str = 'cpu') -> None:
        self.num_bands = num_bands
        self.rescale_interval = rescale_interval
        self.device = device

        # Vectorized code selection statistics for all bands
        self.high_res_accumulators = torch.zeros(num_bands, dtype=torch.float64, device=device)
        self.counters = torch.zeros(num_bands, dtype=torch.long, device=device)

        # Threshold for high/low entropy classification
        self.entropy_threshold = 8.0

        # Initialize low-entropy codes (optimized lookup tables)
        self.low_entropy_codes = self._initialize_optimized_codes()

        # Compressed bitstream components
        self.compressed_bits = []
        self.band_compressed_sizes = torch.zeros(num_bands, dtype=torch.long, device=device)

    def _initialize_optimized_codes(self) -> List['OptimizedLowEntropyCode']:
        """
        Initialize optimized low-entropy code lookup tables

        Uses vectorized operations and pre-computed tables for faster encoding
        """
        codes = []

        # Optimized code specifications (from Table 1)
        code_specs = [
            (0, 12, 105, 3, 13), (1, 10, 144, 3, 13), (2, 8, 118, 3, 12),
            (3, 6, 120, 4, 13), (4, 6, 92, 4, 13), (5, 4, 116, 6, 15),
            (6, 4, 101, 6, 15), (7, 4, 81, 5, 18), (8, 2, 88, 12, 16),
            (9, 2, 106, 12, 17), (10, 2, 103, 12, 18), (11, 2, 127, 16, 20),
            (12, 2, 109, 27, 21), (13, 2, 145, 46, 18), (14, 2, 256, 85, 17),
            (15, 0, 257, 256, 9)
        ]

        for code_idx, input_limit, num_codewords, max_input_len, max_output_len in code_specs:
            code = OptimizedLowEntropyCode(
                code_idx, input_limit, num_codewords, max_input_len, max_output_len
            )
            codes.append(code)

        return codes

    def update_statistics_vectorized(self, mapped_indices_band: torch.Tensor, band: int) -> None:
        """
        Update code selection statistics for a band using vectorized operations

        Args:
            mapped_indices_band: [Y, X] mapped quantizer indices for band
            band: Band index
        """
        # Vectorized accumulator update
        total_increment = torch.sum(mapped_indices_band.float()) * 4
        self.high_res_accumulators[band] += total_increment

        # Update counter
        self.counters[band] += mapped_indices_band.numel()

        # Rescale periodically
        if self.counters[band] % self.rescale_interval == 0:
            self.high_res_accumulators[band] /= 2
            self.counters[band] //= 2

            # Output LSB for decoder synchronization
            lsb = int(self.high_res_accumulators[band]) & 1
            self.compressed_bits.append(lsb)

    def classify_entropy_vectorized(self, mapped_indices_band: torch.Tensor, band: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify samples as high/low entropy using vectorized operations

        Args:
            mapped_indices_band: [Y, X] mapped indices for band
            band: Band index

        Returns:
            entropy_mask: [Y, X] boolean mask (True = high entropy)
            code_indices: [Y, X] code indices to use
        """
        if self.counters[band] == 0:
            # Default to high-entropy for first samples
            entropy_mask = torch.ones_like(mapped_indices_band, dtype=torch.bool)
            code_indices = torch.zeros_like(mapped_indices_band)
            return entropy_mask, code_indices

        # Compute scaled mean ratio for band
        mean_ratio = self.high_res_accumulators[band] / self.counters[band]

        # Vectorized entropy classification
        entropy_mask = mapped_indices_band.float() > self.entropy_threshold

        # Vectorized code index selection
        code_indices = torch.where(
            entropy_mask,
            torch.clamp((mapped_indices_band.float() / 2).long(), 0, 31),  # High-entropy codes
            torch.clamp(mapped_indices_band, 0, 15)  # Low-entropy codes
        )

        return entropy_mask, code_indices

    def encode_high_entropy_vectorized(self, mapped_indices: torch.Tensor, code_indices: torch.Tensor) -> List[int]:
        """
        Vectorized high-entropy sample encoding

        Args:
            mapped_indices: [N] high-entropy mapped indices
            code_indices: [N] corresponding code indices

        Returns:
            encoded_bits: List of encoded bits for all samples
        """
        encoded_bits = []

        # Process in batches for efficiency
        for i in range(len(mapped_indices)):
            mapped_val = int(mapped_indices[i].item())
            code_idx = int(code_indices[i].item())

            if mapped_val == 0:
                encoded_bits.extend([0])
            else:
                # Simplified GPO2 encoding
                k = max(0, code_idx - 4)
                quotient = mapped_val >> k
                remainder = mapped_val & ((1 << k) - 1)

                # Unary + binary encoding
                unary_bits = [1] * quotient + [0]
                binary_bits = [(remainder >> j) & 1 for j in range(k)]

                encoded_bits.extend(unary_bits + binary_bits)

        return encoded_bits

    def encode_low_entropy_batch(self, mapped_indices: torch.Tensor, code_indices: torch.Tensor) -> List[int]:
        """
        Batch encoding of low-entropy samples

        Args:
            mapped_indices: [N] low-entropy mapped indices
            code_indices: [N] corresponding code indices

        Returns:
            encoded_bits: List of encoded bits
        """
        encoded_bits = []

        # Group by code index for efficient processing
        unique_codes = torch.unique(code_indices)

        for code_idx in unique_codes:
            mask = code_indices == code_idx
            samples = mapped_indices[mask]

            if code_idx < len(self.low_entropy_codes):
                code = self.low_entropy_codes[code_idx]
                code_bits = code.encode_batch(samples.tolist())
                if code_bits:
                    encoded_bits.extend(code_bits)

        return encoded_bits

    def encode_band_optimized(self, mapped_indices_band: torch.Tensor, band: int) -> Tuple[List[int], Dict[str, Union[int, float]]]:
        """
        Optimized encoding of entire band

        Args:
            mapped_indices_band: [Y, X] mapped indices for band
            band: Band index

        Returns:
            compressed_bits: List of encoded bits for band
            compression_stats: Dictionary of compression statistics
        """
        start_time = time.time()

        # Update statistics for band
        self.update_statistics_vectorized(mapped_indices_band, band)

        # Classify entropy for all samples in band
        entropy_mask, code_indices = self.classify_entropy_vectorized(mapped_indices_band, band)

        # Flatten for processing
        flat_indices = mapped_indices_band.flatten()
        flat_entropy_mask = entropy_mask.flatten()
        flat_code_indices = code_indices.flatten()

        # Separate high and low entropy samples
        high_entropy_indices = flat_indices[flat_entropy_mask]
        high_entropy_codes = flat_code_indices[flat_entropy_mask]

        low_entropy_indices = flat_indices[~flat_entropy_mask]
        low_entropy_codes = flat_code_indices[~flat_entropy_mask]

        compressed_bits = []

        # Encode high-entropy samples
        if len(high_entropy_indices) > 0:
            high_bits = self.encode_high_entropy_vectorized(high_entropy_indices, high_entropy_codes)
            compressed_bits.extend(high_bits)

        # Encode low-entropy samples
        if len(low_entropy_indices) > 0:
            low_bits = self.encode_low_entropy_batch(low_entropy_indices, low_entropy_codes)
            compressed_bits.extend(low_bits)

        # Compression statistics
        encoding_time = time.time() - start_time
        total_samples = mapped_indices_band.numel()
        high_entropy_ratio = len(high_entropy_indices) / total_samples if total_samples > 0 else 0

        stats = {
            'band': band,
            'total_samples': total_samples,
            'high_entropy_samples': len(high_entropy_indices),
            'low_entropy_samples': len(low_entropy_indices),
            'high_entropy_ratio': high_entropy_ratio,
            'compressed_bits': len(compressed_bits),
            'encoding_time': encoding_time,
            'bits_per_sample': len(compressed_bits) / total_samples if total_samples > 0 else 0
        }

        return compressed_bits, stats

    def encode_image_optimized(self, mapped_indices: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        """
        Optimized encoding of entire image

        Args:
            mapped_indices: [Z, Y, X] mapped quantizer indices

        Returns:
            compressed_data: Bytes of compressed image
            compression_stats: Dictionary of detailed statistics
        """
        Z, Y, X = mapped_indices.shape

        all_compressed_bits = []
        band_stats = []

        total_start_time = time.time()

        # Process each band with optimized encoding
        for z in range(Z):
            band_bits, band_stat = self.encode_band_optimized(mapped_indices[z], z)
            all_compressed_bits.extend(band_bits)
            band_stats.append(band_stat)
            self.band_compressed_sizes[z] = len(band_bits)

        # Write tail information (final accumulator states)
        for z in range(Z):
            final_acc = int(self.high_res_accumulators[z]) & 0xFF
            for i in range(8):
                all_compressed_bits.append((final_acc >> i) & 1)

        # Convert bits to bytes
        compressed_data = self._bits_to_bytes_optimized(all_compressed_bits)

        # Overall compression statistics
        total_time = time.time() - total_start_time
        total_samples = Z * Y * X
        total_bits = len(all_compressed_bits)

        overall_stats = {
            'total_samples': total_samples,
            'total_compressed_bits': total_bits,
            'total_compressed_bytes': len(compressed_data),
            'bits_per_sample': total_bits / total_samples,
            'compression_ratio': (total_samples * 16) / total_bits,  # Assuming 16-bit input
            'encoding_time': total_time,
            'throughput_samples_per_sec': total_samples / total_time,
            'band_stats': band_stats
        }

        return compressed_data, overall_stats

    def _bits_to_bytes_optimized(self, bits: List[int]) -> bytes:
        """
        Efficiently convert list of bits to bytes

        Args:
            bits: List of bits (0 or 1)

        Returns:
            bytes: Compressed data as bytes
        """
        if not bits:
            return b''

        # Pad to byte boundary
        while len(bits) % 8 != 0:
            bits.append(0)

        # Vectorized conversion using numpy for speed
        bit_array = np.array(bits, dtype=np.uint8)
        bit_array = bit_array.reshape(-1, 8)

        # Convert each byte
        byte_values = []
        for byte_bits in bit_array:
            byte_val = 0
            for i, bit in enumerate(byte_bits):
                byte_val |= (bit << i)
            byte_values.append(byte_val)

        return bytes(byte_values)


class OptimizedLowEntropyCode:
    """
    Optimized variable-to-variable length low-entropy code

    Enhanced with batch processing and lookup table optimizations
    """

    def __init__(self, code_index: int, input_limit: int, num_codewords: int, max_input_len: int, max_output_len: int) -> None:
        self.code_index = code_index
        self.input_limit = input_limit
        self.num_codewords = num_codewords
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        # Pre-computed lookup tables for faster encoding
        self.input_to_output = self._generate_optimized_lookup_table()

    def _generate_optimized_lookup_table(self) -> Dict[Tuple[int, ...], List[int]]:
        """
        Generate optimized lookup table with vectorized operations
        """
        table = {}

        # Simplified pattern-based generation (optimized version)
        for input_len in range(1, min(self.max_input_len + 1, 8)):  # Limit for performance
            for pattern_id in range(min(2**input_len, self.num_codewords)):

                # Generate input pattern
                input_pattern = []
                temp_pattern = pattern_id
                for i in range(input_len):
                    symbol = min(temp_pattern % (self.input_limit + 2), self.input_limit + 1)
                    input_pattern.append(symbol)
                    temp_pattern //= (self.input_limit + 2)

                # Generate output bits (simplified but functional)
                output_len = min(8, self.max_output_len)
                output_bits = []
                output_val = pattern_id

                for i in range(output_len):
                    output_bits.append((output_val >> i) & 1)

                table[tuple(input_pattern)] = output_bits

                if len(table) >= self.num_codewords:
                    break

            if len(table) >= self.num_codewords:
                break

        return table

    def encode_batch(self, input_symbols: List[int]) -> List[int]:
        """
        Batch encode multiple symbols efficiently

        Args:
            input_symbols: List of symbols to encode

        Returns:
            output_bits: List of output bits, or None if incomplete
        """
        if not input_symbols:
            return []

        output_bits = []
        i = 0

        while i < len(input_symbols):
            # Try to match patterns of decreasing length
            matched = False

            for pattern_len in range(min(self.max_input_len, len(input_symbols) - i), 0, -1):
                pattern = tuple(input_symbols[i:i+pattern_len])

                if pattern in self.input_to_output:
                    # Found match
                    pattern_bits = self.input_to_output[pattern].copy()

                    # Handle escape symbols
                    for symbol in pattern:
                        if symbol > self.input_limit:
                            residual = symbol - self.input_limit - 1
                            escape_bits = self._encode_residual_fast(residual)
                            output_bits.extend(escape_bits)

                    output_bits.extend(pattern_bits)
                    i += pattern_len
                    matched = True
                    break

            if not matched:
                # Encode single symbol as escape
                if i < len(input_symbols):
                    symbol = input_symbols[i]
                    if symbol > self.input_limit:
                        residual = symbol - self.input_limit - 1
                        escape_bits = self._encode_residual_fast(residual)
                        output_bits.extend(escape_bits)
                    else:
                        # Simple encoding for small symbols
                        output_bits.extend([symbol & 1, (symbol >> 1) & 1])
                    i += 1

        return output_bits

    def _encode_residual_fast(self, residual: int) -> List[int]:
        """
        Fast residual encoding for escape symbols
        """
        max_unary_length = 16

        if residual >= max_unary_length:
            bits = [1] * (max_unary_length - 1) + [0]
            excess = residual - (max_unary_length - 1)
            for i in range(8):
                bits.append((excess >> i) & 1)
            return bits
        else:
            return [1] * residual + [0]


class StreamingOptimizedCoder:
    """
    Memory-efficient streaming version of optimized entropy coder

    For processing very large images without loading everything into memory
    """

    def __init__(self, num_bands: int, chunk_size: Tuple[int, int, int] = (4, 32, 32), device: str = 'cpu') -> None:
        self.base_coder = OptimizedHybridEntropyCoder(num_bands, device=device)
        self.chunk_size = chunk_size

    def encode_streaming(self, mapped_indices: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        """
        Stream-process large image in chunks

        Args:
            mapped_indices: [Z, Y, X] large mapped indices tensor

        Returns:
            compressed_data: Bytes of compressed data
            compression_stats: Statistics dictionary
        """
        Z, Y, X = mapped_indices.shape
        Z_chunk, Y_chunk, X_chunk = self.chunk_size

        all_compressed_data = []
        all_stats = []

        for z_start in range(0, Z, Z_chunk):
            for y_start in range(0, Y, Y_chunk):
                for x_start in range(0, X, X_chunk):
                    # Extract chunk
                    z_end = min(z_start + Z_chunk, Z)
                    y_end = min(y_start + Y_chunk, Y)
                    x_end = min(x_start + X_chunk, X)

                    chunk = mapped_indices[z_start:z_end, y_start:y_end, x_start:x_end]

                    # Encode chunk
                    chunk_coder = OptimizedHybridEntropyCoder(z_end - z_start)
                    chunk_data, chunk_stats = chunk_coder.encode_image_optimized(chunk)

                    all_compressed_data.append(chunk_data)
                    all_stats.append(chunk_stats)

        # Combine all compressed data
        combined_data = b''.join(all_compressed_data)

        # Combine statistics
        total_samples = sum(stats['total_samples'] for stats in all_stats)
        total_bits = sum(stats['total_compressed_bits'] for stats in all_stats)
        total_time = sum(stats['encoding_time'] for stats in all_stats)

        combined_stats = {
            'total_samples': total_samples,
            'total_compressed_bits': total_bits,
            'total_compressed_bytes': len(combined_data),
            'bits_per_sample': total_bits / total_samples if total_samples > 0 else 0,
            'compression_ratio': (total_samples * 16) / total_bits if total_bits > 0 else 0,
            'encoding_time': total_time,
            'throughput_samples_per_sec': total_samples / total_time if total_time > 0 else 0,
            'num_chunks': len(all_stats),
            'chunk_stats': all_stats
        }

        return combined_data, combined_stats


def encode_image_optimized(mapped_indices: torch.Tensor, num_bands: int) -> Tuple[bytes, Dict[str, Any]]:
    """
    Main optimized image encoding function

    Args:
        mapped_indices: [Z, Y, X] tensor of mapped quantizer indices
        num_bands: Number of spectral bands

    Returns:
        compressed_data: Bytes of compressed image data
        compression_stats: Detailed compression statistics
    """
    device = mapped_indices.device.type
    encoder = OptimizedHybridEntropyCoder(num_bands, device=device)
    return encoder.encode_image_optimized(mapped_indices)


def encode_image_streaming(mapped_indices: torch.Tensor, num_bands: int, chunk_size: Tuple[int, int, int] = (4, 32, 32)) -> Tuple[bytes, Dict[str, Any]]:
    """
    Memory-efficient streaming image encoding

    Args:
        mapped_indices: [Z, Y, X] tensor of mapped quantizer indices
        num_bands: Number of spectral bands
        chunk_size: (Z_chunk, Y_chunk, X_chunk) chunk dimensions

    Returns:
        compressed_data: Bytes of compressed image data
        compression_stats: Detailed compression statistics
    """
    device = mapped_indices.device.type
    encoder = StreamingOptimizedCoder(num_bands, chunk_size, device=device)
    return encoder.encode_streaming(mapped_indices)
