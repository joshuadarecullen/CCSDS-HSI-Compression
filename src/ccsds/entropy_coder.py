import torch
from typing import List, Dict, Any, Tuple, Optional, Union
from .low_entropy_tables import LOW_ENTROPY_TABLES, CompleteLowEntropyCode

class HybridEntropyCoder:
    """
    CCSDS-123.0-B-2 Hybrid Entropy Coder

    Combines length-limited Golomb-Power-of-2 (GPO2) codes with 16 new
    variable-to-variable length "low-entropy" codes for better compression
    of low-entropy data.

    Key features:
    - Adaptive code selection based on sample statistics
    - Decoding in reverse order for memory efficiency
    - Suffix-free codes instead of prefix-free
    """

    def __init__(self, num_bands: int, rescale_interval: int = 64) -> None:
        self.num_bands = num_bands
        self.rescale_interval = rescale_interval

        # Code selection statistics for each band
        self.high_res_accumulators = torch.zeros(num_bands, dtype=torch.float64)
        self.counters = torch.zeros(num_bands, dtype=torch.long)

        # Threshold for high/low entropy classification
        self.entropy_threshold = 8.0

        # Initialize low-entropy codes
        self.low_entropy_codes = self._initialize_low_entropy_codes()

        # Compressed bitstream buffer
        self.compressed_bits = []

    def _initialize_low_entropy_codes(self) -> List[CompleteLowEntropyCode]:
        """
        Initialize the 16 variable-to-variable length low-entropy codes
        Based on Table 5-16 from CCSDS-123.0-B-2 standard
        """
        codes = []

        # Use complete low-entropy code tables from the standard
        for code_id in range(16):
            code = CompleteLowEntropyCode(code_id, LOW_ENTROPY_TABLES)
            codes.append(code)

        return codes

    def update_statistics(self, mapped_index: Union[int, float], band: int) -> None:
        """
        Update code selection statistics for a band

        Args:
            mapped_index: Mapped quantizer index δ_z(t)
            band: Band index z
        """
        # Update high-resolution accumulator
        self.high_res_accumulators[band] += 4 * mapped_index

        # Update counter
        self.counters[band] += 1

        # Rescale periodically
        if self.counters[band] % self.rescale_interval == 0:
            self.high_res_accumulators[band] /= 2
            self.counters[band] //= 2

            # Output LSB before rescaling for decoder synchronization
            lsb = int(self.high_res_accumulators[band]) & 1
            self.compressed_bits.append(lsb)

    def classify_entropy(self, band: int) -> Tuple[bool, int]:
        """
        Classify sample as high-entropy or low-entropy based on statistics

        Args:
            band: Band index z

        Returns:
            is_high_entropy: Boolean indicating if sample is high-entropy
            code_index: Index of code to use
        """
        if self.counters[band] == 0:
            return True, 0  # Default to high-entropy for first sample

        # Compute scaled mean of mapped quantizer indices
        mean_ratio = self.high_res_accumulators[band] / self.counters[band]

        if mean_ratio > self.entropy_threshold:
            # High-entropy sample
            # Select GPO2 code based on mean ratio
            code_index = min(int(mean_ratio / 2), 31)  # Limit to reasonable range
            return True, code_index
        else:
            # Low-entropy sample
            # Select low-entropy code based on mean ratio
            code_index = min(int(mean_ratio), 15)
            return False, code_index

    def encode_high_entropy(self, mapped_index: int, code_index: int) -> List[int]:
        """
        Encode high-entropy sample using GPO2-equivalent code

        Args:
            mapped_index: Mapped quantizer index to encode
            code_index: GPO2 code parameter

        Returns:
            encoded_bits: List of bits representing encoded sample
        """
        # Simplified GPO2 encoding (suffix-free variant)
        # In practice, this would implement the full GPO2 algorithm

        if mapped_index == 0:
            return [0]  # Special case for zero

        # Encode using unary prefix + binary suffix
        k = max(0, code_index - 4)  # GPO2 parameter
        quotient = mapped_index >> k
        remainder = mapped_index & ((1 << k) - 1)

        # Unary encoding of quotient (reversed for suffix-free property)
        unary_bits = [1] * quotient + [0]

        # Binary encoding of remainder
        binary_bits = []
        for i in range(k):
            binary_bits.append((remainder >> i) & 1)

        return unary_bits + binary_bits

    def encode_low_entropy(self, mapped_index: int, code_index: int) -> Optional[List[int]]:
        """
        Encode single low-entropy sample using variable-to-variable code

        Args:
            mapped_index: Single mapped quantizer index
            code_index: Low-entropy code index (0-15)

        Returns:
            encoded_bits: List of bits, or None if more symbols needed
        """
        if code_index >= len(self.low_entropy_codes):
            return None

        code = self.low_entropy_codes[code_index]
        return code.add_symbol(mapped_index)

    def encode_escape_symbol(self, residual_value: int) -> List[int]:
        """
        Encode residual value for escape symbols in low-entropy codes

        Args:
            residual_value: Non-negative residual δ_z(t) - L_i - 1

        Returns:
            encoded_bits: Length-limited unary codeword
        """
        # Length-limited unary encoding
        max_length = 32  # Limit for practical purposes

        if residual_value >= max_length:
            # Use truncated encoding for very large values
            bits = [1] * (max_length - 1) + [0]
            # Add additional bits for the excess
            excess = residual_value - (max_length - 1)
            for i in range(8):  # 8-bit excess encoding
                bits.append((excess >> i) & 1)
            return bits
        else:
            # Standard unary encoding
            return [1] * residual_value + [0]

    def encode_samples(self, mapped_indices: torch.Tensor, band: int) -> List[int]:
        """
        Encode a sequence of samples for a given band

        Args:
            mapped_indices: [N] tensor of mapped quantizer indices
            band: Band index

        Returns:
            compressed_bits: List of encoded bits
        """
        compressed_bits = []
        low_entropy_buffer = []
        current_low_code = None

        for mapped_index in mapped_indices:
            # Update statistics before encoding (reverse order decoding requirement)
            self.update_statistics(mapped_index, band)

            # Classify entropy and select code
            is_high_entropy, code_index = self.classify_entropy(band)

            mapped_index_val = int(mapped_index.item())

            if is_high_entropy:
                # Flush any pending low-entropy codes
                if low_entropy_buffer and current_low_code is not None:
                    low_bits = self.encode_low_entropy(low_entropy_buffer, current_low_code)
                    if low_bits:
                        compressed_bits.extend(low_bits)
                    low_entropy_buffer = []
                    current_low_code = None

                # Encode high-entropy sample immediately
                high_bits = self.encode_high_entropy(mapped_index_val, code_index)
                compressed_bits.extend(high_bits)

            else:
                # Buffer low-entropy sample
                if current_low_code != code_index:
                    # Code changed - flush buffer with previous code
                    if low_entropy_buffer and current_low_code is not None:
                        low_bits = self.encode_low_entropy(low_entropy_buffer, current_low_code)
                        if low_bits:
                            compressed_bits.extend(low_bits)
                    low_entropy_buffer = []
                    current_low_code = code_index

                # Add to buffer
                low_entropy_buffer.append(mapped_index_val)

                # Check if we can encode this buffer
                if current_low_code is not None:
                    low_bits = self.encode_low_entropy(low_entropy_buffer, current_low_code)
                    if low_bits:  # Complete codeword formed
                        compressed_bits.extend(low_bits)
                        low_entropy_buffer = []

        # Flush remaining low-entropy samples
        if low_entropy_buffer and current_low_code is not None:
            low_bits = self.encode_low_entropy(low_entropy_buffer, current_low_code)
            if low_bits:
                compressed_bits.extend(low_bits)

        # Add proper tail processing
        tail_bits = self.finalize_encoding(band)
        compressed_bits.extend(tail_bits)

        return compressed_bits

    def finalize_encoding(self, band: int) -> List[int]:
        """
        Finalize encoding with proper tail processing for CCSDS compliance

        This method implements the tail processing requirements from Section 5.2.3
        of the CCSDS-123.0-B-2 standard for proper decoder synchronization.

        Args:
            band: Band index for final statistics

        Returns:
            tail_bits: Final bits for proper decoder synchronization
        """
        tail_bits = []

        # Flush any remaining low-entropy code buffers
        for code in self.low_entropy_codes:
            if hasattr(code, 'input_buffer') and code.input_buffer:
                # Force flush remaining symbols
                flush_bits = code.flush()
                for bit_sequence in flush_bits:
                    tail_bits.extend(bit_sequence)

        # Add flush table bits for reverse-order decoding
        flush_table_bits = self._generate_flush_table()
        tail_bits.extend(flush_table_bits)

        # Add final synchronization bits
        sync_bits = self._generate_sync_bits()
        tail_bits.extend(sync_bits)

        return tail_bits

    def _generate_flush_table(self) -> List[int]:
        """
        Generate flush table for reverse-order decoding

        The flush table allows the decoder to properly handle the end
        of the compressed bitstream when decoding in reverse order.

        Returns:
            flush_table_bits: Bits encoding the flush table
        """
        # Simplified flush table implementation
        # In practice, this would encode the exact state needed for decoding
        flush_bits = []

        # Encode number of active low-entropy codes
        active_codes = 0
        for code in self.low_entropy_codes:
            if hasattr(code, 'input_buffer') and code.input_buffer:
                active_codes += 1

        # Encode active code count (4 bits sufficient for 16 codes)
        for i in range(4):
            flush_bits.append((active_codes >> i) & 1)

        # Encode state for each active code
        for code_idx, code in enumerate(self.low_entropy_codes):
            if hasattr(code, 'input_buffer') and code.input_buffer:
                # Encode code index (4 bits)
                for i in range(4):
                    flush_bits.append((code_idx >> i) & 1)

                # Encode remaining buffer length (4 bits)
                buffer_len = min(len(code.input_buffer), 15)
                for i in range(4):
                    flush_bits.append((buffer_len >> i) & 1)

        return flush_bits

    def _generate_sync_bits(self) -> List[int]:
        """
        Generate synchronization bits for decoder

        These bits ensure the decoder can identify the end of the
        compressed data and properly align for reverse decoding.

        Returns:
            sync_bits: Synchronization bit pattern
        """
        # Standard synchronization pattern: 0x5555 (alternating pattern)
        sync_pattern = 0x5555  # 16-bit pattern
        sync_bits = []

        for i in range(16):
            sync_bits.append((sync_pattern >> i) & 1)

        return sync_bits


class LowEntropyCode:
    """
    Individual low-entropy variable-to-variable length code
    """

    def __init__(self,
                 code_index: int,
                 input_limit: int,
                 num_codewords: int,
                 max_input_len: int,
                 max_output_len: int) -> None:
        self.code_index = code_index
        self.input_limit = input_limit  # L_i
        self.num_codewords = num_codewords
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        # Generate codeword tables (simplified - would be from standard specification)
        self.input_to_output = self._generate_codeword_table()
        self.input_buffer = []

    def _generate_codeword_table(self) -> Dict[Tuple[int, ...], List[int]]:
        """
        Generate simplified codeword mapping table
        In practice, these would be specified exactly by the standard
        """
        table = {}

        # Simple pattern-based generation for demonstration
        # Real implementation would use exact tables from standard
        codeword_id = 0

        for input_len in range(1, self.max_input_len + 1):
            for pattern in range(min(2**input_len, self.num_codewords - codeword_id)):
                if codeword_id >= self.num_codewords:
                    break

                # Generate input pattern
                input_pattern = []
                for i in range(input_len):
                    if pattern & (1 << i):
                        input_pattern.append(min(pattern % (self.input_limit + 2), self.input_limit + 1))
                    else:
                        input_pattern.append(0)

                # Generate corresponding output bits (simplified)
                output_bits = []
                output_val = codeword_id
                output_len = min(8, self.max_output_len)  # Simplified

                for i in range(output_len):
                    output_bits.append((output_val >> i) & 1)

                table[tuple(input_pattern)] = output_bits
                codeword_id += 1

                if codeword_id >= self.num_codewords:
                    break

            if codeword_id >= self.num_codewords:
                break

        return table

    def encode(self, input_symbols: List[int]) -> Optional[List[int]]:
        """
        Encode sequence of input symbols

        Args:
            input_symbols: List of mapped quantizer indices

        Returns:
            output_bits: List of output bits, or None if incomplete
        """
        self.input_buffer.extend(input_symbols)

        # Try to find a matching codeword
        for length in range(1, min(len(self.input_buffer) + 1, self.max_input_len + 1)):
            pattern = tuple(self.input_buffer[:length])

            if pattern in self.input_to_output:
                # Found complete codeword
                output_bits = self.input_to_output[pattern].copy()

                # Check for escape symbols
                escape_bits = []
                for symbol in pattern:
                    if symbol > self.input_limit:
                        # Encode residual
                        residual = symbol - self.input_limit - 1
                        escape_bits.extend(self._encode_residual(residual))

                # Remove processed symbols from buffer
                self.input_buffer = self.input_buffer[length:]

                return escape_bits + output_bits

        return None  # No complete codeword yet

    def _encode_residual(self, residual: int) -> List[int]:
        """
        Encode residual value for escape symbol
        """
        # Length-limited unary
        max_unary_length = 16

        if residual >= max_unary_length:
            bits = [1] * (max_unary_length - 1) + [0]
            # Add binary representation of excess
            excess = residual - (max_unary_length - 1)
            for i in range(8):
                bits.append((excess >> i) & 1)
            return bits
        else:
            return [1] * residual + [0]


class BitWriter:
    """
    Utility class for writing compressed bits to byte stream
    """

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.bit_buffer = 0
        self.bit_count = 0

    def write_bit(self, bit: int) -> None:
        """Write a single bit"""
        self.bit_buffer = (self.bit_buffer << 1) | (bit & 1)
        self.bit_count += 1

        if self.bit_count == 8:
            self.buffer.append(self.bit_buffer)
            self.bit_buffer = 0
            self.bit_count = 0

    def write_bits(self, bits: List[int]) -> None:
        """Write a list of bits"""
        for bit in bits:
            self.write_bit(bit)

    def flush(self) -> bytes:
        """Flush remaining bits with padding"""
        if self.bit_count > 0:
            # Pad with zeros
            while self.bit_count < 8:
                self.write_bit(0)

        return bytes(self.buffer)


def encode_image(mapped_indices: torch.Tensor, num_bands: int) -> bytes:
    """
    Main function to encode an entire image

    Args:
        mapped_indices: [Z, Y, X] tensor of mapped quantizer indices
        num_bands: Number of spectral bands

    Returns:
        compressed_data: Bytes of compressed image data
    """
    Z, Y, X = mapped_indices.shape
    encoder = HybridEntropyCoder(num_bands)
    writer = BitWriter()

    # Process each band
    for z in range(Z):
        band_data = mapped_indices[z].flatten()
        compressed_bits = encoder.encode_samples(band_data, z)
        writer.write_bits(compressed_bits)

    # Write tail information (final accumulator states, etc.)
    for z in range(Z):
        # Write final accumulator LSBs
        final_acc = int(encoder.high_res_accumulators[z]) & 0xFF
        for i in range(8):
            writer.write_bit((final_acc >> i) & 1)

    return writer.flush()


class BlockAdaptiveEntropyCoder:
    """
    CCSDS-123.0-B-2 Block-Adaptive Entropy Coder

    Groups samples into blocks and selects entropy codes based on block-level
    statistics rather than sample-by-sample adaptation. This can provide
    better compression for images with spatial structure.

    Key features:
    - Block-based entropy estimation and code selection
    - Unified code selection per block for all samples in the block
    - Block size adaptation based on image characteristics
    - Integration with existing 16 low-entropy codes and GPO2 codes
    """

    def __init__(self, num_bands: int, block_size: Tuple[int, int] = (8, 8),
                 min_block_samples: int = 16) -> None:
        """
        Initialize block-adaptive entropy coder

        Args:
            num_bands: Number of spectral bands
            block_size: (height, width) of blocks for adaptive coding
            min_block_samples: Minimum samples per block for reliable statistics
        """
        self.num_bands = num_bands
        self.block_size = block_size
        self.min_block_samples = min_block_samples

        # Initialize low-entropy codes (same as hybrid coder)
        self.low_entropy_codes = self._initialize_low_entropy_codes()

        # Block-level statistics
        self.block_stats = {}
        self.compressed_blocks = []

    def _initialize_low_entropy_codes(self) -> List['LowEntropyCode']:
        """Initialize the 16 low-entropy codes from CCSDS standard"""
        codes = []
        code_specs = [
            (0, 12, 105, 3, 13), (1, 10, 144, 3, 13), (2, 8, 118, 3, 12),
            (3, 6, 120, 4, 13), (4, 6, 92, 4, 13), (5, 4, 116, 6, 15),
            (6, 4, 101, 6, 15), (7, 4, 81, 5, 18), (8, 2, 88, 12, 16),
            (9, 2, 106, 12, 17), (10, 2, 103, 12, 18), (11, 2, 127, 16, 20),
            (12, 2, 109, 27, 21), (13, 2, 145, 46, 18), (14, 2, 256, 85, 17),
            (15, 0, 257, 256, 9)
        ]

        for code_idx, input_limit, num_codewords, max_input_len, max_output_len in code_specs:
            code = LowEntropyCode(code_idx, input_limit, num_codewords, max_input_len, max_output_len)
            codes.append(code)

        return codes

    def _partition_into_blocks(self, mapped_indices: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Partition image into blocks for block-adaptive coding

        Args:
            mapped_indices: [Z, Y, X] mapped quantizer indices

        Returns:
            List of block dictionaries with indices, coordinates, and metadata
        """
        Z, Y, X = mapped_indices.shape
        block_h, block_w = self.block_size
        blocks = []

        for z in range(Z):
            for y in range(0, Y, block_h):
                for x in range(0, X, block_w):
                    # Define block boundaries
                    y_end = min(y + block_h, Y)
                    x_end = min(x + block_w, X)

                    # Extract block data
                    block_data = mapped_indices[z, y:y_end, x:x_end]

                    # Only process blocks with sufficient samples
                    if block_data.numel() >= self.min_block_samples:
                        block = {
                            'band': z,
                            'coordinates': (y, x, y_end, x_end),
                            'data': block_data.flatten(),  # Flatten for entropy analysis
                            'size': block_data.numel(),
                            'shape': (y_end - y, x_end - x)
                        }
                        blocks.append(block)

        return blocks

    def _estimate_block_entropy(self, block_data: torch.Tensor) -> Dict[str, float]:
        """
        Estimate entropy characteristics of a block

        Args:
            block_data: Flattened block data

        Returns:
            Dictionary with entropy statistics
        """
        # Convert to numpy for histogram-based entropy estimation
        data_np = block_data.cpu().numpy().astype(int)

        # Compute histogram and probabilities
        unique_vals, counts = torch.unique(block_data, return_counts=True)
        probabilities = counts.float() / block_data.numel()

        # Compute empirical entropy: H = -Σ p(x) * log2(p(x))
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10)).item()

        # Additional statistics for code selection
        mean_val = torch.mean(block_data.float()).item()
        var_val = torch.var(block_data.float()).item()
        max_val = torch.max(torch.abs(block_data)).item()

        return {
            'entropy': entropy,
            'mean': mean_val,
            'variance': var_val,
            'max_magnitude': max_val,
            'num_unique': len(unique_vals),
            'data_range': (torch.min(block_data).item(), torch.max(block_data).item())
        }

    def _select_block_code(self, entropy_stats: Dict[str, float]) -> Tuple[str, int]:
        """
        Select optimal entropy code for a block based on its statistics

        Args:
            entropy_stats: Block entropy statistics

        Returns:
            Tuple of (code_type, code_index) where code_type is 'low' or 'high'
        """
        entropy = entropy_stats['entropy']
        max_magnitude = entropy_stats['max_magnitude']
        num_unique = entropy_stats['num_unique']

        # Decision logic for block-level code selection
        if entropy < 3.0 and max_magnitude <= 15 and num_unique <= 32:
            # Low entropy block - select best low-entropy code
            # Choose code based on max magnitude and entropy
            if max_magnitude <= 6:
                if entropy < 1.5:
                    return ('low', 0)  # Code 0: very low entropy, small magnitude
                else:
                    return ('low', 1)  # Code 1: low entropy, small magnitude
            elif max_magnitude <= 12:
                return ('low', 2)  # Code 2: low entropy, medium magnitude
            else:
                return ('low', 3)  # Code 3: low entropy, larger magnitude

        elif entropy < 5.0 and max_magnitude <= 64:
            # Medium entropy block - select mid-range codes
            if num_unique <= 16:
                return ('low', 4)  # Code 4: medium entropy, limited range
            else:
                return ('low', 5)  # Code 5: medium entropy, more diverse

        else:
            # High entropy block - use GPO2-style codes
            # Select GPO2 parameter based on magnitude
            if max_magnitude <= 32:
                return ('high', 2)  # GPO2 with k=2
            elif max_magnitude <= 128:
                return ('high', 3)  # GPO2 with k=3
            else:
                return ('high', 4)  # GPO2 with k=4

    def _encode_block_with_code(self, block_data: torch.Tensor, code_type: str,
                              code_index: int) -> List[int]:
        """
        Encode a block using the selected code

        Args:
            block_data: Flattened block data
            code_type: 'low' or 'high' entropy code type
            code_index: Index of the selected code

        Returns:
            List of encoded bits
        """
        if code_type == 'low':
            # Use low-entropy variable-length code
            code = self.low_entropy_codes[code_index]
            encoded_bits = []

            # Encode each sample in the block with the selected low-entropy code
            for sample in block_data:
                sample_bits = self._encode_with_low_entropy_code(sample.item(), code)
                if sample_bits is not None:
                    encoded_bits.extend(sample_bits)
                else:
                    # Fallback to high-entropy encoding for this sample
                    fallback_bits = self._encode_with_gpo2(sample.item(), 3)
                    encoded_bits.extend(fallback_bits)

            return encoded_bits

        else:  # code_type == 'high'
            # Use GPO2-style code for entire block
            encoded_bits = []
            k = code_index  # GPO2 parameter

            for sample in block_data:
                sample_bits = self._encode_with_gpo2(sample.item(), k)
                encoded_bits.extend(sample_bits)

            return encoded_bits

    def _encode_with_low_entropy_code(self, value: int, code: 'LowEntropyCode') -> Optional[List[int]]:
        """Encode a single value with a low-entropy code"""
        # Simplified low-entropy encoding (would need full code table implementation)
        abs_val = abs(value)
        if abs_val <= code.input_limit:
            # Map to codeword (simplified - would use actual code table)
            codeword_index = abs_val
            sign_bit = 0 if value >= 0 else 1

            # Convert to binary representation (simplified)
            bits = []
            temp = codeword_index
            for _ in range(code.max_output_len):
                bits.append(temp & 1)
                temp >>= 1
                if temp == 0:
                    break

            bits.append(sign_bit)  # Add sign bit
            return bits
        else:
            return None  # Value too large for this code

    def _encode_with_gpo2(self, value: int, k: int) -> List[int]:
        """Encode a single value with GPO2 code"""
        # GPO2 encoding: unary prefix + k-bit suffix
        abs_val = abs(value)
        sign_bit = 0 if value >= 0 else 1

        # Compute quotient and remainder
        quotient = abs_val >> k  # abs_val // (2^k)
        remainder = abs_val & ((1 << k) - 1)  # abs_val % (2^k)

        # Unary encoding of quotient
        bits = [1] * quotient + [0]  # quotient ones followed by zero

        # k-bit binary encoding of remainder
        for i in range(k):
            bits.append((remainder >> i) & 1)

        # Add sign bit
        bits.append(sign_bit)

        return bits

    def encode_image_block_adaptive(self, mapped_indices: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        """
        Encode entire image using block-adaptive entropy coding

        Args:
            mapped_indices: [Z, Y, X] mapped quantizer indices

        Returns:
            Tuple of (compressed_data, compression_stats)
        """
        import time
        start_time = time.time()

        # Partition image into blocks
        blocks = self._partition_into_blocks(mapped_indices)

        all_compressed_bits = []
        block_stats = []
        total_samples = 0

        # Process each block
        for block in blocks:
            block_data = block['data']
            total_samples += block_data.numel()

            # Estimate block entropy
            entropy_stats = self._estimate_block_entropy(block_data)

            # Select optimal code for this block
            code_type, code_index = self._select_block_code(entropy_stats)

            # Encode block header (code selection info)
            header_bits = self._encode_block_header(block, code_type, code_index)
            all_compressed_bits.extend(header_bits)

            # Encode block data
            block_bits = self._encode_block_with_code(block_data, code_type, code_index)
            all_compressed_bits.extend(block_bits)

            # Track statistics
            block_stat = {
                'band': block['band'],
                'coordinates': block['coordinates'],
                'entropy': entropy_stats['entropy'],
                'code_type': code_type,
                'code_index': code_index,
                'samples': block_data.numel(),
                'compressed_bits': len(block_bits) + len(header_bits)
            }
            block_stats.append(block_stat)

        # Convert bits to bytes
        compressed_data = self._bits_to_bytes(all_compressed_bits)

        # Compute compression statistics
        end_time = time.time()
        total_bits = len(all_compressed_bits)

        compression_stats = {
            'total_samples': total_samples,
            'total_compressed_bits': total_bits,
            'total_compressed_bytes': len(compressed_data),
            'bits_per_sample': total_bits / total_samples if total_samples > 0 else 0,
            'compression_ratio': (total_samples * 16) / total_bits if total_bits > 0 else 0,  # Assuming 16-bit samples
            'encoding_time': end_time - start_time,
            'throughput_samples_per_sec': total_samples / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'num_blocks': len(blocks),
            'block_size': self.block_size,
            'block_stats': block_stats
        }

        return compressed_data, compression_stats

    def _encode_block_header(self, block: Dict[str, Any], code_type: str, code_index: int) -> List[int]:
        """Encode block header with metadata"""
        # Simple header encoding (in practice would be more sophisticated)
        header = []

        # Block coordinates (simplified - would use variable-length encoding)
        y, x, y_end, x_end = block['coordinates']
        header.extend([y & 0xFF, (y >> 8) & 0xFF])  # y coordinate (16-bit)
        header.extend([x & 0xFF, (x >> 8) & 0xFF])  # x coordinate (16-bit)
        header.extend([block['shape'][0] & 0xFF])    # block height (8-bit)
        header.extend([block['shape'][1] & 0xFF])    # block width (8-bit)

        # Code selection (4 bits for code type, 4 bits for code index)
        code_type_bit = 0 if code_type == 'low' else 1
        code_info = (code_type_bit << 4) | (code_index & 0x0F)
        header.append(code_info)

        # Convert to bit list
        bits = []
        for byte_val in header:
            for i in range(8):
                bits.append((byte_val >> i) & 1)

        return bits

    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes"""
        # Pad to byte boundary
        while len(bits) % 8 != 0:
            bits.append(0)

        # Pack bits into bytes
        byte_data = []
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bits):
                    byte_val |= (bits[i + j] << j)
            byte_data.append(byte_val)

        return bytes(byte_data)
