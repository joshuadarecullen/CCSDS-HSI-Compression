"""
CCSDS-121.0-B-2 Rice Coder Implementation

This module implements the Rice coder as specified in CCSDS-121.0-B-2
for block-adaptive entropy coding as referenced in CCSDS-123.0-B-2 Issue 2.
"""

import torch
import math
from typing import List, Dict, Any, Tuple, Optional
from .bitstream import BitWriter, BitReader


class RiceCoder:
    """
    CCSDS-121.0-B-2 Rice Coder

    Implements the Rice coding algorithm with adaptive parameter selection
    as specified in the CCSDS-121.0-B-2 standard for lossless data compression.
    """

    def __init__(self, max_k: int = 14):
        """
        Initialize Rice coder

        Args:
            max_k: Maximum Rice parameter value
        """
        self.max_k = max_k
        self.bit_writer = BitWriter(8)

    def encode_block(self, data: torch.Tensor, k: Optional[int] = None) -> Tuple[List[int], int]:
        """
        Encode a block of data using Rice coding

        Args:
            data: Block of non-negative integers to encode
            k: Rice parameter (auto-selected if None)

        Returns:
            encoded_bits: List of encoded bits
            used_k: Rice parameter used for encoding
        """
        if k is None:
            k = self._select_optimal_k(data)

        encoded_bits = []

        for value in data:
            value = int(value)
            if value < 0:
                raise ValueError("Rice coding requires non-negative integers")

            # Rice encoding: quotient in unary + remainder in k-bit binary
            quotient = value >> k  # value // 2^k
            remainder = value & ((1 << k) - 1)  # value % 2^k

            # Unary encode quotient (quotient 1s followed by 0)
            for _ in range(quotient):
                encoded_bits.append(1)
            encoded_bits.append(0)

            # Binary encode remainder with k bits
            for i in range(k-1, -1, -1):
                encoded_bits.append((remainder >> i) & 1)

        return encoded_bits, k

    def decode_block(self, bit_reader: BitReader, block_size: int, k: int) -> List[int]:
        """
        Decode a block of Rice-encoded data

        Args:
            bit_reader: Bitstream reader
            block_size: Number of values to decode
            k: Rice parameter used for encoding

        Returns:
            decoded_values: List of decoded integers
        """
        decoded_values = []

        for _ in range(block_size):
            # Read unary-encoded quotient
            quotient = 0
            while bit_reader.read_bit() == 1:
                quotient += 1

            # Read k-bit remainder
            remainder = 0
            for i in range(k):
                bit = bit_reader.read_bit()
                if bit is None:
                    break
                remainder = (remainder << 1) | bit

            # Reconstruct original value
            value = (quotient << k) + remainder
            decoded_values.append(value)

        return decoded_values

    def _select_optimal_k(self, data: torch.Tensor) -> int:
        """
        Select optimal Rice parameter k for given data block

        Args:
            data: Block of non-negative integers

        Returns:
            optimal_k: Best Rice parameter for this block
        """
        if len(data) == 0:
            return 0

        # Estimate optimal k based on data statistics
        # k = log2(mean) is a good approximation for Rice coding
        mean_val = float(torch.mean(data.float()))

        if mean_val <= 1.0:
            optimal_k = 0
        else:
            optimal_k = min(self.max_k, max(0, int(math.log2(mean_val))))

        return optimal_k

    def estimate_compressed_size(self, data: torch.Tensor, k: Optional[int] = None) -> float:
        """
        Estimate compressed size in bits for given data block

        Args:
            data: Block of data
            k: Rice parameter (auto-selected if None)

        Returns:
            estimated_bits: Estimated number of bits after compression
        """
        if k is None:
            k = self._select_optimal_k(data)

        total_bits = 0
        for value in data:
            value = int(value)
            quotient = value >> k
            total_bits += quotient + 1 + k  # quotient (unary) + separator + remainder (k bits)

        return float(total_bits)


class CCSDS121BlockAdaptiveEntropyCoder:
    """
    CCSDS-121.0-B-2 Block-Adaptive Entropy Coder

    Implements block-adaptive Rice coding as referenced in CCSDS-123.0-B-2 Issue 2.
    This provides the third entropy coding option alongside sample-adaptive and hybrid coders.
    """

    def __init__(self, block_size: Tuple[int, int] = (16, 16)):
        """
        Initialize block-adaptive Rice coder

        Args:
            block_size: (height, width) of encoding blocks
        """
        self.block_size = block_size
        self.rice_coder = RiceCoder()

    def encode_image(self, mapped_indices: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        """
        Encode entire image using block-adaptive Rice coding

        Args:
            mapped_indices: [Z, Y, X] tensor of mapped quantizer indices

        Returns:
            compressed_data: Compressed bitstream
            compression_stats: Statistics about the compression
        """
        Z, Y, X = mapped_indices.shape
        bit_writer = BitWriter(8)

        total_bits = 0
        total_blocks = 0
        k_parameters = []

        # Process each band separately
        for z in range(Z):
            band_data = mapped_indices[z]

            # Partition band into blocks
            blocks = self._partition_into_blocks(band_data)

            for block_info in blocks:
                block_data = block_info['data']

                # Encode block with Rice coding
                encoded_bits, k_used = self.rice_coder.encode_block(block_data.flatten())

                # Write Rice parameter to bitstream (4 bits sufficient for k <= 15)
                bit_writer.write_uint(k_used, 4)

                # Write encoded block data
                bit_writer.write_bits(encoded_bits)

                total_bits += len(encoded_bits) + 4  # Include parameter overhead
                total_blocks += 1
                k_parameters.append(k_used)

        # Align to byte boundary
        bit_writer.align_to_byte()
        compressed_data = bit_writer.get_bytes()

        compression_stats = {
            'total_bits': total_bits,
            'total_blocks': total_blocks,
            'average_k': sum(k_parameters) / len(k_parameters) if k_parameters else 0,
            'k_parameter_distribution': self._compute_k_distribution(k_parameters),
            'compression_ratio': (Z * Y * X * 16) / (total_bits if total_bits > 0 else 1)  # Assume 16-bit input
        }

        return compressed_data, compression_stats

    def decode_image(self, compressed_data: bytes, image_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Decode Rice-compressed image data

        Args:
            compressed_data: Compressed bitstream
            image_shape: (Z, Y, X) image dimensions

        Returns:
            mapped_indices: [Z, Y, X] tensor of decoded mapped indices
        """
        Z, Y, X = image_shape
        bit_reader = BitReader(compressed_data)

        mapped_indices = torch.zeros(Z, Y, X, dtype=torch.long)

        for z in range(Z):
            # Get block partition info for this band
            band_dummy = torch.zeros(Y, X)  # Just for partitioning
            blocks = self._partition_into_blocks(band_dummy)

            for block_info in blocks:
                y_start, y_end = block_info['y_range']
                x_start, x_end = block_info['x_range']
                block_size = (y_end - y_start) * (x_end - x_start)

                # Read Rice parameter
                k = bit_reader.read_uint(4)
                if k is None:
                    break

                # Decode block data
                decoded_values = self.rice_coder.decode_block(bit_reader, block_size, k)

                # Reshape and place back into image
                block_data = torch.tensor(decoded_values).reshape(y_end - y_start, x_end - x_start)
                mapped_indices[z, y_start:y_end, x_start:x_end] = block_data

        return mapped_indices

    def _partition_into_blocks(self, band_data: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Partition band data into blocks for adaptive coding

        Args:
            band_data: [Y, X] band data

        Returns:
            blocks: List of block information dictionaries
        """
        Y, X = band_data.shape
        block_height, block_width = self.block_size
        blocks = []

        for y in range(0, Y, block_height):
            for x in range(0, X, block_width):
                y_end = min(y + block_height, Y)
                x_end = min(x + block_width, X)

                block_data = band_data[y:y_end, x:x_end]

                blocks.append({
                    'data': block_data,
                    'y_range': (y, y_end),
                    'x_range': (x, x_end),
                    'size': block_data.numel()
                })

        return blocks

    def _compute_k_distribution(self, k_parameters: List[int]) -> Dict[int, int]:
        """Compute distribution of Rice parameters used"""
        distribution = {}
        for k in k_parameters:
            distribution[k] = distribution.get(k, 0) + 1
        return distribution

    def get_coder_info(self) -> Dict[str, Any]:
        """Get information about the Rice coder configuration"""
        return {
            'coder_type': 'CCSDS-121.0-B-2 Rice Coder',
            'block_size': self.block_size,
            'max_k': self.rice_coder.max_k,
            'description': 'Block-adaptive Rice coding as specified in CCSDS-121.0-B-2'
        }


def encode_image_rice(mapped_indices: torch.Tensor, block_size: Tuple[int, int] = (16, 16)) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convenience function for Rice encoding an entire image

    Args:
        mapped_indices: [Z, Y, X] tensor of mapped quantizer indices
        block_size: Block size for adaptive coding

    Returns:
        compressed_data: Compressed bitstream
        compression_stats: Compression statistics
    """
    coder = CCSDS121BlockAdaptiveEntropyCoder(block_size)
    return coder.encode_image(mapped_indices)
