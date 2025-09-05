"""
CCSDS-123.0-B-2 Bitstream Formatting

Implements proper bitstream formatting including output word size handling,
byte alignment, and compressed image structure as specified in Section 5.3
of the CCSDS-123.0-B-2 standard.
"""

from typing import List, Optional
import struct


class BitstreamFormatter:
    """
    Handles formatting of compressed bitstream according to CCSDS-123.0-B-2 specification

    The compressed image structure consists of:
    1. Header (Image + Predictor + Entropy Coder Metadata)
    2. Compressed Image Body (entropy-coded samples)
    3. Optional padding for word alignment
    """

    def __init__(self, output_word_size: int = 8):
        """
        Initialize bitstream formatter

        Args:
            output_word_size: Output word size in bits (8, 16, 32, or 64)
        """
        if output_word_size not in [8, 16, 32, 64]:
            raise ValueError(f"Invalid output word size: {output_word_size}. Must be 8, 16, 32, or 64.")

        self.output_word_size = output_word_size
        self.output_byte_size = output_word_size // 8

    def format_bitstream(self, header_bytes: bytes, compressed_bits: List[int],
                        pad_to_word_boundary: bool = True) -> bytes:
        """
        Format complete compressed bitstream with proper word alignment

        Args:
            header_bytes: Header bytes (already properly formatted)
            compressed_bits: List of compressed bits from entropy coder
            pad_to_word_boundary: Whether to pad to word boundary

        Returns:
            formatted_bitstream: Complete formatted bitstream
        """
        # Convert bits to bytes
        body_bytes = self._bits_to_bytes(compressed_bits, pad_to_word_boundary)

        # Combine header and body
        complete_bitstream = header_bytes + body_bytes

        # Apply final word alignment if needed
        if pad_to_word_boundary:
            complete_bitstream = self._align_to_word_boundary(complete_bitstream)

        return complete_bitstream

    def _bits_to_bytes(self, bits: List[int], pad_to_byte: bool = True) -> bytes:
        """
        Convert list of bits to bytes with optional padding

        Args:
            bits: List of bits (0 or 1)
            pad_to_byte: Whether to pad to byte boundary

        Returns:
            bytes_data: Converted bytes
        """
        if not bits:
            return b''

        # Pad to byte boundary if requested
        if pad_to_byte and len(bits) % 8 != 0:
            padding_bits = 8 - (len(bits) % 8)
            bits = bits + [0] * padding_bits

        # Convert bits to bytes (MSB first)
        bytes_data = bytearray()
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte_bits = bits[i:i+8]
                byte_value = 0
                for j, bit in enumerate(byte_bits):
                    byte_value |= (bit << (7 - j))  # MSB first
                bytes_data.append(byte_value)

        return bytes(bytes_data)

    def _align_to_word_boundary(self, data: bytes) -> bytes:
        """
        Align data to output word boundary with zero padding

        Args:
            data: Input data bytes

        Returns:
            aligned_data: Data aligned to word boundary
        """
        if len(data) % self.output_byte_size == 0:
            return data  # Already aligned

        # Calculate padding needed
        padding_bytes = self.output_byte_size - (len(data) % self.output_byte_size)

        # Add zero padding
        return data + b'\x00' * padding_bytes

    def extract_header_and_body(self, bitstream: bytes, header_size: int) -> tuple[bytes, bytes]:
        """
        Extract header and body from formatted bitstream

        Args:
            bitstream: Complete formatted bitstream
            header_size: Size of header in bytes

        Returns:
            header_bytes, body_bytes: Separated header and body
        """
        if len(bitstream) < header_size:
            raise ValueError(f"Bitstream too short: {len(bitstream)} < {header_size}")

        header_bytes = bitstream[:header_size]
        body_bytes = bitstream[header_size:]

        return header_bytes, body_bytes

    def bytes_to_bits(self, data: bytes, remove_padding: bool = False,
                     expected_bit_count: Optional[int] = None) -> List[int]:
        """
        Convert bytes back to list of bits

        Args:
            data: Input byte data
            remove_padding: Whether to remove zero padding
            expected_bit_count: Expected number of bits (removes trailing padding)

        Returns:
            bits: List of bits (0 or 1)
        """
        bits = []
        for byte_val in data:
            for i in range(8):
                bits.append((byte_val >> (7 - i)) & 1)  # MSB first

        # Remove padding if requested
        if expected_bit_count is not None:
            bits = bits[:expected_bit_count]
        elif remove_padding:
            # Remove trailing zeros
            while bits and bits[-1] == 0:
                bits.pop()

        return bits

    def calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """
        Calculate compression ratio

        Args:
            original_size: Original data size in bytes
            compressed_size: Compressed data size in bytes

        Returns:
            ratio: Compression ratio (original_size / compressed_size)
        """
        if compressed_size == 0:
            return float('inf')
        return original_size / compressed_size

    def get_bitstream_info(self, bitstream: bytes) -> dict:
        """
        Get information about formatted bitstream

        Args:
            bitstream: Formatted bitstream

        Returns:
            info: Dictionary with bitstream information
        """
        return {
            'total_size_bytes': len(bitstream),
            'total_size_bits': len(bitstream) * 8,
            'output_word_size': self.output_word_size,
            'is_word_aligned': len(bitstream) % self.output_byte_size == 0,
            'padding_bytes': (self.output_byte_size - (len(bitstream) % self.output_byte_size)) % self.output_byte_size
        }


class BitWriter:
    """
    Utility class for efficient bit writing with buffering

    This class provides efficient bit-level writing operations with
    automatic buffering and word-aligned output formatting.
    """

    def __init__(self, output_word_size: int = 8):
        """
        Initialize bit writer

        Args:
            output_word_size: Output word size in bits
        """
        self.formatter = BitstreamFormatter(output_word_size)
        self.buffer = []

    def write_bit(self, bit: int) -> None:
        """Write single bit to buffer"""
        self.buffer.append(bit & 1)

    def write_bits(self, bits: List[int]) -> None:
        """Write multiple bits to buffer"""
        self.buffer.extend([bit & 1 for bit in bits])

    def write_uint(self, value: int, bit_count: int) -> None:
        """Write unsigned integer with specified bit count (MSB first)"""
        for i in range(bit_count - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def write_unary(self, value: int) -> None:
        """Write unary encoded value (value ones followed by zero)"""
        for _ in range(value):
            self.write_bit(1)
        self.write_bit(0)

    def get_bit_count(self) -> int:
        """Get current number of bits in buffer"""
        return len(self.buffer)

    def get_bytes(self, pad_to_word_boundary: bool = True) -> bytes:
        """
        Get formatted bytes from buffer

        Args:
            pad_to_word_boundary: Whether to pad to word boundary

        Returns:
            formatted_bytes: Formatted byte data
        """
        return self.formatter._bits_to_bytes(self.buffer, pad_to_word_boundary)

    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer = []

    def align_to_byte(self) -> None:
        """Align buffer to byte boundary by adding zero bits"""
        while len(self.buffer) % 8 != 0:
            self.write_bit(0)

    def align_to_word(self) -> None:
        """Align buffer to word boundary"""
        word_bit_size = self.formatter.output_word_size
        while len(self.buffer) % word_bit_size != 0:
            self.write_bit(0)


class BitReader:
    """
    Utility class for efficient bit reading with reverse-order support

    Supports both forward and reverse-order reading as required by
    CCSDS-123.0-B-2 entropy decoding procedures.
    """

    def __init__(self, data: bytes):
        """
        Initialize bit reader

        Args:
            data: Input byte data
        """
        self.formatter = BitstreamFormatter()
        self.bits = self.formatter.bytes_to_bits(data)
        self.position = 0
        self.reverse_mode = False

    def set_reverse_mode(self, reverse: bool) -> None:
        """Enable/disable reverse reading mode"""
        self.reverse_mode = reverse
        if reverse:
            self.position = len(self.bits) - 1
        else:
            self.position = 0

    def read_bit(self) -> Optional[int]:
        """Read single bit (respects reverse mode)"""
        if self.reverse_mode:
            if self.position < 0:
                return None
            bit = self.bits[self.position]
            self.position -= 1
            return bit
        else:
            if self.position >= len(self.bits):
                return None
            bit = self.bits[self.position]
            self.position += 1
            return bit

    def read_bits(self, count: int) -> Optional[List[int]]:
        """Read multiple bits"""
        bits = []
        for _ in range(count):
            bit = self.read_bit()
            if bit is None:
                return None
            bits.append(bit)
        return bits

    def read_uint(self, bit_count: int) -> Optional[int]:
        """Read unsigned integer (MSB first)"""
        bits = self.read_bits(bit_count)
        if bits is None:
            return None

        value = 0
        for i, bit in enumerate(bits):
            value |= (bit << (bit_count - 1 - i))
        return value

    def read_unary(self) -> Optional[int]:
        """Read unary encoded value"""
        count = 0
        while True:
            bit = self.read_bit()
            if bit is None:
                return None
            if bit == 0:
                break
            count += 1
        return count

    def bits_remaining(self) -> int:
        """Get number of bits remaining"""
        if self.reverse_mode:
            return self.position + 1
        else:
            return len(self.bits) - self.position

    def seek(self, position: int) -> None:
        """Seek to specific bit position"""
        self.position = max(0, min(position, len(self.bits) - 1))

    def tell(self) -> int:
        """Get current bit position"""
        return self.position
