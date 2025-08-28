import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import struct


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
    
    def __init__(self, num_bands, rescale_interval=64):
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
        
    def _initialize_low_entropy_codes(self):
        """
        Initialize the 16 variable-to-variable length low-entropy codes
        Based on Table 1 from the paper
        """
        codes = []
        
        # Code specifications from Table 1 in the paper
        code_specs = [
            (0, 12, 105, 3, 13),   # Code 0: limit=12, 105 codewords, max_in=3, max_out=13
            (1, 10, 144, 3, 13),   # Code 1: limit=10, 144 codewords, max_in=3, max_out=13
            (2, 8, 118, 3, 12),    # Code 2: limit=8, 118 codewords, max_in=3, max_out=12
            (3, 6, 120, 4, 13),    # Code 3: limit=6, 120 codewords, max_in=4, max_out=13
            (4, 6, 92, 4, 13),     # Code 4: limit=6, 92 codewords, max_in=4, max_out=13
            (5, 4, 116, 6, 15),    # Code 5: limit=4, 116 codewords, max_in=6, max_out=15
            (6, 4, 101, 6, 15),    # Code 6: limit=4, 101 codewords, max_in=6, max_out=15
            (7, 4, 81, 5, 18),     # Code 7: limit=4, 81 codewords, max_in=5, max_out=18
            (8, 2, 88, 12, 16),    # Code 8: limit=2, 88 codewords, max_in=12, max_out=16
            (9, 2, 106, 12, 17),   # Code 9: limit=2, 106 codewords, max_in=12, max_out=17
            (10, 2, 103, 12, 18),  # Code 10: limit=2, 103 codewords, max_in=12, max_out=18
            (11, 2, 127, 16, 20),  # Code 11: limit=2, 127 codewords, max_in=16, max_out=20
            (12, 2, 109, 27, 21),  # Code 12: limit=2, 109 codewords, max_in=27, max_out=21
            (13, 2, 145, 46, 18),  # Code 13: limit=2, 145 codewords, max_in=46, max_out=18
            (14, 2, 256, 85, 17),  # Code 14: limit=2, 256 codewords, max_in=85, max_out=17
            (15, 0, 257, 256, 9),  # Code 15: limit=0, 257 codewords, max_in=256, max_out=9
        ]
        
        for code_idx, input_limit, num_codewords, max_input_len, max_output_len in code_specs:
            code = LowEntropyCode(code_idx, input_limit, num_codewords, max_input_len, max_output_len)
            codes.append(code)
            
        return codes
    
    def update_statistics(self, mapped_index, band):
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
    
    def classify_entropy(self, band):
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
    
    def encode_high_entropy(self, mapped_index, code_index):
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
    
    def encode_low_entropy(self, mapped_indices, code_index):
        """
        Encode sequence of low-entropy samples using variable-to-variable code
        
        Args:
            mapped_indices: List of mapped quantizer indices
            code_index: Low-entropy code index (0-15)
            
        Returns:
            encoded_bits: List of bits, or None if more samples needed
        """
        if code_index >= len(self.low_entropy_codes):
            return None
            
        code = self.low_entropy_codes[code_index]
        return code.encode(mapped_indices)
    
    def encode_escape_symbol(self, residual_value):
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
    
    def encode_samples(self, mapped_indices, band):
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
        
        return compressed_bits


class LowEntropyCode:
    """
    Individual low-entropy variable-to-variable length code
    """
    
    def __init__(self, code_index, input_limit, num_codewords, max_input_len, max_output_len):
        self.code_index = code_index
        self.input_limit = input_limit  # L_i
        self.num_codewords = num_codewords
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
        # Generate codeword tables (simplified - would be from standard specification)
        self.input_to_output = self._generate_codeword_table()
        self.input_buffer = []
        
    def _generate_codeword_table(self):
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
    
    def encode(self, input_symbols):
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
    
    def _encode_residual(self, residual):
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
    
    def __init__(self):
        self.buffer = bytearray()
        self.bit_buffer = 0
        self.bit_count = 0
    
    def write_bit(self, bit):
        """Write a single bit"""
        self.bit_buffer = (self.bit_buffer << 1) | (bit & 1)
        self.bit_count += 1
        
        if self.bit_count == 8:
            self.buffer.append(self.bit_buffer)
            self.bit_buffer = 0
            self.bit_count = 0
    
    def write_bits(self, bits):
        """Write a list of bits"""
        for bit in bits:
            self.write_bit(bit)
    
    def flush(self):
        """Flush remaining bits with padding"""
        if self.bit_count > 0:
            # Pad with zeros
            while self.bit_count < 8:
                self.write_bit(0)
        
        return bytes(self.buffer)


def encode_image(mapped_indices, num_bands):
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