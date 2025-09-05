import torch
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Union
from ..ccsds.low_entropy_tables import LOW_ENTROPY_TABLES, CompleteLowEntropyCode
from ..ccsds.rice_coder import RiceCoder


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

    def _initialize_optimized_codes(self) -> List[CompleteLowEntropyCode]:
        """
        Initialize complete CCSDS-123.0-B-2 low-entropy code tables

        Uses the complete Table 5-16 specification with proper codeword mappings
        """
        codes = []

        # Use complete low-entropy code tables from the standard
        for code_id in range(16):
            code = CompleteLowEntropyCode(code_id, LOW_ENTROPY_TABLES)
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

    def encode_high_entropy_vectorized(self,
                                       mapped_indices: torch.Tensor,
                                       code_indices: torch.Tensor) -> List[int]:
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
                # Process samples one by one with the complete low-entropy code
                for sample in samples.tolist():
                    code_bits = code.add_symbol(int(sample))
                    if code_bits:
                        encoded_bits.extend(code_bits)
                        
                # Flush any remaining symbols in the code buffer
                flush_bits = code.flush()
                for bit_sequence in flush_bits:
                    encoded_bits.extend(bit_sequence)

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


class OptimizedBlockAdaptiveEntropyCoder:
    """
    GPU-Accelerated Block-Adaptive Entropy Coder for CCSDS-123.0-B-2
    
    Features:
    - Vectorized block partitioning using tensor operations
    - GPU-accelerated entropy estimation
    - Parallel block processing with batching
    - Memory-efficient streaming for large images
    - Device-aware tensor management
    """
    
    def __init__(self, num_bands: int, block_size: Tuple[int, int] = (8, 8), 
                 min_block_samples: int = 16, device: str = 'cpu',
                 gpu_batch_size: int = 32) -> None:
        self.num_bands = num_bands
        self.block_size = block_size
        self.min_block_samples = min_block_samples
        self.device = device
        self.gpu_batch_size = gpu_batch_size  # Number of blocks to process in parallel
        
        # Pre-compute low-entropy code lookup tables on device
        self.low_entropy_codes = self._initialize_optimized_codes()
        
        # Pre-allocate GPU memory for block statistics
        self.block_stats_buffer = None
        
    def _initialize_optimized_codes(self) -> List[CompleteLowEntropyCode]:
        """Initialize complete CCSDS-123.0-B-2 low-entropy code tables for block-adaptive encoding"""
        codes = []
        
        # Use complete low-entropy code tables from Table 5-16
        for code_id in range(16):
            code = CompleteLowEntropyCode(code_id, LOW_ENTROPY_TABLES)
            codes.append(code)
            
        return codes
    
    def _partition_into_blocks_vectorized(self, mapped_indices: torch.Tensor) -> List[Dict[str, Any]]:
        """
        GPU-accelerated block partitioning using vectorized operations
        
        Args:
            mapped_indices: [Z, Y, X] tensor of mapped indices
            
        Returns:
            List of block dictionaries with pre-computed coordinates and data
        """
        Z, Y, X = mapped_indices.shape
        block_h, block_w = self.block_size
        
        blocks = []
        
        # Vectorized block extraction for each band
        for z in range(Z):
            band_data = mapped_indices[z]  # [Y, X]
            
            # Calculate number of blocks in each dimension
            num_blocks_y = (Y + block_h - 1) // block_h
            num_blocks_x = (X + block_w - 1) // block_w
            
            for block_y in range(num_blocks_y):
                for block_x in range(num_blocks_x):
                    # Calculate block boundaries
                    y_start = block_y * block_h
                    y_end = min(y_start + block_h, Y)
                    x_start = block_x * block_w
                    x_end = min(x_start + block_w, X)
                    
                    # Extract block data (vectorized)
                    block_data = band_data[y_start:y_end, x_start:x_end].contiguous()
                    
                    # Skip blocks that are too small
                    if block_data.numel() < self.min_block_samples:
                        continue
                    
                    blocks.append({
                        'band': z,
                        'coordinates': (y_start, x_start, y_end, x_end),
                        'shape': (y_end - y_start, x_end - x_start),
                        'data': block_data.flatten(),  # Flatten for entropy computation
                        'size': block_data.numel()
                    })
        
        return blocks
    
    def _estimate_block_entropy_batch(self, block_data_list: List[torch.Tensor]) -> List[Dict[str, float]]:
        """
        GPU-accelerated batch entropy estimation for multiple blocks
        
        Args:
            block_data_list: List of flattened block tensors
            
        Returns:
            List of entropy statistics for each block
        """
        entropy_stats = []
        
        # Process blocks in batches for GPU efficiency
        for i in range(0, len(block_data_list), self.gpu_batch_size):
            batch_blocks = block_data_list[i:i + self.gpu_batch_size]
            
            # Compute entropy statistics for batch
            for block_data in batch_blocks:
                # Move to device if needed
                if block_data.device.type != self.device:
                    block_data = block_data.to(self.device)
                
                # Vectorized histogram computation
                unique_vals, counts = torch.unique(block_data, return_counts=True)
                
                # Compute probability distribution
                total_samples = block_data.numel()
                probabilities = counts.float() / total_samples
                
                # Compute entropy: H = -sum(p * log2(p))
                entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-12))
                
                # Additional statistics
                variance = torch.var(block_data.float())
                mean_val = torch.mean(block_data.float())
                max_magnitude = torch.max(torch.abs(block_data)).item()
                
                entropy_stats.append({
                    'entropy': entropy.item(),
                    'variance': variance.item(), 
                    'mean': mean_val.item(),
                    'unique_values': len(unique_vals),
                    'max_magnitude': max_magnitude,
                    'total_samples': total_samples
                })
        
        return entropy_stats
    
    def _select_optimal_code_vectorized(self, entropy_stats: Dict[str, float]) -> Tuple[str, int]:
        """
        Vectorized code selection based on block statistics
        
        Args:
            entropy_stats: Block entropy and statistics
            
        Returns:
            (code_type, code_index) tuple
        """
        entropy = entropy_stats['entropy']
        max_magnitude = entropy_stats['max_magnitude']
        unique_values = entropy_stats['unique_values']
        
        # Classification thresholds (tuned for GPU efficiency)
        low_entropy_threshold = 4.0
        very_low_entropy_threshold = 2.0
        
        if entropy < very_low_entropy_threshold:
            # Very uniform block - use most efficient low-entropy code
            if unique_values <= 2:
                return ('low', 15)  # Most efficient for binary data
            elif unique_values <= 4: 
                return ('low', 14)
            else:
                return ('low', 13)
                
        elif entropy < low_entropy_threshold:
            # Low entropy block - select based on magnitude and diversity
            if max_magnitude <= 16:
                if unique_values <= 8:
                    return ('low', 12)
                elif unique_values <= 32:
                    return ('low', 10)
                else:
                    return ('low', 8)
            else:
                return ('low', 5)  # Moderate entropy, higher magnitude
                
        else:
            # High entropy block - use GPO2 codes
            if max_magnitude <= 32:
                return ('high', 2)  # GPO2 with k=2
            elif max_magnitude <= 128:
                return ('high', 3)  # GPO2 with k=3  
            else:
                return ('high', 4)  # GPO2 with k=4
    
    def _encode_blocks_batch(self, blocks: List[Dict[str, Any]], 
                           entropy_stats_list: List[Dict[str, float]]) -> Tuple[bytes, Dict[str, Any]]:
        """
        GPU-accelerated batch encoding of blocks
        
        Args:
            blocks: List of block dictionaries
            entropy_stats_list: List of entropy statistics for each block
            
        Returns:
            (compressed_data, compression_stats) tuple
        """
        import time
        start_time = time.time()
        
        all_compressed_bits = []
        total_samples = 0
        block_stats = []
        
        # Process blocks in GPU-efficient batches
        for i, (block, entropy_stats) in enumerate(zip(blocks, entropy_stats_list)):
            # Select optimal code for block
            code_type, code_index = self._select_optimal_code_vectorized(entropy_stats)
            
            # Encode block header (metadata)
            header_bits = self._encode_block_header_optimized(block, code_type, code_index)
            all_compressed_bits.extend(header_bits)
            
            # Encode block data with selected code
            block_bits = self._encode_block_data_optimized(block['data'], code_type, code_index)
            all_compressed_bits.extend(block_bits)
            
            # Update statistics
            total_samples += block['size']
            block_stats.append({
                'band': block['band'],
                'coordinates': block['coordinates'],
                'entropy': entropy_stats['entropy'],
                'code_type': code_type,
                'code_index': code_index,
                'samples': block['size'],
                'compressed_bits': len(header_bits) + len(block_bits)
            })
        
        # Convert bits to bytes
        compressed_data = self._bits_to_bytes_optimized(all_compressed_bits)
        
        # Compute final statistics
        end_time = time.time()
        total_bits = len(all_compressed_bits)
        
        compression_stats = {
            'total_samples': total_samples,
            'total_compressed_bits': total_bits,
            'total_compressed_bytes': len(compressed_data),
            'bits_per_sample': total_bits / total_samples if total_samples > 0 else 0,
            'compression_ratio': (total_samples * 16) / total_bits if total_bits > 0 else 0,
            'encoding_time': end_time - start_time,
            'throughput_samples_per_sec': total_samples / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'num_blocks': len(blocks),
            'block_size': self.block_size,
            'gpu_batch_size': self.gpu_batch_size,
            'device': self.device,
            'block_stats': block_stats
        }
        
        return compressed_data, compression_stats
    
    def _encode_block_header_optimized(self, block: Dict[str, Any], 
                                     code_type: str, code_index: int) -> List[int]:
        """Optimized block header encoding"""
        header = []
        
        # Block coordinates and dimensions (optimized bit packing)
        y, x, y_end, x_end = block['coordinates']
        header.extend([(y >> i) & 1 for i in range(16)])        # y coordinate
        header.extend([(x >> i) & 1 for i in range(16)])        # x coordinate  
        header.extend([(block['shape'][0] >> i) & 1 for i in range(8)])  # height
        header.extend([(block['shape'][1] >> i) & 1 for i in range(8)])  # width
        
        # Code selection (4 bits total)
        code_type_bit = 0 if code_type == 'low' else 1
        code_info = (code_type_bit << 3) | (code_index & 0x07)
        header.extend([(code_info >> i) & 1 for i in range(4)])
        
        return header
    
    def _encode_block_data_optimized(self, block_data: torch.Tensor, 
                                   code_type: str, code_index: int) -> List[int]:
        """GPU-optimized block data encoding"""
        if code_type == 'low':
            return self._encode_with_low_entropy_optimized(block_data, code_index)
        else:
            return self._encode_with_gpo2_optimized(block_data, code_index)
    
    def _encode_with_low_entropy_optimized(self, data: torch.Tensor, code_index: int) -> List[int]:
        """Optimized low-entropy encoding with vectorized operations"""
        code = self.low_entropy_codes[code_index]
        bits = []
        
        # Vectorized processing for GPU efficiency
        data_cpu = data.cpu() if data.device.type != 'cpu' else data
        
        for value in data_cpu:
            val = value.item()
            abs_val = abs(val)
            
            if abs_val <= code['input_limit']:
                # Use low-entropy code (simplified bit generation)
                sign_bit = 0 if val >= 0 else 1
                
                # Generate variable-length codeword
                codeword_bits = []
                temp = abs_val
                for _ in range(code['max_output_len']):
                    codeword_bits.append(temp & 1)
                    temp >>= 1
                    if temp == 0:
                        break
                
                bits.extend(codeword_bits)
                bits.append(sign_bit)
            else:
                # Fallback to GPO2 for values outside code range
                fallback_bits = self._gpo2_encode_single(val, 3)
                bits.extend(fallback_bits)
        
        return bits
    
    def _encode_with_gpo2_optimized(self, data: torch.Tensor, k: int) -> List[int]:
        """Optimized GPO2 encoding with vectorized operations"""
        bits = []
        
        # Move to CPU for bit manipulation (more efficient than GPU for this operation)
        data_cpu = data.cpu() if data.device.type != 'cpu' else data
        
        for value in data_cpu:
            val = value.item()
            bits.extend(self._gpo2_encode_single(val, k))
        
        return bits
    
    def _gpo2_encode_single(self, value: int, k: int) -> List[int]:
        """Optimized single value GPO2 encoding"""
        abs_val = abs(value)
        sign_bit = 0 if value >= 0 else 1
        
        # Efficient bit operations
        quotient = abs_val >> k
        remainder = abs_val & ((1 << k) - 1)
        
        # Unary prefix + k-bit suffix + sign
        bits = [1] * quotient + [0]  # Unary
        bits.extend([(remainder >> i) & 1 for i in range(k)])  # k-bit suffix
        bits.append(sign_bit)  # Sign bit
        
        return bits
    
    def _bits_to_bytes_optimized(self, bits: List[int]) -> bytes:
        """Optimized bit-to-byte conversion"""
        # Pad to byte boundary
        while len(bits) % 8 != 0:
            bits.append(0)
        
        # Efficient byte packing using numpy
        byte_array = np.zeros(len(bits) // 8, dtype=np.uint8)
        bits_array = np.array(bits, dtype=np.uint8)
        
        for i in range(8):
            byte_array |= (bits_array[i::8] << i)
        
        return bytes(byte_array)
    
    def encode_image_block_adaptive_optimized(self, mapped_indices: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        """
        Main optimized block-adaptive encoding function
        
        Args:
            mapped_indices: [Z, Y, X] tensor of mapped quantizer indices
            
        Returns:
            (compressed_data, compression_stats) tuple
        """
        # Ensure tensor is on correct device
        if mapped_indices.device.type != self.device:
            mapped_indices = mapped_indices.to(self.device)
        
        # GPU-accelerated block partitioning
        blocks = self._partition_into_blocks_vectorized(mapped_indices)
        
        # Batch entropy estimation
        block_data_list = [block['data'] for block in blocks]
        entropy_stats_list = self._estimate_block_entropy_batch(block_data_list)
        
        # Batch encoding
        compressed_data, compression_stats = self._encode_blocks_batch(blocks, entropy_stats_list)
        
        return compressed_data, compression_stats


def encode_image_block_adaptive_optimized(mapped_indices: torch.Tensor, num_bands: int,
                                        block_size: Tuple[int, int] = (8, 8),
                                        min_block_samples: int = 16,
                                        gpu_batch_size: int = 32) -> Tuple[bytes, Dict[str, Any]]:
    """
    Optimized block-adaptive image encoding function
    
    Args:
        mapped_indices: [Z, Y, X] tensor of mapped quantizer indices
        num_bands: Number of spectral bands
        block_size: (height, width) of blocks
        min_block_samples: Minimum samples per block
        gpu_batch_size: Number of blocks to process in parallel
        
    Returns:
        compressed_data: Bytes of compressed image data
        compression_stats: Detailed compression statistics
    """
    device = mapped_indices.device.type
    encoder = OptimizedBlockAdaptiveEntropyCoder(
        num_bands=num_bands,
        block_size=block_size, 
        min_block_samples=min_block_samples,
        device=device,
        gpu_batch_size=gpu_batch_size
    )
    return encoder.encode_image_block_adaptive_optimized(mapped_indices)


class OptimizedRiceCoder:
    """
    GPU-Optimized CCSDS-121.0-B-2 Rice Coder for Issue 2 Compatibility
    
    Provides vectorized Rice coding with GPU acceleration for the optimized
    CCSDS-123.0-B-2 compressor pipeline.
    """
    
    def __init__(self, block_size: Tuple[int, int] = (16, 16), device: str = 'cpu', gpu_batch_size: int = 32):
        self.block_size = block_size
        self.device = device
        self.gpu_batch_size = gpu_batch_size
        self.rice_coder = RiceCoder()
        
    def _select_optimal_k_batch(self, block_data_list: List[torch.Tensor]) -> List[int]:
        """
        Vectorized optimal Rice parameter selection for multiple blocks
        
        Args:
            block_data_list: List of block data tensors
            
        Returns:
            optimal_k_list: List of optimal k parameters for each block
        """
        optimal_k_list = []
        
        for block_data in block_data_list:
            if len(block_data) == 0:
                optimal_k_list.append(0)
                continue
                
            # Vectorized mean computation
            mean_val = float(torch.mean(block_data.float()))
            
            if mean_val <= 1.0:
                optimal_k = 0
            else:
                optimal_k = min(14, max(0, int(np.log2(mean_val))))
                
            optimal_k_list.append(optimal_k)
            
        return optimal_k_list
        
    def encode_blocks_rice_optimized(self, mapped_indices: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        """
        GPU-optimized Rice encoding of image blocks
        
        Args:
            mapped_indices: [Z, Y, X] tensor of mapped quantizer indices
            
        Returns:
            compressed_data: Compressed bitstream as bytes
            compression_stats: Detailed statistics
        """
        start_time = time.time()
        Z, Y, X = mapped_indices.shape
        
        # Partition into blocks (vectorized)
        blocks = self._partition_into_blocks_optimized(mapped_indices)
        
        total_bits = 0
        total_blocks = len(blocks)
        k_parameters = []
        encoded_data = bytearray()
        
        # Process blocks in GPU-friendly batches
        for batch_start in range(0, len(blocks), self.gpu_batch_size):
            batch_end = min(batch_start + self.gpu_batch_size, len(blocks))
            batch_blocks = blocks[batch_start:batch_end]
            
            # Extract block data for batch processing
            batch_data = [block['data'] for block in batch_blocks]
            
            # Vectorized k parameter selection
            optimal_k_values = self._select_optimal_k_batch(batch_data)
            k_parameters.extend(optimal_k_values)
            
            # Encode each block in the batch
            for i, (block, k) in enumerate(zip(batch_blocks, optimal_k_values)):
                block_data = block['data']
                
                # Rice encode block
                encoded_bits, _ = self.rice_coder.encode_block(block_data, k)
                
                # Convert bits to bytes for storage
                block_bytes = self._bits_to_bytes(encoded_bits)
                
                # Store Rice parameter (4 bits) + block data
                param_byte = k & 0x0F  # 4 bits for k parameter
                encoded_data.append(param_byte)
                encoded_data.extend(block_bytes)
                
                total_bits += len(encoded_bits) + 4  # Include parameter overhead
        
        # Final compression statistics
        encoding_time = time.time() - start_time
        original_size = Z * Y * X * 16  # Assume 16-bit input
        compression_ratio = original_size / (total_bits if total_bits > 0 else 1)
        
        compression_stats = {
            'encoding_time': encoding_time,
            'total_bits': total_bits,
            'total_blocks': total_blocks,
            'average_k': sum(k_parameters) / len(k_parameters) if k_parameters else 0,
            'k_parameter_distribution': self._compute_k_distribution(k_parameters),
            'compression_ratio': compression_ratio,
            'original_size': original_size,
            'encoded_size': len(encoded_data) * 8,
            'throughput_blocks_per_sec': total_blocks / encoding_time if encoding_time > 0 else 0
        }
        
        return bytes(encoded_data), compression_stats
        
    def _partition_into_blocks_optimized(self, mapped_indices: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Optimized block partitioning with vectorized operations
        
        Args:
            mapped_indices: [Z, Y, X] tensor of mapped indices
            
        Returns:
            List of block dictionaries
        """
        Z, Y, X = mapped_indices.shape
        block_height, block_width = self.block_size
        blocks = []
        
        for z in range(Z):
            band_data = mapped_indices[z]
            
            for y in range(0, Y, block_height):
                for x in range(0, X, block_width):
                    y_end = min(y + block_height, Y)
                    x_end = min(x + block_width, X)
                    
                    block_data = band_data[y:y_end, x:x_end]
                    
                    blocks.append({
                        'data': block_data.flatten(),
                        'coordinates': (z, y, x, y_end, x_end),
                        'size': block_data.numel()
                    })
                    
        return blocks
        
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes"""
        if not bits:
            return b''
            
        # Pad to byte boundary
        while len(bits) % 8 != 0:
            bits.append(0)
            
        # Convert to bytes
        byte_data = []
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bits):
                    byte_val |= (bits[i + j] << (7 - j))
            byte_data.append(byte_val)
            
        return bytes(byte_data)
        
    def _compute_k_distribution(self, k_parameters: List[int]) -> Dict[int, int]:
        """Compute distribution of Rice parameters used"""
        distribution = {}
        for k in k_parameters:
            distribution[k] = distribution.get(k, 0) + 1
        return distribution


def encode_image_rice_optimized(mapped_indices: torch.Tensor, 
                               block_size: Tuple[int, int] = (16, 16),
                               device: str = 'auto',
                               gpu_batch_size: int = 32) -> Tuple[bytes, Dict[str, Any]]:
    """
    Optimized Rice encoding for entire image with GPU acceleration
    
    Args:
        mapped_indices: [Z, Y, X] tensor of mapped quantizer indices
        block_size: Block size for Rice coding
        device: Processing device ('auto', 'cpu', 'cuda')
        gpu_batch_size: Number of blocks to process in parallel
        
    Returns:
        compressed_data: Compressed bitstream as bytes
        compression_stats: Detailed compression statistics  
    """
    if device == 'auto':
        device = mapped_indices.device.type
        
    encoder = OptimizedRiceCoder(block_size=block_size, device=device, gpu_batch_size=gpu_batch_size)
    return encoder.encode_blocks_rice_optimized(mapped_indices)
