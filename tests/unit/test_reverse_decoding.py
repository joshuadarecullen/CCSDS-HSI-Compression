"""
Test script to verify reverse-order decoding implementation
"""

import torch
import numpy as np
from .ccsds_entropy_coder import CCSDS123HybridEntropyCoder, CCSDS123HybridDecoder


def test_reverse_decoding():
    """Test that encoding and reverse-order decoding work correctly"""

    # Create test data
    num_bands = 4
    height, width = 8, 8
    test_data = torch.randint(0, 16, (num_bands, height, width), dtype=torch.long)

    print(f"Testing reverse-order decoding with {num_bands}x{height}x{width} data")
    print(f"Original data sample: {test_data[0, 0, :4]}")

    # Encode with hybrid entropy coder
    encoder = CCSDS123HybridEntropyCoder(num_bands, gamma_star=1, k=1)
    compressed_data = encoder.encode_image(test_data)

    print(f"Compressed data size: {len(compressed_data)} bytes")
    print(f"Compression ratio: {(num_bands * height * width * 2) / len(compressed_data):.2f}:1")

    # Decode with reverse-order decoder
    decoder = CCSDS123HybridDecoder(num_bands, gamma_star=1, k=1)
    decoded_data = decoder.decode_bitstream(compressed_data, (num_bands, height, width))

    print(f"Decoded data sample: {decoded_data[0, 0, :4]}")

    # Verify correctness
    matches = torch.equal(test_data, decoded_data)
    print(f"Decoding successful: {matches}")

    if not matches:
        diff = torch.sum(torch.abs(test_data - decoded_data))
        print(f"Total absolute difference: {diff}")
        print(f"Max difference: {torch.max(torch.abs(test_data - decoded_data))}")

        # Show first few differences
        diff_mask = test_data != decoded_data
        if torch.any(diff_mask):
            diff_indices = torch.where(diff_mask)
            for i in range(min(5, len(diff_indices[0]))):
                z, y, x = diff_indices[0][i], diff_indices[1][i], diff_indices[2][i]
                print(f"Difference at [{z},{y},{x}]: {test_data[z,y,x]} -> {decoded_data[z,y,x]}")

    return matches


def test_reverse_decoding_properties():
    """Test specific reverse-order decoding properties"""

    print("\n=== Testing Reverse-Order Decoding Properties ===")

    # Test 1: Suffix-free codes
    encoder = CCSDS123HybridEntropyCoder(1, gamma_star=1, k=1)

    # Encode some specific patterns that should use different code types
    test_patterns = [
        torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=torch.long),  # Low entropy
        torch.tensor([[[10, 15, 20, 25, 30, 35, 40, 45]]], dtype=torch.long),  # High entropy
        torch.tensor([[[0, 0, 1, 1, 2, 2, 3, 3]]], dtype=torch.long)  # Very low entropy
    ]

    for i, pattern in enumerate(test_patterns):
        print(f"\nTest pattern {i+1}: {pattern.flatten()[:8]}")

        compressed = encoder.encode_image(pattern)
        decoder = CCSDS123HybridDecoder(1, gamma_star=1, k=1)
        decoded = decoder.decode_bitstream(compressed, pattern.shape)

        match = torch.equal(pattern, decoded)
        print(f"Pattern {i+1} decode success: {match}")

        if not match:
            print(f"Original: {pattern.flatten()}")
            print(f"Decoded:  {decoded.flatten()}")

    # Test 2: Accumulator synchronization
    print(f"\n=== Testing Accumulator Synchronization ===")

    # Create data that will trigger accumulator rescaling
    rescale_data = torch.randint(5, 15, (2, 16, 16), dtype=torch.long)  # Moderate values

    encoder = CCSDS123HybridEntropyCoder(2, rescale_interval=32)  # Small rescale interval
    compressed = encoder.encode_image(rescale_data)

    decoder = CCSDS123HybridDecoder(2, rescale_interval=32)
    decoded = decoder.decode_bitstream(compressed, rescale_data.shape)

    sync_match = torch.equal(rescale_data, decoded)
    print(f"Accumulator sync test: {sync_match}")

    # Get final accumulator states
    encoder_stats = encoder.get_compression_statistics()
    print(f"Final encoder accumulator states: {encoder_stats['accumulator_states']}")

    return sync_match


if __name__ == "__main__":
    # Run basic reverse decoding test
    basic_success = test_reverse_decoding()

    # Run advanced property tests
    properties_success = test_reverse_decoding_properties()

    print(f"\n=== Final Results ===")
    print(f"Basic reverse decoding: {'PASS' if basic_success else 'FAIL'}")
    print(f"Advanced properties: {'PASS' if properties_success else 'FAIL'}")
    print(f"Overall reverse decoding: {'PASS' if basic_success and properties_success else 'FAIL'}")
