"""
Test Suite for CCSDS-123.0-B-2 Implementation

This package contains comprehensive tests for the CCSDS-123.0-B-2 compressor
implementation, including unit tests, performance tests, and integration tests.
"""

# Test discovery and utilities
import os
import sys
from .utils import generate_simple_test_image

__all__ = [
        'generate_simple_test_image'
        ]

# Add src directory to path for testing
test_dir = os.path.dirname(__file__)
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
