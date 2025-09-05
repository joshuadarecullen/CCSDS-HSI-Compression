#!/usr/bin/env python3
"""
Setup script for CCSDS-123.0-B-2 Compressor Implementation

This setup script allows the package to be installed in development mode,
which fixes import issues and makes the code easily accessible.

Usage:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ccsds-123-compressor",
    version="1.0.0",
    author="CCSDS Implementation Team",
    description="PyTorch implementation of CCSDS-123.0-B-2 standard for multispectral image compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/ccsds-123-compressor",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencies
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
    ],
    
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.812",
        ],
        "gpu": [
            "torch[cu118]>=1.9.0",  # CUDA 11.8 support
        ],
        "benchmarks": [
            "memory_profiler>=0.60.0",
            "line_profiler>=3.0.0",
        ]
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "ccsds-compress=ccsds.cli:compress_cli",
            "ccsds-benchmark=ccsds.cli:benchmark_cli",
        ],
    },
    
    # Include additional files
    include_package_data=True,
    
    # Test suite
    test_suite="tests",
)