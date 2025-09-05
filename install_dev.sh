#!/bin/bash
# Development Installation Script for CCSDS-123.0-B-2 Compressor

echo "🚀 Installing CCSDS-123.0-B-2 Compressor in development mode..."
echo

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if pip is available  
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed or not in PATH"
    exit 1
fi

echo "📦 Installing package in development mode..."
pip install -e .

if [ $? -eq 0 ]; then
    echo "✅ Installation successful!"
else
    echo "❌ Installation failed!"
    exit 1
fi

echo
echo "🧪 Testing installation..."

# Test core imports
python -c "
try:
    from ccsds import create_lossless_compressor, calculate_psnr
    print('✅ Core imports working')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test CLI tools
if command -v ccsds-compress &> /dev/null && command -v ccsds-benchmark &> /dev/null; then
    echo "✅ CLI tools installed successfully"
else
    echo "⚠️  CLI tools may not be available (check PATH)"
fi

echo
echo "🎉 Setup complete! You can now:"
echo "   • Import: from ccsds import CCSDS123Compressor"
echo "   • Run tests: pytest tests/"
echo "   • Use CLI: ccsds-compress --help"
echo "   • Benchmark: ccsds-benchmark --help"
echo