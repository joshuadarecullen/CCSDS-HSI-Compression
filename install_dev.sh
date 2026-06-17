#!/bin/bash
# Editable install + smoke test for the CCSDS-123.0-B-2 codec.
# Override the interpreter with:  PYTHON=python3.10 ./install_dev.sh
set -e

PY=${PYTHON:-python3}

echo "Installing (editable, with dev extras) using $PY ..."
"$PY" -m pip install -e ".[dev]"

echo "Smoke test ..."
"$PY" - <<'EOF'
import numpy as np
from ccsds import CCSDS123, quality_report
img = np.random.default_rng(0).integers(0, 1 << 16, (8, 16, 16)).astype(np.int64)
codec = CCSDS123.from_image(img, num_prediction_bands=3)
assert np.array_equal(codec.decompress(codec.compress(img)), img), "round-trip failed"
print("lossless round-trip OK")
EOF

echo "Done. Optional extras: .[numba] (fast path), .[torch] (tensor I/O), .[data] (scipy/.mat)."
echo "  e.g.  $PY -m pip install -e \".[numba]\""
