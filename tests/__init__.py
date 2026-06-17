"""Tests for the CCSDS-123.0-B-2 reference codec."""

import os
import sys

# Make the repo root importable (so `import src.ccsds` works under pytest).
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)
