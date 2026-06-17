# CCSDS-123.0-B-2 image codec

A pure-integer implementation of the CCSDS-123.0-B-2 standard for lossless and
near-lossless multispectral / hyperspectral image compression.

The encoder and decoder share a single prediction loop, so `compress -> decompress`
is bit-exact by construction. Lossless really means lossless, verified through the
actual entropy-coded bitstream rather than by stashing the reconstruction. The core
is numpy-only; with [numba](https://numba.pydata.org/) installed a JIT fast path
runs automatically (byte-identical to the reference, ~100-200x faster).

## Quick start

```python
import numpy as np
from src.ccsds import CCSDS123

img = np.random.randint(0, 1 << 16, size=(100, 64, 64)).astype(np.int64)  # [Z, Y, X]

codec = CCSDS123.from_image(img, num_prediction_bands=3)  # lossless by default
blob  = codec.compress(img)        # -> bytes (CCSDS 5.3 header + body)
recon = codec.decompress(blob)     # -> np.ndarray

assert np.array_equal(img, recon)
```

Near-lossless, with an absolute (and/or relative, per-band) error limit:

```python
codec = CCSDS123.from_image(img, lossless=False, absolute_error_limit=4)
recon = codec.decompress(codec.compress(img))
assert np.abs(img - recon).max() <= 4
```

Quality metrics for near-lossless reconstructions (numpy-only: PSNR, MSSIM, SAM):

```python
from src.ccsds import quality_report
print(quality_report(img, recon, dynamic_range=16))
# {'psnr_db': ..., 'mssim': ..., 'sam_rad': ..., 'max_abs_error': ...}
```

Decode a blob without holding a configured codec; the parameters come from the
header:

```python
recon = CCSDS123.decompress_bytes(blob)
```

## What it implements

- **Adaptive predictor** (4.x): local sums (eq 20-23, wide/narrow × neighbor/column),
  directional and central local differences, the full weight-update rule with the
  time-varying scaling exponent ρ(t), high-resolution prediction, the quantizer, and
  the fold-over mapped quantizer index.
- **Fidelity control** (4.8.2): lossless, absolute, relative, and combined error
  limits, each band-independent or band-dependent.
- **Encoding order** (5.4.2): BSQ (default) or band-interleaved (BI: BIP, BIL, or an
  intermediate sub-frame depth M), with optional **periodic error-limit updating**
  (4.8.2.4) that carries per-period limits in the body.
- **Two entropy coders**: sample-adaptive (5.4.3.2, default) and hybrid (5.4.3.3,
  `entropy_coder="hybrid"`) using the real annex-B low-entropy tables with
  reverse-order suffix-free decoding.
- **Bit-exact CCSDS 5.3 header** (`io/ccsds_header.py`): packs/parses every supported
  parameter; the lossless header is 19 bytes.

Not implemented: the block-adaptive coder (5.4.3.4 / CCSDS-121) and the optional
supplementary / weight tables.

## Layout

```
src/ccsds/
  codec.py                 CCSDS123 high-level wrapper (numpy or torch)
  metrics.py               PSNR / MSSIM / SAM (numpy-only)
  core/reference_codec.py  Ccsds123 / CodecParams (the codec)
  core/_codec_numba.py     numba kernel (byte-identical fast path)
  entropy/hybrid.py        hybrid entropy coder (5.4.3.3)
  entropy/_hybrid_numba.py numba hybrid kernels
  entropy/annexb_tables.json   annex-B low-entropy code tables
  io/ccsds_header.py       CCSDS 5.3 header pack/parse
tests/
  test_reference_codec.py  end-to-end suite (synthetic cube; numba checked if present)
  synthetic_hsi.py         deterministic test cube (pure numpy)
  run_full_cube.py         full-size headline run
tools/extract_annexb_tables.py   regenerate annexb_tables.json from the standard text
```

## Testing

```bash
python3 tests/test_reference_codec.py     # self-contained; no data files needed
python3 tests/run_full_cube.py            # full 145×145×220-size cube
```

The suite runs on a deterministic synthetic cube, so it works from a fresh clone with
only numpy. If a local Indian Pines `.mat` is found it is exercised too (lossless,
~2.49:1). With numba installed, the JIT kernels are checked byte-for-byte against the
pure-Python reference.

## References

1. *Low-Complexity Lossless and Near-Lossless Multispectral and Hyperspectral Image
   Compression*, CCSDS 123.0-B-2, February 2019.
2. Kiely et al., *The New CCSDS Standard for Low-Complexity Lossless and Near-Lossless
   Multispectral and Hyperspectral Image Compression*.
3. Klimesh, *Fast Lossless Compression of Multispectral Images*, 2005.

For research and educational use. The CCSDS specifications are publicly available from
the Consultative Committee for Space Data Systems.
