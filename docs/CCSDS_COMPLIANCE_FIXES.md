# CCSDS-123.0-B-2 compliance notes

This codec (`src/ccsds/core/reference_codec.py`, `Ccsds123` / `CodecParams`)
replaced an earlier torch prototype that reproduced the *vocabulary* of
CCSDS-123.0-B-2 but not its *algorithm*: the predictor never adapted its weights,
"full" mode was disabled by a constructor bug, the mapped quantizer index ignored
the standard's fold-over mapping, the hybrid coder produced an undecodable
bitstream, and `decompress()` returned the encoder-side reconstruction without
ever decoding the body — so "lossless" was never tested through the entropy coder.

The replacement is pure-integer; the encoder and decoder share a single prediction
loop, so encode → decode is bit-exact by construction. Both standard entropy
coders are implemented and forward/reverse-decodable.

## What it implements (exact integer arithmetic throughout)

- **Local sums** (eq 20-23): wide/narrow × neighbor/column, with the correct
  boundary cases (right-column `W+NW+2·N` and the previous-band terms).
- **Local differences** (eq 24-27) and the difference vector in standard order —
  directionals, then `P*_z = min(z,P)` centrals (eq 28-29).
- **Weights**: default init (eq 33-34) with `ω_min=−2^(Ω+2)`, `ω_max=2^(Ω+2)−1`
  (eq 30); update (eq 49-54) with the double-resolution error and time-varying
  scaling exponent ρ(t), applied after every sample in both encode and decode.
- **Prediction** (eq 36-39): `mod*_R`, the high-resolution `s̃` with the
  `2^(Ω+2)·s_mid` constant, double-resolution `s̆`, and `ŝ = ⌊s̆/2⌋`.
- **Quantizer** (eq 40-45): `q = sgn(Δ)·⌊(|Δ|+m)/(2m+1)⌋`; absolute / relative /
  combined limits; the first sample of each band coded losslessly (eq 41, t=0).
- **Sample representatives** (eq 46-48): the full φ/ψ/Θ formula, reducing to `s′`
  when φ=ψ=0.
- **Mapped quantizer index** (eq 55-56): the fold-over mapping via θ_z(t) and the
  parity of ŝ, with an exact decoder inverse.
- **Sample-adaptive coder** (eq 57-62): length-limited GPO2 codewords with
  adaptive accumulator/counter statistics and a matching forward decoder.
- **Hybrid coder** (5.4.3.3): high/low-entropy classification, reversed
  length-limited GPO2 high-entropy codes, the 16 annex-B variable-to-variable
  low-entropy codes (`entropy/annexb_tables.json` — complete, prefix-free,
  suffix-free, matching the spec's worked examples), the escape mechanism, the
  compressed-image tail, and reverse-order decoding (reverse the body and decode
  forward through prefix-free tries). Selected with `entropy_coder="hybrid"`.

## Verification

Through the real bitstream (`python3 tests/test_reference_codec.py`):

| Test | Result |
|---|---|
| 220×64×64 Indian Pines, lossless | bit-exact, 2.49:1 |
| near-lossless (abs limit a=4) | max\|err\|=4 (≤ bound) |
| reduced mode · narrow local sums · φ≠0 sample reps | all bit-exact lossless |
| hybrid coder, lossless + near-lossless | bit-exact |
| mapped-index map/unmap | exhaustively invertible (8-bit, m∈{0,1,3}) |
| numba kernels vs pure-Python | byte-identical |

## Near-lossless fidelity control (4.8.2)

Lossless (`m=0`), **absolute** (`a_z`), **relative** (`⌊r_z|ŝ|/2^D⌋`) and
**combined** (`min`) error limits, each band-independent (a scalar) or
band-dependent (a length-`num_bands` list). The bound is enforced exactly per band.
Not implemented: periodic error-limit updating (4.8.2.4), which needs
band-interleaved (BI) order — this codec processes in BSQ order.

## Scope

- **Entropy coders**: sample-adaptive (default) and hybrid, both bit-exact and
  numba-accelerated. The block-adaptive option (5.4.3.4 / CCSDS-121) is not
  implemented.
- **Header**: a bit-exact CCSDS 5.3 header (`io/ccsds_header.py`) — Image Metadata
  Essential, Predictor Metadata Primary + Quantization + Sample Representative, and
  the Entropy Coder Metadata. Pack/parse round-trips every parameter; the lossless
  header is 19 bytes. Not emitted: supplementary information tables, custom
  weight-init / weight-exponent tables, periodic error-limit updating, BI order.
- **Performance**: a numba kernel (`_codec_numba.py`) mirrors the pure-Python `_run`
  byte-for-byte and runs ~100-200× faster (the full 145×145×220 cube compresses and
  decompresses in ~1 s each way); the pure-Python path is the readable fallback.
