"""
End-to-end tests for the CCSDS-123.0-B-2 reference codec.

Runs on a deterministic synthetic cube (synthetic_hsi.py) so the suite needs no
data files; every check goes through the real compressed bitstream
(compress -> bytes -> decompress). Uses a local Indian Pines .mat too if present.

    python3 tests/test_reference_codec.py
"""
import importlib.util
import os
import sys
import time

import numpy as np

try:
    import pytest as _pytest
except ImportError:                      # the file also runs as a plain script
    _pytest = None


def _skip(msg):
    """Report a skip visibly under pytest instead of passing vacuously.
    Only raise when pytest is actually driving the run: pytest.skip() raises
    Skipped (a BaseException) even outside a test session, which would abort
    the plain 'python3 tests/test_reference_codec.py' invocation."""
    print(f"  SKIP: {msg}")
    if _pytest is not None and os.environ.get("PYTEST_CURRENT_TEST"):
        _pytest.skip(msg)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from synthetic_hsi import make_synthetic_hsi   # noqa: E402

# Optional real-data path (local only; not required for the suite to pass).
INDIAN_PINES = ("/home/joshua/Documents/phd_university/code/deepdynamichsicompression"
                "/data/indian_pines/mat/indian_pines.mat")


def _load_codec_module():
    """Import the codec by file path (avoids the torch-heavy package __init__)."""
    path = os.path.join(REPO, "src", "ccsds", "core", "reference_codec.py")
    spec = importlib.util.spec_from_file_location("reference_codec", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reference_codec"] = mod      # needed for dataclass annotation resolution
    spec.loader.exec_module(mod)
    return mod


rc = _load_codec_module()
Ccsds123, CodecParams = rc.Ccsds123, rc.CodecParams


def cube(Z, Y, X, seed=0):
    """Deterministic synthetic HSI cube shaped [Z, Y, X]."""
    return make_synthetic_hsi(num_bands=Z, height=Y, width=X, seed=seed)


def _roundtrip(img, **kw):
    Nz, Ny, Nx = img.shape
    codec = Ccsds123(CodecParams(num_bands=Nz, height=Ny, width=Nx, dynamic_range=16, **kw))
    t0 = time.time(); blob = codec.compress(img); t1 = time.time()
    out = Ccsds123.decompress_standalone(blob); t2 = time.time()
    raw_bits = Nz * Ny * Nx * 16
    return out, {"ratio": raw_bits / (len(blob) * 8), "bpppb": len(blob) * 8 / (Nz * Ny * Nx),
                 "enc_s": t1 - t0, "dec_s": t2 - t1, "samples": Nz * Ny * Nx}


def test_map_unmap_roundtrip():
    """Mapped quantizer index (Eq 55-56) must invert exactly for every valid q.
    The sign parity keys on the double-resolution predicted sample s_breve."""
    codec = Ccsds123(CodecParams(num_bands=2, height=2, width=2, dynamic_range=8))
    bad = 0
    for s_breve in range(2 * codec.s_min, 2 * codec.s_max + 2):
        s_hat = s_breve >> 1
        for m in (0, 1, 3):
            lo, hi, theta = codec._theta(s_hat, m, t=1)
            for q in range(-lo, hi + 1):
                delta = codec._map_index(q, s_breve, theta)
                if not (0 <= delta < (1 << 8)):
                    bad += 1
                if codec._unmap_index(delta, s_breve, lo, hi, theta) != q:
                    bad += 1
    assert bad == 0, f"{bad} map/unmap failures"
    print("  map/unmap round-trip: OK (exhaustive over 8-bit s_breve, m in {0,1,3})")


def test_eq55_parity_conformance():
    """Eq (55) selects 2|q| vs 2|q|-1 by the parity of the DOUBLE-resolution
    predicted sample s~ (code var s_breve), not of s_hat = s_breve >> 1."""
    codec = Ccsds123(CodecParams(num_bands=2, height=2, width=2, dynamic_range=8))
    # s_breve 100 (even) and 101 (odd) share s_hat = 50 but must map differently
    assert codec._map_index(1, 100, 5) == 2, "even s~: (+1)*q >= 0 -> 2|q|"
    assert codec._map_index(1, 101, 5) == 1, "odd s~: (-1)*q < 0 -> 2|q|-1"
    assert codec._map_index(-1, 100, 5) == 1
    assert codec._map_index(-1, 101, 5) == 2
    assert codec._map_index(0, 100, 5) == 0 and codec._map_index(0, 101, 5) == 0
    # beyond theta the sign is carried by lo/hi, parity must not matter
    assert codec._map_index(7, 100, 5) == codec._map_index(7, 101, 5) == 12
    # Call-site conformance through the real prediction loop: at t=0 with z>0,
    # s~ = 2*s''_{z-1}(0) is always EVEN (parity +1) even though
    # s_hat = s''_{z-1}(0) is odd here; keying on s_hat gives the wrong delta.
    p = CodecParams(num_bands=2, height=2, width=2, dynamic_range=8)
    img = np.zeros((2, 2, 2), dtype=np.int64)
    img[0] = 101                                   # odd -> s_hat odd at (z=1, t=0)
    img[1] = 102                                   # q = +1, within theta
    delta = Ccsds123(p)._run(encode=True, image=img, collect_delta=True)
    assert delta[1, 0, 0] == 2, f"q=+1 with even s~ must map to 2, got {delta[1, 0, 0]}"
    img[1] = 100                                   # q = -1
    delta = Ccsds123(p)._run(encode=True, image=img, collect_delta=True)
    assert delta[1, 0, 0] == 1, f"q=-1 with even s~ must map to 1, got {delta[1, 0, 0]}"
    print("  Eq (55) parity keys on the double-resolution sample s_breve (unit + call site): OK")


def test_zero_band_limit_stays_lossless():
    """Eq (45): with both limit types in use, a per-band limit of 0 forces that
    band lossless (m = min(0, rel) = 0), it does not mean 'limit unused'."""
    rng = np.random.default_rng(9)
    # large sample values so the relative term (r*|s_hat| >> D) is nonzero: a
    # small-valued cube would make this test pass even without the Eq (45) fix
    img = rng.integers(20000, 40000, size=(4, 16, 16)).astype(np.int64)
    out, _ = _roundtrip(img, absolute_error_limit=[0, 5, 5, 5], relative_error_limit=64)
    assert np.array_equal(img[0], out[0]), "a_z=0 band must be exactly lossless"
    for z in (1, 2, 3):
        e = int(np.abs(img[z] - out[z]).max())
        assert e <= 5, f"band {z}: err {e} > 5"
    assert np.abs(img[1:] - out[1:]).max() > 0, "limits should actually bite on this data"
    out2, _ = _roundtrip(img, absolute_error_limit=7, relative_error_limit=[0, 64, 64, 64])
    assert np.array_equal(img[0], out2[0]), "r_z=0 band must be exactly lossless"
    for z in (1, 2, 3):
        e = int(np.abs(img[z] - out2[z]).max())
        assert e <= 7, f"band {z}: err {e} > 7"
    # explicit fc flags: used-but-all-zero absolute limits force every band
    # lossless even though a relative limit is also in use
    p = CodecParams(num_bands=4, height=16, width=16, dynamic_range=16,
                    absolute_error_limit=[0, 0, 0, 0], relative_error_limit=64,
                    abs_limit_used=True)
    blob = Ccsds123(p).compress(img)
    out3 = Ccsds123.decompress_standalone(blob)
    assert np.array_equal(img, out3), "all-zero used abs limits must decode lossless"
    print("  zero per-band limits under Eq (45): forced-lossless bands exact, others bounded")


def _raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc:
        return True
    return False


def test_param_validation():
    """validate() must reject non-conformant configs instead of emitting broken streams."""
    base = dict(num_bands=4, height=8, width=8)
    # error limits must fit in min(D-1,16) bits (DA/DR header fields)
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(**base, dynamic_range=24, absolute_error_limit=1 << 16))
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(**base, dynamic_range=24, relative_error_limit=1 << 20))
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(**base, dynamic_range=8, absolute_error_limit=[0, 1, 200, 3]))
    # boundary value 2^16-1 at D=24 is fine
    Ccsds123(CodecParams(**base, dynamic_range=24, absolute_error_limit=(1 << 16) - 1))
    # image dimensions bounded to 2^16 (3.2)
    assert _raises(AssertionError, Ccsds123, CodecParams(num_bands=2, height=4, width=(1 << 16) + 1))
    # Nx=1 requires reduced prediction mode (4.3.1)
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(num_bands=4, height=8, width=1, local_sum_type="narrow_column"))
    # k_init defaults must be consistent for every valid D (D=2..4 used to fail)
    for D in (2, 3, 4, 5):
        Ccsds123(CodecParams(num_bands=2, height=4, width=4, dynamic_range=D))
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(num_bands=2, height=4, width=4, dynamic_range=4, k_init=3))
    # periodic updating with no limit lists would emit an unparseable header
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(num_bands=2, height=8, width=8, encoding_order="BI",
                               update_period_exp=0))
    # explicit fc flags must not contradict the limit values
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(**base, absolute_error_limit=4, abs_limit_used=False))
    bi = dict(**base, encoding_order="BI", update_period_exp=1)
    assert _raises(AssertionError, Ccsds123, CodecParams(**bi,          # periodic too
                   absolute_error_limit=[1, 2, 3, 4], abs_limit_used=False))
    # period entries must be uniformly scalar or per-band; nested lists need periodic
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(**bi, absolute_error_limit=[3, [1, 2, 3, 4], 2, 1]))
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(**base, absolute_error_limit=[[1, 2, 3, 4]] * 4))
    print("  validate(): DA/DR bit-depth bound, size bounds, Nx=1 reduced mode, k_init defaults, "
          "limit shape/flag consistency OK")


def test_width_one_reduced_mode():
    """Nx=1 images must round-trip in reduced mode with column-oriented sums."""
    img = cube(4, 16, 1)
    out, _ = _roundtrip(img, full=False, local_sum_type="narrow_column")
    assert np.array_equal(img, out), "Nx=1 lossless round-trip failed"
    print("  Nx=1 (reduced mode, narrow_column): lossless round-trip OK")


def test_low_dynamic_range():
    """D=2..4 must construct with defaults and round-trip through both coders,
    standalone-decoded from the header (hybrid used to fabricate k_init=3)."""
    rng = np.random.default_rng(7)
    for D in (2, 4):
        img = rng.integers(0, 1 << D, size=(3, 8, 8)).astype(np.int64)
        for ec in ("sample_adaptive", "hybrid"):
            p = CodecParams(num_bands=3, height=8, width=8, dynamic_range=D, entropy_coder=ec)
            blob = Ccsds123(p).compress(img)
            out = Ccsds123.decompress_standalone(blob)
            assert np.array_equal(img, out), f"D={D} {ec} standalone round-trip failed"
    print("  D=2/D=4 sample-adaptive + hybrid, standalone decode: OK")


def test_header_edge_cases():
    """Header sizes are stored mod 2^16 (0 means 65536); the reserved coder type
    must be rejected, not misparsed."""
    p = CodecParams(num_bands=2, height=4, width=1 << 16)
    parsed, _ = rc.parse_header(rc.pack_header(p))
    assert parsed["width"] == 1 << 16, f"width 65536 parsed as {parsed['width']}"
    hdr = bytearray(rc.pack_header(CodecParams(num_bands=2, height=4, width=4)))
    hdr[10] |= 0x06                                   # Entropy Coder Type bits 85-86 -> '11'
    assert _raises(ValueError, rc.parse_header, bytes(hdr)), \
        "reserved coder type must raise, not misparse"
    # DA=16 boundary: the 4-bit header field wraps to 0 and must parse back as 16
    p16 = CodecParams(num_bands=2, height=4, width=4, dynamic_range=24,
                      absolute_error_limit=(1 << 16) - 1)
    parsed, _ = rc.parse_header(rc.pack_header(p16))
    assert parsed["absolute_error_limit"] == (1 << 16) - 1, \
        f"DA=16 limit parsed as {parsed['absolute_error_limit']}"
    # Umax field 1..7 is invalid (legal: 0 meaning 32, or 8..31): reject, not default
    hdr = bytearray(rc.pack_header(CodecParams(num_bands=2, height=4, width=4)))
    hdr[-2] = (hdr[-2] & 0x07) | 0x28                     # Umax field bits -> 5
    assert _raises(ValueError, rc.parse_header, bytes(hdr)), \
        "invalid Umax field must raise, not misparse"
    print("  header edge cases: 65536-size inversion + reserved coder/Umax rejection "
          "+ DA=16 wrap OK")


def test_ndarray_error_limits():
    """numpy arrays are natural inputs for band-dependent limits and must work."""
    img = cube(4, 16, 16)
    out, _ = _roundtrip(img, absolute_error_limit=np.array([0, 1, 2, 3]))
    for z in range(4):
        e = int(np.abs(img[z] - out[z]).max())
        assert e <= z, f"band {z}: err {e} > {z}"
    print("  ndarray band-dependent limits: accepted and bounds respected")


def test_out_of_range_samples_rejected():
    """compress() must fail loudly on samples outside [s_min, s_max] instead of
    silently emitting a corrupt bitstream."""
    rng = np.random.default_rng(11)
    p = CodecParams(num_bands=4, height=8, width=8, dynamic_range=8)
    img = rng.integers(0, 1 << 8, size=(4, 8, 8)).astype(np.int64)
    codec = Ccsds123(p)
    codec.decompress(codec.compress(img))             # in-range baseline works
    img[0, 2, 3] = 300
    assert _raises(ValueError, codec.compress, img), "over-range sample must raise"
    img[0, 2, 3] = -7
    assert _raises(ValueError, codec.compress, img), "negative sample must raise (unsigned)"
    print("  out-of-range samples rejected with ValueError: OK")


def test_truncated_stream():
    """A truncated body must raise instead of reading past the buffer, and the numba
    decoders must behave exactly like the pure ones on corrupt input."""
    img = cube(4, 12, 12)
    for coder in ("sample_adaptive", "hybrid", "block_adaptive"):
        for order, extra in (("BSQ", {}), ("BI", {"interleave_depth": 2})):
            p = CodecParams(num_bands=4, height=12, width=12,
                            entropy_coder=coder, encoding_order=order, **extra)
            blob = Ccsds123(p).compress(img)
            hlen = rc.parse_header(blob)[1]
            assert _raises(IndexError, Ccsds123(p).decompress, blob[:hlen]), \
                f"{coder}/{order}: empty body must raise"
            assert _raises(IndexError, Ccsds123(p).decompress,
                           blob[:hlen + (len(blob) - hlen) // 2]), \
                f"{coder}/{order}: half body must raise"
    # coder-level: the numba decoders must agree with the pure ones bit for bit
    rng = np.random.default_rng(4)
    vals = rng.integers(0, 1 << 12, 4000).astype(np.int64)
    vals[rng.random(4000) < 0.6] = 0
    bc = rc.BlockAdaptiveCoder(12)
    body = bc.encode(vals)
    for cut in (0, 1, len(body) // 4, len(body) // 2, len(body) - 1):
        got = []
        for use_numba in (False, True):
            bc.use_numba = use_numba and rc.BlockAdaptiveCoder(12).use_numba
            try:
                got.append(bc.decode(body[:cut], len(vals)).tobytes())
            except IndexError:
                got.append(b"raise")
        bc.use_numba = rc.BlockAdaptiveCoder(12).use_numba
        assert got[0] == got[1], f"block-adaptive numba differs from pure at cut={cut}"
    print("  truncated bodies raise IndexError; numba decoders match pure on corrupt input")


def test_float_input_integrality():
    """The numpy front-end must reject fractional floats, not truncate them."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    from ccsds.codec import CCSDS123
    img = cube(3, 8, 8).astype(np.float64)
    c = CCSDS123.from_image(img)
    assert np.array_equal(c.decompress(c.compress(img)), img)   # integral floats fine
    assert _raises(ValueError, c.compress, img + 0.5), "fractional input must raise"
    print("  float input: integral accepted, fractional rejected")


def test_ccsds_header():
    """Bit-exact CCSDS 5.3 header packs and parses every supported param exactly."""
    cases = [
        dict(num_bands=10, height=8, width=8, dynamic_range=16),                          # lossless
        dict(num_bands=10, height=8, width=8, dynamic_range=16, absolute_error_limit=4),  # +quant (abs)
        dict(num_bands=10, height=8, width=8, dynamic_range=16, theta=2, phi=1),          # +sample rep
        dict(num_bands=6, height=4, width=4, dynamic_range=12, full=False,
             local_sum_type="narrow_column", relative_error_limit=64, num_prediction_bands=2),
        dict(num_bands=5, height=4, width=4, dynamic_range=16,
             absolute_error_limit=[i % 4 for i in range(5)]),                             # band-dependent
        dict(num_bands=8, height=8, width=8, dynamic_range=16, entropy_coder="hybrid"),   # hybrid coder
        dict(num_bands=8, height=8, width=8, dynamic_range=16, encoding_order="BI",
             interleave_depth=8, absolute_error_limit=4),                                  # BI order
    ]
    fields = ("num_bands", "height", "width", "dynamic_range", "signed", "num_prediction_bands",
              "full", "local_sum_type", "omega", "register_size", "theta", "phi", "psi",
              "absolute_error_limit", "relative_error_limit", "v_min", "v_max", "t_inc",
              "gamma0", "gamma_star", "u_max", "k_init", "entropy_coder",
              "encoding_order", "interleave_depth", "update_period_exp")
    for kw in cases:
        p = CodecParams(**kw)
        hdr = rc.pack_header(p)
        parsed, hlen = rc.parse_header(hdr)
        assert hlen == len(hdr), (hlen, len(hdr))
        for k in fields:
            if k == "k_init" and kw.get("entropy_coder") == "hybrid":
                continue                       # not carried in the hybrid header (table 5-14)
            assert parsed[k] == getattr(p, k), f"{k}: {parsed[k]} != {getattr(p, k)} for {kw}"
    n = len(rc.pack_header(CodecParams(num_bands=10, height=8, width=8, dynamic_range=16)))
    assert n == 19, f"lossless header should be 12+5+2=19 bytes, got {n}"
    print("  CCSDS 5.3 header: pack/parse round-trips all params; lossless header = 19 bytes")


def test_numba_byte_identical():
    """The numba kernel must produce a byte-identical bitstream to the reference."""
    if not getattr(rc, "NUMBA_OK", False):
        _skip("numba not available in this interpreter (pure-Python path in use)")
        return
    img = cube(20, 24, 24)
    configs = [
        dict(num_prediction_bands=3, full=True),
        dict(num_prediction_bands=3, full=True, absolute_error_limit=4),
        dict(num_prediction_bands=2, full=False, local_sum_type="narrow_neighbor"),
        dict(num_prediction_bands=3, full=True, theta=2, phi=1),
        dict(num_prediction_bands=3, full=True, theta=2, phi=1, psi=2, absolute_error_limit=4),
        dict(num_prediction_bands=2, full=True, weight_init_resolution=5,
             weight_init=[[3, -7, 12] + [-9, 4][:min(z, 2)] for z in range(20)]),
        dict(num_prediction_bands=4, full=True,
             weight_exp_offset=[[(z + i) % 12 - 6 for i in range(1 + min(z, 4))]
                                for z in range(20)]),
        dict(num_prediction_bands=3, full=True, k_init=[(3 * z) % 13 for z in range(20)]),
        dict(num_prediction_bands=3, full=True, theta=2, phi=[z % 4 for z in range(20)],
             psi=[(z + 1) % 4 for z in range(20)], absolute_error_limit=3),
    ]
    for kw in configs:
        p = CodecParams(num_bands=20, height=24, width=24, dynamic_range=16, **kw)
        cn = Ccsds123(p)
        assert cn.use_numba, f"numba path should self-select for {kw}"
        cp = Ccsds123(p); cp.use_numba = False
        assert cn.compress(img) == cp.compress(img), f"numba bitstream differs for {kw}"
        dn = Ccsds123(p)
        dp = Ccsds123(p); dp.use_numba = False
        assert np.array_equal(dn.decompress(cp.compress(img)), dp.decompress(cn.compress(img)))
    print(f"  numba kernel BYTE-IDENTICAL to pure-Python reference ({len(configs)} configs) "
          "+ cross-decode OK")


def test_hybrid_codec():
    """Full codec through the hybrid entropy coder (real Annex-B tables, reverse-order
    decode): lossless + near-lossless, with the CCSDS header carrying the coder type."""
    img = cube(20, 24, 24)
    out, st = _roundtrip(img, num_prediction_bands=3, full=True, entropy_coder="hybrid")
    assert np.array_equal(img, out), "hybrid lossless round-trip failed"
    out2, st2 = _roundtrip(img, num_prediction_bands=3, full=True,
                           entropy_coder="hybrid", absolute_error_limit=4)
    err = int(np.abs(img - out2).max())
    assert err <= 4, f"hybrid near-lossless error {err} exceeds 4"
    # sample-adaptive for comparison
    _, sa = _roundtrip(img, num_prediction_bands=3, full=True, absolute_error_limit=4)
    print(f"  hybrid codec: lossless max|err|={np.abs(img - out).max()} ratio={st['ratio']:.3f}:1  "
          f"near-lossless(a=4) max|err|={err} ratio={st2['ratio']:.3f}:1 "
          f"(sample-adaptive {sa['ratio']:.3f}:1)")


def test_hybrid_numba_identical():
    """The numba hybrid kernels must be byte-identical to the pure-Python coder."""
    import importlib.util as ilu
    hp = os.path.join(REPO, "src/ccsds/entropy/hybrid.py")
    hspec = ilu.spec_from_file_location("hybrid", hp)
    hm = ilu.module_from_spec(hspec)
    sys.modules["hybrid"] = hm
    hspec.loader.exec_module(hm)
    hc = hm.HybridCoder(dynamic_range=16)
    if not hc.use_numba:
        _skip("numba not available (hybrid coder runs pure-Python)")
        return
    rng = np.random.default_rng(1)
    for shp in [(8, 16, 16), (20, 24, 24)]:
        for hi in [3, 100, 1 << 16]:
            d = rng.integers(0, hi, shp).astype(np.int64)
            for M in (0, 1, 5):                 # BSQ, BIL, intermediate BI depth
                hc.use_numba = True;  bn = hc.encode(d, M); on = hc.decode(bn, shp, M)
                hc.use_numba = False; bp = hc.encode(d, M); op = hc.decode(bp, shp, M)
                hc.use_numba = True
                assert bn == bp, f"hybrid numba encode differs from pure-Python for {shp}, {hi}, M={M}"
                assert np.array_equal(on, d) and np.array_equal(op, d) and np.array_equal(on, op)
    print("  hybrid numba kernels BYTE-IDENTICAL to pure-Python + round-trip OK (BSQ + BI)")


def test_package_imports_without_torch():
    """The correct codec must import and round-trip (torch optional)."""
    import importlib
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    pkg = importlib.import_module("src.ccsds")
    assert hasattr(pkg, "CCSDS123") and hasattr(pkg, "Ccsds123") and hasattr(pkg, "CodecParams")
    img = cube(8, 16, 16)
    codec = pkg.CCSDS123.from_image(img, num_prediction_bands=3)
    assert np.array_equal(codec.decompress(codec.compress(img)), img)
    # near-lossless is inferred from the limits; contradictions raise
    nl = pkg.CCSDS123.from_image(img, absolute_error_limit=4)
    assert not nl.params.lossless and nl.params.absolute_error_limit == 4
    err = int(np.abs(nl.decompress(nl.compress(img)).astype(np.int64) - img).max())
    assert err <= 4, f"front-end near-lossless error {err} > 4"
    assert _raises(ValueError, pkg.CCSDS123, num_bands=8, height=16, width=16,
                   lossless=True, absolute_error_limit=4)
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    print(f"  package import via src.ccsds.CCSDS123 + round-trip: OK (torch present={has_torch})")


def test_torch_wrapper():
    """CCSDS123Module: batched round-trip through the real bitstream, rate
    reporting, near-lossless bound, and the straight-through gradient path."""
    try:
        import torch
    except ImportError:
        _skip("torch not available in this interpreter")
        return
    sys.path.insert(0, os.path.join(REPO, "src"))
    from ccsds.torch_wrapper import CCSDS123Module

    img = np.stack([cube(6, 16, 16, seed=s) for s in (0, 1)])   # [B, Z, Y, X]
    x = torch.from_numpy(img)
    m = CCSDS123Module(dynamic_range=16)
    out = m(x)
    rec = out["reconstruction"]
    assert rec.shape == x.shape and rec.dtype == x.dtype
    assert torch.equal(rec, x), "lossless torch round-trip failed"
    assert out["bits"].tolist() == [len(b) * 8 for b in out["bitstreams"]]
    assert torch.allclose(out["bpppb"], out["bits"].double() / img[0].size)
    # blobs are self-contained: static decode matches, on the requested dtype
    dec = CCSDS123Module.decompress(out["bitstreams"], dtype=x.dtype)
    assert torch.equal(dec, x)
    # unbatched input keeps its rank
    assert m(x[0])["reconstruction"].shape == x[0].shape
    # near-lossless bound holds through the wrapper
    m2 = CCSDS123Module(dynamic_range=16, absolute_error_limit=3)
    err = (m2(x)["reconstruction"] - x).abs().max().item()
    assert 0 < err <= 3, f"near-lossless error {err} outside (0, 3]"
    # integral float input works; fractional input is rejected
    xf = x.float()
    assert torch.equal(m(xf)["reconstruction"], xf)
    if not _raises(ValueError, m, xf + 0.5):
        raise AssertionError("fractional input must be rejected")
    # straight-through: gradients pass as identity, values are the codec's
    xg = x.float().requires_grad_(True)
    out_st = CCSDS123Module(dynamic_range=16, absolute_error_limit=3,
                            straight_through=True)(xg)
    assert not torch.equal(out_st["reconstruction"].detach(), xg.detach())
    out_st["reconstruction"].sum().backward()
    assert torch.equal(xg.grad, torch.ones_like(xg)), "STE gradient must be identity"
    # per-call overrides beat the constructor settings and cache separately
    o = m(x, absolute_error_limit=3)
    err = (o["reconstruction"] - x).abs().max().item()
    assert 0 < err <= 3 and o["bitstreams"] != out["bitstreams"]
    assert torch.equal(m(x)["reconstruction"], x), "override must not stick"
    assert m.codec_for((6, 16, 16), entropy_coder="hybrid").params.entropy_coder == "hybrid"
    o2 = m(x, entropy_coder="block_adaptive", block_size=16)
    assert torch.equal(o2["reconstruction"], x)
    # worker processes give the same bitstreams as the serial path
    mp = CCSDS123Module(dynamic_range=16, num_workers=2)
    op = mp(x)
    assert op["bitstreams"] == out["bitstreams"] and torch.equal(op["reconstruction"], x)
    assert mp.compress(x, absolute_error_limit=3) == o["bitstreams"]
    print("  torch wrapper: batched lossless + near-lossless, static decode, STE grad, "
          "per-call overrides, worker pool OK")


def test_metrics():
    """numpy quality metrics (PSNR/MSSIM/SAM): exact match is the ceiling, error degrades them."""
    import importlib.util as ilu
    mspec = ilu.spec_from_file_location("metrics", os.path.join(REPO, "src/ccsds/metrics.py"))
    m = ilu.module_from_spec(mspec)
    mspec.loader.exec_module(m)
    img = cube(12, 32, 32)
    assert m.calculate_psnr(img, img, 16) == float("inf")
    assert m.calculate_mssim(img, img) > 0.999999
    assert m.calculate_spectral_angle(img, img) < 1e-4
    rng = np.random.default_rng(3)
    noisy = np.clip(img + rng.integers(-4, 5, img.shape), 0, (1 << 16) - 1)
    rep = m.quality_report(img, noisy, 16)
    assert np.isfinite(rep["psnr_db"]) and rep["mssim"] < 1.0
    assert rep["sam_rad"] > 0 and rep["max_abs_error"] <= 4
    print(f"  metrics: identical=(inf dB, MSSIM~1, SAM~0); noisy(a<=4) "
          f"PSNR={rep['psnr_db']:.1f}dB MSSIM={rep['mssim']:.4f} SAM={rep['sam_rad']:.2e}rad")


def test_bi_order():
    """Band-interleaved (BI) encoding order (5.4.2.2): BIP, BIL and intermediate M,
    lossless and near-lossless, decoded through the header."""
    img = cube(16, 24, 24)
    Nz = img.shape[0]
    for M, tag in [(Nz, "BIP"), (1, "BIL"), (4, "M=4")]:
        out, st = _roundtrip(img, encoding_order="BI", interleave_depth=M)
        assert np.array_equal(img, out), f"BI {tag} lossless round-trip failed"
    out, st = _roundtrip(img, encoding_order="BI", interleave_depth=Nz, absolute_error_limit=4)
    err = int(np.abs(img - out).max())
    assert err <= 4, f"BI near-lossless error {err} exceeds 4"
    print(f"  BI order (BIP/BIL/M=4) lossless + near-lossless(a=4, max|err|={err}): OK  "
          f"ratio={st['ratio']:.3f}:1")


def test_periodic_error_limits():
    """Periodic error-limit updating (4.8.2.4): per-period limits carried in the body,
    decoded standalone; each period must respect its own bound."""
    img = cube(12, 16, 16)
    Nz, Ny, Nx = img.shape
    u = 1
    nper = (Ny + (1 << u) - 1) >> u
    abs_bi = [(p % 5) for p in range(nper)]                      # band-independent, varies per period
    out, st = _roundtrip(img, encoding_order="BI", interleave_depth=Nz,
                         update_period_exp=u, absolute_error_limit=abs_bi)
    for y in range(Ny):
        e = int(np.abs(img[:, y, :] - out[:, y, :]).max())
        assert e <= abs_bi[y >> u], f"row {y} (period {y >> u}): err {e} > limit {abs_bi[y >> u]}"
    abs_bd = [[(z + p) % 4 for z in range(Nz)] for p in range((Ny + 3) >> 2)]   # band-dependent
    out2, _ = _roundtrip(img, encoding_order="BI", interleave_depth=1,
                         update_period_exp=2, absolute_error_limit=abs_bd)
    for y in range(Ny):
        for z in range(Nz):
            e = int(np.abs(img[z, y, :] - out2[z, y, :]).max())
            assert e <= abs_bd[y >> 2][z], f"band {z} row {y}: err {e} > {abs_bd[y >> 2][z]}"
    print(f"  periodic error limits (band-indep {abs_bi} + band-dep, decoded standalone): "
          f"every period bound respected  ratio={st['ratio']:.3f}:1")


def test_hybrid_bi():
    """Hybrid coder under BI encoding order (5.4.2.2 + 5.4.3.3), including periodic
    error-limit updating with the limits carried in the body (5.4.3.3.5.1.1)."""
    img = cube(12, 16, 16)
    Nz, Ny, Nx = img.shape
    bodies = set()
    for M in (0, Nz, 1, 4):                      # BSQ then BIP/BIL/intermediate
        kw = dict(encoding_order="BI", interleave_depth=M) if M else {}
        p = CodecParams(num_bands=Nz, height=Ny, width=Nx, dynamic_range=16,
                        entropy_coder="hybrid", **kw)
        blob = Ccsds123(p).compress(img)
        assert np.array_equal(img, Ccsds123.decompress_standalone(blob)), \
            f"hybrid BI M={M} lossless round-trip failed"
        bodies.add(blob[rc.parse_header(blob)[1]:])
    assert len(bodies) == 4, "BSQ/BIP/BIL/M=4 hybrid bodies must all differ"
    out, _ = _roundtrip(img, entropy_coder="hybrid", encoding_order="BI",
                        interleave_depth=Nz, absolute_error_limit=4)
    err = int(np.abs(img - out).max())
    assert err <= 4, f"hybrid BI near-lossless error {err} exceeds 4"
    u = 1
    abs_bi = [p % 5 for p in range((Ny + 1) >> 1)]
    rel_bd = [[8 * ((z + p) % 3 + 1) for z in range(Nz)] for p in range((Ny + 1) >> 1)]
    out2, st = _roundtrip(img, entropy_coder="hybrid", encoding_order="BI", interleave_depth=1,
                          update_period_exp=u, absolute_error_limit=abs_bi,
                          relative_error_limit=rel_bd)
    for y in range(Ny):
        e = int(np.abs(img[:, y, :] - out2[:, y, :]).max())
        assert e <= abs_bi[y >> u], f"row {y} (period {y >> u}): err {e} > {abs_bi[y >> u]}"
    print(f"  hybrid BI (BIP/BIL/M=4) + periodic abs+rel limits: every bound respected  "
          f"ratio={st['ratio']:.3f}:1")


def test_bi_numba_identical():
    """The numba BI sample-adaptive coder must be byte-identical to the pure-Python one."""
    if not getattr(rc, "NUMBA_OK", False):
        _skip("numba not available (BI coder runs pure-Python)")
        return
    img = cube(10, 16, 16, seed=5)
    Nz, Ny, Nx = img.shape
    nper, nper2 = (Ny + 1) >> 1, (Ny + 3) >> 2
    configs = [
        dict(interleave_depth=Nz),                                              # BIP lossless
        dict(interleave_depth=1),                                               # BIL lossless
        dict(interleave_depth=4),                                               # intermediate M
        dict(interleave_depth=Nz, absolute_error_limit=4),                      # near-lossless
        dict(interleave_depth=Nz, update_period_exp=1,
             absolute_error_limit=[(p % 5) for p in range(nper)]),              # periodic band-indep
        dict(interleave_depth=1, update_period_exp=2,
             absolute_error_limit=[[(z + p) % 4 for z in range(Nz)] for p in range(nper2)]),  # band-dep
        dict(interleave_depth=Nz, update_period_exp=2,
             relative_error_limit=[32 * (p + 1) for p in range(nper2)]),        # periodic relative
        dict(interleave_depth=2, k_init=[(3 * z) % 11 for z in range(Nz)]),     # per-band acc init
    ]
    for kw in configs:
        p = CodecParams(num_bands=Nz, height=Ny, width=Nx, dynamic_range=16,
                        encoding_order="BI", **kw)
        cn = Ccsds123(p)
        assert cn.use_numba, f"numba path should self-select for {kw}"
        cp = Ccsds123(p); cp.use_numba = False
        bn, bp = cn.compress(img), cp.compress(img)
        assert bn == bp, f"BI numba bytes differ from pure-Python for {kw}"
        assert np.array_equal(cn.decompress(bn), cp.decompress(bp)), f"BI numba decode differs for {kw}"
    print(f"  BI numba coder BYTE-IDENTICAL to pure-Python + round-trip OK ({len(configs)} configs)")


def test_lossless_crop():
    img = cube(24, 32, 32)
    out, st = _roundtrip(img, num_prediction_bands=3, full=True)
    assert np.array_equal(img, out), "LOSSLESS ROUND-TRIP FAILED"
    print(f"  lossless 24x32x32: max|err|={np.abs(img - out).max()}  ratio={st['ratio']:.3f}:1  "
          f"bpppb={st['bpppb']:.3f}  enc={st['enc_s']:.2f}s dec={st['dec_s']:.2f}s")


def test_reduced_mode():
    img = cube(24, 32, 32)
    out, st = _roundtrip(img, num_prediction_bands=3, full=False)
    assert np.array_equal(img, out), "reduced-mode lossless round-trip failed"
    print(f"  reduced-mode lossless: max|err|={np.abs(img - out).max()}  ratio={st['ratio']:.3f}:1")


def test_custom_weight_init():
    """Custom weight initialization (Eq 35) + Weight Tables header subpart
    (5.3.3.3.2): vectors must survive the header round-trip and decode standalone."""
    rng = np.random.default_rng(7)
    img = cube(8, 20, 20)
    Q = 6
    half = 1 << (Q - 1)
    for full in (True, False):
        p0 = CodecParams(num_bands=8, height=20, width=20, dynamic_range=16,
                         num_prediction_bands=3, full=full)
        lam = [[int(rng.integers(-half, half))
                for _ in range(p0.band_components(z))] for z in range(8)]
        p = CodecParams(num_bands=8, height=20, width=20, dynamic_range=16,
                        num_prediction_bands=3, full=full,
                        weight_init=lam, weight_init_resolution=Q)
        blob = Ccsds123(p).compress(img)
        out = Ccsds123.decompress_standalone(blob)
        assert np.array_equal(img, out), f"custom weight init round-trip failed (full={full})"
        parsed, _ = rc.parse_header(blob)
        assert parsed["weight_init"] == lam and parsed["weight_init_resolution"] == Q, \
            "weight init vectors did not survive the header round-trip"
        # custom stream must differ from default-init stream (weights actually used)
        blob_def = Ccsds123(p0).compress(img)
        assert blob != blob_def, "custom weight init produced the default bitstream"
    # Eq (35) NOTE: in the (Omega+3)-bit two's complement of each w, the Q MSBs
    # equal Lambda and the rest are '0' followed by '1's
    p = CodecParams(num_bands=2, height=4, width=4, dynamic_range=16, omega=13,
                    num_prediction_bands=1, full=False,
                    weight_init=[[], [-17]], weight_init_resolution=6)
    w = Ccsds123(p)._init_weights(1)[0]
    nbits, Q = 13 + 3, 6
    tc = w & ((1 << nbits) - 1)
    assert tc >> (nbits - Q) == (-17) & ((1 << Q) - 1), "Q MSBs must equal Lambda"
    assert tc & ((1 << (nbits - Q)) - 1) == (1 << (nbits - Q - 1)) - 1, \
        "low bits must be '0' then all '1's"
    # Q = Omega+3 means w = Lambda exactly
    p = CodecParams(num_bands=2, height=4, width=4, dynamic_range=16, omega=13,
                    num_prediction_bands=1, full=False,
                    weight_init=[[], [-17]], weight_init_resolution=16)
    assert Ccsds123(p)._init_weights(1) == [-17]
    print("  custom weight init: full+reduced round-trips, header table, Eq 35 bit layout OK")


def test_weight_exp_offset():
    """Per-band weight exponent offsets (4.10.4) + Weight Exponent Offset Table
    (5.3.3.3.3): offsets alter the stream, survive the header, decode standalone."""
    img = cube(8, 20, 20)
    for full in (True, False):
        zoff = [[(z + i) % 12 - 6 for i in range((1 if full else 0) + min(z, 3))]
                for z in range(8)]
        p = CodecParams(num_bands=8, height=20, width=20, dynamic_range=16,
                        num_prediction_bands=3, full=full, weight_exp_offset=zoff)
        blob = Ccsds123(p).compress(img)
        assert np.array_equal(img, Ccsds123.decompress_standalone(blob)), \
            f"weight exponent offset round-trip failed (full={full})"
        parsed, _ = rc.parse_header(blob)
        assert parsed["weight_exp_offset"] == zoff, \
            "offset table did not survive the header round-trip"
        p0 = CodecParams(num_bands=8, height=20, width=20, dynamic_range=16,
                         num_prediction_bands=3, full=full)
        assert blob != Ccsds123(p0).compress(img), "offsets did not change the stream"
    # scalar zetas are a band-independent convenience for the same table
    ps = CodecParams(num_bands=8, height=20, width=20, dynamic_range=16,
                     num_prediction_bands=3, zeta_inter=3, zeta_intra=-2)
    pt = CodecParams(num_bands=8, height=20, width=20, dynamic_range=16,
                     num_prediction_bands=3,
                     weight_exp_offset=[[-2] + [3] * min(z, 3) for z in range(8)])
    assert Ccsds123(ps).compress(img) == Ccsds123(pt).compress(img)
    # offset flag without a table is mission-defined and must be rejected loudly
    hdr = bytearray(rc.pack_header(pt))
    hdr[16] &= 0x7F                             # clear the Offset Table Flag (byte 16 MSB)
    assert _raises(NotImplementedError, rc.parse_header, bytes(hdr)), \
        "offset flag without table must not parse"
    assert _raises(AssertionError, Ccsds123, CodecParams(
        num_bands=2, height=4, width=4, weight_exp_offset=[[7], [7, 0, 0, 0]]))
    print("  weight exponent offsets: full+reduced round-trips, header table, scalar equiv OK")


def test_supplementary_tables():
    """Supplementary information tables (3.5, tables 5-4): all types and
    structures must survive the header and leave the body untouched."""
    Z, Y, X = 4, 6, 5
    img = cube(Z, Y, X)
    tables = [
        dict(type="unsigned", purpose=2, structure="1d", bit_depth=14,
             data=[400 + 3 * z for z in range(Z)]),
        dict(type="signed", purpose=1, structure="0d", bit_depth=9, data=[-100]),
        dict(type="unsigned", purpose=4, structure="yx", bit_depth=1,
             data=[(y ^ x) & 1 for y in range(Y) for x in range(X)]),
        dict(type="signed", purpose=15, structure="zx", bit_depth=32,
             data=[(z * X + x) * 12345 - (1 << 30) for z in range(Z) for x in range(X)]),
        dict(type="float", purpose=10, structure="1d", user=5, df=10, de=5, bias=15,
             data=[(z & 1, 12 + z, 700 + z) for z in range(Z)]),
        dict(type="float", purpose=0, structure="0d", df=23, de=8, bias=127,
             data=[(1, 255, 1)]),               # DE=8 packs mod 8; element is a raw NaN
    ]
    p = CodecParams(num_bands=Z, height=Y, width=X, dynamic_range=16,
                    supplementary_tables=tables)
    blob = Ccsds123(p).compress(img)
    assert np.array_equal(img, Ccsds123.decompress_standalone(blob))
    parsed, _ = rc.parse_header(blob)
    got = parsed["supplementary_tables"]
    assert len(got) == len(tables)
    for a, b in zip(got, tables):
        for k in b:
            assert a[k] == b[k], f"table field {k}: {a[k]} != {b[k]}"
    # header-only: the body must be identical with and without tables
    p0 = CodecParams(num_bands=Z, height=Y, width=X, dynamic_range=16)
    blob0 = Ccsds123(p0).compress(img)
    n = rc.parse_header(blob)[1]
    n0 = rc.parse_header(blob0)[1]
    assert blob[n:] == blob0[n0:], "supplementary tables must not touch the body"
    # limits: reserved purpose values and tau > 15 must be rejected
    bad = dict(type="unsigned", purpose=7, structure="0d", bit_depth=4, data=[1])
    assert _raises(AssertionError, Ccsds123, CodecParams(
        num_bands=Z, height=Y, width=X, supplementary_tables=[bad]))
    ok = dict(type="unsigned", purpose=0, structure="0d", bit_depth=4, data=[1])
    assert _raises(AssertionError, Ccsds123, CodecParams(
        num_bands=Z, height=Y, width=X, supplementary_tables=[ok] * 16))
    print("  supplementary tables: 6 tables (u/s/float, 0d/1d/zx/yx) header round-trip OK")


def test_accumulator_init_table():
    """Per-band accumulator init (5.4.3.2.3.3 + table 5.3.4.2.2): k''_z alter the
    stream, survive the header, decode standalone; K=15 without a table rejects."""
    img = cube(8, 20, 20)
    ktab = [(3 * z) % 13 for z in range(8)]
    base = dict(num_bands=8, height=20, width=20, dynamic_range=16)
    for kw in (dict(), dict(encoding_order="BI", interleave_depth=2)):
        p = CodecParams(**base, k_init=ktab, **kw)
        blob = Ccsds123(p).compress(img)
        assert np.array_equal(img, Ccsds123.decompress_standalone(blob)), \
            f"accumulator init table round-trip failed ({kw or 'BSQ'})"
        assert rc.parse_header(blob)[0]["k_init"] == ktab
        assert blob != Ccsds123(CodecParams(**base, **kw)).compress(img), \
            "per-band k_init did not change the stream"
    # an equal-valued table changes only the header, never the body
    bt = Ccsds123(CodecParams(**base, k_init=[5] * 8)).compress(img)
    bc = Ccsds123(CodecParams(**base, k_init=5)).compress(img)
    assert bt[rc.parse_header(bt)[1]:] == bc[rc.parse_header(bc)[1]:]
    # last header byte holds gamma0(3) K(4) table-flag(1) (table 5-13)
    hdr = bytearray(rc.pack_header(CodecParams(**base, k_init=5)))
    hdr[-1] |= 0x01                             # table flag alongside a constant K
    assert _raises(ValueError, rc.parse_header, bytes(hdr))
    hdr[-1] = (hdr[-1] & 0xE0) | 0x1E           # K=15 with no table
    assert _raises(NotImplementedError, rc.parse_header, bytes(hdr))
    assert _raises(AssertionError, Ccsds123, CodecParams(**base, k_init=[15] * 8))
    assert _raises(AssertionError, Ccsds123, CodecParams(**base, k_init=[3, 3]))
    print("  accumulator init table: BSQ+BI round-trips, header table, loud rejects OK")


def test_band_varying_sample_rep():
    """Band-varying damping/offset (4.9, table 5-12): per-band phi_z/psi_z carried as
    Damping/Offset Table subblocks, decoded standalone; flag abuse rejected loudly."""
    img = cube(6, 16, 16)
    base = dict(num_bands=6, height=16, width=16, dynamic_range=16)
    phis = [z % 4 for z in range(6)]
    psis = [(z + 1) % 4 for z in range(6)]
    for kw in (dict(phi=phis, psi=1), dict(phi=1, psi=psis), dict(phi=phis, psi=psis)):
        p = CodecParams(**base, theta=2, absolute_error_limit=3, **kw)
        blob = Ccsds123(p).compress(img)
        out = Ccsds123.decompress_standalone(blob)
        assert int(np.abs(img - out).max()) <= 3
        parsed, _ = rc.parse_header(blob)
        assert parsed["phi"] == kw["phi"] and parsed["psi"] == kw["psi"]
    fixed = Ccsds123(CodecParams(**base, theta=2, phi=1, psi=2,
                                 absolute_error_limit=3)).compress(img)
    var = Ccsds123(CodecParams(**base, theta=2, phi=phis, psi=psis,
                               absolute_error_limit=3)).compress(img)
    assert var != fixed, "band-varying phi/psi did not change the stream"
    # lossless subpart is header bytes 17-19: theta, damping flags, offset flags
    hdr = bytearray(rc.pack_header(CodecParams(**base, theta=2, phi=phis, psi=0)))
    hdr[18] &= 0xDF                             # band-varying without a table
    assert _raises(NotImplementedError, rc.parse_header, bytes(hdr))
    hdr2 = bytearray(rc.pack_header(CodecParams(**base, theta=2, phi=1, psi=0)))
    hdr2[18] |= 0x20                            # table flag without band-varying
    assert _raises(ValueError, rc.parse_header, bytes(hdr2))
    assert _raises(AssertionError, Ccsds123, CodecParams(**base, theta=2, phi=[1, 2], psi=0))
    print("  band-varying sample rep: phi/psi tables round-trip, loud rejects OK")


def test_hybrid_sigma_init():
    """User-specified hybrid initial accumulators (5.4.3.3.4.3): out-of-band, alter
    the stream, and decoding never needs them (finals travel in the stream tail)."""
    img = cube(8, 16, 16)
    sig = [(37 * z + 5) % (1 << 17) for z in range(8)]
    base = dict(num_bands=8, height=16, width=16, dynamic_range=16, entropy_coder="hybrid")
    for kw in (dict(), dict(encoding_order="BI", interleave_depth=2)):
        blob = Ccsds123(CodecParams(**base, hybrid_sigma_init=sig, **kw)).compress(img)
        assert blob != Ccsds123(CodecParams(**base, **kw)).compress(img), \
            "custom accumulators did not change the stream"
        assert np.array_equal(img, Ccsds123.decompress_standalone(blob)), \
            "standalone decode must work without knowing Sigma_z(0)"
    assert _raises(AssertionError, Ccsds123,
                   CodecParams(**base, hybrid_sigma_init=[1 << 17] * 8))
    print("  hybrid sigma init: BSQ+BI streams differ, standalone decode OK")


def test_output_word_size():
    """Output word size B (5.2.2): the compressed image is padded to a multiple of
    B bytes; the User-Defined Data byte survives the header."""
    img = cube(6, 12, 12)
    base = dict(num_bands=6, height=12, width=12, dynamic_range=16)
    for kw in (dict(), dict(entropy_coder="hybrid"), dict(encoding_order="BI")):
        for B in (3, 8):
            p = CodecParams(**base, output_word_size=B, user_data=0xA5, **kw)
            blob = Ccsds123(p).compress(img)
            assert len(blob) % B == 0, f"blob not a multiple of B={B} for {kw}"
            assert np.array_equal(img, Ccsds123.decompress_standalone(blob))
            parsed, _ = rc.parse_header(blob)
            assert parsed["output_word_size"] == B and parsed["user_data"] == 0xA5
    assert _raises(AssertionError, Ccsds123, CodecParams(**base, output_word_size=9))
    print("  output word size: B=3/8 padding for all three coder paths + user data OK")


def test_block_adaptive():
    """Block-adaptive entropy coder (5.4.3.4 / CCSDS-121): all block sizes, zero-run
    and ROS paths, restricted options, BI and periodic; decoded through the header."""
    img = cube(6, 16, 16)
    base = dict(num_bands=6, height=16, width=16, dynamic_range=16,
                entropy_coder="block_adaptive")
    for J in (8, 16, 32, 64):
        blob = Ccsds123(CodecParams(**base, block_size=J)).compress(img)
        assert np.array_equal(img, Ccsds123.decompress_standalone(blob)), f"J={J}"
        parsed, _ = rc.parse_header(blob)
        assert parsed["entropy_coder"] == "block_adaptive" and parsed["block_size"] == J
    # constant image: long zero-block runs incl. ROS; small r resets segments
    flat = np.full((6, 16, 16), 777, np.int64)
    for r in (2, 4096):
        blob = Ccsds123(CodecParams(**base, block_size=8, ref_sample_interval=r)).compress(flat)
        assert np.array_equal(flat, Ccsds123.decompress_standalone(blob)), f"r={r}"
    # both option sets at D=4
    img4 = img >> 12
    for restricted in (False, True):
        p = CodecParams(num_bands=6, height=16, width=16, dynamic_range=4,
                        entropy_coder="block_adaptive", block_size=16, restricted=restricted)
        blob = Ccsds123(p).compress(img4)
        assert np.array_equal(img4, Ccsds123.decompress_standalone(blob))
        assert rc.parse_header(blob)[0]["restricted"] is restricted
    # BI with the periodic limit values in the input sequence
    u = 2
    abs_bi = [p % 4 for p in range((16 + 3) >> 2)]
    out, st = _roundtrip(img, entropy_coder="block_adaptive", block_size=16,
                         encoding_order="BI", interleave_depth=2,
                         update_period_exp=u, absolute_error_limit=abs_bi)
    for y in range(16):
        e = int(np.abs(img[:, y, :] - out[:, y, :]).max())
        assert e <= abs_bi[y >> u], f"row {y}: err {e} > {abs_bi[y >> u]}"
    assert _raises(AssertionError, Ccsds123, CodecParams(**base, block_size=12))
    assert _raises(AssertionError, Ccsds123, CodecParams(**base, restricted=True))
    print(f"  block-adaptive coder: J=8/16/32/64, zero runs, restricted, BI+periodic OK  "
          f"ratio={st['ratio']:.3f}:1")


def test_block_adaptive_numba_identical():
    """The numba block-adaptive kernels must be byte-identical to the pure coder."""
    if not rc.BlockAdaptiveCoder(16).use_numba:
        _skip("numba not available (block-adaptive coder runs pure-Python)")
        return
    rng = np.random.default_rng(2)
    for D, J, r, restricted in [(16, 64, 4096, False), (16, 8, 2, False),
                                (4, 16, 3, True), (20, 32, 64, False)]:
        bc = rc.BlockAdaptiveCoder(D, J, r, restricted)
        for hi in (1, 3, 1 << D):               # hi=1: all zeros -> runs + ROS
            v = rng.integers(0, hi, 1000)
            v[rng.random(1000) < 0.5] = 0
            v = v.astype(np.int64)
            bc.use_numba = True;  bn = bc.encode(v); on = bc.decode(bn, len(v))
            bc.use_numba = False; bp = bc.encode(v); op = bc.decode(bp, len(v))
            assert bn == bp, f"block-adaptive numba bytes differ for {(D, J, r, restricted, hi)}"
            assert np.array_equal(on, v) and np.array_equal(op, v)
    print("  block-adaptive numba kernels BYTE-IDENTICAL to pure-Python + round-trip OK")


def test_sample_rep_phi():
    """Exercise the full Eq (47) sample-representative path (phi != 0)."""
    img = cube(16, 24, 24)
    out, st = _roundtrip(img, num_prediction_bands=3, full=True, theta=2, phi=1)
    assert np.array_equal(img, out), "phi!=0 lossless round-trip failed"
    print(f"  sample-rep phi=1,Theta=2 lossless: max|err|={np.abs(img - out).max()}  "
          f"ratio={st['ratio']:.3f}:1")


def test_narrow_local_sums():
    img = cube(16, 24, 24)
    out, st = _roundtrip(img, num_prediction_bands=3, full=True, local_sum_type="narrow_neighbor")
    assert np.array_equal(img, out), "narrow local sums lossless round-trip failed"
    print(f"  narrow_neighbor lossless: max|err|={np.abs(img - out).max()}  ratio={st['ratio']:.3f}:1")


def test_column_local_sums():
    """Column-oriented local sums (Eq 22-23) end-to-end, lossless + near-lossless."""
    img = cube(8, 24, 24)
    for lst in ("wide_column", "narrow_column"):
        out, st = _roundtrip(img, num_prediction_bands=3, full=True, local_sum_type=lst)
        assert np.array_equal(img, out), f"{lst} lossless round-trip failed"
        out, _ = _roundtrip(img, num_prediction_bands=3, full=True, local_sum_type=lst,
                            absolute_error_limit=4)
        e = int(np.abs(img - out).max())
        assert e <= 4, f"{lst} near-lossless error {e} > 4"
    print("  wide_column + narrow_column: lossless + near-lossless(a=4) round-trips OK")


def test_high_dynamic_range_and_signed():
    """D>16 (both the numba-safe D=24 and the pure-only D=32) and signed samples."""
    rng = np.random.default_rng(21)
    for D, shape in ((24, (4, 12, 12)), (32, (2, 8, 8))):
        img = rng.integers(0, 1 << D, size=shape).astype(np.int64)
        p = CodecParams(num_bands=shape[0], height=shape[1], width=shape[2], dynamic_range=D)
        codec = Ccsds123(p)
        if getattr(rc, "NUMBA_OK", False):     # pin the path: numba caps at D=24
            assert codec.use_numba == (D <= 24), f"unexpected dispatch for D={D}"
        blob = codec.compress(img)
        out = Ccsds123.decompress_standalone(blob)
        assert np.array_equal(img, out), f"D={D} lossless round-trip failed"
    img = rng.integers(-2048, 2048, size=(4, 12, 12)).astype(np.int64)
    p = CodecParams(num_bands=4, height=12, width=12, dynamic_range=12, signed=True)
    out = Ccsds123.decompress_standalone(Ccsds123(p).compress(img))
    assert np.array_equal(img, out), "signed D=12 lossless round-trip failed"
    out2 = Ccsds123.decompress_standalone(
        Ccsds123(CodecParams(num_bands=4, height=12, width=12, dynamic_range=12, signed=True,
                             absolute_error_limit=3)).compress(img))
    e = int(np.abs(img - out2).max())
    assert e <= 3, f"signed near-lossless error {e} > 3"
    print("  D=24, D=32 (pure-only) and signed D=12 lossless + near-lossless: OK")


def test_sample_rep_psi():
    """Nonzero offset psi (Eq 47) requires near-lossless; must respect the limit
    AND actually change the bitstream (a psi-ignored regression would otherwise
    stay round-trip-consistent and pass silently)."""
    img = cube(8, 24, 24)
    out, _ = _roundtrip(img, theta=2, phi=1, psi=2, absolute_error_limit=4)
    e = int(np.abs(img - out).max())
    assert e <= 4, f"psi=2 near-lossless error {e} > 4"
    base = dict(num_bands=8, height=24, width=24, dynamic_range=16,
                theta=2, phi=1, absolute_error_limit=4)
    b0 = Ccsds123(CodecParams(**base, psi=0)).compress(img)
    b2 = Ccsds123(CodecParams(**base, psi=2)).compress(img)
    hlen0, hlen2 = rc.parse_header(b0)[1], rc.parse_header(b2)[1]
    assert b0[hlen0:] != b2[hlen2:], "psi=2 must change the body vs psi=0 (Eq 47 offset)"
    print(f"  sample-rep psi=2 (theta=2, phi=1, a=4): max|err|={e} (<= 4), body differs from psi=0")


def test_relative_error_bound():
    """Relative-only limits through the full bitstream: per Eq (44) every error is
    bounded by (r * |s_hat|) >> D <= (r * s_max) >> D."""
    rng = np.random.default_rng(17)
    img = rng.integers(20000, 40000, size=(6, 24, 24)).astype(np.int64)
    r = 64
    out, st = _roundtrip(img, relative_error_limit=r)
    e = int(np.abs(img - out).max())
    bound = (r * ((1 << 16) - 1)) >> 16
    assert 0 < e <= bound, f"relative-limit error {e} outside (0, {bound}]"
    print(f"  relative-only limit r={r}: 0 < max|err|={e} <= {bound}  ratio={st['ratio']:.3f}:1")


def test_near_lossless():
    img = cube(24, 32, 32)
    a = 4
    out, st = _roundtrip(img, num_prediction_bands=3, full=True, absolute_error_limit=a)
    err = int(np.abs(img - out).max())
    assert err <= a, f"near-lossless error {err} exceeds limit {a}"
    print(f"  near-lossless (a={a}): max|err|={err} (<= {a})  ratio={st['ratio']:.3f}:1")


def test_band_dependent_error_limits():
    """Per-band {a_z} absolute limits: each band must respect its own bound."""
    img = cube(24, 32, 32)
    nz = img.shape[0]
    limits = [i % 5 for i in range(nz)]            # 0,1,2,3,4,0,1,...
    out, st = _roundtrip(img, num_prediction_bands=3, full=True, absolute_error_limit=limits)
    for z in range(nz):
        e = int(np.abs(img[z] - out[z]).max())
        assert e <= limits[z], f"band {z}: error {e} exceeds its limit {limits[z]}"
    print(f"  band-dependent abs limits {{a_z}}: every per-band bound respected  "
          f"ratio={st['ratio']:.3f}:1")


def test_lossless_region():
    img = cube(100, 64, 64, seed=2)                # larger, at-scale
    out, st = _roundtrip(img, num_prediction_bands=3, full=True)
    assert np.array_equal(img, out), "LOSSLESS ROUND-TRIP FAILED"
    print(f"  lossless 100x64x64: max|err|={np.abs(img - out).max()}  ratio={st['ratio']:.3f}:1  "
          f"({st['samples']} samples, enc={st['enc_s']:.1f}s dec={st['dec_s']:.1f}s)")


def test_indian_pines_if_present():
    """Optional: real Indian Pines round-trip (only if the .mat is available locally)."""
    if not os.path.exists(INDIAN_PINES):
        _skip("Indian Pines .mat not present (synthetic covers CI)")
        return
    import scipy.io as sio
    img = np.transpose(sio.loadmat(INDIAN_PINES)["indian_pines"], (2, 0, 1)).astype(np.int64)
    img = img[:, 40:104, 40:104]
    out, st = _roundtrip(img, num_prediction_bands=3, full=True)
    assert np.array_equal(img, out), "Indian Pines lossless round-trip failed"
    print(f"  REAL Indian Pines {img.shape}: max|err|=0  ratio={st['ratio']:.3f}:1")


if __name__ == "__main__":
    for fn in (test_map_unmap_roundtrip, test_eq55_parity_conformance,
               test_zero_band_limit_stays_lossless, test_param_validation,
               test_width_one_reduced_mode, test_low_dynamic_range,
               test_header_edge_cases, test_ndarray_error_limits,
               test_out_of_range_samples_rejected, test_float_input_integrality,
               test_truncated_stream,
               test_ccsds_header, test_hybrid_codec,
               test_numba_byte_identical, test_hybrid_numba_identical,
               test_package_imports_without_torch, test_metrics,
               test_bi_order, test_periodic_error_limits, test_hybrid_bi,
               test_bi_numba_identical, test_accumulator_init_table,
               test_band_varying_sample_rep, test_hybrid_sigma_init, test_output_word_size,
               test_block_adaptive, test_block_adaptive_numba_identical,
               test_lossless_crop, test_reduced_mode,
               test_sample_rep_phi, test_sample_rep_psi, test_narrow_local_sums,
               test_column_local_sums, test_high_dynamic_range_and_signed,
               test_relative_error_bound, test_near_lossless,
               test_band_dependent_error_limits, test_lossless_region,
               test_custom_weight_init, test_weight_exp_offset,
               test_supplementary_tables, test_torch_wrapper,
               test_indian_pines_if_present):
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nALL TESTS PASSED")
