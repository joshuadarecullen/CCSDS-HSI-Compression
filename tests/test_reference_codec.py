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
    """Mapped quantizer index (Eq 55-56) must invert exactly for every valid q."""
    codec = Ccsds123(CodecParams(num_bands=2, height=2, width=2, dynamic_range=8))
    bad = 0
    for s_hat in range(codec.s_min, codec.s_max + 1):
        for m in (0, 1, 3):
            lo, hi, theta = codec._theta(s_hat, m, t=1)
            for q in range(-lo, hi + 1):
                delta = codec._map_index(q, s_hat, theta)
                if not (0 <= delta < (1 << 8)):
                    bad += 1
                if codec._unmap_index(delta, s_hat, lo, hi, theta) != q:
                    bad += 1
    assert bad == 0, f"{bad} map/unmap failures"
    print("  map/unmap round-trip: OK (exhaustive over 8-bit s_hat, m in {0,1,3})")


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
    ]
    fields = ("num_bands", "height", "width", "dynamic_range", "signed", "num_prediction_bands",
              "full", "local_sum_type", "omega", "register_size", "theta", "phi", "psi",
              "absolute_error_limit", "relative_error_limit", "v_min", "v_max", "t_inc",
              "gamma0", "gamma_star", "u_max", "k_init", "entropy_coder")
    for kw in cases:
        p = CodecParams(**kw)
        hdr = rc.pack_header(p)
        parsed, hlen = rc.parse_header(hdr)
        assert hlen == len(hdr), (hlen, len(hdr))
        for k in fields:
            assert parsed[k] == getattr(p, k), f"{k}: {parsed[k]} != {getattr(p, k)} for {kw}"
    n = len(rc.pack_header(CodecParams(num_bands=10, height=8, width=8, dynamic_range=16)))
    assert n == 19, f"lossless header should be 12+5+2=19 bytes, got {n}"
    print("  CCSDS 5.3 header: pack/parse round-trips all params; lossless header = 19 bytes")


def test_numba_byte_identical():
    """The numba kernel must produce a byte-identical bitstream to the reference."""
    if not getattr(rc, "NUMBA_OK", False):
        print("  numba not available in this interpreter - skipping (pure-Python path in use)")
        return
    img = cube(20, 24, 24)
    configs = [
        dict(num_prediction_bands=3, full=True),
        dict(num_prediction_bands=3, full=True, absolute_error_limit=4),
        dict(num_prediction_bands=2, full=False, local_sum_type="narrow_neighbor"),
        dict(num_prediction_bands=3, full=True, theta=2, phi=1),
    ]
    for kw in configs:
        p = CodecParams(num_bands=20, height=24, width=24, dynamic_range=16, **kw)
        cn = Ccsds123(p); cn.use_numba = True
        cp = Ccsds123(p); cp.use_numba = False
        assert cn.use_numba, "numba should be active here"
        assert cn.compress(img) == cp.compress(img), f"numba bitstream differs for {kw}"
        dn = Ccsds123(p); dn.use_numba = True
        dp = Ccsds123(p); dp.use_numba = False
        assert np.array_equal(dn.decompress(cp.compress(img)), dp.decompress(cn.compress(img)))
    print("  numba kernel BYTE-IDENTICAL to pure-Python reference (4 configs) + cross-decode OK")


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
        print("  numba not available - hybrid coder runs pure-Python (identity check skipped)")
        return
    rng = np.random.default_rng(1)
    for shp in [(8, 16, 16), (20, 24, 24)]:
        for hi in [3, 100, 1 << 16]:
            d = rng.integers(0, hi, shp).astype(np.int64)
            hc.use_numba = True;  bn = hc.encode(d); on = hc.decode(bn, shp)
            hc.use_numba = False; bp = hc.encode(d); op = hc.decode(bp, shp)
            hc.use_numba = True
            assert bn == bp, f"hybrid numba encode differs from pure-Python for {shp}, {hi}"
            assert np.array_equal(on, d) and np.array_equal(op, d) and np.array_equal(on, op)
    print("  hybrid numba kernels BYTE-IDENTICAL to pure-Python + round-trip OK")


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
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    print(f"  package import via src.ccsds.CCSDS123 + round-trip: OK (torch present={has_torch})")


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
        print("  Indian Pines .mat not present - skipping real-data test (synthetic covers CI)")
        return
    import scipy.io as sio
    img = np.transpose(sio.loadmat(INDIAN_PINES)["indian_pines"], (2, 0, 1)).astype(np.int64)
    img = img[:, 40:104, 40:104]
    out, st = _roundtrip(img, num_prediction_bands=3, full=True)
    assert np.array_equal(img, out), "Indian Pines lossless round-trip failed"
    print(f"  REAL Indian Pines {img.shape}: max|err|=0  ratio={st['ratio']:.3f}:1")


if __name__ == "__main__":
    for fn in (test_map_unmap_roundtrip, test_ccsds_header, test_hybrid_codec,
               test_numba_byte_identical, test_hybrid_numba_identical,
               test_package_imports_without_torch, test_metrics,
               test_lossless_crop, test_reduced_mode,
               test_sample_rep_phi, test_narrow_local_sums, test_near_lossless,
               test_band_dependent_error_limits, test_lossless_region,
               test_indian_pines_if_present):
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nALL TESTS PASSED")
