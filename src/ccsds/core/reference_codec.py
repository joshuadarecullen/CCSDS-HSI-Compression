"""
Pure-integer CCSDS-123.0-B-2 predictor and entropy coders.

Encoder and decoder share one prediction loop (`_run`), so encode -> decode is
bit-exact by construction. Arithmetic is exact Python integers throughout
(mod*_R, floor division, clipping); equation numbers refer to CCSDS 123.0-B-2.
numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def _load_sibling(name, *parts):
    """Load a sibling module by file path. Fallback for when this file is exec'd
    standalone, with no package context (as the tests do)."""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), *parts))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from ._codec_numba import (run_numba, run_numba_delta, numba_safe, NUMBA_OK,
                               encode_bi_numba, decode_bi_numba)
except ImportError:
    _m = _load_sibling("_codec_numba", "_codec_numba.py")
    run_numba, run_numba_delta, numba_safe, NUMBA_OK = (
        _m.run_numba, _m.run_numba_delta, _m.numba_safe, _m.NUMBA_OK)
    encode_bi_numba, decode_bi_numba = _m.encode_bi_numba, _m.decode_bi_numba

try:
    from ..io.ccsds_header import pack_header, parse_header
except ImportError:
    _m = _load_sibling("ccsds_header", "..", "io", "ccsds_header.py")
    pack_header, parse_header = _m.pack_header, _m.parse_header

try:
    from ..entropy.hybrid import HybridCoder
except ImportError:
    HybridCoder = _load_sibling("hybrid", "..", "entropy", "hybrid.py").HybridCoder

try:
    from ..entropy.block_adaptive import BlockAdaptiveCoder
except ImportError:
    BlockAdaptiveCoder = _load_sibling(
        "block_adaptive", "..", "entropy", "block_adaptive.py").BlockAdaptiveCoder


def _clip(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _mod_star(x: int, R: int) -> int:
    """mod*_R[x]: the representative of x mod 2^R lying in [-2^(R-1), 2^(R-1)). (4.7.2)"""
    half = 1 << (R - 1)
    return ((x + half) % (1 << R)) - half


# Bit I/O, MSB-first.
class BitWriter:
    def __init__(self) -> None:
        self._out = bytearray()
        self._acc = 0
        self._n = 0

    def write_bits(self, value: int, n: int) -> None:
        """Append the n low bits of `value`, most-significant bit first."""
        if n == 0:
            return
        self._acc = (self._acc << n) | (value & ((1 << n) - 1))
        self._n += n
        while self._n >= 8:
            self._n -= 8
            self._out.append((self._acc >> self._n) & 0xFF)
        self._acc &= (1 << self._n) - 1

    def write_zeros(self, n: int) -> None:
        while n >= 8 and self._n == 0:
            self._out.append(0)
            n -= 8
        for _ in range(n):
            self.write_bits(0, 1)

    def to_bytes(self) -> bytes:
        if self._n > 0:
            self._out.append((self._acc << (8 - self._n)) & 0xFF)  # fill bits = 0 (5.4.3.2.4.4)
            self._acc = 0
            self._n = 0
        return bytes(self._out)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_bit(self) -> int:
        byte = self._data[self._pos >> 3]
        bit = (byte >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit

    def read_bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v


@dataclass
class CodecParams:
    # image geometry / sample format
    num_bands: int
    height: int
    width: int
    dynamic_range: int = 16          # D
    signed: bool = False             # signed vs unsigned samples (Eq 9/10)

    # predictor
    num_prediction_bands: int = 3    # P  (0..15)
    full: bool = True                # full vs reduced prediction mode
    local_sum_type: str = "wide_neighbor"   # wide_neighbor|narrow_neighbor|wide_column|narrow_column
    omega: int = 14                  # Omega, weight resolution (4..19)
    register_size: int = 64          # R  (max{32,D+Omega+2}..64)

    # sample representatives (4.9); phi/psi: int or per-band list (5.3.3.5.2/3)
    theta: int = 0                   # Theta resolution (0..4)
    phi: object = 0                  # damping phi_z (0..2^Theta-1)
    psi: object = 0                  # offset psi_z (0..2^Theta-1, 0 if lossless)

    # quantizer fidelity (4.8.2); lossless => all zero. Each limit may be a single
    # int (band-independent) or a length-num_bands list (band-dependent {a_z}/{r_z}).
    absolute_error_limit: object = 0   # a_z
    relative_error_limit: object = 0   # r_z
    # explicit fidelity-control mode (4.8.2.1): None = infer from the limit values
    # (any nonzero => used). parse_header sets these from the header's fc field so
    # that used-but-all-zero limits (legal; they force lossless bands) round-trip.
    abs_limit_used: object = None
    rel_limit_used: object = None

    # weight update (4.10)
    v_min: int = -1
    v_max: int = 3
    t_inc: int = 64                  # power of 2 in [2^4, 2^11]
    zeta_inter: int = 0              # inter-band weight exponent offset (central comps)
    zeta_intra: int = 0              # intra-band weight exponent offset (directional comps)
    # per-band weight exponent offsets (4.10.4), each in -6..5, in table order
    # (5.3.3.3.3.2): [zeta*_z (full mode only)] + [zeta_z^(1..min(z,P))].
    # None with nonzero scalar zetas above expands them per band.
    weight_exp_offset: object = None

    # custom weight initialization (4.6.3.3): per-band vectors Lambda_z, each of
    # length C_z = (3 if full else 0) + min(z, P), components signed Q-bit ints.
    # None = default initialization (Eq 33-34).
    weight_init: object = None       # list of per-band lists, or None
    weight_init_resolution: int = 0  # Q (3..Omega+3); 0 with weight_init => Omega+3

    # sample-adaptive entropy coder (5.4.3.2)
    gamma0: int = 1                  # initial count exponent (1..8)
    gamma_star: int = 6              # rescaling counter size (max{4,gamma0+1}..11)
    u_max: int = 18                  # unary length limit (8..32)
    # accumulator init K (5.4.3.2.3.3): 0..min(D-2,14), None = min(3, D-2),
    # or a per-band list (table 5.3.4.2.2)
    k_init: object = None

    # hybrid initial accumulators Sigma_z(0) per band (5.4.3.3.4.3): in
    # [0, 2^(D+gamma0)), not carried in the stream; None = 4 * 2^gamma0
    hybrid_sigma_init: object = None

    # entropy coder selection
    entropy_coder: str = "sample_adaptive"   # 'sample_adaptive' | 'hybrid' | 'block_adaptive'

    # block-adaptive coder (5.4.3.4 / CCSDS-121)
    block_size: int = 64             # J (8/16/32/64)
    ref_sample_interval: int = 4096  # r (1..4096, encoded mod 2^12)
    restricted: bool = False         # restricted code options (D <= 4 only)

    # framing (5.2.2 / table 5-3)
    user_data: int = 0               # User-Defined Data header byte
    output_word_size: int = 1        # B: compressed image padded to a multiple of B bytes

    # input order (5.4.2): 'BSQ' (default) or 'BI' (band-interleaved)
    encoding_order: str = "BSQ"      # 'BSQ' | 'BI'
    interleave_depth: int = 0        # M for BI: 1=BIL .. num_bands=BIP; 0 => num_bands
    # periodic error-limit updating (4.8.2.4): u in 0..9 (-1 = off). When on, the
    # error-limit params are per-period lists of length ceil(height / 2^u), BI order
    # is required, and the limit values travel in the body rather than the header.
    update_period_exp: int = -1
    abs_bits: int = 0                # DA override (0 => derive from values); set on header parse
    rel_bits: int = 0                # DR override

    # supplementary information tables (3.5): header-only metadata, no effect on
    # the body. List of dicts {type: unsigned|signed|float, purpose, structure:
    # 0d|1d|zx|yx, user, bit_depth (int types) or df/de/bias (float), data} with
    # flat row-major data; float elements are raw (sign, exponent, significand).
    supplementary_tables: object = None

    @staticmethod
    def _norm_limit(v):
        """Accept numpy arrays (including entries of periodic lists); -> plain lists."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)) and any(isinstance(e, np.ndarray) for e in v):
            return [e.tolist() if isinstance(e, np.ndarray) else e for e in v]
        return v

    def __post_init__(self):
        self.absolute_error_limit = self._norm_limit(self.absolute_error_limit)
        self.relative_error_limit = self._norm_limit(self.relative_error_limit)
        if self.k_init is None:
            self.k_init = min(3, max(0, self.dynamic_range - 2))
        elif isinstance(self.k_init, (list, tuple, np.ndarray)):
            self.k_init = [int(v) for v in self.k_init]
        if isinstance(self.phi, (list, tuple, np.ndarray)):
            self.phi = [int(v) for v in self.phi]
        if isinstance(self.psi, (list, tuple, np.ndarray)):
            self.psi = [int(v) for v in self.psi]
        if self.hybrid_sigma_init is not None:
            self.hybrid_sigma_init = [int(v) for v in self.hybrid_sigma_init]
        if self.weight_init is not None:
            self.weight_init = [[int(v) for v in row] for row in self.weight_init]
            if self.weight_init_resolution == 0:
                self.weight_init_resolution = self.omega + 3
        if self.weight_exp_offset is not None:
            self.weight_exp_offset = [[int(v) for v in row] for row in self.weight_exp_offset]
        elif self.zeta_inter or self.zeta_intra:    # expand the scalar zetas per band
            self.weight_exp_offset = [
                ([self.zeta_intra] if self.full else []) +
                [self.zeta_inter] * min(z, self.num_prediction_bands)
                for z in range(self.num_bands)]

    def band_components(self, z: int) -> int:
        """C_z: number of weight/local-difference components for band z (4.6.1)."""
        return (3 if self.full else 0) + min(z, self.num_prediction_bands)

    @staticmethod
    def _limit_max(v) -> int:
        return max(v) if isinstance(v, (list, tuple)) else v

    @property
    def periodic(self) -> bool:
        return self.update_period_exp >= 0

    @property
    def lossless(self) -> bool:
        if self.periodic:
            return False
        return self._limit_max(self.absolute_error_limit) == 0 and \
            self._limit_max(self.relative_error_limit) == 0

    def fidelity_layout(self):
        """(abs_used, rel_used, band_dep_abs, band_dep_rel, DA, DR) for the quantizer
        metadata, covering both fixed and periodic (per-period) error limits."""
        a, r = self.absolute_error_limit, self.relative_error_limit
        if self.periodic:
            au = (a != 0) if self.abs_limit_used is None else bool(self.abs_limit_used)
            ru = (r != 0) if self.rel_limit_used is None else bool(self.rel_limit_used)
            da = au and isinstance(a[0], (list, tuple))
            dr = ru and isinstance(r[0], (list, tuple))
            def depth(used, arr, override):
                if not used:
                    return 0
                if override:
                    return override
                m = max(max(e) if isinstance(e, (list, tuple)) else e for e in arr)
                return max(1, int(m).bit_length())
            return au, ru, da, dr, depth(au, a, self.abs_bits), depth(ru, r, self.rel_bits)
        au = self._limit_max(a) > 0 if self.abs_limit_used is None else bool(self.abs_limit_used)
        ru = self._limit_max(r) > 0 if self.rel_limit_used is None else bool(self.rel_limit_used)
        da, dr = au and isinstance(a, (list, tuple)), ru and isinstance(r, (list, tuple))
        DA = max(1, int(self._limit_max(a)).bit_length()) if au else 0
        DR = max(1, int(self._limit_max(r)).bit_length()) if ru else 0
        return au, ru, da, dr, DA, DR

    def validate(self) -> None:
        D = self.dynamic_range
        assert 2 <= D <= 32
        assert 1 <= self.num_bands <= (1 << 16), "Nz must be in 1..2^16 (3.2)"
        assert 1 <= self.height <= (1 << 16), "Ny must be in 1..2^16 (3.2)"
        assert 1 <= self.width <= (1 << 16), "Nx must be in 1..2^16 (3.2)"
        assert 0 <= self.num_prediction_bands <= 15
        assert 4 <= self.omega <= 19
        assert max(32, D + self.omega + 2) <= self.register_size <= 64
        assert 0 <= self.theta <= 4
        for f in (self.phi, self.psi):
            if isinstance(f, list):
                assert len(f) == self.num_bands, "per-band phi/psi need num_bands entries"
            assert all(0 <= v <= (1 << self.theta) - 1
                       for v in (f if isinstance(f, list) else [f]))
        if self.lossless:
            assert not any(self.psi if isinstance(self.psi, list) else [self.psi])
        assert -6 <= self.v_min <= self.v_max <= 9
        assert (self.t_inc & (self.t_inc - 1)) == 0 and 16 <= self.t_inc <= 2048
        assert 1 <= self.gamma0 <= 8
        assert max(4, self.gamma0 + 1) <= self.gamma_star <= 11
        assert 8 <= self.u_max <= 32
        if self.hybrid_sigma_init is not None:
            assert len(self.hybrid_sigma_init) == self.num_bands and \
                all(0 <= v < (1 << (D + self.gamma0)) for v in self.hybrid_sigma_init), \
                "hybrid_sigma_init needs num_bands values in 0..2^(D+gamma0)-1"
        assert 1 <= self.output_word_size <= 8
        assert 0 <= self.user_data <= 255
        ki = self.k_init if isinstance(self.k_init, list) else [self.k_init]
        assert all(0 <= k <= min(D - 2, 14) for k in ki), \
            f"k_init {self.k_init} outside 0..min(D-2,14); use k_init=None to derive it from D"
        if isinstance(self.k_init, list):
            assert len(self.k_init) == self.num_bands, "per-band k_init needs num_bands entries"
        assert self.entropy_coder in ("sample_adaptive", "hybrid", "block_adaptive")
        assert self.block_size in (8, 16, 32, 64)
        assert 1 <= self.ref_sample_interval <= 4096
        if self.restricted:
            assert D <= 4, "restricted code options require D <= 4"
        assert self.encoding_order in ("BSQ", "BI")
        if self.encoding_order == "BI":
            M = self.interleave_depth or self.num_bands
            assert 1 <= M <= self.num_bands, "interleave_depth M must be in 1..num_bands"
        if self.periodic:
            assert 0 <= self.update_period_exp <= 9, "update period exponent u must be 0..9"
            assert self.encoding_order == "BI", "periodic error-limit updating requires BI order"
            assert self.absolute_error_limit != 0 or self.relative_error_limit != 0, \
                "periodic updating requires at least one per-period error-limit list"
            if self.abs_limit_used:
                assert self.absolute_error_limit != 0, \
                    "abs_limit_used=True requires per-period absolute error-limit lists"
            if self.rel_limit_used:
                assert self.relative_error_limit != 0, \
                    "rel_limit_used=True requires per-period relative error-limit lists"
            if self.abs_limit_used is not None and not self.abs_limit_used:
                assert self.absolute_error_limit == 0, \
                    "abs_limit_used=False conflicts with absolute error-limit lists"
            if self.rel_limit_used is not None and not self.rel_limit_used:
                assert self.relative_error_limit == 0, \
                    "rel_limit_used=False conflicts with relative error-limit lists"
            nper = (self.height + (1 << self.update_period_exp) - 1) >> self.update_period_exp
            for lim in (self.absolute_error_limit, self.relative_error_limit):
                if lim != 0:
                    assert isinstance(lim, (list, tuple)), \
                        "periodic error limits must be per-period lists, not scalars"
                    assert len(lim) == nper, f"periodic error-limit list must have length {nper}"
                    dep = isinstance(lim[0], (list, tuple))
                    for v in lim:
                        assert isinstance(v, (list, tuple)) == dep, \
                            "period entries must be uniformly scalar or per-band"
                        if dep:
                            assert len(v) == self.num_bands, "band-dependent period entry needs num_bands values"
        else:
            for lim in (self.absolute_error_limit, self.relative_error_limit):
                if isinstance(lim, (list, tuple)):
                    assert len(lim) == self.num_bands, \
                        "per-band error-limit list must have length num_bands"
                    assert not any(isinstance(v, (list, tuple)) for v in lim), \
                        "per-period nested lists require periodic updating"
            if self.abs_limit_used is not None and not self.abs_limit_used:
                assert self._limit_max(self.absolute_error_limit) == 0, \
                    "abs_limit_used=False conflicts with a nonzero absolute_error_limit"
            if self.rel_limit_used is not None and not self.rel_limit_used:
                assert self._limit_max(self.relative_error_limit) == 0, \
                    "rel_limit_used=False conflicts with a nonzero relative_error_limit"
        # error-limit values must fit in DA/DR <= min(D-1, 16) bits (4.8.2.2, 4.8.2.3)
        lim_bits = min(D - 1, 16)
        for lim in (self.absolute_error_limit, self.relative_error_limit):
            flat = []
            if isinstance(lim, (list, tuple)):
                for e in lim:
                    if isinstance(e, (list, tuple)):
                        flat.extend(e)
                    else:
                        flat.append(e)
            else:
                flat.append(lim)
            for v in flat:
                assert 0 <= int(v) < (1 << lim_bits), \
                    f"error limit {v} does not fit in min(D-1,16)={lim_bits} bits"
        if self.weight_init is not None:
            Q = self.weight_init_resolution
            assert 3 <= Q <= self.omega + 3, "weight init resolution Q must be 3..Omega+3 (4.6.3.3.2)"
            assert len(self.weight_init) == self.num_bands, \
                "weight_init needs one vector per band"
            half = 1 << (Q - 1)
            for z, row in enumerate(self.weight_init):
                assert len(row) == self.band_components(z), \
                    f"weight_init[{z}] must have C_z={self.band_components(z)} components"
                for v in row:
                    assert -half <= v < half, f"weight init component {v} not a signed {Q}-bit int"
        assert -6 <= self.zeta_inter <= 5 and -6 <= self.zeta_intra <= 5, \
            "weight exponent offsets must be in -6..5 (4.10.4)"
        if self.weight_exp_offset is not None:
            assert len(self.weight_exp_offset) == self.num_bands, \
                "weight_exp_offset needs one vector per band"
            for z, row in enumerate(self.weight_exp_offset):
                n = (1 if self.full else 0) + min(z, self.num_prediction_bands)
                assert len(row) == n, \
                    f"weight_exp_offset[{z}] must have {n} components (intra + min(z,P) inter)"
                for v in row:
                    assert -6 <= v <= 5, f"weight exponent offset {v} outside -6..5 (4.10.4)"
        if self.supplementary_tables is not None:
            assert len(self.supplementary_tables) <= 15, \
                "at most 15 supplementary information tables (3.5.2.1)"
            counts = {"0d": 1, "1d": self.num_bands,
                      "zx": self.num_bands * self.width, "yx": self.height * self.width}
            for t in self.supplementary_tables:
                assert t["type"] in ("unsigned", "signed", "float")
                assert 0 <= t["purpose"] <= 15 and not 5 <= t["purpose"] <= 9, \
                    "table purpose values 5..9 are reserved (3.5.2.2)"
                assert 0 <= t.get("user", 0) <= 15
                assert len(t["data"]) == counts[t["structure"]], \
                    f"{t['structure']} table needs {counts[t['structure']]} elements"
                if t["type"] == "float":
                    DF, DE, beta = t["df"], t["de"], t["bias"]
                    assert 1 <= DF <= 23 and 2 <= DE <= 8 and 0 <= beta < (1 << DE), \
                        "float table needs 1<=DF<=23, 2<=DE<=8, 0<=bias<2^DE (3.5.2.3.3)"
                    for b, al, j in t["data"]:
                        assert b in (0, 1) and 0 <= al < (1 << DE) and 0 <= j < (1 << DF)
                else:
                    DI = t["bit_depth"]
                    assert 1 <= DI <= 32, "integer table bit depth must be 1..32 (3.5.2.3.1)"
                    lo, hi = ((-(1 << (DI - 1)), 1 << (DI - 1)) if t["type"] == "signed"
                              else (0, 1 << DI))
                    for v in t["data"]:
                        assert lo <= v < hi, f"table element {v} not a {t['type']} {DI}-bit int"
        if self.local_sum_type not in (
            "wide_neighbor", "narrow_neighbor", "wide_column", "narrow_column"
        ):
            raise ValueError(f"bad local_sum_type {self.local_sum_type}")
        if self.width == 1:
            assert "column" in self.local_sum_type, "Nx=1 requires column-oriented local sums"
            assert not self.full, "Nx=1 requires reduced prediction mode (4.3.1)"


class Ccsds123:
    """CCSDS-123.0-B-2 compressor/decompressor (predictor + sample-adaptive coder)."""

    def __init__(self, params: CodecParams) -> None:
        params.validate()
        self.p = params
        D = params.dynamic_range
        if params.signed:                                   # Eq (10)
            self.s_min = -(1 << (D - 1))
            self.s_max = (1 << (D - 1)) - 1
            self.s_mid = 0
        else:                                               # Eq (9)
            self.s_min = 0
            self.s_max = (1 << D) - 1
            self.s_mid = 1 << (D - 1)
        self.w_min = -(1 << (params.omega + 2))             # Eq (30)
        self.w_max = (1 << (params.omega + 2)) - 1
        # image-level fidelity mode (4.8.2.1): which limit types are in use
        self._abs_used, self._rel_used = params.fidelity_layout()[:2]
        # numba kernel when available and int64-safe; byte-identical to _run.
        # Set False to force the pure-Python path.
        self.use_numba = bool(NUMBA_OK and numba_safe(params))

    # weight initialization (default Eq 33-34, custom Eq 35)
    def _init_weights(self, z: int) -> List[int]:
        p = self.p
        if p.weight_init is not None:                       # custom (Eq 35)
            Q = p.weight_init_resolution
            scale = 1 << (p.omega + 3 - Q)
            extra = (1 << (p.omega + 2 - Q)) - 1 if Q <= p.omega + 2 else 0
            return [scale * lam + extra for lam in p.weight_init[z]]
        p_star = min(z, p.num_prediction_bands)
        centrals: List[int] = []
        if p_star > 0:
            first = (7 * (1 << p.omega)) // 8               # floor(7/8 * 2^Omega)
            centrals.append(first)
            for _ in range(1, p_star):
                centrals.append(centrals[-1] // 8)          # floor(1/8 * previous)
        if p.full:
            return [0, 0, 0] + centrals                     # [w^N, w^W, w^NW, w^(1..)]
        return centrals

    # per-component weight exponent offsets zeta, j-indexed like the weights:
    # the 3 directional components share zeta*_z (4.10.4)
    def _zeta_row(self, z: int) -> List[int]:
        p = self.p
        if p.weight_exp_offset is None:
            return [0] * p.band_components(z)
        o = p.weight_exp_offset[z]
        return ([o[0]] * 3 + list(o[1:])) if p.full else list(o)

    # local sum (Eq 20-23)
    def _local_sum(self, spp: np.ndarray, z: int, y: int, x: int) -> int:
        Nx = self.p.width
        g = lambda zz, yy, xx: int(spp[zz, yy, xx])
        t = self.p.local_sum_type
        if t == "wide_neighbor":                            # Eq (20)
            if y > 0 and 0 < x < Nx - 1:
                return g(z, y, x - 1) + g(z, y - 1, x - 1) + g(z, y - 1, x) + g(z, y - 1, x + 1)
            if y == 0 and x > 0:
                return 4 * g(z, y, x - 1)
            if y > 0 and x == 0:
                return 2 * (g(z, y - 1, x) + g(z, y - 1, x + 1))
            if y > 0 and x == Nx - 1:
                return g(z, y, x - 1) + g(z, y - 1, x - 1) + 2 * g(z, y - 1, x)
            return 0  # (0,0): undefined, never used
        if t == "narrow_neighbor":                          # Eq (21)
            if y > 0 and 0 < x < Nx - 1:
                return g(z, y - 1, x - 1) + 2 * g(z, y - 1, x) + g(z, y - 1, x + 1)
            if y == 0 and x > 0 and z > 0:
                return 4 * g(z - 1, y, x - 1)
            if y > 0 and x == 0:
                return 2 * (g(z, y - 1, x) + g(z, y - 1, x + 1))
            if y > 0 and x == Nx - 1:
                return 2 * (g(z, y - 1, x - 1) + g(z, y - 1, x))
            if y == 0 and x > 0 and z == 0:
                return 4 * self.s_mid
            return 0
        if t == "wide_column":                              # Eq (22)
            if y > 0:
                return 4 * g(z, y - 1, x)
            return 4 * g(z, y, x - 1)                        # y==0, x>0
        # narrow_column                                       Eq (23)
        if y > 0:
            return 4 * g(z, y - 1, x)
        if x > 0 and z > 0:
            return 4 * g(z - 1, y, x - 1)
        return 4 * self.s_mid                                # y==0, x>0, z==0

    # local difference vector U_z(t) (Eq 24-29)
    def _local_diffs(self, spp: np.ndarray, cdiff: np.ndarray,
                     z: int, y: int, x: int, sigma: int) -> List[int]:
        p = self.p
        g = lambda zz, yy, xx: int(spp[zz, yy, xx])
        U: List[int] = []
        if p.full:                                          # directional, current band (Eq 25-27)
            d_n = (4 * g(z, y - 1, x) - sigma) if y > 0 else 0
            if y > 0 and x > 0:
                d_w = 4 * g(z, y, x - 1) - sigma
                d_nw = 4 * g(z, y - 1, x - 1) - sigma
            elif y > 0 and x == 0:
                d_w = 4 * g(z, y - 1, x) - sigma
                d_nw = 4 * g(z, y - 1, x) - sigma
            else:
                d_w = 0
                d_nw = 0
            U += [d_n, d_w, d_nw]
        # central diffs from previous bands (Eq 24). These were computed and cached
        # when each previous band was processed (a band's central differences are
        # fixed once its sample representatives are final), so no recomputation here.
        p_star = min(z, p.num_prediction_bands)
        for i in range(1, p_star + 1):
            U.append(int(cdiff[z - i, y, x]))
        return U

    # prediction (Eq 36-39)
    def _predict(self, spp, weights, U, sigma, z, y, x, t):
        p = self.p
        Om = p.omega
        if t == 0:                                          # Eq (38) first-sample cases
            if z > 0 and p.num_prediction_bands > 0:
                s_breve = 2 * int(spp[z - 1, y, x])
            else:
                s_breve = 2 * self.s_mid
            return None, s_breve, s_breve >> 1
        d_hat = 0                                           # Eq (36) inner product
        for w, u in zip(weights, U):
            d_hat += w * u
        inner = d_hat + (1 << Om) * (sigma - 4 * self.s_mid)
        hr = _mod_star(inner, p.register_size) + (1 << (Om + 2)) * self.s_mid + (1 << (Om + 1))
        s_tilde = _clip(hr, (1 << (Om + 2)) * self.s_min,
                        (1 << (Om + 2)) * self.s_max + (1 << (Om + 1)))   # Eq (37)
        s_breve = s_tilde >> (Om + 1)                       # Eq (38)  t>0
        s_hat = s_breve >> 1                                # Eq (39)
        return s_tilde, s_breve, s_hat

    # quantizer fidelity (Eq 42-45). al/rl are this sample's active limits; they
    # default to the per-image limits, or are the period's limits when periodic.
    # Which limit types are in use is an image-level mode (4.8.2.1, matching the
    # header's fidelity control field), not a per-value test: a limit of 0 in a
    # band-dependent list is a valid value forcing that band lossless.
    def _max_error(self, s_hat: int, z: int, al=None, rl=None) -> int:
        p = self.p
        if al is None:
            al, rl = p.absolute_error_limit, p.relative_error_limit
        au, ru = self._abs_used, self._rel_used
        if not au and not ru:
            return 0
        a = al[z] if isinstance(al, (list, tuple)) else al      # band-dependent or -independent
        r = rl[z] if isinstance(rl, (list, tuple)) else rl
        if not ru:
            return a                                            # Eq (43)
        rel = (r * abs(s_hat)) >> p.dynamic_range               # Eq (44)
        if not au:
            return rel
        return min(a, rel)                                      # Eq (45)

    # mapped quantizer index (Eq 55-56)
    def _theta(self, s_hat: int, m: int, t: int):
        if t == 0 or m == 0:
            lo = s_hat - self.s_min
            hi = self.s_max - s_hat
        else:
            step = 2 * m + 1
            lo = (s_hat - self.s_min + m) // step
            hi = (self.s_max - s_hat + m) // step
        return lo, hi, min(lo, hi)

    # Eq (55) keys the sign on the parity of the double-resolution predicted
    # sample (the standard's s~_z(t), Eq 38; held here in s_breve), not on the
    # predicted sample s_hat = s_breve >> 1.
    def _map_index(self, q: int, s_breve: int, theta: int) -> int:
        aq = abs(q)
        if aq > theta:
            return aq + theta
        parity = -1 if (s_breve & 1) else 1                 # (-1)^{s~_z(t)}
        if parity * q >= 0:
            return 2 * aq
        return 2 * aq - 1

    def _unmap_index(self, delta: int, s_breve: int, lo: int, hi: int, theta: int) -> int:
        if delta > 2 * theta:                               # |q| > theta : sign forced
            aq = delta - theta
            return aq if lo < hi else -aq
        parity = -1 if (s_breve & 1) else 1
        if delta % 2 == 0:
            return (delta // 2) * parity
        return -((delta + 1) // 2) * parity

    # sample representative (Eq 46-48)
    def _sample_rep(self, s_prime: int, s_tilde: int, q: int, m: int, t: int, z: int) -> int:
        p = self.p
        if t == 0:
            return s_prime                                  # s''_z(0) = s_z(0) = s'_z(0)
        phi = p.phi[z] if isinstance(p.phi, list) else p.phi
        psi = p.psi[z] if isinstance(p.psi, list) else p.psi
        if phi == 0 and psi == 0:
            return s_prime                                  # Note 2: s'' = s'
        Om, Th = p.omega, p.theta
        sgn_q = (q > 0) - (q < 0)
        num = (4 * ((1 << Th) - phi)
               * (s_prime * (1 << Om) - sgn_q * m * psi * (1 << (Om - Th)))
               + phi * s_tilde - phi * (1 << (Om + 1)))                      # Eq (47) numerator
        s_breve_pp = num // (1 << (Om + Th + 1))            # double-resolution representative
        return (s_breve_pp + 1) // 2                        # Eq (46)

    # weight update (Eq 49-54)
    def _update_weights(self, weights, U, s_prime, s_breve, t, zrow):
        p = self.p
        e = 2 * s_prime - s_breve                           # Eq (49) double-resolution error
        rho = _clip(p.v_min + (t - p.width) // p.t_inc, p.v_min, p.v_max) + p.dynamic_range - p.omega
        sgn_e = 1 if e >= 0 else -1                         # sgn+ (Eq 7)
        for j in range(len(weights)):
            pw = rho + zrow[j]
            val = sgn_e * U[j]
            if pw < 0:
                inc = ((val << (-pw)) + 1) >> 1
            else:
                inc = (val + (1 << pw)) >> (pw + 1)         # floor(1/2 (sgn+ 2^-pw d + 1))
            weights[j] = _clip(weights[j] + inc, self.w_min, self.w_max)

    # entropy coder statistics (5.4.3.2.3)
    def _sigma_init(self, z: int = 0) -> int:
        p = self.p
        kpp = p.k_init[z] if isinstance(p.k_init, list) else p.k_init
        kprime = kpp if kpp <= 30 - p.dynamic_range else 2 * kpp + p.dynamic_range - 30   # Eq (59)
        gamma1 = 1 << p.gamma0
        return ((3 * (1 << (kprime + 6)) - 49) * gamma1) >> 7                              # Eq (58)

    def _code_param(self, sigma: int, gamma: int) -> int:
        p = self.p
        thresh = sigma + ((49 * gamma) >> 7)               # Sigma + floor(49 Gamma / 128)
        if 2 * gamma > thresh:                             # Eq (62)
            return 0
        k = 0
        kmax = p.dynamic_range - 2
        while k < kmax and (gamma << (k + 1)) <= thresh:
            k += 1
        return k

    def _gpo2_encode(self, w: BitWriter, j: int, k: int) -> None:
        p = self.p
        u = j >> k
        if u < p.u_max:                                    # 5.4.3.2.2.1 a)
            w.write_zeros(u)
            w.write_bits(1, 1)
            if k:
                w.write_bits(j & ((1 << k) - 1), k)
        else:                                              # 5.4.3.2.2.1 b) escape
            w.write_zeros(p.u_max)
            w.write_bits(j, p.dynamic_range)

    def _gpo2_decode(self, r: BitReader, k: int) -> int:
        p = self.p
        c = 0
        while c < p.u_max:
            if r.read_bit():
                rem = r.read_bits(k) if k else 0
                return (c << k) | rem
            c += 1
        return r.read_bits(p.dynamic_range)                # escape

    # sample-adaptive counter schedule (data-independent), Eq 57/60/61
    def _gamma_seq(self, N: int):
        full = (1 << self.p.gamma_star) - 1
        G = [0] * N
        resc = [False] * N
        g = 1 << self.p.gamma0
        for t in range(1, N):
            G[t] = g
            if g < full:
                g += 1
            else:
                resc[t] = True
                g = (g + 1) >> 1
        return G, resc

    def _bi_blocks(self):
        Nz, M = self.p.num_bands, (self.p.interleave_depth or self.p.num_bands)
        return M, (Nz + M - 1) // M

    def _period_arrays(self):
        """Per-period error limits (4.8.2.4): lists indexed by period p = y >> u, each
        entry a scalar (band-independent) or a [num_bands] list (band-dependent)."""
        p = self.p
        u = p.update_period_exp
        nper = (p.height + (1 << u) - 1) >> u
        expand = lambda lim: list(lim) if lim != 0 else [0] * nper
        return expand(p.absolute_error_limit), expand(p.relative_error_limit), u

    def _limit_layout(self, *_):
        # Bit depths DA/DR are fixed for the whole image (4.8.2.4.3).
        return self.p.fidelity_layout()

    def _emit_limits(self, w, av, rv, lay):
        au, ru, da, dr, DA, DR = lay
        if au:
            for v in (av if da else [av]):
                w.write_bits(int(v), DA)
        if ru:
            for v in (rv if dr else [rv]):
                w.write_bits(int(v), DR)

    def _read_limits(self, r, lay):
        au, ru, da, dr, DA, DR = lay
        Nz = self.p.num_bands
        av = ([r.read_bits(DA) for _ in range(Nz)] if da else r.read_bits(DA)) if au else 0
        rv = ([r.read_bits(DR) for _ in range(Nz)] if dr else r.read_bits(DR)) if ru else 0
        return av, rv

    # Band-interleaved (BI) sample-adaptive coding of the mapped-index array (5.4.2.2).
    # Each band's accumulator evolves over its own raster sequence exactly as in BSQ;
    # only the order in which codewords are emitted differs. With periodic updating, the
    # limit values for each period are written into the body at y mod 2^u == 0 (5.4.3.2.4.1).
    def _encode_bi(self, delta, period_abs=None, period_rel=None, u=0) -> bytes:
        if self.use_numba:                                  # byte-identical fast path
            return encode_bi_numba(self, delta, period_abs, period_rel, u)
        p = self.p
        Nz, Ny, Nx, D = p.num_bands, p.height, p.width, p.dynamic_range
        M, n_i = self._bi_blocks()
        G, resc = self._gamma_seq(Ny * Nx)
        sigma = [self._sigma_init(z) for z in range(Nz)]
        periodic = period_abs is not None
        lay = self._limit_layout(period_abs, period_rel) if periodic else None
        w = BitWriter()
        for y in range(Ny):
            if periodic and y % (1 << u) == 0:
                pi = y >> u
                self._emit_limits(w, period_abs[pi], period_rel[pi], lay)
            for i in range(n_i):
                for x in range(Nx):
                    for z in range(i * M, min((i + 1) * M, Nz)):
                        t = y * Nx + x
                        d = int(delta[z, y, x])
                        if t == 0:
                            w.write_bits(d, D)
                        else:
                            self._gpo2_encode(w, d, self._code_param(sigma[z], G[t]))
                            if not resc[t]:
                                sigma[z] += d
                            else:
                                sigma[z] = (sigma[z] + d + 1) >> 1
        return w.to_bytes()

    def _decode_bi(self, body):
        """Decode a BI body to (delta, period_abs, period_rel, u). The last three are
        None/0 unless periodic updating recovered per-period limits from the body."""
        if self.use_numba:                                  # byte-identical fast path
            return decode_bi_numba(self, body)
        p = self.p
        Nz, Ny, Nx, D = p.num_bands, p.height, p.width, p.dynamic_range
        M, n_i = self._bi_blocks()
        G, resc = self._gamma_seq(Ny * Nx)
        sigma = [self._sigma_init(z) for z in range(Nz)]
        r = BitReader(body)
        delta = np.zeros((Nz, Ny, Nx), dtype=np.int64)
        periodic = p.periodic
        if periodic:
            period_abs, period_rel, u = self._period_arrays()
            lay = self._limit_layout(period_abs, period_rel)
            got_abs, got_rel = [0] * len(period_abs), [0] * len(period_rel)
        for y in range(Ny):
            if periodic and y % (1 << u) == 0:
                pi = y >> u
                got_abs[pi], got_rel[pi] = self._read_limits(r, lay)
            for i in range(n_i):
                for x in range(Nx):
                    for z in range(i * M, min((i + 1) * M, Nz)):
                        t = y * Nx + x
                        if t == 0:
                            d = r.read_bits(D)
                        else:
                            d = self._gpo2_decode(r, self._code_param(sigma[z], G[t]))
                            if not resc[t]:
                                sigma[z] += d
                            else:
                                sigma[z] = (sigma[z] + d + 1) >> 1
                        delta[z, y, x] = d
        if periodic:
            return delta, got_abs, got_rel, u
        return delta, None, None, 0

    # shared encode/decode loop
    def _run(self, encode: bool, image=None, body=None, collect_delta=False, delta_in=None,
             period_abs=None, period_rel=None, u_period=0):
        # collect_delta (encode) returns the mapped-index array instead of a body;
        # delta_in (decode) reconstructs from a mapped-index array. Both are used by the
        # hybrid and BI coders, whose entropy I/O is decoupled from the causal predictor.
        # period_abs/period_rel supply per-period error limits (4.8.2.4, pure-Python only).
        hybrid_mode = collect_delta or (delta_in is not None)
        periodic = period_abs is not None
        # refresh the image-level fidelity flags so params mutated after
        # construction behave identically on the pure and numba paths
        self._abs_used, self._rel_used = self.p.fidelity_layout()[:2]
        if self.use_numba and not periodic:                  # byte-identical fast path
            if not hybrid_mode:
                return run_numba(self, encode, image, body)
            return run_numba_delta(self, encode, image=image, delta=delta_in)
        p = self.p
        Nz, Ny, Nx, D = p.num_bands, p.height, p.width, p.dynamic_range
        spp = np.zeros((Nz, Ny, Nx), dtype=np.int64)        # sample representatives s''
        recon = np.zeros((Nz, Ny, Nx), dtype=np.int64)      # reconstructed samples s'
        cdiff = np.zeros((Nz, Ny, Nx), dtype=np.int64)      # cached central local differences
        delta_out = np.zeros((Nz, Ny, Nx), dtype=np.int64) if collect_delta else None
        gstar_full = (1 << p.gamma_star) - 1

        writer = BitWriter() if (encode and not collect_delta) else None
        reader = BitReader(body) if (not encode and delta_in is None) else None

        for z in range(Nz):
            weights = self._init_weights(z)
            zrow = self._zeta_row(z)
            gamma = 1 << p.gamma0                            # Gamma(1)  (Eq 57)
            sigma_acc = self._sigma_init(z)                 # Sigma_z(1) (Eq 58)
            for y in range(Ny):
                if periodic:                                # active limits for this row's period
                    al, rl = period_abs[y >> u_period], period_rel[y >> u_period]
                else:
                    al, rl = p.absolute_error_limit, p.relative_error_limit
                for x in range(Nx):
                    t = y * Nx + x
                    if t == 0:
                        s_tilde, s_breve, s_hat = self._predict(spp, weights, None, 0, z, y, x, t)
                        U = None
                    else:
                        sigma = self._local_sum(spp, z, y, x)
                        U = self._local_diffs(spp, cdiff, z, y, x, sigma)
                        s_tilde, s_breve, s_hat = self._predict(spp, weights, U, sigma, z, y, x, t)

                    m = self._max_error(s_hat, z, al, rl)
                    lo, hi, theta = self._theta(s_hat, m, t)

                    if encode:
                        s_val = int(image[z, y, x])
                        d = s_val - s_hat                    # Eq (40)
                        if t == 0:
                            q = d                            # Eq (41) t=0
                        else:
                            sgn = (d > 0) - (d < 0)
                            q = sgn * ((abs(d) + m) // (2 * m + 1))
                        delta = self._map_index(q, s_breve, theta)
                        if collect_delta:
                            delta_out[z, y, x] = delta
                        elif t == 0:
                            assert 0 <= delta < (1 << D), f"delta {delta} not D-bit at t=0"
                            writer.write_bits(delta, D)
                        else:
                            k = self._code_param(sigma_acc, gamma)
                            self._gpo2_encode(writer, delta, k)
                    else:
                        if delta_in is not None:
                            delta = int(delta_in[z, y, x])
                        elif t == 0:
                            delta = reader.read_bits(D)
                        else:
                            k = self._code_param(sigma_acc, gamma)
                            delta = self._gpo2_decode(reader, k)
                        q = self._unmap_index(delta, s_breve, lo, hi, theta)

                    # Eq (41): q_z(0)=Delta is the raw residual, so the first sample of
                    # each band is reconstructed losslessly (step 1, not 2m+1).
                    step = 1 if t == 0 else (2 * m + 1)
                    s_prime = _clip(s_hat + q * step, self.s_min, self.s_max)          # Eq (48)
                    recon[z, y, x] = s_prime
                    spp[z, y, x] = self._sample_rep(s_prime, s_tilde, q, m, t, z)

                    if t > 0:
                        cdiff[z, y, x] = 4 * int(spp[z, y, x]) - sigma   # cache central diff (Eq 24)
                        self._update_weights(weights, U, s_prime, s_breve, t, zrow)
                        if not hybrid_mode and gamma < gstar_full:   # Eq (60)/(61): stats for next
                            sigma_acc += delta
                            gamma += 1
                        elif not hybrid_mode:
                            sigma_acc = (sigma_acc + delta + 1) >> 1
                            gamma = (gamma + 1) >> 1

        if encode:
            return delta_out if collect_delta else writer.to_bytes()
        return recon

    # block-adaptive coder I/O: the full input sequence (5.4.2) as a flat array,
    # periodic limit values inline at row starts
    def _seq_pack(self, delta, pa=None, pr=None, u=0):
        p = self.p
        if p.encoding_order != "BI":
            return delta.reshape(-1)
        Nz, Ny, Nx = delta.shape
        M, n_i = self._bi_blocks()
        lay = self._limit_layout() if pa is not None else None
        out = []
        for y in range(Ny):
            if pa is not None and y % (1 << u) == 0:
                pi = y >> u
                au, ru, da, dr, _, _ = lay
                for used, dep, vals in ((au, da, pa), (ru, dr, pr)):
                    if used:
                        out.extend(vals[pi] if dep else [vals[pi]])
            for i in range(n_i):
                for x in range(Nx):
                    for z in range(i * M, min((i + 1) * M, Nz)):
                        out.append(int(delta[z, y, x]))
        return np.asarray(out, dtype=np.int64)

    def _seq_unpack(self, vals):
        """Inverse of _seq_pack: -> (delta, period_abs, period_rel)."""
        p = self.p
        Nz, Ny, Nx = p.num_bands, p.height, p.width
        if p.encoding_order != "BI":
            return vals.reshape(Nz, Ny, Nx).copy(), None, None
        M, n_i = self._bi_blocks()
        delta = np.zeros((Nz, Ny, Nx), np.int64)
        got_a = got_r = None
        if p.periodic:
            pa0, pr0, u = self._period_arrays()
            got_a, got_r = [0] * len(pa0), [0] * len(pr0)
            au, ru, da, dr, _, _ = self._limit_layout()
        pos = 0
        for y in range(Ny):
            if p.periodic and y % (1 << u) == 0:
                pi = y >> u
                for used, dep, got in ((au, da, got_a), (ru, dr, got_r)):
                    if used:
                        got[pi] = [int(v) for v in vals[pos:pos + Nz]] if dep else int(vals[pos])
                        pos += Nz if dep else 1
            for i in range(n_i):
                for x in range(Nx):
                    for z in range(i * M, min((i + 1) * M, Nz)):
                        delta[z, y, x] = vals[pos]
                        pos += 1
        return delta, got_a, got_r

    # public API
    def _hybrid(self):
        if getattr(self, "_hybrid_coder", None) is None:          # cache (flattened tables reused)
            p = self.p
            self._hybrid_coder = HybridCoder(p.dynamic_range, p.gamma0, p.gamma_star, p.u_max,
                                             sigma_init=p.hybrid_sigma_init)
        return self._hybrid_coder

    def _block_adaptive(self):
        p = self.p
        return BlockAdaptiveCoder(p.dynamic_range, p.block_size,
                                  p.ref_sample_interval, p.restricted)

    def _seq_len(self) -> int:
        """Entropy coder input sequence length (samples + periodic limit values)."""
        p = self.p
        n = p.num_bands * p.height * p.width
        if p.periodic:
            au, ru, da, dr, _, _ = self._limit_layout()
            nper = (p.height + (1 << p.update_period_exp) - 1) >> p.update_period_exp
            n += nper * ((p.num_bands if da else 1) * au + (p.num_bands if dr else 1) * ru)
        return n

    def compress(self, image: np.ndarray) -> bytes:
        """Compress a [Z, Y, X] integer image to a CCSDS-123 header + body byte string."""
        p = self.p
        if image.shape != (p.num_bands, p.height, p.width):
            raise ValueError(f"image shape {image.shape} != {(p.num_bands, p.height, p.width)}")
        header = pack_header(p)                              # bit-exact CCSDS 5.3 header
        img = image.astype(np.int64)
        lo, hi = int(img.min()), int(img.max())
        if lo < self.s_min or hi > self.s_max:
            raise ValueError(
                f"sample values span [{lo}, {hi}], outside [{self.s_min}, {self.s_max}] "
                f"for dynamic_range={p.dynamic_range}, signed={p.signed}")
        if p.entropy_coder == "hybrid":                     # predictor -> mapped indices -> hybrid
            M = (p.interleave_depth or p.num_bands) if p.encoding_order == "BI" else 0
            if p.periodic:
                pa, pr, uu = self._period_arrays()
                delta = self._run(encode=True, image=img, collect_delta=True,
                                  period_abs=pa, period_rel=pr, u_period=uu)
                body = self._hybrid().encode(delta, M, (uu, self._limit_layout(), pa, pr))
            else:
                delta = self._run(encode=True, image=img, collect_delta=True)
                body = self._hybrid().encode(delta, M)
        elif p.entropy_coder == "block_adaptive":           # predictor -> input sequence -> 121 coder
            if p.periodic:
                pa, pr, uu = self._period_arrays()
                delta = self._run(encode=True, image=img, collect_delta=True,
                                  period_abs=pa, period_rel=pr, u_period=uu)
                seq = self._seq_pack(delta, pa, pr, uu)
            else:
                delta = self._run(encode=True, image=img, collect_delta=True)
                seq = self._seq_pack(delta)
            body = self._block_adaptive().encode(seq)
        elif p.encoding_order == "BI":                      # predictor -> mapped indices -> BI coder
            if p.periodic:
                pa, pr, u = self._period_arrays()
                delta = self._run(encode=True, image=img, collect_delta=True,
                                  period_abs=pa, period_rel=pr, u_period=u)
                body = self._encode_bi(delta, period_abs=pa, period_rel=pr, u=u)
            else:
                delta = self._run(encode=True, image=img, collect_delta=True)
                body = self._encode_bi(delta)
        else:
            body = self._run(encode=True, image=img)        # sample-adaptive, BSQ (interleaved)
        blob = header + body
        if len(blob) % p.output_word_size:                  # fill to the output word size (5.2.2)
            blob += bytes(p.output_word_size - len(blob) % p.output_word_size)
        return blob

    def decompress(self, blob: bytes) -> np.ndarray:
        """Decode a byte string produced by compress() back to the [Z, Y, X] image."""
        p = self.p
        body = blob[parse_header(blob)[1]:]
        if p.entropy_coder == "hybrid":
            M = (p.interleave_depth or p.num_bands) if p.encoding_order == "BI" else 0
            shape = (p.num_bands, p.height, p.width)
            if p.periodic:
                pa0, _, uu = self._period_arrays()
                delta, pa, pr = self._hybrid().decode(
                    body, shape, M, (uu, self._limit_layout(), len(pa0)))
                return self._run(encode=False, delta_in=delta,
                                 period_abs=pa, period_rel=pr, u_period=uu)
            delta = self._hybrid().decode(body, shape, M)
            return self._run(encode=False, delta_in=delta)
        if p.entropy_coder == "block_adaptive":
            vals = self._block_adaptive().decode(body, self._seq_len())
            delta, pa, pr = self._seq_unpack(vals)
            if pa is not None:
                return self._run(encode=False, delta_in=delta, period_abs=pa,
                                 period_rel=pr, u_period=p.update_period_exp)
            return self._run(encode=False, delta_in=delta)
        if p.encoding_order == "BI":
            delta, pa, pr, u = self._decode_bi(body)
            if pa is not None:
                return self._run(encode=False, delta_in=delta,
                                 period_abs=pa, period_rel=pr, u_period=u)
            return self._run(encode=False, delta_in=delta)
        return self._run(encode=False, body=body)

    @staticmethod
    def decompress_standalone(blob: bytes) -> np.ndarray:
        """Decode without already having a codec (rebuilds params from the CCSDS header)."""
        params, _ = parse_header(blob)
        return Ccsds123(CodecParams(**params)).decompress(blob)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Nz, Ny, Nx, D = 6, 12, 10, 16
    img = rng.integers(0, 1 << D, size=(Nz, Ny, Nx), dtype=np.int64)
    codec = Ccsds123(CodecParams(num_bands=Nz, height=Ny, width=Nx, dynamic_range=D))
    blob = codec.compress(img)
    out = Ccsds123.decompress_standalone(blob)
    ok = np.array_equal(img, out)
    raw = Nz * Ny * Nx * D
    print(f"lossless={ok}  ratio={raw / (len(blob) * 8):.3f}:1  bytes={len(blob)}")
    assert ok
