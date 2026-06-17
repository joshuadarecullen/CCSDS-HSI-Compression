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
    from ._codec_numba import run_numba, run_numba_delta, numba_safe, NUMBA_OK
except ImportError:
    _m = _load_sibling("_codec_numba", "_codec_numba.py")
    run_numba, run_numba_delta, numba_safe, NUMBA_OK = (
        _m.run_numba, _m.run_numba_delta, _m.numba_safe, _m.NUMBA_OK)

try:
    from ..io.ccsds_header import pack_header, parse_header
except ImportError:
    _m = _load_sibling("ccsds_header", "..", "io", "ccsds_header.py")
    pack_header, parse_header = _m.pack_header, _m.parse_header

try:
    from ..entropy.hybrid import HybridCoder
except ImportError:
    HybridCoder = _load_sibling("hybrid", "..", "entropy", "hybrid.py").HybridCoder


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

    def bit_length(self) -> int:
        return len(self._out) * 8 + self._n


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

    # sample representatives (4.9)
    theta: int = 0                   # Theta resolution (0..4)
    phi: int = 0                     # damping phi_z (0..2^Theta-1)
    psi: int = 0                     # offset psi_z (0..2^Theta-1, 0 if lossless)

    # quantizer fidelity (4.8.2); lossless => all zero. Each limit may be a single
    # int (band-independent) or a length-num_bands list (band-dependent {a_z}/{r_z}).
    absolute_error_limit: object = 0   # a_z
    relative_error_limit: object = 0   # r_z

    # weight update (4.10)
    v_min: int = -1
    v_max: int = 3
    t_inc: int = 64                  # power of 2 in [2^4, 2^11]
    zeta_inter: int = 0              # inter-band weight exponent offset (central comps)
    zeta_intra: int = 0              # intra-band weight exponent offset (directional comps)

    # sample-adaptive entropy coder (5.4.3.2)
    gamma0: int = 1                  # initial count exponent (1..8)
    gamma_star: int = 6              # rescaling counter size (max{4,gamma0+1}..11)
    u_max: int = 18                  # unary length limit (8..32)
    k_init: int = 3                  # accumulator init constant K (0..min(D-2,14))

    # entropy coder selection
    entropy_coder: str = "sample_adaptive"   # 'sample_adaptive' | 'hybrid'

    @staticmethod
    def _limit_max(v) -> int:
        return max(v) if isinstance(v, (list, tuple)) else v

    @property
    def lossless(self) -> bool:
        return self._limit_max(self.absolute_error_limit) == 0 and \
            self._limit_max(self.relative_error_limit) == 0

    def validate(self) -> None:
        D = self.dynamic_range
        assert 2 <= D <= 32
        assert 0 <= self.num_prediction_bands <= 15
        assert 4 <= self.omega <= 19
        assert max(32, D + self.omega + 2) <= self.register_size <= 64
        assert 0 <= self.theta <= 4
        assert 0 <= self.phi <= (1 << self.theta) - 1
        assert 0 <= self.psi <= (1 << self.theta) - 1
        if self.lossless:
            assert self.psi == 0
        assert -6 <= self.v_min <= self.v_max <= 9
        assert (self.t_inc & (self.t_inc - 1)) == 0 and 16 <= self.t_inc <= 2048
        assert 1 <= self.gamma0 <= 8
        assert max(4, self.gamma0 + 1) <= self.gamma_star <= 11
        assert 8 <= self.u_max <= 32
        assert 0 <= self.k_init <= min(D - 2, 14)
        assert self.entropy_coder in ("sample_adaptive", "hybrid")
        for lim in (self.absolute_error_limit, self.relative_error_limit):
            if isinstance(lim, (list, tuple)):
                assert len(lim) == self.num_bands, \
                    "per-band error-limit list must have length num_bands"
        if self.local_sum_type not in (
            "wide_neighbor", "narrow_neighbor", "wide_column", "narrow_column"
        ):
            raise ValueError(f"bad local_sum_type {self.local_sum_type}")
        if self.width == 1:
            assert "column" in self.local_sum_type, "Nx=1 requires column-oriented local sums"


class Ccsds123:
    """CCSDS-123.0-B-2 compressor/decompressor (predictor + sample-adaptive coder)."""

    MAGIC = b"C123"

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
        # numba kernel when available and int64-safe; byte-identical to _run.
        # Set False to force the pure-Python path.
        self.use_numba = bool(NUMBA_OK and numba_safe(params))

    # weight initialization (default, Eq 33-34)
    def _init_weights(self, z: int) -> List[int]:
        p = self.p
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

    # quantizer fidelity (Eq 42-45)
    def _max_error(self, s_hat: int, z: int) -> int:
        p = self.p
        if p.lossless:
            return 0
        al, rl = p.absolute_error_limit, p.relative_error_limit
        a = al[z] if isinstance(al, (list, tuple)) else al      # band-dependent or -independent
        r = rl[z] if isinstance(rl, (list, tuple)) else rl
        if r == 0:
            return a
        rel = (r * abs(s_hat)) >> p.dynamic_range
        if a == 0:
            return rel
        return min(a, rel)

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

    def _map_index(self, q: int, s_hat: int, theta: int) -> int:
        aq = abs(q)
        if aq > theta:
            return aq + theta
        parity = -1 if (s_hat & 1) else 1                   # (-1)^{s_hat}
        if parity * q >= 0:
            return 2 * aq
        return 2 * aq - 1

    def _unmap_index(self, delta: int, s_hat: int, lo: int, hi: int, theta: int) -> int:
        if delta > 2 * theta:                               # |q| > theta : sign forced
            aq = delta - theta
            return aq if lo < hi else -aq
        parity = -1 if (s_hat & 1) else 1
        if delta % 2 == 0:
            return (delta // 2) * parity
        return -((delta + 1) // 2) * parity

    # sample representative (Eq 46-48)
    def _sample_rep(self, s_prime: int, s_tilde: int, q: int, m: int, t: int) -> int:
        p = self.p
        if t == 0:
            return s_prime                                  # s''_z(0) = s_z(0) = s'_z(0)
        if p.phi == 0 and p.psi == 0:
            return s_prime                                  # Note 2: s'' = s'
        Om, Th = p.omega, p.theta
        sgn_q = (q > 0) - (q < 0)
        num = (4 * ((1 << Th) - p.phi)
               * (s_prime * (1 << Om) - sgn_q * m * p.psi * (1 << (Om - Th)))
               + p.phi * s_tilde - p.phi * (1 << (Om + 1)))                  # Eq (47) numerator
        s_breve_pp = num // (1 << (Om + Th + 1))            # double-resolution representative
        return (s_breve_pp + 1) // 2                        # Eq (46)

    # weight update (Eq 49-54)
    def _update_weights(self, weights, U, s_prime, s_breve, t):
        p = self.p
        e = 2 * s_prime - s_breve                           # Eq (49) double-resolution error
        rho = _clip(p.v_min + (t - p.width) // p.t_inc, p.v_min, p.v_max) + p.dynamic_range - p.omega
        sgn_e = 1 if e >= 0 else -1                         # sgn+ (Eq 7)
        for j in range(len(weights)):
            zeta = p.zeta_intra if (p.full and j < 3) else p.zeta_inter
            pw = rho + zeta
            val = sgn_e * U[j]
            if pw < 0:
                inc = ((val << (-pw)) + 1) >> 1
            else:
                inc = (val + (1 << pw)) >> (pw + 1)         # floor(1/2 (sgn+ 2^-pw d + 1))
            weights[j] = _clip(weights[j] + inc, self.w_min, self.w_max)

    # entropy coder statistics (5.4.3.2.3)
    def _sigma_init(self) -> int:
        p = self.p
        kpp = p.k_init
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

    # shared encode/decode loop
    def _run(self, encode: bool, image=None, body=None, collect_delta=False, delta_in=None):
        # collect_delta (encode) returns the mapped-index array instead of a body;
        # delta_in (decode) reconstructs from a mapped-index array. Both are used by
        # the hybrid coder, whose decode is reverse-order and cannot interleave with
        # the causal predictor. Sample-adaptive entropy I/O is skipped in those modes.
        hybrid_mode = collect_delta or (delta_in is not None)
        if self.use_numba:                                   # byte-identical fast path
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
            gamma = 1 << p.gamma0                            # Gamma(1)  (Eq 57)
            sigma_acc = self._sigma_init()                  # Sigma_z(1) (Eq 58)
            for y in range(Ny):
                for x in range(Nx):
                    t = y * Nx + x
                    if t == 0:
                        s_tilde, s_breve, s_hat = self._predict(spp, weights, None, 0, z, y, x, t)
                        U = None
                    else:
                        sigma = self._local_sum(spp, z, y, x)
                        U = self._local_diffs(spp, cdiff, z, y, x, sigma)
                        s_tilde, s_breve, s_hat = self._predict(spp, weights, U, sigma, z, y, x, t)

                    m = self._max_error(s_hat, z)
                    lo, hi, theta = self._theta(s_hat, m, t)

                    if encode:
                        s_val = int(image[z, y, x])
                        d = s_val - s_hat                    # Eq (40)
                        if t == 0:
                            q = d                            # Eq (41) t=0
                        else:
                            sgn = (d > 0) - (d < 0)
                            q = sgn * ((abs(d) + m) // (2 * m + 1))
                        delta = self._map_index(q, s_hat, theta)
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
                        q = self._unmap_index(delta, s_hat, lo, hi, theta)

                    # Eq (41): q_z(0)=Delta is the raw residual, so the first sample of
                    # each band is reconstructed losslessly (step 1, not 2m+1).
                    step = 1 if t == 0 else (2 * m + 1)
                    s_prime = _clip(s_hat + q * step, self.s_min, self.s_max)          # Eq (48)
                    recon[z, y, x] = s_prime
                    spp[z, y, x] = self._sample_rep(s_prime, s_tilde, q, m, t)

                    if t > 0:
                        cdiff[z, y, x] = 4 * int(spp[z, y, x]) - sigma   # cache central diff (Eq 24)
                        self._update_weights(weights, U, s_prime, s_breve, t)
                        if not hybrid_mode and gamma < gstar_full:   # Eq (60)/(61): stats for next
                            sigma_acc += delta
                            gamma += 1
                        elif not hybrid_mode:
                            sigma_acc = (sigma_acc + delta + 1) >> 1
                            gamma = (gamma + 1) >> 1

        if encode:
            return delta_out if collect_delta else writer.to_bytes()
        return recon

    # public API
    def _hybrid(self):
        if getattr(self, "_hybrid_coder", None) is None:          # cache (flattened tables reused)
            p = self.p
            self._hybrid_coder = HybridCoder(p.dynamic_range, p.gamma0, p.gamma_star, p.u_max)
        return self._hybrid_coder

    def compress(self, image: np.ndarray) -> bytes:
        """Compress a [Z, Y, X] integer image to a CCSDS-123 header + body byte string."""
        p = self.p
        if image.shape != (p.num_bands, p.height, p.width):
            raise ValueError(f"image shape {image.shape} != {(p.num_bands, p.height, p.width)}")
        header = pack_header(p)                              # bit-exact CCSDS 5.3 header
        img = image.astype(np.int64)
        if p.entropy_coder == "hybrid":                     # predictor -> mapped indices -> hybrid
            delta = self._run(encode=True, image=img, collect_delta=True)
            body = self._hybrid().encode(delta)
        else:
            body = self._run(encode=True, image=img)        # sample-adaptive (interleaved)
        return header + body

    def decompress(self, blob: bytes) -> np.ndarray:
        """Decode a byte string produced by compress() back to the [Z, Y, X] image."""
        p = self.p
        body = blob[parse_header(blob)[1]:]
        if p.entropy_coder == "hybrid":
            delta = self._hybrid().decode(body, (p.num_bands, p.height, p.width))
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
