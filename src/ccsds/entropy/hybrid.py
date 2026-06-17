"""
CCSDS-123.0-B-2 hybrid entropy coder (5.4.3.3).

Codes the mapped quantizer indices with the high-entropy reversed length-limited
GPO2 codes and the 16 low-entropy variable-to-variable codes from annex B (real
tables in `annexb_tables.json`); the body ends with flush words, final
accumulators and a '1' marker. The codewords are suffix-free, so decoding
reverses the whole bitstream and walks ordinary prefix-free tries forward.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    from . import _hybrid_numba as _HN
except Exception:
    try:
        import importlib.util as _ilu

        _spec = _ilu.spec_from_file_location(
            "_hybrid_numba", os.path.join(os.path.dirname(__file__), "_hybrid_numba.py"))
        _HN = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_HN)
    except Exception:
        _HN = None

# Table 5-16: input symbol limit L_i and threshold T_i.
_L = [12, 10, 8, 6, 6, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 0]
_T = [303336, 225404, 166979, 128672, 95597, 69670, 50678, 34898,
      23331, 14935, 9282, 5510, 3195, 1928, 1112, 408]

_X = "X"  # escape input symbol


def _load_tables():
    path = os.path.join(os.path.dirname(__file__), "annexb_tables.json")
    raw = json.load(open(path))
    code, flush = [], []
    for i in range(16):
        ci = {}
        for k, v in raw[str(i)]["code"].items():
            ci[tuple(int(s) if s != _X else _X for s in k.split(","))] = v
        fi = {}
        for k, v in raw[str(i)]["flush"].items():
            key = tuple(int(s) if s != _X else _X for s in k.split(",")) if k else ()
            fi[key] = v
        code.append(ci)
        flush.append(fi)
    return code, flush


def _build_trie(mapping_keys_bits):
    """Trie keyed on REVERSED codeword bits -> value (input/prefix tuple)."""
    root: Dict = {}
    for value, bits in mapping_keys_bits:
        node = root
        for c in reversed(bits):                 # reverse: suffix-free -> prefix-free
            node = node.setdefault(int(c), {})   # int keys to match the bit reader
        node["$"] = value
    return root


class _Bits:
    """Forward bit reader over the reversed body bitstream."""
    def __init__(self, rev_bits: bytes):
        self.b = rev_bits
        self.pos = 0

    def one(self) -> int:
        v = self.b[self.pos]
        self.pos += 1
        return v

    def val(self, n: int) -> int:
        """Read an n-bit field; bits arrive LSB-first in the reversed stream."""
        v = 0
        for j in range(n):
            v |= self.b[self.pos] << j
            self.pos += 1
        return v

    def match(self, trie):
        node = trie
        while "$" not in node:
            node = node[self.b[self.pos]]
            self.pos += 1
        return node["$"]


class HybridCoder:
    def __init__(self, dynamic_range: int, gamma0: int = 1, gamma_star: int = 6,
                 u_max: int = 18) -> None:
        self.D = dynamic_range
        self.g0 = gamma0
        self.gstar = gamma_star
        self.umax = u_max
        self._L, self._T = _L, _T
        self.code, self.flush = _load_tables()
        # decode tries (reversed-bit) for each code's outputs and flush words
        self._out_trie = [_build_trie([(k, v) for k, v in self.code[i].items()]) for i in range(16)]
        self._flush_trie = [_build_trie([(k, v) for k, v in self.flush[i].items()]) for i in range(16)]
        self.use_numba = bool(_HN is not None and getattr(_HN, "NUMBA_OK", False))
        self._A = None

    def _arrays(self):
        if self._A is None:
            self._A = _HN.build_arrays(self)
        return self._A

    # statistics helpers
    def _gamma_seq(self, N: int):
        full = (1 << self.gstar) - 1
        G = [0] * N
        resc = [False] * N
        G[0] = 1 << self.g0
        for t in range(1, N):
            prev = G[t - 1]
            if prev == full:
                resc[t] = True
                G[t] = (prev + 1) >> 1
            else:
                G[t] = prev + 1
        return G, resc

    def _sigma_init(self) -> int:
        return 4 << self.g0                       # small, in [0, 2^(D+gamma0)); decoder-independent

    def _high_k(self, Sigma: int, Gamma: int) -> int:
        thresh = Sigma + ((49 * Gamma) >> 5)      # Eq (66)
        kmax = max(self.D - 2, 2)
        for k in range(kmax, 0, -1):
            if (Gamma << (k + 2)) <= thresh:
                return k
        return 1

    def _low_index(self, Sigma: int, Gamma: int) -> int:
        val = Sigma << 14                         # 5.4.3.3.5.3.1: largest i with Sigma*2^14 < Gamma*T_i
        best = 0
        for i in range(16):
            if val < Gamma * _T[i]:
                best = i
        return best

    def _is_high(self, Sigma: int, Gamma: int) -> bool:
        return Sigma * (1 << 14) >= _T[0] * Gamma   # 5.4.3.3.5.1.4

    # GPO2 (reversed length-limited)
    def _rev_gpo2_fwd(self, j: int, k: int) -> List[int]:
        q = j >> k
        if q < self.umax:                          # 5.4.3.3.3.2.1 a)
            out = [(j >> (k - 1 - b)) & 1 for b in range(k)]   # k LSBs, MSB-first
            out.append(1)
            out.extend([0] * q)
        else:                                      # b) escape
            out = [(j >> (self.D - 1 - b)) & 1 for b in range(self.D)]
            out.extend([0] * self.umax)
        return out

    def _rev_gpo2_dec(self, r: _Bits, k: int) -> int:
        q = 0
        while q < self.umax:
            if r.one() == 1:
                lsbs = 0
                for jj in range(k):
                    lsbs |= r.one() << jj
                return q * (1 << k) + lsbs
            q += 1
        v = 0                                      # escape
        for jj in range(self.D):
            v |= r.one() << jj
        return v

    # encode
    def encode(self, delta: np.ndarray) -> bytes:
        if self.use_numba:
            return _HN.encode_numba(self, self._arrays(), delta)
        Nz, Ny, Nx = delta.shape
        N = Ny * Nx
        D, gstar = self.D, self.gstar
        full = (1 << gstar) - 1
        G, resc = self._gamma_seq(N)
        bits: List[int] = []
        active: List[List] = [[] for _ in range(16)]
        sigma_final: List[int] = []

        for z in range(Nz):
            Sigma = self._sigma_init()
            d0 = int(delta[z, 0, 0])
            bits.extend((d0 >> (D - 1 - b)) & 1 for b in range(D))     # 5.4.3.3.5.1.3
            for t in range(1, N):
                y, x = divmod(t, Nx)
                d = int(delta[z, y, x])
                rescale = resc[t]
                if rescale:
                    bits.append(Sigma & 1)                            # 5.4.3.3.5.1.2 (LSB before delta)
                    Sigma = (Sigma + 4 * d + 1) >> 1
                else:
                    Sigma = Sigma + 4 * d
                Gamma = G[t]
                if self._is_high(Sigma, Gamma):
                    bits.extend(self._rev_gpo2_fwd(d, self._high_k(Sigma, Gamma)))
                else:
                    i = self._low_index(Sigma, Gamma)
                    if d <= _L[i]:
                        sym = d
                    else:
                        sym = _X
                        bits.extend(self._rev_gpo2_fwd(d - _L[i] - 1, 0))   # residual
                    active[i].append(sym)
                    key = tuple(active[i])
                    cw = self.code[i].get(key)
                    if cw is not None:
                        bits.extend(int(c) for c in cw)
                        active[i] = []
            sigma_final.append(Sigma)

        # tail (5.4.3.3.5.4)
        for i in range(16):
            bits.extend(int(c) for c in self.flush[i][tuple(active[i])])
        nbits_sigma = 2 + D + gstar
        for z in range(Nz):
            s = sigma_final[z]
            bits.extend((s >> (nbits_sigma - 1 - b)) & 1 for b in range(nbits_sigma))
        bits.append(1)                                                # marker
        while len(bits) % 8:
            bits.append(0)                                            # fill
        out = bytearray()
        for p in range(0, len(bits), 8):
            byte = 0
            for q in range(8):
                byte = (byte << 1) | bits[p + q]
            out.append(byte)
        return bytes(out)

    # decode (reverse-order)
    def decode(self, body: bytes, shape: Tuple[int, int, int]) -> np.ndarray:
        if self.use_numba:
            return _HN.decode_numba(self, self._arrays(), body, shape)
        Nz, Ny, Nx = shape
        N = Ny * Nx
        D, gstar = self.D, self.gstar
        G, resc = self._gamma_seq(N)
        # forward bits, then reverse the whole stream and decode forward
        fwd = [(body[p >> 3] >> (7 - (p & 7))) & 1 for p in range(len(body) * 8)]
        r = _Bits(fwd[::-1])
        while r.one() == 0:                                           # skip fill
            pass
        # (the '1' just consumed is the marker)
        nbits_sigma = 2 + D + gstar
        sigma_final = [0] * Nz
        for z in range(Nz - 1, -1, -1):
            sigma_final[z] = r.val(nbits_sigma)
        active = [None] * 16
        for i in range(15, -1, -1):
            active[i] = r.match(self._flush_trie[i])
        sym_buf = [list(active[i]) for i in range(16)]               # pop() -> reverse order

        delta = np.zeros(shape, dtype=np.int64)
        for z in range(Nz - 1, -1, -1):
            Sigma = sigma_final[z]
            for t in range(N - 1, 0, -1):
                y, x = divmod(t, Nx)
                Gamma = G[t]
                if self._is_high(Sigma, Gamma):
                    d = self._rev_gpo2_dec(r, self._high_k(Sigma, Gamma))
                else:
                    i = self._low_index(Sigma, Gamma)
                    if not sym_buf[i]:
                        sym_buf[i] = list(r.match(self._out_trie[i]))
                    sym = sym_buf[i].pop()
                    if sym == _X:
                        d = self._rev_gpo2_dec(r, 0) + _L[i] + 1
                    else:
                        d = sym
                delta[z, y, x] = d
                if resc[t]:
                    b = r.one()
                    Sigma = 2 * Sigma - 4 * d - b
                else:
                    Sigma = Sigma - 4 * d
            delta[z, 0, 0] = r.val(D)
        return delta


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    shapes = [(4, 8, 8), (6, 10, 10), (3, 5, 7), (1, 1, 9), (2, 1, 16), (5, 16, 4)]
    n = 0
    for D in (8, 12, 16):
        hc = HybridCoder(dynamic_range=D)
        peak = 1 << D
        for shp in shapes:
            for dist in ("low", "mid", "high", "zeros", "mixed"):
                if dist == "low":
                    d = rng.integers(0, 3, shp)
                elif dist == "mid":                       # exercises escape symbols (delta > L_i)
                    d = rng.integers(0, min(40, peak), shp)
                elif dist == "high":
                    d = rng.integers(0, peak, shp)
                elif dist == "zeros":
                    d = np.zeros(shp, dtype=int)
                else:
                    d = rng.integers(0, peak, shp)
                    d[d > peak // 4] //= 64
                d = d.astype(np.int64)
                assert np.array_equal(hc.decode(hc.encode(d), shp), d), (D, shp, dist)
                n += 1
    print(f"hybrid coder self-test OK ({n} round-trips: D in 8/12/16, 6 shapes x 5 distributions)")
