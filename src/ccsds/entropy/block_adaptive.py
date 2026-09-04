"""
CCSDS-123.0-B-2 block-adaptive entropy coder (5.4.3.4): the CCSDS-121 adaptive
coder over the input sequence, preprocessor bypassed. Each J-sample block picks
the shortest of zero-block (run-length with ROS), second extension, sample
splitting (k=0 is the fundamental sequence) and no compression.
"""

from __future__ import annotations

import math

import numpy as np


class BlockAdaptiveCoder:
    SEG = 64                                     # blocks per zero-run segment

    def __init__(self, dynamic_range: int, block_size: int = 64,
                 ref_interval: int = 4096, restricted: bool = False) -> None:
        self.D, self.J, self.r = dynamic_range, block_size, ref_interval
        self.id_bits = max((dynamic_range - 1).bit_length(), 1 if restricted else 3)
        if restricted:                           # CCSDS-121 5.1.2 option sets
            self.max_k = -1 if dynamic_range <= 2 else 1
        else:
            self.max_k = 5 if dynamic_range <= 8 else (13 if dynamic_range <= 16 else 29)

    def _seg_start(self, num: int) -> bool:
        return (num % self.r) % self.SEG == 0

    def _next_seg(self, num: int) -> int:
        b = num + 1
        while not self._seg_start(b):
            b += 1
        return b

    def _emit_block(self, bits, blk) -> None:
        D, J, ids = self.D, self.J, self.id_bits
        cap = J * D
        # candidate lengths; an option over the 121 length caps never wins
        lens = [ids + cap]                       # no compression
        pairs = [(int(blk[i]) + int(blk[i + 1])) * (int(blk[i]) + int(blk[i + 1]) + 1) // 2
                 + int(blk[i + 1]) for i in range(0, J, 2)]
        lens.append(ids + 1 + sum(pairs) + J // 2 if max(pairs) < cap else 1 << 30)
        for k in range(self.max_k + 1):
            fslen, ok = 0, True
            for s in blk:
                q = int(s) >> k
                if q > cap or fslen > cap:
                    ok = False
                    break
                fslen += q + 1
            lens.append(ids + fslen + J * k if ok else 1 << 30)
        best = min(range(len(lens)), key=lambda i: lens[i])   # ties -> earliest
        if best == 0:
            bits.extend([1] * ids)
            for s in blk:
                bits.extend((int(s) >> (D - 1 - b)) & 1 for b in range(D))
        elif best == 1:                          # second extension
            bits.extend([0] * ids)
            bits.append(1)
            for t in pairs:
                bits.extend([0] * t)
                bits.append(1)
        else:                                    # sample splitting, ID = k + 1
            k = best - 2
            bits.extend(((k + 1) >> b) & 1 for b in range(ids - 1, -1, -1))
            for s in blk:
                bits.extend([0] * (int(s) >> k))
                bits.append(1)
            for s in blk:
                bits.extend((int(s) >> (k - 1 - b)) & 1 for b in range(k))

    def encode(self, vals: np.ndarray) -> bytes:
        J, ids = self.J, self.id_bits
        v = np.asarray(vals, np.int64)
        blocks = np.concatenate([v, np.zeros((-len(v)) % J, np.int64)]).reshape(-1, J)
        bits: list = []

        def zero_run(c, at_seg):                 # 121 zero-block: FS(c-1), ROS, or FS(c)
            bits.extend([0] * (ids + 1))
            bits.extend([0] * (c - 1 if c <= 4 else (4 if at_seg else c)))
            bits.append(1)

        run = 0
        for num in range(len(blocks)):
            seg = self._seg_start(num)
            zero = not blocks[num].any()
            if run and (seg or not zero):
                zero_run(run, seg)
                run = 0
            if zero:
                run += 1
            else:
                self._emit_block(bits, blocks[num])
        if run:
            zero_run(run, True)
        while len(bits) % 8:
            bits.append(0)
        return bytes(sum(b << (7 - i) for i, b in enumerate(bits[p:p + 8]))
                     for p in range(0, len(bits), 8))

    def decode(self, body: bytes, n: int) -> np.ndarray:
        D, J, ids = self.D, self.J, self.id_bits
        fwd = np.unpackbits(np.frombuffer(body, np.uint8))
        pos = 0

        def rd(nb):
            nonlocal pos
            v = 0
            for _ in range(nb):
                v = (v << 1) | int(fwd[pos])
                pos += 1
            return v

        def fs():
            nonlocal pos
            c = 0
            while fwd[pos] == 0:
                c += 1
                pos += 1
            pos += 1
            return c

        nblk = -(-n // J)
        out = np.zeros(nblk * J, np.int64)
        num = 0
        while num < nblk:
            idv = rd(ids)
            if idv == 0:
                if rd(1):                        # second extension
                    for i in range(0, J, 2):
                        m = fs()
                        b = (math.isqrt(8 * m + 1) - 1) // 2
                        d1 = m - b * (b + 1) // 2
                        out[num * J + i] = b - d1
                        out[num * J + i + 1] = d1
                else:                            # zero-block run (blocks stay zero)
                    v = fs()
                    num = min(self._next_seg(num), nblk) if v == 4 else \
                        num + (v + 1 if v <= 3 else v)
                    continue
            elif idv == (1 << ids) - 1:          # no compression
                for i in range(J):
                    out[num * J + i] = rd(D)
            else:                                # sample splitting k = ID - 1
                k = idv - 1
                q = [fs() for _ in range(J)]
                for i in range(J):
                    out[num * J + i] = (q[i] << k) | (rd(k) if k else 0)
            num += 1
        return out[:n]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = 0
    for D in (2, 4, 8, 16, 20):
        for J in (8, 16, 64):
            for r in (1, 3, 64, 4096):
                for restricted in ([False, True] if D <= 4 else [False]):
                    bc = BlockAdaptiveCoder(D, J, r, restricted)
                    for dist in ("zeros", "low", "high", "mixed"):
                        n = int(rng.integers(1, 700))
                        if dist == "zeros":
                            v = np.zeros(n, np.int64)
                        elif dist == "low":
                            v = rng.integers(0, 3, n)
                        elif dist == "high":
                            v = rng.integers(0, 1 << D, n)
                        else:
                            v = rng.integers(0, 1 << D, n)
                            v[rng.random(n) < 0.8] = 0
                        v = v.astype(np.int64)
                        assert np.array_equal(bc.decode(bc.encode(v), n), v), \
                            (D, J, r, restricted, dist)
                        t += 1
    print(f"block-adaptive coder self-test OK ({t} round-trips)")
