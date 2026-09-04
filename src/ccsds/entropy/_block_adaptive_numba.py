"""
Numba kernels for the block-adaptive entropy coder. Byte-identical to the
pure-Python `BlockAdaptiveCoder`.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_OK = True
except Exception:  # pragma: no cover
    NUMBA_OK = False

    def njit(*a, **k):
        if a and callable(a[0]):                 # bare @njit form
            return a[0]

        def wrap(f):
            return f
        return wrap


@njit
def _w(buf, pos, v, nb):                         # v in nb bits, MSB first
    for b in range(nb - 1, -1, -1):
        if (v >> b) & 1:
            buf[pos >> 3] |= 1 << (7 - (pos & 7))
        pos += 1
    return pos


@njit
def _wfs(buf, pos, c):                           # FS(c): c zeros then a 1
    pos += c
    buf[pos >> 3] |= 1 << (7 - (pos & 7))
    return pos + 1


@njit
def _enc_kernel(blocks, D, J, ids, max_k, r, buf):
    cap = J * D
    pairs = np.zeros(J // 2, np.int64)
    pos = 0
    run = 0
    for num in range(blocks.shape[0]):
        seg = (num % r) % 64 == 0
        zero = True
        for i in range(J):
            if blocks[num, i] != 0:
                zero = False
                break
        if run != 0 and (seg or not zero):       # flush zero-block run
            pos += ids + 1
            pos = _wfs(buf, pos, run - 1 if run <= 4 else (4 if seg else run))
            run = 0
        if zero:
            run += 1
            continue
        best, blen = 0, ids + cap                # no compression
        se_len = ids + 1 + J // 2
        se_ok = True
        for i in range(J // 2):
            s = blocks[num, 2 * i] + blocks[num, 2 * i + 1]
            # s >= 2^20 always aborts anyway; the sentinel dodges int64 overflow
            pairs[i] = s * (s + 1) // 2 + blocks[num, 2 * i + 1] if s < (1 << 20) else cap
            if pairs[i] >= cap:
                se_ok = False
            se_len += pairs[i]
        if se_ok and se_len < blen:
            best, blen = 1, se_len
        for k in range(max_k + 1):
            fslen = 0
            ok = True
            for i in range(J):
                q = blocks[num, i] >> k
                if q > cap or fslen > cap:
                    ok = False
                    break
                fslen += q + 1
            if ok and ids + fslen + J * k < blen:
                best, blen = k + 2, ids + fslen + J * k
        if best == 0:
            pos = _w(buf, pos, (1 << ids) - 1, ids)
            for i in range(J):
                pos = _w(buf, pos, blocks[num, i], D)
        elif best == 1:                          # second extension
            pos = _wfs(buf, pos, ids)            # ids zeros + '1'
            for i in range(J // 2):
                pos = _wfs(buf, pos, pairs[i])
        else:                                    # sample splitting, ID = k + 1
            k = best - 2
            pos = _w(buf, pos, k + 1, ids)
            for i in range(J):
                pos = _wfs(buf, pos, blocks[num, i] >> k)
            for i in range(J):
                pos = _w(buf, pos, blocks[num, i] & ((1 << k) - 1), k)
    if run != 0:
        pos += ids + 1
        pos = _wfs(buf, pos, run - 1 if run <= 4 else 4)
    return pos


@njit
def _dec_kernel(fwd, nbits, nblk, D, J, ids, r, out):
    pos = 0
    num = 0
    while num < nblk:
        if pos >= nbits:                         # body ran out: truncated
            return -1
        idv = 0
        for _ in range(ids):
            idv = (idv << 1) | fwd[pos]
            pos += 1
        if idv == 0:
            marker = fwd[pos]
            pos += 1
            if marker == 1:                      # second extension
                for i in range(0, J, 2):
                    m = 0
                    while fwd[pos] == 0:
                        m += 1
                        pos += 1
                    pos += 1
                    b = 0
                    acc = 0
                    while acc + b + 1 <= m:      # largest b with b(b+1)/2 <= m
                        b += 1
                        acc += b
                    d1 = m - acc
                    out[num * J + i] = b - d1
                    out[num * J + i + 1] = d1
            else:                                # zero-block run (blocks stay zero)
                v = 0
                while fwd[pos] == 0:
                    v += 1
                    pos += 1
                pos += 1
                if v == 4:                       # ROS: to the next segment start
                    nn = num + 1
                    while (nn % r) % 64 != 0:
                        nn += 1
                    num = nn if nn < nblk else nblk
                else:
                    num += v + 1 if v <= 3 else v
                continue
        elif idv == (1 << ids) - 1:              # no compression
            for i in range(J):
                v = 0
                for _ in range(D):
                    v = (v << 1) | fwd[pos]
                    pos += 1
                out[num * J + i] = v
        else:                                    # sample splitting k = ID - 1
            k = idv - 1
            for i in range(J):
                q = 0
                while fwd[pos] == 0:
                    q += 1
                    pos += 1
                pos += 1
                out[num * J + i] = q << k
            for i in range(J):
                v = 0
                for _ in range(k):
                    v = (v << 1) | fwd[pos]
                    pos += 1
                out[num * J + i] |= v
        num += 1
    return pos


def encode_numba(bc, vals):
    J, ids = bc.J, bc.id_bits
    v = np.asarray(vals, np.int64).ravel()
    blocks = np.ascontiguousarray(
        np.concatenate([v, np.zeros((-len(v)) % J, np.int64)]).reshape(-1, J))
    buf = np.zeros(len(blocks) * (ids + J * bc.D) // 8 + 1024, np.uint8)
    nbits = _enc_kernel(blocks, bc.D, J, ids, bc.max_k, bc.r, buf)
    return bytes(buf[:(nbits + 7) // 8])


def decode_numba(bc, body, n):
    J, nbits = bc.J, 8 * len(body)
    pad = bc.id_bits + 1 + J * max(bc.D, bc.max_k + 1) + 64   # bounds one block's overrun
    fwd = np.concatenate([np.unpackbits(np.frombuffer(body, np.uint8)),
                          np.ones(pad, np.uint8)])
    nblk = -(-n // J)
    out = np.zeros(nblk * J, np.int64)
    pos = _dec_kernel(fwd, nbits, nblk, bc.D, J, bc.id_bits, bc.r, out)
    if pos < 0 or pos > nbits:
        raise IndexError("truncated block-adaptive body")
    return out[:n]
