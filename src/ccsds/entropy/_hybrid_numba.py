"""
Numba kernels for the hybrid entropy coder. The annex-B tables are flattened
into flat-array tries (encoding tracks a trie node per code; decoding walks
reversed-bit tries). Byte-identical to the pure-Python `HybridCoder`.
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


def _symidx(s):
    return 13 if s == "X" else int(s)            # alphabet 0..12 plus X->13


def build_arrays(hc):
    """Flatten a HybridCoder's tables into numba-friendly flat arrays."""
    # --- encode: one unified trie per code (output codeword at leaves, flush word
    #     at every proper-prefix node), children indexed by symbol (0..13) ---
    e_child, e_op, e_ol, e_fp, e_fl, e_bits, e_root = [], [], [], [], [], [], []

    def e_new():
        e_child.append([-1] * 14)
        e_op.append(-1); e_ol.append(-1); e_fp.append(-1); e_fl.append(-1)
        return len(e_child) - 1

    for i in range(16):
        root = e_new(); e_root.append(root)
        for inp, bits in hc.code[i].items():
            node = root
            for s in inp:
                si = _symidx(s)
                if e_child[node][si] == -1:
                    e_child[node][si] = e_new()
                node = e_child[node][si]
            e_op[node] = len(e_bits); e_ol[node] = len(bits)
            e_bits.extend(int(c) for c in bits)
        for pref, bits in hc.flush[i].items():
            node = root
            for s in pref:
                si = _symidx(s)
                if e_child[node][si] == -1:
                    e_child[node][si] = e_new()
                node = e_child[node][si]
            e_fp[node] = len(e_bits); e_fl[node] = len(bits)
            e_bits.extend(int(c) for c in bits)

    # --- decode: reversed-output trie (-> input symbols) and reversed-flush trie
    #     (-> active-prefix symbols), children indexed by bit (0/1) ---
    def build_bit_trie(table):
        child, sp, sl, syms, root = [], [], [], [], []

        def new():
            child.append([-1, -1]); sp.append(-1); sl.append(-1)
            return len(child) - 1

        for i in range(16):
            r = new(); root.append(r)
            for key, bits in table[i].items():
                node = r
                for c in reversed(bits):
                    b = int(c)
                    if child[node][b] == -1:
                        child[node][b] = new()
                    node = child[node][b]
                sp[node] = len(syms); sl[node] = len(key)
                syms.extend(_symidx(s) for s in key)
        return (np.array(child, np.int32), np.array(sp, np.int32),
                np.array(sl, np.int32), np.array(syms, np.int32), np.array(root, np.int32))

    d_child, d_sp, d_sl, d_syms, d_root = build_bit_trie(hc.code)
    f_child, f_sp, f_sl, f_syms, f_root = build_bit_trie(hc.flush)

    return dict(
        e_child=np.array(e_child, np.int32), e_op=np.array(e_op, np.int32),
        e_ol=np.array(e_ol, np.int32), e_fp=np.array(e_fp, np.int32),
        e_fl=np.array(e_fl, np.int32), e_bits=np.array(e_bits, np.uint8),
        e_root=np.array(e_root, np.int32),
        d_child=d_child, d_sp=d_sp, d_sl=d_sl, d_syms=d_syms, d_root=d_root,
        f_child=f_child, f_sp=f_sp, f_sl=f_sl, f_syms=f_syms, f_root=f_root,
        Tarr=np.array(hc._T, np.int64), Larr=np.array(hc._L, np.int64),
    )


@njit
def _wgpo2(buf, pos, j, k, umax, D):
    q = j >> k
    if q < umax:
        for b in range(k - 1, -1, -1):
            if (j >> b) & 1:
                buf[pos >> 3] |= (1 << (7 - (pos & 7)))
            pos += 1
        buf[pos >> 3] |= (1 << (7 - (pos & 7)))         # the '1'
        pos += 1 + q                                    # q trailing zeros
    else:
        for b in range(D - 1, -1, -1):
            if (j >> b) & 1:
                buf[pos >> 3] |= (1 << (7 - (pos & 7)))
            pos += 1
        pos += umax
    return pos


@njit
def _rgpo2(rev, pos, k, umax, D):
    q = 0
    while q < umax:
        if rev[pos] == 1:
            pos += 1
            lsbs = 0
            for jj in range(k):
                lsbs |= rev[pos] << jj
                pos += 1
            return q * (1 << k) + lsbs, pos
        pos += 1
        q += 1
    v = 0
    for jj in range(D):
        v |= rev[pos] << jj
        pos += 1
    return v, pos


@njit
def _high_k(Sigma, Gamma, D):
    thresh = Sigma + ((49 * Gamma) >> 5)
    kmax = D - 2 if D - 2 > 2 else 2
    for k in range(kmax, 0, -1):
        if (Gamma << (k + 2)) <= thresh:
            return k
    return 1


@njit
def _low_index(Sigma, Gamma, Tarr):
    val = Sigma << 14
    best = 0
    for i in range(16):
        if val < Gamma * Tarr[i]:
            best = i
    return best


@njit
def _e_samp(buf, pos, d, t, z, Sigma, cur, G, resc, D, umax,
            Tarr, Larr, e_child, e_op, e_ol, e_bits, e_root):
    if t == 0:
        for b in range(D - 1, -1, -1):
            if (d >> b) & 1:
                buf[pos >> 3] |= (1 << (7 - (pos & 7)))
            pos += 1
        return pos
    if resc[t]:
        if Sigma[z] & 1:
            buf[pos >> 3] |= (1 << (7 - (pos & 7)))
        pos += 1
        Sigma[z] = (Sigma[z] + 4 * d + 1) >> 1
    else:
        Sigma[z] = Sigma[z] + 4 * d
    Gamma = G[t]
    if Sigma[z] * (1 << 14) >= Tarr[0] * Gamma:                  # high-entropy
        pos = _wgpo2(buf, pos, d, _high_k(Sigma[z], Gamma, D), umax, D)
    else:                                                        # low-entropy
        i = _low_index(Sigma[z], Gamma, Tarr)
        if d <= Larr[i]:
            sym = d
        else:
            sym = 13
            pos = _wgpo2(buf, pos, d - Larr[i] - 1, 0, umax, D)
        node = e_child[cur[i]][sym]
        cur[i] = node
        if e_op[node] != -1:                                     # complete codeword
            p = e_op[node]
            for b in range(e_ol[node]):
                if e_bits[p + b]:
                    buf[pos >> 3] |= (1 << (7 - (pos & 7)))
                pos += 1
            cur[i] = e_root[i]
    return pos


@njit
def _enc_kernel(delta, Nz, Ny, Nx, D, gamma0, gstar, umax, sigma_init, M,
                G, resc, Tarr, Larr, e_child, e_op, e_ol, e_fp, e_fl, e_bits, e_root, buf):
    N = Ny * Nx
    pos = 0
    cur = e_root.copy()
    Sigma = sigma_init.copy()
    if M == 0:                                                   # BSQ order
        for z in range(Nz):
            for t in range(N):
                pos = _e_samp(buf, pos, delta[z, t // Nx, t % Nx], t, z, Sigma, cur, G,
                              resc, D, umax, Tarr, Larr, e_child, e_op, e_ol, e_bits, e_root)
    else:                                                        # BI order (5.4.2.2)
        for y in range(Ny):
            for i in range((Nz + M - 1) // M):
                z1 = min((i + 1) * M, Nz)
                for x in range(Nx):
                    for z in range(i * M, z1):
                        pos = _e_samp(buf, pos, delta[z, y, x], y * Nx + x, z, Sigma, cur, G,
                                      resc, D, umax, Tarr, Larr, e_child, e_op, e_ol, e_bits, e_root)
    # ---- tail ----
    for i in range(16):
        node = cur[i]
        p = e_fp[node]
        for b in range(e_fl[node]):
            if e_bits[p + b]:
                buf[pos >> 3] |= (1 << (7 - (pos & 7)))
            pos += 1
    nbs = 2 + D + gstar
    for z in range(Nz):
        s = Sigma[z]
        for b in range(nbs - 1, -1, -1):
            if (s >> b) & 1:
                buf[pos >> 3] |= (1 << (7 - (pos & 7)))
            pos += 1
    buf[pos >> 3] |= (1 << (7 - (pos & 7)))                          # marker
    pos += 1
    return pos


@njit
def _d_samp(rev, pos, nbits, t, z, Nx, Sigma, symbuf, symtop, out, G, resc, D, umax,
            Tarr, Larr, d_child, d_sp, d_sl, d_syms, d_root):
    if pos >= nbits:                                            # body ran out: truncated
        return -1
    if t == 0:
        v = 0
        for j in range(D):
            v |= rev[pos] << j
            pos += 1
        out[z, 0, 0] = v
        return pos
    Gamma = G[t]
    if Sigma[z] * (1 << 14) >= Tarr[0] * Gamma:                 # high
        d, pos = _rgpo2(rev, pos, _high_k(Sigma[z], Gamma, D), umax, D)
    else:                                                       # low
        i = _low_index(Sigma[z], Gamma, Tarr)
        if symtop[i] == 0:                                      # refill from output codeword
            node = d_root[i]
            while d_sp[node] == -1:
                node = d_child[node][rev[pos]]
                pos += 1
            p = d_sp[node]
            for b in range(d_sl[node]):
                symbuf[i][symtop[i]] = d_syms[p + b]
                symtop[i] += 1
        symtop[i] -= 1
        sym = symbuf[i][symtop[i]]
        if sym == 13:
            r, pos = _rgpo2(rev, pos, 0, umax, D)
            d = r + Larr[i] + 1
        else:
            d = sym
    out[z, t // Nx, t % Nx] = d
    if resc[t]:
        b = rev[pos]
        pos += 1
        Sigma[z] = 2 * Sigma[z] - 4 * d - b
    else:
        Sigma[z] = Sigma[z] - 4 * d
    return pos


@njit
def _dec_kernel(rev, nbits, Nz, Ny, Nx, D, gamma0, gstar, umax, M, G, resc, Tarr, Larr,
                d_child, d_sp, d_sl, d_syms, d_root,
                f_child, f_sp, f_sl, f_syms, f_root, out):
    N = Ny * Nx
    pos = 0
    while rev[pos] == 0:                                             # skip fill
        pos += 1
    pos += 1                                                        # marker
    nbs = 2 + D + gstar
    Sigma = np.zeros(Nz, np.int64)
    for z in range(Nz - 1, -1, -1):
        if pos + nbs > nbits:
            return -1
        v = 0
        for j in range(nbs):
            v |= rev[pos] << j
            pos += 1
        Sigma[z] = v
    symbuf = np.zeros((16, 320), np.int64)
    symtop = np.zeros(16, np.int64)
    for i in range(15, -1, -1):                                     # flush -> active prefix
        if pos >= nbits:
            return -1
        node = f_root[i]
        while f_sp[node] == -1:
            node = f_child[node][rev[pos]]
            pos += 1
        p = f_sp[node]
        for b in range(f_sl[node]):
            symbuf[i][symtop[i]] = f_syms[p + b]
            symtop[i] += 1
    if M == 0:                                                      # BSQ order
        for z in range(Nz - 1, -1, -1):
            for t in range(N - 1, -1, -1):
                pos = _d_samp(rev, pos, nbits, t, z, Nx, Sigma, symbuf, symtop, out, G, resc,
                              D, umax, Tarr, Larr, d_child, d_sp, d_sl, d_syms, d_root)
                if pos < 0:
                    return -1
    else:                                                           # BI order (5.4.2.2)
        for y in range(Ny - 1, -1, -1):
            for i in range((Nz + M - 1) // M - 1, -1, -1):
                z1 = min((i + 1) * M, Nz)
                for x in range(Nx - 1, -1, -1):
                    for z in range(z1 - 1, i * M - 1, -1):
                        pos = _d_samp(rev, pos, nbits, y * Nx + x, z, Nx, Sigma, symbuf, symtop,
                                      out, G, resc, D, umax, Tarr, Larr, d_child, d_sp, d_sl,
                                      d_syms, d_root)
                        if pos < 0:
                            return -1
    return pos


def _gamma_seq(hc, N):
    G, resc = hc._gamma_seq(N)
    return np.array(G, np.int64), np.array([1 if r else 0 for r in resc], np.int64)


def encode_numba(hc, A, delta, M=0):
    Nz, Ny, Nx = delta.shape
    G, resc = _gamma_seq(hc, Ny * Nx)
    cap = (Nz * Ny * Nx * (hc.umax + hc.D) + Nz * (2 + hc.D + hc.gstar) + 4096) // 8 + 64
    buf = np.zeros(cap, np.uint8)
    sig0 = np.array([hc._sigma_init(z) for z in range(Nz)], np.int64)
    nbits = _enc_kernel(np.ascontiguousarray(delta, np.int64), Nz, Ny, Nx, hc.D,
                        hc.g0, hc.gstar, hc.umax, sig0, M, G, resc,
                        A["Tarr"], A["Larr"], A["e_child"], A["e_op"], A["e_ol"],
                        A["e_fp"], A["e_fl"], A["e_bits"], A["e_root"], buf)
    while nbits % 8:                                                 # fill to byte
        nbits += 1
    return bytes(buf[:nbits // 8])


def decode_numba(hc, A, body, shape, M=0):
    Nz, Ny, Nx = shape
    G, resc = _gamma_seq(hc, Ny * Nx)
    nbits = 8 * len(body)
    fwd = np.unpackbits(np.frombuffer(body, np.uint8))
    # all-ones tail ends the fill skip; tries are complete, so bounds one sample's overrun
    rev = np.concatenate([np.ascontiguousarray(fwd[::-1]),
                          np.ones(2 * (hc.umax + hc.D + hc.gstar) + 512, np.uint8)])
    out = np.zeros(shape, np.int64)
    pos = _dec_kernel(rev, nbits, Nz, Ny, Nx, hc.D, hc.g0, hc.gstar, hc.umax, M, G, resc,
                      A["Tarr"], A["Larr"], A["d_child"], A["d_sp"], A["d_sl"], A["d_syms"],
                      A["d_root"], A["f_child"], A["f_sp"], A["f_sl"], A["f_syms"], A["f_root"], out)
    if pos < 0 or pos > nbits:
        raise IndexError("truncated hybrid body")
    return out
