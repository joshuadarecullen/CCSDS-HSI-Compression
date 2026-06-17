"""
Numba port of `Ccsds123._run` — byte-identical to the pure-Python reference
(verified in the tests), ~100-200x faster. Selected automatically when numba is
installed and the config is int64-safe, otherwise the pure-Python path runs.

All arithmetic is int64; mod*_R is applied only when R<=62 (for the default R=64
it is the identity).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_OK = True
except Exception:  # pragma: no cover
    NUMBA_OK = False

    def njit(*a, **k):  # no-op fallback decorator
        def wrap(f):
            return f
        return wrap if (a and callable(a[0])) is False else a[0]


@njit
def _mod_star(x, R):
    half = 1 << (R - 1)
    m = 1 << R
    r = x % m
    if r >= half:
        r -= m
    return r


@njit
def _lsum(spp, lst, z, y, x, Nx, s_mid):
    if lst == 0:  # wide neighbor-oriented (Eq 20)
        if y > 0 and 0 < x < Nx - 1:
            return spp[z, y, x - 1] + spp[z, y - 1, x - 1] + spp[z, y - 1, x] + spp[z, y - 1, x + 1]
        elif y == 0 and x > 0:
            return 4 * spp[z, y, x - 1]
        elif y > 0 and x == 0:
            return 2 * (spp[z, y - 1, x] + spp[z, y - 1, x + 1])
        elif y > 0 and x == Nx - 1:
            return spp[z, y, x - 1] + spp[z, y - 1, x - 1] + 2 * spp[z, y - 1, x]
        return 0
    elif lst == 1:  # narrow neighbor-oriented (Eq 21)
        if y > 0 and 0 < x < Nx - 1:
            return spp[z, y - 1, x - 1] + 2 * spp[z, y - 1, x] + spp[z, y - 1, x + 1]
        elif y == 0 and x > 0 and z > 0:
            return 4 * spp[z - 1, y, x - 1]
        elif y > 0 and x == 0:
            return 2 * (spp[z, y - 1, x] + spp[z, y - 1, x + 1])
        elif y > 0 and x == Nx - 1:
            return 2 * (spp[z, y - 1, x - 1] + spp[z, y - 1, x])
        elif y == 0 and x > 0 and z == 0:
            return 4 * s_mid
        return 0
    elif lst == 2:  # wide column-oriented (Eq 22)
        if y > 0:
            return 4 * spp[z, y - 1, x]
        return 4 * spp[z, y, x - 1]
    else:           # narrow column-oriented (Eq 23)
        if y > 0:
            return 4 * spp[z, y - 1, x]
        elif x > 0 and z > 0:
            return 4 * spp[z - 1, y, x - 1]
        return 4 * s_mid


@njit
def _kparam(sigma, gamma, D):
    thresh = sigma + ((49 * gamma) >> 7)
    if 2 * gamma > thresh:
        return 0
    k = 0
    kmax = D - 2
    while k < kmax and (gamma << (k + 1)) <= thresh:
        k += 1
    return k


@njit
def _kernel(encode, mode, delta_arr, image, body, buf, spp, recon, cdiff,
            Nz, Ny, Nx, D, P, full, lst, Omega, R, do_mod,
            Theta, phi, psi, abs_lim, rel_lim, lossless,
            vmin, vmax, tinc, zinter, zintra,
            gamma0, gstar_full, sigma_init, umax,
            s_min, s_max, s_mid, w_min, w_max):
    pos = 0
    Cmax = 3 + P
    U = np.zeros(Cmax, dtype=np.int64)
    w = np.zeros(Cmax, dtype=np.int64)
    lo_clip = (1 << (Omega + 2)) * s_min
    hi_clip = (1 << (Omega + 2)) * s_max + (1 << (Omega + 1))

    for z in range(Nz):
        pstar = z if z < P else P
        ci0 = 3 if full == 1 else 0
        Cz = ci0 + pstar
        if full == 1:
            w[0] = 0; w[1] = 0; w[2] = 0
        if pstar > 0:
            w[ci0] = (7 * (1 << Omega)) // 8
            for i in range(1, pstar):
                w[ci0 + i] = w[ci0 + i - 1] // 8
        gamma = 1 << gamma0
        sigma_acc = sigma_init

        for y in range(Ny):
            for x in range(Nx):
                t = y * Nx + x

                if t == 0:
                    if z > 0 and P > 0:
                        s_breve = 2 * spp[z - 1, y, x]
                    else:
                        s_breve = 2 * s_mid
                    s_hat = s_breve >> 1
                    s_tilde = 0
                    sigma = 0
                else:
                    sigma = _lsum(spp, lst, z, y, x, Nx, s_mid)
                    if full == 1:                                   # directional (Eq 25-27)
                        U[0] = (4 * spp[z, y - 1, x] - sigma) if y > 0 else 0
                        if y > 0 and x > 0:
                            U[1] = 4 * spp[z, y, x - 1] - sigma
                            U[2] = 4 * spp[z, y - 1, x - 1] - sigma
                        elif y > 0 and x == 0:
                            U[1] = 4 * spp[z, y - 1, x] - sigma
                            U[2] = 4 * spp[z, y - 1, x] - sigma
                        else:
                            U[1] = 0
                            U[2] = 0
                    for i in range(1, pstar + 1):                  # central diffs (cached)
                        U[ci0 + i - 1] = cdiff[z - i, y, x]
                    dhat = 0
                    for j in range(Cz):
                        dhat += w[j] * U[j]
                    inner = dhat + (1 << Omega) * (sigma - 4 * s_mid)
                    if do_mod == 1:
                        inner = _mod_star(inner, R)
                    s_tilde = inner + (1 << (Omega + 2)) * s_mid + (1 << (Omega + 1))
                    if s_tilde < lo_clip:
                        s_tilde = lo_clip
                    elif s_tilde > hi_clip:
                        s_tilde = hi_clip
                    s_breve = s_tilde >> (Omega + 1)
                    s_hat = s_breve >> 1

                # maximum error (Eq 42-45)
                if lossless == 1:
                    m = 0
                else:
                    a = abs_lim[z]
                    r = rel_lim[z]
                    if r == 0:
                        m = a
                    else:
                        ah = s_hat if s_hat >= 0 else -s_hat
                        rel = (r * ah) >> D
                        if a == 0:
                            m = rel
                        else:
                            m = a if a < rel else rel

                # theta (Eq 56)
                if t == 0 or m == 0:
                    lo = s_hat - s_min
                    hi = s_max - s_hat
                else:
                    step_t = 2 * m + 1
                    lo = (s_hat - s_min + m) // step_t
                    hi = (s_max - s_hat + m) // step_t
                theta = lo if lo < hi else hi

                if encode == 1:
                    d = image[z, y, x] - s_hat
                    if t == 0:
                        q = d
                    else:
                        if d > 0:
                            sgn = 1
                        elif d < 0:
                            sgn = -1
                        else:
                            sgn = 0
                        ad = d if d >= 0 else -d
                        q = sgn * ((ad + m) // (2 * m + 1))
                    aq = q if q >= 0 else -q
                    if aq > theta:
                        delta = aq + theta
                    else:
                        parity = -1 if (s_hat & 1) else 1
                        if parity * q >= 0:
                            delta = 2 * aq
                        else:
                            delta = 2 * aq - 1
                    if mode != 0:                               # delta-collect (hybrid)
                        delta_arr[z, y, x] = delta
                    elif t == 0:
                        for i in range(D - 1, -1, -1):
                            if (delta >> i) & 1:
                                buf[pos >> 3] |= (1 << (7 - (pos & 7)))
                            pos += 1
                    else:
                        k = _kparam(sigma_acc, gamma, D)
                        u = delta >> k
                        if u < umax:
                            pos += u                                   # zeros (buf pre-zeroed)
                            buf[pos >> 3] |= (1 << (7 - (pos & 7)))     # the '1'
                            pos += 1
                            for i in range(k - 1, -1, -1):
                                if (delta >> i) & 1:
                                    buf[pos >> 3] |= (1 << (7 - (pos & 7)))
                                pos += 1
                        else:
                            pos += umax
                            for i in range(D - 1, -1, -1):
                                if (delta >> i) & 1:
                                    buf[pos >> 3] |= (1 << (7 - (pos & 7)))
                                pos += 1
                else:
                    if mode != 0:                               # delta-consume (hybrid)
                        delta = delta_arr[z, y, x]
                    elif t == 0:
                        delta = 0
                        for _ in range(D):
                            b = (body[pos >> 3] >> (7 - (pos & 7))) & 1
                            pos += 1
                            delta = (delta << 1) | b
                    else:
                        k = _kparam(sigma_acc, gamma, D)
                        c = 0
                        got = 0
                        delta = 0
                        while c < umax and got == 0:
                            b = (body[pos >> 3] >> (7 - (pos & 7))) & 1
                            pos += 1
                            if b == 1:
                                rem = 0
                                for _ in range(k):
                                    bb = (body[pos >> 3] >> (7 - (pos & 7))) & 1
                                    pos += 1
                                    rem = (rem << 1) | bb
                                delta = (c << k) | rem
                                got = 1
                            else:
                                c += 1
                        if got == 0:
                            delta = 0
                            for _ in range(D):
                                bb = (body[pos >> 3] >> (7 - (pos & 7))) & 1
                                pos += 1
                                delta = (delta << 1) | bb
                    if delta > 2 * theta:
                        aq = delta - theta
                        q = aq if lo < hi else -aq
                    else:
                        parity = -1 if (s_hat & 1) else 1
                        if (delta & 1) == 0:
                            q = (delta // 2) * parity
                        else:
                            q = -((delta + 1) // 2) * parity

                # reconstruct (Eq 48) and sample representative (Eq 46-47)
                step = 1 if t == 0 else (2 * m + 1)
                s_prime = s_hat + q * step
                if s_prime < s_min:
                    s_prime = s_min
                elif s_prime > s_max:
                    s_prime = s_max
                recon[z, y, x] = s_prime
                if t == 0 or (phi == 0 and psi == 0):
                    spp[z, y, x] = s_prime
                else:
                    sgn_q = 1 if q > 0 else (-1 if q < 0 else 0)
                    num = (4 * ((1 << Theta) - phi)
                           * (s_prime * (1 << Omega) - sgn_q * m * psi * (1 << (Omega - Theta)))
                           + phi * s_tilde - phi * (1 << (Omega + 1)))
                    sbpp = num // (1 << (Omega + Theta + 1))
                    spp[z, y, x] = (sbpp + 1) // 2

                if t > 0:
                    cdiff[z, y, x] = 4 * spp[z, y, x] - sigma
                    e = 2 * s_prime - s_breve                         # weight update (Eq 49-54)
                    rho_t = vmin + (t - Nx) // tinc
                    if rho_t < vmin:
                        rho_t = vmin
                    elif rho_t > vmax:
                        rho_t = vmax
                    rho = rho_t + D - Omega
                    sgn_e = 1 if e >= 0 else -1
                    for j in range(Cz):
                        zeta = zintra if (full == 1 and j < 3) else zinter
                        pw = rho + zeta
                        val = sgn_e * U[j]
                        if pw < 0:
                            inc = ((val << (-pw)) + 1) >> 1
                        else:
                            inc = (val + (1 << pw)) >> (pw + 1)
                        nw = w[j] + inc
                        if nw < w_min:
                            nw = w_min
                        elif nw > w_max:
                            nw = w_max
                        w[j] = nw
                    if mode == 0 and gamma < gstar_full:              # Eq 60/61 (sample-adaptive)
                        sigma_acc += delta
                        gamma += 1
                    elif mode == 0:
                        sigma_acc = (sigma_acc + delta + 1) >> 1
                        gamma = (gamma + 1) >> 1
    return pos


def _limit_array(v, n):
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=np.int64)
    return np.full(n, int(v), dtype=np.int64)


_LST = {"wide_neighbor": 0, "narrow_neighbor": 1, "wide_column": 2, "narrow_column": 3}


def numba_safe(p) -> bool:
    """int64-safe regime that the kernel matches the reference on."""
    return (NUMBA_OK
            and p.dynamic_range <= 24
            and p.omega <= 19
            and (p.register_size == 64 or p.register_size <= 62))


def _setup(codec):
    p = codec.p
    Nz, Ny, Nx, D = p.num_bands, p.height, p.width, p.dynamic_range
    spp = np.zeros((Nz, Ny, Nx), dtype=np.int64)
    recon = np.zeros((Nz, Ny, Nx), dtype=np.int64)
    cdiff = np.zeros((Nz, Ny, Nx), dtype=np.int64)
    abs_lim = _limit_array(p.absolute_error_limit, Nz)
    rel_lim = _limit_array(p.relative_error_limit, Nz)
    kprime = p.k_init if p.k_init <= 30 - D else 2 * p.k_init + D - 30
    sigma_init = ((3 * (1 << (kprime + 6)) - 49) * (1 << p.gamma0)) >> 7
    gstar_full = (1 << p.gamma_star) - 1
    do_mod = 1 if p.register_size <= 62 else 0
    args = (Nz, Ny, Nx, D, p.num_prediction_bands, 1 if p.full else 0, _LST[p.local_sum_type],
            p.omega, p.register_size, do_mod, p.theta, p.phi, p.psi, abs_lim, rel_lim,
            1 if p.lossless else 0, p.v_min, p.v_max, p.t_inc, p.zeta_inter, p.zeta_intra,
            p.gamma0, gstar_full, sigma_init, p.u_max,
            codec.s_min, codec.s_max, codec.s_mid, codec.w_min, codec.w_max)
    return spp, recon, cdiff, args


def run_numba(codec, encode: bool, image=None, body=None):
    p = codec.p
    Nz, Ny, Nx, D = p.num_bands, p.height, p.width, p.dynamic_range
    spp, recon, cdiff, args = _setup(codec)
    none_delta = np.zeros((1, 1, 1), dtype=np.int64)
    if encode:
        img = np.ascontiguousarray(image, dtype=np.int64)
        buf = np.zeros((Nz * Ny * Nx * (p.u_max + D)) // 8 + 64, dtype=np.uint8)
        nbits = _kernel(1, 0, none_delta, img, np.zeros(1, np.uint8), buf, spp, recon, cdiff, *args)
        return bytes(buf[:(nbits + 7) // 8])
    body_arr = np.frombuffer(body, dtype=np.uint8).copy()
    _kernel(0, 0, none_delta, np.zeros((1, 1, 1), np.int64), body_arr,
            np.zeros(1, np.uint8), spp, recon, cdiff, *args)
    return recon


def run_numba_delta(codec, encode: bool, image=None, delta=None):
    """Predictor-only numba pass for the hybrid coder (no entropy I/O):
    encode -> mapped-index array; decode -> reconstruction from a mapped-index array."""
    p = codec.p
    Nz, Ny, Nx = p.num_bands, p.height, p.width
    spp, recon, cdiff, args = _setup(codec)
    dummy_buf = np.zeros(1, dtype=np.uint8)
    if encode:
        img = np.ascontiguousarray(image, dtype=np.int64)
        delta_arr = np.zeros((Nz, Ny, Nx), dtype=np.int64)
        _kernel(1, 1, delta_arr, img, np.zeros(1, np.uint8), dummy_buf, spp, recon, cdiff, *args)
        return delta_arr
    delta_arr = np.ascontiguousarray(delta, dtype=np.int64)
    _kernel(0, 2, delta_arr, np.zeros((1, 1, 1), np.int64), np.zeros(1, np.uint8),
            dummy_buf, spp, recon, cdiff, *args)
    return recon
