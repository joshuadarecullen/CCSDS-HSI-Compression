"""
Bit-exact CCSDS-123.0-B-2 compressed-image header (section 5.3).

`pack_header(params)` -> header bytes; `parse_header(data)` -> (params, n_bytes).
Field widths follow 5.3 exactly. Covers the parts this codec emits:

  * Image Metadata, Essential subpart (table 5-3)
  * Predictor Metadata, Primary (table 5-6) + Quantization (tables 5-8..5-11,
    near-lossless) + Sample Representative (table 5-12, Theta > 0)
  * Entropy Coder Metadata (table 5-13 sample-adaptive / 5-14 hybrid)

Out of scope: supplementary tables, custom weight init/exponent tables, periodic
error-limit updating, BI order.
"""

from __future__ import annotations

from typing import Dict, Tuple

_LST = {"wide_neighbor": 0, "narrow_neighbor": 1, "wide_column": 2, "narrow_column": 3}
_LST_INV = {v: k for k, v in _LST.items()}


class _BitPacker:
    def __init__(self) -> None:
        self.bits = bytearray()

    def w(self, value: int, n: int) -> None:
        for i in range(n - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def align(self) -> None:
        while len(self.bits) % 8:
            self.bits.append(0)

    def to_bytes(self) -> bytes:
        assert len(self.bits) % 8 == 0, "header not byte-aligned"
        out = bytearray()
        for i in range(0, len(self.bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.bits[i + j]
            out.append(byte)
        return bytes(out)


class _BitUnpacker:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def r(self, n: int) -> int:
        v = 0
        for _ in range(n):
            bit = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
            v = (v << 1) | bit
            self.pos += 1
        return v

    def align(self) -> None:
        while self.pos % 8:
            self.pos += 1

    @property
    def nbytes(self) -> int:
        return self.pos // 8


def pack_header(p) -> bytes:
    """Pack a CodecParams-like object into a bit-exact CCSDS 5.3 header."""
    if p.zeta_inter or p.zeta_intra:
        raise NotImplementedError("non-zero weight-exponent offsets need the Weight Tables subpart")
    D = p.dynamic_range
    a, r = p.absolute_error_limit, p.relative_error_limit
    au, ru, dep_a, dep_r, DA, DR = p.fidelity_layout()
    fc = 0 if p.lossless else (3 if (au and ru) else 1 if au else 2)
    bi = getattr(p, "encoding_order", "BSQ") == "BI"
    periodic = getattr(p, "periodic", False)

    bp = _BitPacker()
    # Image Metadata, Essential (table 5-3)
    bp.w(0, 8)                                  # User-Defined Data
    bp.w(p.width & 0xFFFF, 16)                  # X Size
    bp.w(p.height & 0xFFFF, 16)                 # Y Size
    bp.w(p.num_bands & 0xFFFF, 16)              # Z Size
    bp.w(1 if p.signed else 0, 1)               # Sample Type
    bp.w(0, 1)                                  # Reserved
    bp.w(1 if D > 16 else 0, 1)                 # Large Dynamic Range Flag
    bp.w(D & 0xF, 4)                            # Dynamic Range (D mod 16)
    bp.w(0 if bi else 1, 1)                     # Sample Encoding Order: 0=BI, 1=BSQ
    M = (p.interleave_depth or p.num_bands) if bi else 0
    bp.w(M & 0xFFFF, 16)                         # Sub-Frame Interleaving Depth (M for BI)
    bp.w(0, 2)                                  # Reserved
    bp.w(1 & 0x7, 3)                            # Output Word Size B=1 (byte-aligned body)
    bp.w(1 if getattr(p, "entropy_coder", "sample_adaptive") == "hybrid" else 0, 2)  # Entropy Coder Type
    bp.w(0, 1)                                  # Reserved
    bp.w(fc, 2)                                 # Quantizer Fidelity Control Method
    bp.w(0, 2)                                  # Reserved
    bp.w(0, 4)                                  # Supplementary Information Table Count (tau=0)

    # Predictor Metadata, Primary (table 5-6)
    t_inc_log = p.t_inc.bit_length() - 1        # log2(t_inc)
    bp.w(0, 1)                                  # Reserved
    bp.w(1 if p.theta > 0 else 0, 1)            # Sample Representative Flag
    bp.w(p.num_prediction_bands, 4)             # Number of Prediction Bands P
    bp.w(0 if p.full else 1, 1)                 # Prediction Mode
    bp.w(0, 1)                                  # Weight Exponent Offset Flag
    bp.w(_LST[p.local_sum_type], 2)             # Local Sum Type
    bp.w(p.register_size & 0x3F, 6)             # Register Size (R mod 64)
    bp.w((p.omega - 4) & 0xF, 4)                # Weight Component Resolution
    bp.w((t_inc_log - 4) & 0xF, 4)              # Weight Update Change Interval
    bp.w((p.v_min + 6) & 0xF, 4)                # Weight Update Initial Parameter
    bp.w((p.v_max + 6) & 0xF, 4)                # Weight Update Final Parameter
    bp.w(0, 1)                                  # Weight Exponent Offset Table Flag
    bp.w(0, 1)                                  # Weight Initialization Method (default)
    bp.w(0, 1)                                  # Weight Initialization Table Flag
    bp.w(0, 5)                                  # Weight Initialization Resolution

    # Predictor Metadata, Quantization subpart (near-lossless only)
    if not p.lossless:
        if bi:                                      # Error Limit Update Period block (table 5-9)
            bp.w(0, 1)                              # Reserved
            bp.w(1 if periodic else 0, 1)          # Periodic Updating Flag
            bp.w(0, 2)                             # Reserved
            bp.w((p.update_period_exp if periodic else 0) & 0xF, 4)   # update period exponent u
        if au:                                      # Absolute Error Limit block (table 5-10)
            bp.w(0, 1); bp.w(1 if dep_a else 0, 1); bp.w(0, 2); bp.w(DA & 0xF, 4)
            if not periodic:                        # values in header (else carried in the body)
                for v in (a if dep_a else [a]):
                    bp.w(int(v), DA)
                bp.align()
        if ru:                                      # Relative Error Limit block (table 5-11)
            bp.w(0, 1); bp.w(1 if dep_r else 0, 1); bp.w(0, 2); bp.w(DR & 0xF, 4)
            if not periodic:
                for v in (r if dep_r else [r]):
                    bp.w(int(v), DR)
                bp.align()

    # Predictor Metadata, Sample Representative subpart (Theta > 0)
    if p.theta > 0:
        bp.w(0, 5); bp.w(p.theta, 3)                                    # Reserved + Theta
        bp.w(0, 1); bp.w(0, 1); bp.w(0, 1); bp.w(0, 1); bp.w(p.phi, 4)  # damping (band-indep)
        bp.w(0, 1); bp.w(0, 1); bp.w(0, 1); bp.w(0, 1); bp.w(p.psi, 4)  # offset (band-indep)

    # Entropy Coder Metadata (table 5-13 sample-adaptive / 5-14 hybrid)
    bp.w(p.u_max & 0x1F, 5)                     # Unary Length Limit (Umax mod 32)
    bp.w((p.gamma_star - 4) & 0x7, 3)           # Rescaling Counter Size
    bp.w(p.gamma0 & 0x7, 3)                     # Initial Count Exponent
    if getattr(p, "entropy_coder", "sample_adaptive") == "hybrid":
        bp.w(0, 5)                              # Reserved (table 5-14)
    else:
        bp.w(p.k_init & 0xF, 4)                 # Accumulator Initialization Constant K
        bp.w(0, 1)                              # Accumulator Initialization Table Flag

    bp.align()
    return bp.to_bytes()


def parse_header(data: bytes) -> Tuple[Dict, int]:
    """Parse a CCSDS 5.3 header. Returns (CodecParams kwargs, header length in bytes)."""
    u = _BitUnpacker(data)
    # Image Metadata, Essential
    u.r(8)                                      # User-Defined Data
    Nx = u.r(16); Ny = u.r(16); Nz = u.r(16)
    signed = bool(u.r(1)); u.r(1)
    large = u.r(1); drange = u.r(4)
    if large:
        D = drange + 16 if drange != 0 else 32
    else:
        D = drange if drange != 0 else 16
    order_bit = u.r(1)                          # Sample Encoding Order: 0=BI, 1=BSQ
    M_field = u.r(16)                           # Sub-Frame Interleaving Depth
    u.r(2); u.r(3); ect = u.r(2); u.r(1)        # Reserved, Output Word Size, Entropy Coder Type, Reserved
    fc = u.r(2)                                 # Quantizer Fidelity Control Method
    u.r(2); u.r(4)                              # Reserved, Supplementary Information Table Count

    # Predictor Metadata, Primary
    u.r(1)                                      # Reserved
    sample_rep_flag = u.r(1)
    P = u.r(4)
    full = (u.r(1) == 0)
    u.r(1)                                      # Weight Exponent Offset Flag
    lst = _LST_INV[u.r(2)]
    R = u.r(6); R = R if R != 0 else 64
    omega = u.r(4) + 4
    t_inc = 1 << (u.r(4) + 4)
    v_min = u.r(4) - 6
    v_max = u.r(4) - 6
    u.r(1); u.r(1); u.r(1); u.r(5)             # weight exponent/init table flags + init resolution

    # Quantization subpart
    bi = (order_bit == 0)
    periodic = False
    u_exp = -1
    abs_bits = rel_bits = 0
    abs_lim, rel_lim = 0, 0
    if fc != 0:
        if bi:                                  # Error Limit Update Period block (table 5-9)
            u.r(1); pflag = u.r(1); u.r(2); uu = u.r(4)
            if pflag:
                periodic = True
                u_exp = uu
        nper = (Ny + (1 << u_exp) - 1) >> u_exp if periodic else 0
        if fc in (1, 3):
            u.r(1); band_dep = u.r(1); u.r(2); DA = u.r(4)
            DA = DA if DA != 0 else 16
            if periodic:                        # values carried in the body, not the header
                abs_bits = DA
                abs_lim = [[0] * Nz for _ in range(nper)] if band_dep else [0] * nper
            else:
                abs_lim = [u.r(DA) for _ in range(Nz)] if band_dep else u.r(DA)
                u.align()
        if fc in (2, 3):
            u.r(1); band_dep = u.r(1); u.r(2); DR = u.r(4)
            DR = DR if DR != 0 else 16
            if periodic:
                rel_bits = DR
                rel_lim = [[0] * Nz for _ in range(nper)] if band_dep else [0] * nper
            else:
                rel_lim = [u.r(DR) for _ in range(Nz)] if band_dep else u.r(DR)
                u.align()

    # Sample Representative subpart
    theta, phi, psi = 0, 0, 0
    if sample_rep_flag:
        u.r(5); theta = u.r(3)
        u.r(1); u.r(1); u.r(1); u.r(1); phi = u.r(4)
        u.r(1); u.r(1); u.r(1); u.r(1); psi = u.r(4)

    # Entropy Coder Metadata
    u_max = u.r(5); u_max = u_max if u_max >= 8 else 32
    gamma_star = u.r(3) + 4
    gamma0 = u.r(3); gamma0 = gamma0 if gamma0 != 0 else 8
    if ect == 1:                                # hybrid (table 5-14)
        u.r(5)                                  # Reserved
        k_init = 3                              # unused by the hybrid coder
        entropy_coder = "hybrid"
    else:                                       # sample-adaptive (table 5-13)
        k_init = u.r(4)
        u.r(1)                                  # Accumulator Initialization Table Flag
        entropy_coder = "sample_adaptive"

    u.align()
    params = dict(
        num_bands=Nz, height=Ny, width=Nx, dynamic_range=D, signed=signed,
        num_prediction_bands=P, full=full, local_sum_type=lst, omega=omega,
        register_size=R, theta=theta, phi=phi, psi=psi,
        absolute_error_limit=abs_lim, relative_error_limit=rel_lim,
        v_min=v_min, v_max=v_max, t_inc=t_inc,
        gamma0=gamma0, gamma_star=gamma_star, u_max=u_max, k_init=k_init,
        entropy_coder=entropy_coder,
        encoding_order=("BI" if bi else "BSQ"),
        interleave_depth=(M_field if bi else 0),
        update_period_exp=u_exp, abs_bits=abs_bits, rel_bits=rel_bits,
    )
    return params, u.nbytes
