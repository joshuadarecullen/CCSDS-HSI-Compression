"""
Bit-exact CCSDS-123.0-B-2 compressed-image header (section 5.3).

`pack_header(params)` -> header bytes; `parse_header(data)` -> (params, n_bytes).
Field widths follow 5.3 exactly. Covers the parts this codec emits:

  * Image Metadata, Essential subpart (table 5-3) + Supplementary Information
    Tables (table 5-4)
  * Predictor Metadata, Primary (table 5-6) + Weight Tables (table 5-7: custom
    weight initialization and weight exponent offsets) + Quantization (tables
    5-8..5-11, near-lossless) + Sample Representative (table 5-12, Theta > 0)
  * Entropy Coder Metadata (table 5-13 sample-adaptive / 5-14 hybrid)

Out of scope: the block-adaptive entropy coder and streams whose weight exponent
offsets, accumulator init or damping/offset values are mission-defined rather
than carried in a header table; parse_header rejects both.
"""

from __future__ import annotations

from typing import Dict, Tuple

_LST = {"wide_neighbor": 0, "narrow_neighbor": 1, "wide_column": 2, "narrow_column": 3}
_LST_INV = {v: k for k, v in _LST.items()}
_TT = {"unsigned": 0, "signed": 1, "float": 2}              # supplementary Table Type
_TT_INV = {v: k for k, v in _TT.items()}
_TS = {"0d": 0, "1d": 1, "zx": 2, "yx": 3}                  # supplementary Table Structure
_TS_INV = {v: k for k, v in _TS.items()}


def _s2c(v: int, n: int) -> int:
    """n-bit two's complement -> signed int."""
    return v - (1 << n) if v >= (1 << (n - 1)) else v


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
    D = p.dynamic_range
    a, r = p.absolute_error_limit, p.relative_error_limit
    au, ru, dep_a, dep_r, DA, DR = p.fidelity_layout()
    fc = 0 if p.lossless else (3 if (au and ru) else 1 if au else 2)
    bi = getattr(p, "encoding_order", "BSQ") == "BI"
    periodic = getattr(p, "periodic", False)

    bp = _BitPacker()
    # Image Metadata, Essential (table 5-3)
    bp.w(getattr(p, "user_data", 0) & 0xFF, 8)  # User-Defined Data
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
    bp.w(getattr(p, "output_word_size", 1) & 0x7, 3)   # Output Word Size (B mod 8)
    bp.w(1 if getattr(p, "entropy_coder", "sample_adaptive") == "hybrid" else 0, 2)  # Entropy Coder Type
    bp.w(0, 1)                                  # Reserved
    bp.w(fc, 2)                                 # Quantizer Fidelity Control Method
    tables = getattr(p, "supplementary_tables", None) or []
    bp.w(0, 2)                                  # Reserved
    bp.w(len(tables) & 0xF, 4)                  # Supplementary Information Table Count tau

    # Image Metadata, Supplementary Information Tables (tables 5-4, 5.3.2.3)
    for t in tables:
        bp.w(_TT[t["type"]], 2); bp.w(0, 2); bp.w(t["purpose"], 4)
        bp.w(0, 1); bp.w(_TS[t["structure"]], 2); bp.w(0, 1); bp.w(t.get("user", 0), 4)
        if t["type"] == "float":                # data subblock (5.3.2.3.2.3)
            DF, DE = t["df"], t["de"]
            bp.w(DF, 5); bp.w(DE & 0x7, 3); bp.w(t["bias"], DE)
            for b, alpha, j in t["data"]:
                bp.w(b, 1); bp.w(alpha, DE); bp.w(j, DF)
        else:                                   # integer data subblock (5.3.2.3.2.2)
            DI = t["bit_depth"]
            bp.w(DI & 0x1F, 5)
            for v in t["data"]:
                bp.w(int(v) & ((1 << DI) - 1), DI)
        bp.align()

    # Predictor Metadata, Primary (table 5-6)
    t_inc_log = p.t_inc.bit_length() - 1        # log2(t_inc)
    bp.w(0, 1)                                  # Reserved
    bp.w(1 if p.theta > 0 else 0, 1)            # Sample Representative Flag
    bp.w(p.num_prediction_bands, 4)             # Number of Prediction Bands P
    bp.w(0 if p.full else 1, 1)                 # Prediction Mode
    custom_z = getattr(p, "weight_exp_offset", None) is not None
    bp.w(1 if custom_z else 0, 1)               # Weight Exponent Offset Flag
    bp.w(_LST[p.local_sum_type], 2)             # Local Sum Type
    bp.w(p.register_size & 0x3F, 6)             # Register Size (R mod 64)
    bp.w((p.omega - 4) & 0xF, 4)                # Weight Component Resolution
    bp.w((t_inc_log - 4) & 0xF, 4)              # Weight Update Change Interval
    bp.w((p.v_min + 6) & 0xF, 4)                # Weight Update Initial Parameter
    bp.w((p.v_max + 6) & 0xF, 4)                # Weight Update Final Parameter
    custom_w = getattr(p, "weight_init", None) is not None
    bp.w(1 if custom_z else 0, 1)               # Weight Exponent Offset Table Flag
    bp.w(1 if custom_w else 0, 1)               # Weight Initialization Method
    bp.w(1 if custom_w else 0, 1)               # Weight Initialization Table Flag
    bp.w(p.weight_init_resolution if custom_w else 0, 5)   # Weight Initialization Resolution Q

    # Predictor Metadata, Weight Tables subpart (table 5-7): custom weight
    # initialization vectors Lambda_z as Q-bit two's complement, z-major (5.3.3.3.2)
    if custom_w:
        Q = p.weight_init_resolution
        for row in p.weight_init:
            for v in row:
                bp.w(int(v) & ((1 << Q) - 1), Q)
        bp.align()
    if custom_z:                                # Weight Exponent Offset Table (5.3.3.3.3):
        for row in p.weight_exp_offset:         # [zeta*_z (full)] + zeta_z^(i), 4-bit two's comp
            for v in row:
                bp.w(int(v) & 0xF, 4)
        bp.align()

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

    # Predictor Metadata, Sample Representative subpart (table 5-12)
    if p.theta > 0:
        bp.w(0, 5); bp.w(p.theta, 3)                                    # Reserved + Theta
        for f in (p.phi, p.psi):                # damping then offset: flags + fixed value
            bv = isinstance(f, list)
            bp.w(0, 1); bp.w(1 if bv else 0, 1); bp.w(1 if bv else 0, 1); bp.w(0, 1)
            bp.w(0 if bv else f, 4)
        for f in (p.phi, p.psi):                # Damping/Offset Table subblocks (5.3.3.5.2/3)
            if isinstance(f, list):
                for v in f:
                    bp.w(int(v), p.theta)
                bp.align()

    # Entropy Coder Metadata (table 5-13 sample-adaptive / 5-14 hybrid)
    bp.w(p.u_max & 0x1F, 5)                     # Unary Length Limit (Umax mod 32)
    bp.w((p.gamma_star - 4) & 0x7, 3)           # Rescaling Counter Size
    bp.w(p.gamma0 & 0x7, 3)                     # Initial Count Exponent
    if getattr(p, "entropy_coder", "sample_adaptive") == "hybrid":
        bp.w(0, 5)                              # Reserved (table 5-14)
    else:
        table = isinstance(p.k_init, list)      # Accumulator Initialization Table (5.3.4.2.2)
        bp.w(15 if table else p.k_init & 0xF, 4)   # Accumulator Initialization Constant K
        bp.w(1 if table else 0, 1)              # Accumulator Initialization Table Flag
        if table:
            for k in p.k_init:
                bp.w(int(k) & 0xF, 4)

    bp.align()
    return bp.to_bytes()


def parse_header(data: bytes) -> Tuple[Dict, int]:
    """Parse a CCSDS 5.3 header. Returns (CodecParams kwargs, header length in bytes)."""
    u = _BitUnpacker(data)
    # Image Metadata, Essential
    user_data = u.r(8)                          # User-Defined Data
    # sizes are stored mod 2^16 (table 5-3); a field value of 0 means 65536
    Nx = u.r(16) or (1 << 16)
    Ny = u.r(16) or (1 << 16)
    Nz = u.r(16) or (1 << 16)
    signed = bool(u.r(1)); u.r(1)
    large = u.r(1); drange = u.r(4)
    if large:
        D = drange + 16 if drange != 0 else 32
    else:
        D = drange if drange != 0 else 16
    order_bit = u.r(1)                          # Sample Encoding Order: 0=BI, 1=BSQ
    M_field = u.r(16)                           # Sub-Frame Interleaving Depth
    u.r(2); B = u.r(3) or 8; ect = u.r(2); u.r(1)   # Reserved, Output Word Size, Entropy Coder Type, Reserved
    if ect > 1:                                 # '10' block-adaptive / '11' reserved (table 5-3)
        raise ValueError(f"unsupported Entropy Coder Type {ect} "
                         "(only sample-adaptive (0) and hybrid (1) are implemented)")
    fc = u.r(2)                                 # Quantizer Fidelity Control Method
    u.r(2); tau = u.r(4)                        # Reserved, Supplementary Information Table Count

    # Image Metadata, Supplementary Information Tables (table 5-4)
    tables = []
    for _ in range(tau):
        tt = u.r(2)
        if tt > 2:
            raise ValueError("reserved supplementary table type '11'")
        ttype = _TT_INV[tt]
        u.r(2); purpose = u.r(4)
        u.r(1); struct = _TS_INV[u.r(2)]; u.r(1); user = u.r(4)
        n = {"0d": 1, "1d": Nz, "zx": Nz * Nx, "yx": Ny * Nx}[struct]
        t = dict(type=ttype, purpose=purpose, structure=struct, user=user)
        if ttype == "float":
            DF = u.r(5); DE = u.r(3) or 8
            t["df"], t["de"], t["bias"] = DF, DE, u.r(DE)
            t["data"] = [(u.r(1), u.r(DE), u.r(DF)) for _ in range(n)]
        else:
            DI = u.r(5) or 32
            t["bit_depth"] = DI
            t["data"] = [u.r(DI) for _ in range(n)]
            if ttype == "signed":
                t["data"] = [_s2c(v, DI) for v in t["data"]]
        u.align()
        tables.append(t)

    # Predictor Metadata, Primary
    u.r(1)                                      # Reserved
    sample_rep_flag = u.r(1)
    P = u.r(4)
    full = (u.r(1) == 0)
    offset_flag = u.r(1)                        # Weight Exponent Offset Flag
    lst = _LST_INV[u.r(2)]
    R = u.r(6); R = R if R != 0 else 64
    omega = u.r(4) + 4
    t_inc = 1 << (u.r(4) + 4)
    v_min = u.r(4) - 6
    v_max = u.r(4) - 6
    offset_table_flag = u.r(1)                  # Weight Exponent Offset Table Flag
    custom_w = u.r(1) == 1                      # Weight Initialization Method
    w_table_flag = u.r(1)                       # Weight Initialization Table Flag
    Q = u.r(5)                                  # Weight Initialization Resolution
    if offset_flag and not offset_table_flag:
        raise NotImplementedError(
            "nonzero weight exponent offsets without a Weight Exponent Offset Table "
            "cannot be decoded (offsets are mission-defined)")
    if offset_table_flag and not offset_flag:
        raise ValueError("Weight Exponent Offset Table present without the offset flag")

    # Weight Tables subpart (custom weight initialization vectors)
    weight_init = None
    if custom_w:
        if not w_table_flag:
            raise ValueError("custom weight initialization without a Weight Initialization "
                             "Table in the header cannot be decoded (vectors are mission-defined)")
        weight_init = []
        for z in range(Nz):
            cz = (3 if full else 0) + min(z, P)
            weight_init.append([_s2c(u.r(Q), Q) for _ in range(cz)])
        u.align()

    # Weight Exponent Offset Table (5.3.3.3.3): [zeta*_z (full)] + zeta_z^(1..min(z,P))
    weight_exp_offset = None
    if offset_table_flag:
        weight_exp_offset = []
        for z in range(Nz):
            n = (1 if full else 0) + min(z, P)
            weight_exp_offset.append([_s2c(u.r(4), 4) for _ in range(n)])
        u.align()

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

    # Sample Representative subpart (table 5-12)
    theta, phi, psi = 0, 0, 0
    if sample_rep_flag:
        u.r(5); theta = u.r(3)
        u.r(1); bv_phi = u.r(1); tf_phi = u.r(1); u.r(1); phi = u.r(4)
        u.r(1); bv_psi = u.r(1); tf_psi = u.r(1); u.r(1); psi = u.r(4)
        for tag, bv, tf in (("damping", bv_phi, tf_phi), ("offset", bv_psi, tf_psi)):
            if tf and not bv:
                raise ValueError(f"{tag} table present without the band-varying flag")
            if bv and not tf:
                raise NotImplementedError(
                    f"band-varying {tag} without a table cannot be decoded "
                    "(values are mission-defined)")
        if tf_phi:                              # Damping Table subblock (5.3.3.5.2)
            phi = [u.r(theta) for _ in range(Nz)]
            u.align()
        if tf_psi:                              # Offset Table subblock (5.3.3.5.3)
            psi = [u.r(theta) for _ in range(Nz)]
            u.align()

    # Entropy Coder Metadata
    u_max = u.r(5); u_max = u_max if u_max >= 8 else 32
    gamma_star = u.r(3) + 4
    gamma0 = u.r(3); gamma0 = gamma0 if gamma0 != 0 else 8
    if ect == 1:                                # hybrid (table 5-14)
        u.r(5)                                  # Reserved
        k_init = 0                              # unused by the hybrid coder; 0 validates for any D
        entropy_coder = "hybrid"
    else:                                       # sample-adaptive (table 5-13)
        k_init = u.r(4)
        if u.r(1):                              # Accumulator Initialization Table Flag
            if k_init != 15:
                raise ValueError("Accumulator Initialization Table alongside a constant K")
            k_init = [u.r(4) for _ in range(Nz)]
            u.align()
        elif k_init == 15:
            raise NotImplementedError(
                "accumulator initialization without a table cannot be decoded "
                "(per-band values are mission-defined)")
        entropy_coder = "sample_adaptive"

    u.align()
    params = dict(
        num_bands=Nz, height=Ny, width=Nx, dynamic_range=D, signed=signed,
        num_prediction_bands=P, full=full, local_sum_type=lst, omega=omega,
        register_size=R, theta=theta, phi=phi, psi=psi,
        weight_init=weight_init, weight_init_resolution=(Q if custom_w else 0),
        weight_exp_offset=weight_exp_offset, supplementary_tables=(tables or None),
        absolute_error_limit=abs_lim, relative_error_limit=rel_lim,
        abs_limit_used=(fc in (1, 3)), rel_limit_used=(fc in (2, 3)),
        v_min=v_min, v_max=v_max, t_inc=t_inc,
        gamma0=gamma0, gamma_star=gamma_star, u_max=u_max, k_init=k_init,
        entropy_coder=entropy_coder,
        user_data=user_data, output_word_size=B,
        encoding_order=("BI" if bi else "BSQ"),
        interleave_depth=(M_field if bi else 0),
        update_period_exp=u_exp, abs_bits=abs_bits, rel_bits=rel_bits,
    )
    return params, u.nbytes
