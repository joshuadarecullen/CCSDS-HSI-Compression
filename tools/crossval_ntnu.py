"""Cross-validate this encoder against the NTNU CCSDS-123.0-B-2 high-level model.

The NTNU model (github.com/NTNU-SmallSat-Lab/ccsds123_issue_2_verification_model,
MIT) is an independent Python implementation that was itself verified against the
official CCSDS test vector set Test1-20190201, so byte-identical streams here tie
this codec to the official vectors transitively.

For each config: compress with this codec, write the image as a raw BSQ file plus
our packed CCSDS header as the NTNU settings file, run the NTNU compressor, and
compare full bitstreams byte-for-byte. The hybrid coder's initial accumulator is
user-specified by the standard (5.4.3.3.4.3) and not carried in the stream, so it
is passed to the NTNU tool explicitly via --accu to match our choice.

Usage:
    git clone https://github.com/NTNU-SmallSat-Lab/ccsds123_issue_2_verification_model
    pip install bitarray psutil
    python3 tools/crossval_ntnu.py path/to/ccsds123_issue_2_verification_model

Config names must not contain 'x' or '-' (the NTNU tool parses dimensions and
sample format out of the file name).
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))
from ccsds.core.reference_codec import Ccsds123, CodecParams   # noqa: E402
from ccsds.io.ccsds_header import pack_header                  # noqa: E402
from synthetic_hsi import make_synthetic_hsi                   # noqa: E402


def accu_bytes(p: CodecParams) -> bytes:
    """Our hybrid initial accumulators (4 * 2^gamma0 per band, HybridCoder._sigma_init),
    D+gamma0 bits each, MSB-first, zero-padded to a byte."""
    nbits = p.dynamic_range + p.gamma0
    val = min(4 << p.gamma0, (1 << nbits) - 1)
    bits = []
    for _ in range(p.num_bands):
        bits.extend((val >> (nbits - 1 - b)) & 1 for b in range(nbits))
    while len(bits) % 8:
        bits.append(0)
    return bytes(sum(bit << (7 - i) for i, bit in enumerate(bits[k:k + 8]))
                 for k in range(0, len(bits), 8))


def run_one(ntnu: str, work: str, name: str, img: np.ndarray, params: CodecParams) -> str:
    blob = Ccsds123(params).compress(img)
    tag = "s16be" if params.signed else "u16be"
    Nz, Ny, Nx = img.shape
    raw_path = os.path.join(work, f"{name}-{tag}-{Nz}x{Ny}x{Nx}.raw")
    img.astype(np.int64).astype(">i2" if params.signed else ">u2").tofile(raw_path)
    hdr_path = os.path.join(work, f"{name}.hdr.bin")
    with open(hdr_path, "wb") as f:
        f.write(pack_header(params))
    cmd = [sys.executable, "ccsds123_0_b_2_high_level_model.py", raw_path, "--header", hdr_path]
    if params.entropy_coder == "hybrid":
        accu_path = os.path.join(work, f"{name}.accu.bin")
        with open(accu_path, "wb") as f:
            f.write(accu_bytes(params))
        cmd += ["--accu", accu_path]
    r = subprocess.run(cmd, cwd=ntnu, capture_output=True, text=True, timeout=3600)
    out_bin = os.path.join(ntnu, "output", "z-output-bitstream.bin")
    if r.returncode != 0 or not os.path.exists(out_bin):
        print(f"  {name}: NTNU tool failed\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}")
        return "ntnu-error"
    with open(out_bin, "rb") as f:
        theirs = f.read()
    os.remove(out_bin)
    if theirs == blob:
        print(f"  {name}: MATCH ({len(blob)} bytes)")
        return "match"
    n = min(len(blob), len(theirs))
    diff = next((i for i in range(n) if blob[i] != theirs[i]), n)
    print(f"  {name}: MISMATCH ours={len(blob)}B theirs={len(theirs)}B first diff at byte {diff}")
    return "mismatch"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ntnu_repo", help="path to a clone of ccsds123_issue_2_verification_model")
    args = ap.parse_args()
    ntnu = os.path.abspath(args.ntnu_repo)

    img = make_synthetic_hsi(num_bands=5, height=20, width=24).astype(np.int64) >> 2
    base = dict(num_bands=5, height=20, width=24, dynamic_range=14)

    configs = [
        ("lossless_full_wide", img, dict(**base, full=True)),
        ("lossless_reduced_narrow", img, dict(**base, full=False, local_sum_type="narrow_neighbor")),
        ("lossless_widecol", img, dict(**base, local_sum_type="wide_column")),
        ("lossless_narrowcol", img, dict(**base, local_sum_type="narrow_column")),
        ("lossless_P0", img, dict(**base, num_prediction_bands=0)),
        ("lossless_P4_omega19", img, dict(**base, num_prediction_bands=4, omega=19)),
        ("nearlossless_abs2", img, dict(**base, absolute_error_limit=2)),
        ("nearlossless_rel32", img, dict(**base, relative_error_limit=32)),
        ("nearlossless_bands", img, dict(**base, absolute_error_limit=[1, 2, 3, 0, 2],
                                         relative_error_limit=16)),
        ("samplerep_theta2", img, dict(**base, absolute_error_limit=3, theta=2, phi=1, psi=2)),
        ("signed", (img - 4096), dict(**base, signed=True)),
        ("vparams", img, dict(**base, v_min=-2, v_max=5, t_inc=256, gamma0=2, gamma_star=8,
                              u_max=12, k_init=6)),
        ("bi_bil", img, dict(**base, encoding_order="BI", interleave_depth=1)),
        ("bi_bip", img, dict(**base, encoding_order="BI")),
        ("hybrid_lossless", img, dict(**base, entropy_coder="hybrid")),
        ("hybrid_nearlossless", img, dict(**base, entropy_coder="hybrid", absolute_error_limit=2)),
    ]
    # custom weight initialization (Eq 35 + Weight Tables header subpart)
    wrng = np.random.default_rng(3)
    for name, full, Q in [("customw_full_Q6", True, 6), ("customw_reduced_Q4", False, 4),
                          ("customw_full_Q17", True, 17)]:
        p0 = CodecParams(**base, full=full)
        half = 1 << (Q - 1)
        lam = [[int(wrng.integers(-half, half)) for _ in range(p0.band_components(z))]
               for z in range(base["num_bands"])]
        configs.append((name, img, dict(**base, full=full, weight_init=lam,
                                        weight_init_resolution=Q)))

    results = {}
    with tempfile.TemporaryDirectory(prefix="ccsds_crossval_") as work:
        for name, im, kw in configs:
            results[name] = run_one(ntnu, work, name, im, CodecParams(**kw))
    good = sum(1 for v in results.values() if v == "match")
    print(f"\n{good}/{len(results)} configs byte-identical")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
