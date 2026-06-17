#!/usr/bin/env python3
"""Compress a hyperspectral cube and report compression ratio + PSNR / MSSIM / SAM.

Runs the codec at lossless and a sweep of near-lossless absolute error limits and
prints a quality table. With no --input it uses a local Indian Pines .mat if present,
otherwise a synthetic cube, so it runs from a fresh clone.

    python3 tools/evaluate.py
    python3 tools/evaluate.py --input cube.npy --limits 0 2 4 8 16
    python3 tools/evaluate.py --input scene.mat --mat-key data --transpose 2,0,1
    python3 tools/evaluate.py --peak data            # PSNR vs the cube's actual peak
    python3 tools/evaluate.py --order BI
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.ccsds import Ccsds123, CodecParams, quality_report   # noqa: E402

INDIAN_PINES = ("/home/joshua/Documents/phd_university/code/deepdynamichsicompression"
                "/data/indian_pines/mat/indian_pines.mat")


def load_cube(args):
    """Return (cube [Z, Y, X] int64, label)."""
    if args.input:
        if args.input.endswith(".npy"):
            cube = np.load(args.input)
        elif args.input.endswith(".mat"):
            import scipy.io as sio
            cube = np.asarray(sio.loadmat(args.input)[args.mat_key])
        else:
            raise SystemExit("input must be a .npy or .mat file")
        if args.transpose:
            cube = np.transpose(cube, tuple(int(i) for i in args.transpose.split(",")))
        return cube.astype(np.int64), os.path.basename(args.input)
    if os.path.exists(INDIAN_PINES):
        import scipy.io as sio
        cube = np.transpose(sio.loadmat(INDIAN_PINES)["indian_pines"], (2, 0, 1))
        return cube.astype(np.int64), "Indian Pines"
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from synthetic_hsi import make_synthetic_hsi
    cube = make_synthetic_hsi(num_bands=args.bands, height=args.height, width=args.width)
    return cube.astype(np.int64), "synthetic"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="input cube (.npy or .mat); default: Indian Pines or synthetic")
    ap.add_argument("--mat-key", default="indian_pines", help="variable name inside a .mat file")
    ap.add_argument("--transpose", help="axis order to apply, e.g. 2,0,1 to make (Z,Y,X)")
    ap.add_argument("--limits", type=int, nargs="+", default=[0, 2, 4, 8, 16],
                    help="absolute error limits to test (0 = lossless)")
    ap.add_argument("--dynamic-range", type=int, default=16, help="sample bit depth D")
    ap.add_argument("--order", choices=["BSQ", "BI"], default="BSQ", help="encoding order")
    ap.add_argument("--pred-bands", type=int, default=3, help="number of prediction bands P (0..15)")
    ap.add_argument("--entropy", choices=["sample_adaptive", "hybrid"],
                    default="sample_adaptive", help="entropy coder (hybrid needs BSQ order)")
    ap.add_argument("--peak", choices=["full", "data"], default="full",
                    help="PSNR peak: full = 2^D-1 (default), data = the cube's actual maximum")
    ap.add_argument("--bands", type=int, default=100, help="synthetic-fallback band count")
    ap.add_argument("--height", type=int, default=128, help="synthetic-fallback height")
    ap.add_argument("--width", type=int, default=128, help="synthetic-fallback width")
    args = ap.parse_args()

    cube, label = load_cube(args)
    if cube.ndim != 3:
        raise SystemExit(f"expected a 3-D [Z, Y, X] cube, got shape {cube.shape}")
    Nz, Ny, Nx = cube.shape
    D = args.dynamic_range
    peak = int(cube.max()) if args.peak == "data" else (1 << D) - 1
    raw_bits = Nz * Ny * Nx * D

    print(f"{label}: (Z,Y,X)={cube.shape}  values [{cube.min()}, {cube.max()}]  "
          f"D={D}  P={args.pred_bands}  order={args.order}  PSNR peak={peak}\n")
    print(f"{'config':18s} {'ratio':>8s} {'bpppb':>6s} {'maxerr':>6s} "
          f"{'PSNR dB':>8s} {'MSSIM':>8s} {'SAM rad':>9s} {'enc/dec s':>11s}")
    for a in args.limits:
        kw = {} if a == 0 else {"absolute_error_limit": a}
        codec = Ccsds123(CodecParams(num_bands=Nz, height=Ny, width=Nx, dynamic_range=D,
                                     num_prediction_bands=args.pred_bands,
                                     entropy_coder=args.entropy, encoding_order=args.order, **kw))
        t0 = time.time(); blob = codec.compress(cube); t1 = time.time()
        recon = codec.decompress(blob); t2 = time.time()
        r = quality_report(cube, recon, D, peak=peak)
        ratio = raw_bits / (len(blob) * 8)
        bpppb = len(blob) * 8 / (Nz * Ny * Nx)
        psnr = "inf" if np.isinf(r["psnr_db"]) else f"{r['psnr_db']:.2f}"
        name = "lossless" if a == 0 else f"near-lossless a={a}"
        print(f"{name:18s} {ratio:6.3f}:1 {bpppb:6.3f} {r['max_abs_error']:6.0f} "
              f"{psnr:>8s} {r['mssim']:8.5f} {r['sam_rad']:9.5f} {t1 - t0:5.1f}/{t2 - t1:<4.1f}")


if __name__ == "__main__":
    main()
