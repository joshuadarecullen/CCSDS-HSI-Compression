#!/usr/bin/env python3
"""Benchmark the CCSDS-123.0-B-2 codec across a directory of HSI cubes (zarr).

Each <dataset>/zarr/cube.zarr is a raw integer DN cube stored as [Y, X, Z] float32.
For every cube the codec is run at lossless + a sweep of near-lossless limits, with
the chosen entropy coders, and one CSV row per (dataset, coder, limit) is written:
ratio, bpppb, max error, PSNR, MSSIM, SAM and timing.

Cubes whose values are not integer (e.g. normalised to [0, 1]) are skipped -- lossless
integer compression does not apply to them. Negative integers are shifted to a
non-negative range (lossless, and the quality metrics are shift-invariant).

    python3 examples/benchmark.py --root /srv/data/ml/datasets --out results.csv
    python3 examples/benchmark.py --root ../deepdynamichsicompression/data \
        --datasets indian_pines pavia_university --coders hybrid --limits 0 4 16
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.ccsds import (Ccsds123, CodecParams,                        # noqa: E402
                       calculate_psnr, calculate_mssim, calculate_spectral_angle)


def find_cubes(root):
    """(name, cube.zarr path) for every cube under root, including nested variants."""
    paths = sorted(glob.glob(os.path.join(root, "*", "zarr", "cube.zarr")) +
                   glob.glob(os.path.join(root, "*", "*", "zarr", "cube.zarr")))
    out = []
    for p in paths:
        rel = os.path.relpath(os.path.dirname(os.path.dirname(p)), root)
        out.append((rel.replace(os.sep, "_"), p))
    return out


def load_cube(path):
    """Return (cube [Z,Y,X] int64, D) for an integer cube, or (None, reason)."""
    import zarr
    a = np.asarray(zarr.open(path, mode="r"))                # [Y, X, Z] float
    if not np.array_equal(a, np.rint(a)):
        return None, "non-integer values (normalised float)"
    a = np.rint(a).astype(np.int64).transpose(2, 0, 1)       # -> [Z, Y, X]
    a = a - int(a.min())                                     # shift to non-negative (lossless)
    D = max(2, int(a.max()).bit_length())                   # bits needed for the range
    return (a, D), None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir containing <dataset>/zarr/cube.zarr")
    ap.add_argument("--out", default="benchmark.csv", help="output CSV path")
    ap.add_argument("--datasets", nargs="+", help="only these dataset names")
    ap.add_argument("--coders", nargs="+", default=["sample_adaptive", "hybrid"],
                    choices=["sample_adaptive", "hybrid"])
    ap.add_argument("--limits", type=int, nargs="+", default=[0, 2, 4, 8, 16],
                    help="absolute error limits (0 = lossless)")
    ap.add_argument("--pred-bands", type=int, default=15)
    ap.add_argument("--max-samples", type=float, default=None,
                    help="skip cubes with more than this many samples (e.g. 2e8)")
    ap.add_argument("--no-mssim", action="store_true",
                    help="skip MSSIM (the slow metric) -- useful for very large cubes")
    args = ap.parse_args()

    cubes = find_cubes(args.root)
    if args.datasets:
        cubes = [(n, p) for n, p in cubes if n in args.datasets]
    if not cubes:
        raise SystemExit(f"no cube.zarr found under {args.root}")

    fields = ["dataset", "bands", "height", "width", "samples", "D", "entropy", "limit",
              "ratio", "bpppb", "max_err", "psnr_db", "mssim", "sam_rad", "enc_s", "dec_s"]
    skipped, n_rows = [], 0
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name, path in cubes:
            loaded, reason = load_cube(path)
            if loaded is None:
                print(f"[skip] {name}: {reason}", flush=True)
                skipped.append((name, reason))
                continue
            cube, D = loaded
            Nz, Ny, Nx = cube.shape
            nsamp = Nz * Ny * Nx
            if args.max_samples and nsamp > args.max_samples:
                print(f"[skip] {name}: {nsamp:,} samples > --max-samples", flush=True)
                skipped.append((name, "over --max-samples"))
                continue
            peak = int(cube.max())
            raw_bits = nsamp * D
            print(f"[run]  {name}  shape={cube.shape}  D={D}  ({nsamp:,} samples)", flush=True)
            for coder in args.coders:
                for lim in args.limits:
                    kw = {} if lim == 0 else {"absolute_error_limit": lim}
                    codec = Ccsds123(CodecParams(
                        num_bands=Nz, height=Ny, width=Nx, dynamic_range=D,
                        num_prediction_bands=args.pred_bands, entropy_coder=coder, **kw))
                    t0 = time.time(); blob = codec.compress(cube); t1 = time.time()
                    recon = codec.decompress(blob); t2 = time.time()
                    psnr = calculate_psnr(cube, recon, D, peak=peak)
                    mssim = None if args.no_mssim else calculate_mssim(cube, recon)
                    row = dict(
                        dataset=name, bands=Nz, height=Ny, width=Nx, samples=nsamp, D=D,
                        entropy=coder, limit=lim,
                        ratio=round(raw_bits / (len(blob) * 8), 4),
                        bpppb=round(len(blob) * 8 / nsamp, 4),
                        max_err=int(np.abs(cube - recon).max()),
                        psnr_db=("inf" if np.isinf(psnr) else round(psnr, 3)),
                        mssim=("" if mssim is None else round(mssim, 6)),
                        sam_rad=round(calculate_spectral_angle(cube, recon), 6),
                        enc_s=round(t1 - t0, 2), dec_s=round(t2 - t1, 2))
                    writer.writerow(row); fh.flush(); n_rows += 1
                    print(f"    {coder:15s} a={lim:<3} ratio={row['ratio']:7.3f}  "
                          f"bpppb={row['bpppb']:6.3f}  PSNR={row['psnr_db']}  "
                          f"MSSIM={row['mssim']}  SAM={row['sam_rad']}", flush=True)

    print(f"\nwrote {n_rows} rows to {args.out}")
    if skipped:
        print("skipped:", ", ".join(f"{n} ({r})" for n, r in skipped))


if __name__ == "__main__":
    main()
