#!/usr/bin/env python3
"""Are the sample representatives s'' (4.9) a better estimate of the original than the
reconstruction s'?

In near-lossless mode the decoder outputs s', the centre of the quantizer bin. The
predictor instead runs on s'', which is s' pulled toward the prediction by the
per-band damping phi (blend) and offset psi (shift inside the bin) at resolution
Theta. This script sweeps the absolute error limit and (Theta, phi, psi) and, for
every setting, scores BOTH s' and s'' against the original: PSNR, SAM, max|err|,
and the fraction of samples where s'' is strictly closer than s'. The rate (bpppb)
is reported too, since phi/psi also change the bitstream.

A positive "psnr_gain" column means the representatives are the better input for a
downstream model fed from the codec (denoised, compressed-domain input).

    python3 examples/sample_representatives.py                       # Indian Pines or synthetic
    python3 examples/sample_representatives.py --input scene.mat     # .mat key auto-detected
    python3 examples/sample_representatives.py --input cube.zarr     # bare [Y, X, Z] zarr array
    python3 examples/sample_representatives.py --limits 2 8 32 --theta 4 --phi 0 8 15 --psi 0 8 15
    python3 examples/sample_representatives.py --csv out.csv --json out.json
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.ccsds import CCSDS123, calculate_psnr, calculate_spectral_angle   # noqa: E402

# local real-data defaults (sibling research repo); the first that exists is used
_DATA = os.path.join(os.path.dirname(REPO), "deepdynamichsicompression", "data")
DEFAULT_INPUTS = [
    os.path.join(_DATA, "indian_pines", "zarr", "cube_v2_purr.zarr"),
    os.path.join(_DATA, "indianpines", "Indian_pines.mat"),
]


def load_cube(path, mat_key=None, transpose=None):
    """Return a [Z, Y, X] int64 cube from .npy / .npz / .mat (key auto-detected) or a
    bare .zarr array stored [Y, X, Z] (the sibling repo's cube format)."""
    path = path.rstrip("/")
    if path.endswith(".zarr"):
        import zarr
        cube = np.asarray(zarr.open(path, mode="r")[:])
        transpose = (2, 0, 1) if transpose is None else transpose
    elif path.endswith(".npy"):
        cube = np.load(path)
    elif path.endswith(".npz"):
        z = np.load(path)
        cube = z[mat_key or list(z.keys())[0]]
    elif path.endswith(".mat"):
        import scipy.io as sio
        d = sio.loadmat(path)
        if mat_key is None:
            cands = [k for k, v in d.items()
                     if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3]
            if len(cands) != 1:
                raise SystemExit(f"cannot pick a 3-D variable in {path}: {cands}; use --mat-key")
            mat_key = cands[0]
        cube = np.asarray(d[mat_key])
    else:
        raise SystemExit("input must be a .npy, .npz, .mat file or a .zarr array")
    if transpose is None and path.endswith(".mat"):
        transpose = (2, 0, 1)                       # .mat scenes are stored [Y, X, Z]
    if transpose is not None:
        cube = np.transpose(cube, transpose)
    if cube.ndim != 3:
        raise SystemExit(f"expected a 3-D cube, got shape {cube.shape}")
    return np.ascontiguousarray(np.rint(cube).astype(np.int64))


def score(orig, est, peak):
    """PSNR (dB), SAM (rad), max|err|, MSE between an estimate and the original."""
    x, y = orig.astype(np.float64), est.astype(np.float64)
    return {
        "psnr": calculate_psnr(x, y, None, peak),
        "sam": calculate_spectral_angle(x, y),
        "maxerr": int(np.abs(orig - est).max()),
        "mse": float(np.mean((x - y) ** 2)),
    }


def run(cube, limit, theta, phi, psi, dynamic_range, peak, codec_kwargs):
    """One codec pass. Returns a flat result row for (limit, theta, phi, psi)."""
    Nz, Ny, Nx = cube.shape
    kw = dict(dynamic_range=dynamic_range, absolute_error_limit=limit,
              signed=bool(cube.min() < 0), **codec_kwargs)
    if theta:
        kw.update(theta=theta, phi=phi, psi=psi)
    codec = CCSDS123(num_bands=Nz, height=Ny, width=Nx, **kw)
    t0 = time.time()
    enc = codec.encode(cube)
    secs = time.time() - t0
    rec, rep = enc["reconstruction"], enc["representatives"]
    a, b = score(cube, rec, peak), score(cube, rep, peak)
    d_rec, d_rep = np.abs(cube - rec), np.abs(cube - rep)
    row = {
        "limit": limit, "theta": theta, "phi": phi, "psi": psi,
        "bpppb": len(enc["bitstream"]) * 8 / cube.size,
        "psnr_rec": a["psnr"], "psnr_rep": b["psnr"], "psnr_gain": b["psnr"] - a["psnr"],
        "sam_rec": a["sam"], "sam_rep": b["sam"],
        "mse_rec": a["mse"], "mse_rep": b["mse"],
        "maxerr_rec": a["maxerr"], "maxerr_rep": b["maxerr"],
        "frac_rep_closer": float(np.mean(d_rep < d_rec)),
        "frac_rep_farther": float(np.mean(d_rep > d_rec)),
        "frac_rep_outside_limit": float(np.mean(d_rep > limit)),
        "enc_s": secs,
    }
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="cube (.npy/.npz/.mat); default: Indian Pines if found, "
                                    "else synthetic")
    ap.add_argument("--mat-key", help="variable name inside a .mat/.npz (default: auto)")
    ap.add_argument("--transpose", help="axis order to apply, e.g. 2,0,1 (default for .mat)")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                    help="spatial crop [Y0:Y1, X0:X1] to speed things up")
    ap.add_argument("--limits", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                    help="absolute error limits (near-lossless; 0 is pointless here)")
    ap.add_argument("--theta", type=int, default=4, help="representative resolution Theta (1..4)")
    ap.add_argument("--phi", type=int, nargs="+", default=[0, 4, 8, 12, 15],
                    help="damping values to sweep (0 .. 2^Theta-1)")
    ap.add_argument("--psi", type=int, nargs="+", default=[0, 4, 8, 12, 15],
                    help="offset values to sweep (0 .. 2^Theta-1)")
    ap.add_argument("--dynamic-range", type=int, default=16, help="sample bit depth D")
    ap.add_argument("--pred-bands", type=int, default=3, help="prediction bands P")
    ap.add_argument("--entropy", choices=["sample_adaptive", "hybrid", "block_adaptive"],
                    default="sample_adaptive")
    ap.add_argument("--peak", choices=["full", "data"], default="data",
                    help="PSNR peak: the cube's max (default) or 2^D-1")
    ap.add_argument("--csv", help="write every row to this CSV")
    ap.add_argument("--json", help="write every row (plus metadata) to this JSON")
    ap.add_argument("--bands", type=int, default=60, help="synthetic-fallback band count")
    ap.add_argument("--height", type=int, default=96, help="synthetic-fallback height")
    ap.add_argument("--width", type=int, default=96, help="synthetic-fallback width")
    args = ap.parse_args()

    path = args.input or next((p for p in DEFAULT_INPUTS if os.path.exists(p)), None)
    if path:
        tr = tuple(int(i) for i in args.transpose.split(",")) if args.transpose else None
        cube, label = load_cube(path, args.mat_key, tr), os.path.basename(path)
    else:
        sys.path.insert(0, os.path.join(REPO, "tests"))
        from synthetic_hsi import make_synthetic_hsi
        cube, label = make_synthetic_hsi(args.bands, args.height, args.width), "synthetic"
    if args.crop:
        y0, y1, x0, x1 = args.crop
        cube = np.ascontiguousarray(cube[:, y0:y1, x0:x1])
    D = args.dynamic_range
    lo, hi = int(cube.min()), int(cube.max())
    smin, smax = (-(1 << (D - 1)), (1 << (D - 1)) - 1) if lo < 0 else (0, (1 << D) - 1)
    if lo < smin or hi > smax:
        raise SystemExit(f"values [{lo}, {hi}] exceed D={D} ({'signed' if lo < 0 else 'unsigned'})"
                         f"; use --dynamic-range")
    peak = (hi - min(lo, 0)) if args.peak == "data" else (1 << D) - 1
    codec_kwargs = dict(num_prediction_bands=args.pred_bands, entropy_coder=args.entropy)
    vmax = (1 << args.theta) - 1
    for v in args.phi + args.psi:
        if not 0 <= v <= vmax:
            raise SystemExit(f"phi/psi must lie in 0..{vmax} for Theta={args.theta}")

    print(f"{label}: (Z,Y,X)={cube.shape}  values [{cube.min()}, {cube.max()}]  D={D}  "
          f"PSNR peak={peak}  Theta={args.theta}  P={args.pred_bands}  coder={args.entropy}")
    hdr = (f"{'a':>3} {'phi':>3} {'psi':>3} {'bpppb':>6} | {'PSNR s_rec':>10} {'PSNR s_rep':>10} "
           f"{'gain':>6} | {'SAM rec':>8} {'SAM rep':>8} | {'max rec':>7} {'max rep':>7} "
           f"| {'closer':>6} {'farther':>7} | {'s':>5}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for limit in args.limits:
        # Theta = 0 baseline: s'' == s' by definition, also gives the plain rate
        combos = [(0, 0, 0)] + [(args.theta, f, s) for f in args.phi for s in args.psi
                                if (f, s) != (0, 0)]
        best = None
        for theta, phi, psi in combos:
            r = run(cube, limit, theta, phi, psi, D, peak, codec_kwargs)
            rows.append(r)
            print(f"{limit:>3} {phi:>3} {psi:>3} {r['bpppb']:>6.3f} | {r['psnr_rec']:>10.3f} "
                  f"{r['psnr_rep']:>10.3f} {r['psnr_gain']:>+6.2f} | {r['sam_rec']:>8.5f} "
                  f"{r['sam_rep']:>8.5f} | {r['maxerr_rec']:>7d} {r['maxerr_rep']:>7d} "
                  f"| {r['frac_rep_closer']:>6.3f} {r['frac_rep_farther']:>7.3f} "
                  f"| {r['enc_s']:>5.1f}", flush=True)
            if best is None or r["psnr_rep"] > best["psnr_rep"]:
                best = r
        base = rows[-len(combos)]
        print(f"  => a={limit}: best s'' PSNR {best['psnr_rep']:.3f} dB at phi={best['phi']} "
              f"psi={best['psi']} ({best['psnr_gain']:+.2f} dB vs its own s', "
              f"{best['psnr_rep'] - base['psnr_rec']:+.2f} dB vs Theta=0 s'; rate "
              f"{best['bpppb']:.3f} vs {base['bpppb']:.3f} bpppb)", flush=True)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    if args.json:
        meta = {"input": label, "shape": list(cube.shape), "dynamic_range": D, "peak": peak,
                "theta": args.theta, "pred_bands": args.pred_bands, "entropy": args.entropy}
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
