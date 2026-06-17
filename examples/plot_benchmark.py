#!/usr/bin/env python3
"""Turn a benchmark.py CSV into PNGs for the README (written to assets/).

    python3 examples/plot_benchmark.py --csv assets/benchmark_local.csv --out assets
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pretty(name):
    return (name.replace("cuprite_Cuprite_S1_", "cuprite_")
                .replace("jasper_ridge_jasperRidge2_", "jasper_")
                .replace("urban_urban_", "urban_")
                .replace("washington_dc_mall", "washington_dc")).lower()


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        r["limit"] = int(r["limit"])
        for k in ("ratio", "bpppb", "sam_rad"):
            r[k] = float(r[k])
        r["psnr_db"] = float("inf") if r["psnr_db"] == "inf" else float(r["psnr_db"])
        rows.append(r)
    return rows


def datasets_by_ratio(rows):
    loss = {r["dataset"]: r["ratio"] for r in rows if r["limit"] == 0 and r["entropy"] == "hybrid"}
    return sorted(loss, key=loss.get, reverse=True)


def _colors(n):
    cmap = plt.cm.tab20 if n > 10 else plt.cm.tab10
    return [cmap(i % cmap.N) for i in range(n)]


def _lines(rows, datasets, xkey, ykey, only_near_lossless, title, xlabel, ylabel, fname, out):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for ds, c in zip(datasets, _colors(len(datasets))):
        pts = [r for r in rows if r["dataset"] == ds and r["entropy"] == "hybrid"]
        if only_near_lossless:
            pts = [r for r in pts if r["limit"] > 0]
        pts.sort(key=lambda r: r[xkey])
        if pts:
            ax.plot([p[xkey] for p in pts], [p[ykey] for p in pts],
                    marker="o", ms=4, lw=1.5, color=c, label=pretty(ds))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=140)
    plt.close(fig)


def low_rate(rows, datasets, out):
    """Rate-distortion zoomed into the 0-1 bpppb low-bitrate region."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for ds, c in zip(datasets, _colors(len(datasets))):
        pts = [r for r in rows if r["dataset"] == ds and r["entropy"] == "hybrid"
               and 0 < r["bpppb"] <= 1.05]
        pts.sort(key=lambda r: r["bpppb"])
        if pts:
            ax.plot([p["bpppb"] for p in pts], [p["psnr_db"] for p in pts],
                    marker="o", ms=4, lw=1.5, color=c, label=pretty(ds))
    ax.set_xlim(0, 1)
    ax.set_xlabel("bits per pixel per band (bpppb)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate-distortion at low bitrate (hybrid coder)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "low_rate.png"), dpi=140)
    plt.close(fig)


def lossless_ratio(rows, datasets, out):
    def loss(d, coder):
        return next((r["ratio"] for r in rows if r["dataset"] == d
                     and r["entropy"] == coder and r["limit"] == 0), 0)
    hy = [loss(d, "hybrid") for d in datasets]
    sa = [loss(d, "sample_adaptive") for d in datasets]
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x - 0.2, sa, 0.4, label="sample-adaptive", color="#7aa6c2")
    ax.bar(x + 0.2, hy, 0.4, label="hybrid", color="#d98c5f")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(d) for d in datasets], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("lossless compression ratio (x:1)")
    ax.set_title("Lossless ratio by dataset")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "lossless_ratio.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--lowrate-csv", help="extended-limit hybrid sweep -> low_rate.png")
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = load(args.csv)
    datasets = datasets_by_ratio(rows)
    if args.lowrate_csv:
        lr = load(args.lowrate_csv)
        low_rate(lr, datasets_by_ratio(lr), args.out)
        print(f"wrote low_rate.png to {args.out}/")
    _lines(rows, datasets, "bpppb", "psnr_db", True,
           "Rate–distortion (hybrid coder, near-lossless)",
           "bits per pixel per band (bpppb)", "PSNR (dB)", "rate_distortion.png", args.out)
    _lines(rows, datasets, "limit", "ratio", False,
           "Compression ratio vs error limit (hybrid coder)",
           "absolute error limit (0 = lossless)", "compression ratio (x:1)",
           "ratio_vs_limit.png", args.out)
    lossless_ratio(rows, datasets, args.out)
    print(f"wrote rate_distortion.png, ratio_vs_limit.png, lossless_ratio.png "
          f"to {args.out}/ for {len(datasets)} datasets")


if __name__ == "__main__":
    main()
