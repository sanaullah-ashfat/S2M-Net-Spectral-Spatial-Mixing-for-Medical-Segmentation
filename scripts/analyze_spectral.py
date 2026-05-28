#!/usr/bin/env python3
"""
Standalone Spectral Truncation Analysis
=========================================

Analyses spectral energy retention and reconstruction quality across
truncation sizes K on any dataset.  Generates publication-quality figures.

Usage::

    python scripts/analyze_spectral.py \
        --data-dir data/Kvasir-SEG/train/images \
        --output-dir runs/spectral_analysis \
        --input-size 352 \
        --ks 16 24 32 48 64 128 \
        --samples 50
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s2mnet.utils.spectral import analyze_dataset_spectrum, reconstruction_rmse

import cv2
from glob import glob


# =============================================================================
# Plot helpers
# =============================================================================

_COLORS = {"opt": "#27ae60", "line": "#2980b9", "error": "#e74c3c"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
})


def _highlight(ax, K_values, opt_K, y_values):
    if opt_K in K_values:
        idx = K_values.index(opt_K)
        ax.scatter([opt_K], [y_values[idx]], s=200, zorder=5, color=_COLORS["opt"],
                   marker="D", edgecolors="black", linewidths=1.5, label=f"K={opt_K} (Optimal)")


def figure_energy_rmse(stats: dict, save_dir: str, opt_K: int = 32):
    K_values    = sorted(stats)
    energy_mean = [stats[K]["energy_mean"] for K in K_values]
    energy_std  = [stats[K]["energy_std"]  for K in K_values]
    rmse_mean   = [stats[K]["rmse_mean"]   for K in K_values]
    rmse_std    = [stats[K]["rmse_std"]    for K in K_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Energy
    ax1.plot(K_values, energy_mean, "o-", lw=2.5, ms=8, color=_COLORS["line"], label="Mean Energy")
    ax1.fill_between(K_values,
                     [m - s for m, s in zip(energy_mean, energy_std)],
                     [m + s for m, s in zip(energy_mean, energy_std)],
                     alpha=0.2, color=_COLORS["line"])
    ax1.axhline(95, ls="--", lw=2, color=_COLORS["error"], alpha=0.7, label="95% threshold")
    _highlight(ax1, K_values, opt_K, energy_mean)
    ax1.set_xlabel("Truncation Size K"); ax1.set_ylabel("Energy Retention (%)")
    ax1.set_title("(A) Spectral Energy Retention", loc="left", fontweight="bold")
    ax1.legend(framealpha=0.95); ax1.grid(True, ls="--", alpha=0.3)

    # RMSE
    ax2.plot(K_values, rmse_mean, "s-", lw=2.5, ms=8, color=_COLORS["error"], label="Mean RMSE")
    ax2.fill_between(K_values,
                     [m - s for m, s in zip(rmse_mean, rmse_std)],
                     [m + s for m, s in zip(rmse_mean, rmse_std)],
                     alpha=0.2, color=_COLORS["error"])
    _highlight(ax2, K_values, opt_K, rmse_mean)
    ax2.set_xlabel("Truncation Size K"); ax2.set_ylabel("Reconstruction RMSE")
    ax2.set_title("(B) Reconstruction Error (RMSE)", loc="left", fontweight="bold")
    ax2.set_yscale("log"); ax2.legend(framealpha=0.95); ax2.grid(True, ls="--", alpha=0.3)

    fig.suptitle("Spectral Truncation Analysis", fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "spectral_energy_rmse")


def figure_reconstruction(image_paths: list, K_viz: list, save_dir: str, input_size: int):
    samples = []
    for p in image_paths[:3]:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        samples.append(cv2.resize(rgb, (input_size, input_size)).astype(np.float32) / 255.0)

    if not samples:
        return

    rows = len(samples)
    cols = len(K_viz) + 2   # original + K images + error
    fig  = plt.figure(figsize=(4 * cols, 4 * rows))
    gs   = gridspec.GridSpec(rows, cols, hspace=0.15, wspace=0.1)

    for r, img in enumerate(samples):
        ax0 = fig.add_subplot(gs[r, 0])
        ax0.imshow(img); ax0.axis("off")
        ax0.set_title("Original" if r == 0 else "", fontweight="bold")
        ax0.set_ylabel(f"Sample {r+1}", fontweight="bold")

        for c, K in enumerate(K_viz):
            rmse, recon = reconstruction_rmse(img, K)
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(np.stack([recon]*3, axis=-1), cmap="gray")
            ax.axis("off")
            color = _COLORS["opt"] if K == 32 else "black"
            ax.set_title(f"K={K}" if r == 0 else "", fontweight="bold", color=color)
            ax.text(0.5, 0.04, f"RMSE: {rmse:.4f}", transform=ax.transAxes, ha="center",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8), fontsize=9)

        # Error map (vs K=32)
        _, recon32 = reconstruction_rmse(img, 32)
        import cv2 as _cv2
        gray = _cv2.cvtColor((_cv2.resize(img, (input_size, input_size)) * 255).astype(np.uint8),
                              _cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        err  = np.abs(gray - recon32)
        ax_e = fig.add_subplot(gs[r, -1])
        im   = ax_e.imshow(err, cmap="hot", vmin=0, vmax=0.1)
        ax_e.axis("off")
        ax_e.set_title("Error (K=32)" if r == 0 else "", fontweight="bold")

    fig.suptitle("Reconstruction Quality", fontweight="bold")
    _save(fig, save_dir, "reconstruction_quality")


def save_tables(results: dict, stats: dict, save_dir: str, opt_K: int = 32):
    K_values = sorted(stats)

    rows = []
    for K in K_values:
        s = stats[K]
        rows.append({
            "K":              K,
            "Energy (%)":     f"{s['energy_mean']:.2f} ± {s['energy_std']:.2f}",
            "RMSE":           f"{s['rmse_mean']:.4f} ± {s['rmse_std']:.4f}",
            "Coefficients":   f"{K**2:,}",
            "Compression (%)":f"{K**2 / 352**2 * 100:.2f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "spectral_energy_table.csv"), index=False)
    print("\n" + df.to_string(index=False))

    # Statistical significance vs opt_K
    if opt_K in results:
        sig_rows = []
        for K in K_values:
            if K == opt_K or K not in results:
                continue
            _, p = ttest_ind(results[opt_K]["energy"], results[K]["energy"])
            sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            sig_rows.append({"K": K, "p-value": f"{p:.4f}", "Significance": sig})

        if sig_rows:
            df2 = pd.DataFrame(sig_rows)
            df2.to_csv(os.path.join(save_dir, "statistical_significance.csv"), index=False)
            print("\nStatistical significance vs K=32:")
            print(df2.to_string(index=False))


def _save(fig, save_dir: str, stem: str):
    for ext in ["pdf", "png"]:
        p = os.path.join(save_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--output-dir", default="runs/spectral_analysis")
    parser.add_argument("--input-size", type=int, default=352)
    parser.add_argument("--ks",         type=int, nargs="+", default=[16, 24, 32, 48, 64, 128])
    parser.add_argument("--samples",    type=int, default=50)
    parser.add_argument("--opt-k",      type=int, default=32, help="K to highlight as optimal")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect image paths
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif"]
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(args.data_dir, e)))
    paths = sorted(paths)[: args.samples]

    if not paths:
        print(f"[Error] No images found in '{args.data_dir}'")
        sys.exit(1)

    print(f"[Spectral] Analysing {len(paths)} images | K={args.ks} | size={args.input_size}")

    results = analyze_dataset_spectrum(
        image_paths     = paths,
        truncation_sizes= tuple(args.ks),
        input_size      = args.input_size,
        max_samples     = args.samples,
    )

    # Compute stats
    stats = {}
    for K, vals in results.items():
        if vals["energy"]:
            stats[K] = {
                "energy_mean": np.mean(vals["energy"]),
                "energy_std":  np.std(vals["energy"]),
                "rmse_mean":   np.mean(vals["rmse"]),
                "rmse_std":    np.std(vals["rmse"]),
            }

    print("\n[Spectral] Generating figures...")
    figure_energy_rmse(stats, args.output_dir, opt_K=args.opt_k)
    figure_reconstruction(paths, K_viz=[16, 32, 64, 128], save_dir=args.output_dir,
                          input_size=args.input_size)
    save_tables(results, stats, args.output_dir, opt_K=args.opt_k)
    print(f"\n[Done] Output saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
