"""
plot_epoch_f1_curves.py
=======================
Generates 10 individual F1-score vs. epoch graphs (one per radius) from the
deep GNN training history stored in deep_sweep_history.json.

Radii available: 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 miles
We plot all 11 in a 2-column x 5-row layout (plus 1 extra panel for legend/summary),
and also save each as an individual PNG.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "/Users/andrewsmac/CCIR/Traffic"
JSON_PATH = os.path.join(BASE, "output", "deep_sweep_history.json")
OUT_DIR   = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    raw = json.load(f)

# Sort radii numerically; use the 10 integer radii + 4.5 (11 total) — show 10
# by dropping 4.5 to keep exactly 10 clean panels.
all_radii = sorted(raw.keys(), key=float)
print("All radii in JSON:", all_radii)

# Select 10 radii (drop 4.5 to keep clean integer set + 10.0)
RADII_10 = [r for r in all_radii if r != "4.5"][:10]
print("Plotting these 10 radii:", RADII_10)

# ── Palette ────────────────────────────────────────────────────────────────────
# Perceptually distinct palette cycling through the 10 plots
COLORS = [
    "#00C9FF", "#FF6B6B", "#A8FF78", "#FFD93D", "#C77DFF",
    "#FF9A3C", "#4ECDC4", "#F7797D", "#B8E994", "#6C63FF",
]

# ── Smoothing helper ───────────────────────────────────────────────────────────
def smooth(arr, window=5):
    """Simple moving-average smoothing."""
    if len(arr) <= window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 2×5 combined dashboard
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 28))
fig.suptitle(
    "LSTM-GNN Validation F1 Score per Epoch\nby Spatial Connection Radius · PEMS-BAY Dataset",
    fontsize=20, fontweight="bold", color='black', y=0.995
)

gs = GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.32)
axes = [fig.add_subplot(gs[r, c]) for r in range(5) for c in range(2)]

for ax_idx, (radius_str, color) in enumerate(zip(RADII_10, COLORS)):
    ax   = axes[ax_idx]
    hist = raw[radius_str]
    epochs = np.arange(1, len(hist) + 1)
    smoothed = smooth(np.array(hist))

    best_f1  = max(hist)
    best_ep  = hist.index(best_f1) + 1

    # Dark background per panel
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    # Raw F1 (faint)
    ax.plot(epochs, hist, color=color, alpha=0.25, linewidth=1.0, zorder=1)
    # Smoothed F1
    ax.plot(epochs, smoothed, color=color, linewidth=2.5, zorder=2, label="F1 (smoothed)")

    # Best point marker
    ax.scatter([best_ep], [best_f1], color='black', s=90, zorder=5, edgecolors=color, linewidths=2)
    ax.annotate(
        f"Best: {best_f1:.4f}\n(ep {best_ep})",
        xy=(best_ep, best_f1),
        xytext=(best_ep + max(1, len(hist) * 0.07), best_f1 - 0.04),
        fontsize=7.5, color='black',
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.85),
    )

    # Horizontal reference line at best F1
    ax.axhline(best_f1, color=color, linewidth=0.8, linestyle="--", alpha=0.45, zorder=0)

    # Filled area under curve
    ax.fill_between(epochs, hist, alpha=0.10, color=color, zorder=0)

    # Labels
    ax.set_title(f"{radius_str} Mile Radius", fontsize=13, fontweight="bold",
                 color=color, pad=8)
    ax.set_xlabel("Epoch", fontsize=9, color="black")
    ax.set_ylabel("F1 Score", fontsize=9, color="black")
    ax.tick_params(colors="black", labelsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_xlim(1, max(len(hist), 20))
    ax.set_ylim(max(0, min(hist) - 0.02), min(1.0, best_f1 + 0.05))
    ax.grid(True, color="lightgray", linewidth=0.6, zorder=0)

    # Mini stats box
    stats_text = (
        f"Epochs run: {len(hist)}\n"
        f"Final F1:   {hist[-1]:.4f}\n"
        f"Best F1:    {best_f1:.4f}"
    )
    ax.text(
        0.97, 0.05, stats_text,
        transform=ax.transAxes,
        fontsize=7.2, color="black",
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor="black", alpha=0.9),
    )

COMBINED_PATH = os.path.join(OUT_DIR, "f1_epoch_curves_dashboard.png")
plt.savefig(COMBINED_PATH, dpi=150, bbox_inches="tight", facecolor='white')
plt.close()
print(f"✅ Combined dashboard saved → {COMBINED_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 10 individual PNGs (one per radius)
# ══════════════════════════════════════════════════════════════════════════════
for radius_str, color in zip(RADII_10, COLORS):
    hist    = raw[radius_str]
    epochs  = np.arange(1, len(hist) + 1)
    smoothed = smooth(np.array(hist))
    best_f1  = max(hist)
    best_ep  = hist.index(best_f1) + 1

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    ax.plot(epochs, hist, color=color, alpha=0.22, linewidth=1.2, label="Raw F1", zorder=1)
    ax.plot(epochs, smoothed, color=color, linewidth=2.8, label="Smoothed F1 (5-ep MA)", zorder=2)
    ax.fill_between(epochs, hist, alpha=0.12, color=color, zorder=0)
    ax.axhline(best_f1, color=color, linewidth=1.0, linestyle="--", alpha=0.5, zorder=0,
               label=f"Best F1 = {best_f1:.4f}")

    ax.scatter([best_ep], [best_f1], color='black', s=120, zorder=5, edgecolors=color, linewidths=2.5)
    ax.annotate(
        f"Best: {best_f1:.4f}  (epoch {best_ep})",
        xy=(best_ep, best_f1),
        xytext=(best_ep + max(1, len(hist) * 0.06), best_f1 - 0.035),
        fontsize=9, color='black',
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.9),
    )

    ax.set_title(
        f"LSTM-GNN · {radius_str} Mile Radius\nValidation F1 Score over Epochs",
        fontsize=14, fontweight="bold", color='black', pad=12
    )
    ax.set_xlabel("Epoch", fontsize=11, color="black")
    ax.set_ylabel("F1 Score", fontsize=11, color="black")
    ax.tick_params(colors="black", labelsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_xlim(1, max(len(hist), 20))
    ax.set_ylim(max(0, min(hist) - 0.02), min(1.0, best_f1 + 0.06))
    ax.grid(True, color="lightgray", linewidth=0.7)

    legend = ax.legend(fontsize=9, facecolor='white', edgecolor="black",
                       labelcolor='black', loc="lower right")

    # Footer stats
    fig.text(
        0.5, 0.01,
        f"Epochs run: {len(hist)}   |   Final F1: {hist[-1]:.4f}   |   Best F1: {best_f1:.4f}   "
        f"|   PEMS-BAY · LSTM-GNN · Early-stop patience=10",
        ha="center", fontsize=8, color="black",
    )

    fname = f"f1_epoch_{radius_str.replace('.', '_')}mi.png"
    fpath = os.path.join(OUT_DIR, fname)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(fpath, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"  ✅ Saved → {fpath}")

print("\nAll plots generated!")
