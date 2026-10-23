import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
OUT = os.path.join(BASE, 'output')

# ─────────────────────────────────────────────────────
# PLOT 1: Concentrated Dataset Performance
# ─────────────────────────────────────────────────────
conc_data = {
    1.0:  {'f1': 0.7040, 'epochs': 71,  'edges': 3490},
    2.0:  {'f1': 0.7046, 'epochs': 75,  'edges': 8852},
    3.0:  {'f1': 0.7040, 'epochs': 62,  'edges': 16712},
    4.0:  {'f1': 0.7087, 'epochs': 104, 'edges': 26280},
    5.0:  {'f1': 0.7095, 'epochs': 76,  'edges': 37346},
    6.0:  {'f1': 0.7112, 'epochs': 108, 'edges': 48318},
    7.0:  {'f1': 0.7040, 'epochs': 64,  'edges': 57798},
    8.0:  {'f1': 0.7040, 'epochs': 105, 'edges': 68142},
    9.0:  {'f1': 0.7040, 'epochs': 68,  'edges': 78218},
    10.0: {'f1': 0.7040, 'epochs': 67,  'edges': 85984},
}

radii = list(conc_data.keys())
f1_vals = [conc_data[r]['f1'] for r in radii]
epoch_vals = [conc_data[r]['epochs'] for r in radii]
edge_vals = [conc_data[r]['edges'] for r in radii]
best_idx = f1_vals.index(max(f1_vals))

fig, ax1 = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')

ax2 = ax1.twinx()
edge_bars = ax2.bar(radii, edge_vals, alpha=0.12, color='#FF6B6B', width=0.6, label='Edge Count', zorder=1)
ax2.set_ylabel('Number of Graph Edges', color='#FF6B6B', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#FF6B6B')
ax2.set_ylim([0, max(edge_vals) * 1.5])
ax2.grid(False)

# Color bars: gold for best, varying cyan shades otherwise
bar_colors = ['#00BFBF' if i != best_idx else 'gold' for i in range(len(radii))]
bars = ax1.bar(radii, f1_vals, color=bar_colors, alpha=0.9, width=0.5, zorder=3, edgecolor='black', linewidth=0.4)

# Annotate each bar
for bar, f1, ep, i in zip(bars, f1_vals, epoch_vals, range(len(radii))):
    yval = bar.get_height()
    color = 'gold' if i == best_idx else 'white'
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.0003,
             f'{f1:.4f}\n({ep}ep)', ha='center', va='bottom',
             fontsize=8.5, color=color, fontweight='bold')

# Highlight best with arrow annotation
ax1.annotate('★ Best: 6.0 mi\nF1 = 0.7112',
             xy=(6.0, conc_data[6.0]['f1']),
             xytext=(7.8, 0.7108),
             fontsize=10, color='gold', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='gold', lw=1.5))

ax1.set_xlabel('Spatial Connection Radius (Miles)', fontsize=13, color='black', labelpad=10)
ax1.set_ylabel('Best Validation F1 Score', fontsize=13, color='black', labelpad=10)
ax1.set_ylim([0.695, max(f1_vals) + 0.020])
ax1.set_xticks(radii)
ax1.set_xticklabels([f'{r} mi' for r in radii], fontsize=10)
ax1.tick_params(colors='black')
ax1.grid(True, alpha=0.08, axis='y', linestyle='--')

lines2, labels2 = ax2.get_legend_handles_labels()
legend_patches = [
    mpatches.Patch(facecolor='gold', label='Peak Radius (6.0 mi)'),
    mpatches.Patch(facecolor='#00BFBF', label='LSTM-GNN F1 Score'),
    mpatches.Patch(facecolor='#FF6B6B', alpha=0.4, label='Edge Count'),
]
ax1.legend(handles=legend_patches, loc='lower right', fontsize=10,
           facecolor='white', edgecolor='#444', labelcolor='black', framealpha=0.85)

ax1.set_title('LSTM-GNN Concentrated Transition Dataset — Best F1 by Connection Radius\n'
              '200 Epochs (Early Stopping, patience=15) | MPS GPU Optimized | All 10 Radii',
              fontsize=13, color='black', pad=15)

for spine in ax1.spines.values():
    spine.set_edgecolor('lightgray')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'concentrated_performance_updated.png', facecolor='white'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved concentrated_performance_updated.png")


# ─────────────────────────────────────────────────────
# PLOT 2: Regular Dataset Performance
# ─────────────────────────────────────────────────────
reg_data = {
    1.0:  {'f1': 0.7164, 'epochs': 54,  'edges': 3490},
    3.0:  {'f1': 0.7199, 'epochs': 44,  'edges': 16712},
    4.0:  {'f1': 0.7272, 'epochs': 123, 'edges': 26280},
    5.0:  {'f1': 0.7273, 'epochs': 129, 'edges': 37346},
    6.0:  {'f1': 0.7169, 'epochs': 50,  'edges': 48318},
    7.0:  {'f1': 0.7215, 'epochs': 64,  'edges': 57798},
    9.0:  {'f1': 0.7201, 'epochs': 53,  'edges': 78218},
    10.0: {'f1': 0.7250, 'epochs': 146, 'edges': 85984},
}

radii_r = list(reg_data.keys())
f1_vals_r = [reg_data[r]['f1'] for r in radii_r]
epoch_vals_r = [reg_data[r]['epochs'] for r in radii_r]
edge_vals_r = [reg_data[r]['edges'] for r in radii_r]
best_idx_r = f1_vals_r.index(max(f1_vals_r))

fig, ax1 = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')

ax2 = ax1.twinx()
ax2.bar(radii_r, edge_vals_r, alpha=0.12, color='#FF6B6B', width=0.6, label='Edge Count', zorder=1)
ax2.set_ylabel('Number of Graph Edges', color='#FF6B6B', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#FF6B6B')
ax2.set_ylim([0, max(edge_vals_r) * 1.5])
ax2.grid(False)

bar_colors_r = ['#00BFBF' if i != best_idx_r else 'gold' for i in range(len(radii_r))]
bars_r = ax1.bar(radii_r, f1_vals_r, color=bar_colors_r, alpha=0.9, width=0.5,
                 zorder=3, edgecolor='black', linewidth=0.4)

for bar, f1, ep, i in zip(bars_r, f1_vals_r, epoch_vals_r, range(len(radii_r))):
    yval = bar.get_height()
    color = 'gold' if i == best_idx_r else 'white'
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.0002,
             f'{f1:.4f}\n({ep}ep)', ha='center', va='bottom',
             fontsize=8.5, color=color, fontweight='bold')

ax1.annotate('★ Best: 5.0 mi\nF1 = 0.7273',
             xy=(5.0, reg_data[5.0]['f1']),
             xytext=(6.5, 0.7270),
             fontsize=10, color='gold', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='gold', lw=1.5))

ax1.set_xlabel('Spatial Connection Radius (Miles)', fontsize=13, color='black', labelpad=10)
ax1.set_ylabel('Best Validation F1 Score', fontsize=13, color='black', labelpad=10)
ax1.set_ylim([0.710, max(f1_vals_r) + 0.018])
ax1.set_xticks(radii_r)
ax1.set_xticklabels([f'{r} mi' for r in radii_r], fontsize=10)
ax1.tick_params(colors='black')
ax1.grid(True, alpha=0.08, axis='y', linestyle='--')

legend_patches_r = [
    mpatches.Patch(facecolor='gold', label='Peak Radius (5.0 mi)'),
    mpatches.Patch(facecolor='#00BFBF', label='LSTM-GNN F1 Score'),
    mpatches.Patch(facecolor='#FF6B6B', alpha=0.4, label='Edge Count'),
]
ax1.legend(handles=legend_patches_r, loc='lower right', fontsize=10,
           facecolor='white', edgecolor='#444', labelcolor='black', framealpha=0.85)

ax1.set_title('LSTM-GNN Regular Dataset — Best F1 by Connection Radius\n'
              '200 Epochs (Early Stopping, patience=15) | 8 Radii Tested | PEMS-BAY Dataset',
              fontsize=13, color='black', pad=15)

for spine in ax1.spines.values():
    spine.set_edgecolor('lightgray')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'regular_performance_updated.png', facecolor='white'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved regular_performance_updated.png")


# ─────────────────────────────────────────────────────
# PLOT 3: Side-by-Side Comparison (shared radii only)
# ─────────────────────────────────────────────────────
shared_radii = [r for r in radii if r in reg_data]
conc_f1 = [conc_data[r]['f1'] for r in shared_radii]
reg_f1 = [reg_data[r]['f1'] for r in shared_radii]

x = np.arange(len(shared_radii))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bars1 = ax.bar(x - width/2, reg_f1, width, label='Regular Dataset', color='#00BFBF', alpha=0.9,
               edgecolor='black', linewidth=0.4)
bars2 = ax.bar(x + width/2, conc_f1, width, label='Concentrated Transitions', color='#FF6B6B', alpha=0.9,
               edgecolor='black', linewidth=0.4)

for bar, v in zip(bars1, reg_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f'{v:.4f}', ha='center', va='bottom', fontsize=8, color='#00BFBF', fontweight='bold')
for bar, v in zip(bars2, conc_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f'{v:.4f}', ha='center', va='bottom', fontsize=8, color='#FF6B6B', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f'{r} mi' for r in shared_radii], fontsize=11)
ax.set_xlabel('Spatial Connection Radius (Miles)', fontsize=13, color='black', labelpad=10)
ax.set_ylabel('Best Validation F1 Score', fontsize=13, color='black', labelpad=10)
ax.set_ylim([0.695, max(max(reg_f1), max(conc_f1)) + 0.020])
ax.tick_params(colors='black')
ax.grid(True, alpha=0.08, axis='y', linestyle='--')
ax.legend(fontsize=11, facecolor='white', edgecolor='#444', labelcolor='black', framealpha=0.85)

for spine in ax.spines.values():
    spine.set_edgecolor('lightgray')

ax.set_title('LSTM-GNN Performance Comparison: Regular vs. Concentrated Dataset\n'
             '200 Epochs with Early Stopping | PEMS-BAY (San Francisco Bay Area)',
             fontsize=13, color='black', pad=15)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'dataset_comparison_updated.png', facecolor='white'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved dataset_comparison_updated.png")
print("Done!")
