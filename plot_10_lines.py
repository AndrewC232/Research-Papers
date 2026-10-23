import json
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
OUT = os.path.join(BASE, 'output')
JSON_FILE = os.path.join(OUT, 'deep_sweep_history.json')

with open(JSON_FILE, 'r') as f:
    history_data = json.load(f)

# Convert string keys back to float and sort
radii = sorted([float(k) for k in history_data.keys()])

fig, ax = plt.subplots(figsize=(16, 9))
colors = cm.get_cmap('tab20', len(radii))

max_epochs = 0
for i, r in enumerate(radii):
    raw_f1 = history_data[str(r)]
    if len(raw_f1) > max_epochs:
        max_epochs = len(raw_f1)
        
    # Calculate cumulative maximum (i.e. "best F1 score during those epochs")
    cum_max_f1 = np.maximum.accumulate(raw_f1)
    epochs = range(1, len(cum_max_f1) + 1)
    
    # Highlight the absolute best lines
    linewidth = 2.5 if r in [4.0, 4.5, 5.0] else 1.5
    alpha = 1.0 if r in [4.0, 4.5, 5.0] else 0.6
    
    label = f"{r} mi (Best: {cum_max_f1[-1]:.4f})"
    if r == 5.0: label += " 🏆"
    
    ax.plot(epochs, cum_max_f1, color=colors(i), linewidth=linewidth, alpha=alpha, label=label)

ax.set_xlabel('Epochs Run', fontsize=13)
ax.set_ylabel('Best F1 Score Achieved (Cumulative Max)', color='black', fontsize=13)
ax.set_title('Spatio-Temporal GNN Deep Learning Curves by Geographic Radius\n(Tracking the Cumulative Maximum F1 Score)', fontsize=16, pad=15)

ax.grid(True, alpha=0.15)
ax.set_xlim([1, max_epochs])

# Legend styling
ax.legend(loc='lower right', fontsize=11, facecolor='white', edgecolor='gray', framealpha=0.9, title="Spatial Radius")

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_10_lines.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("Saved st_gnn_10_lines.png")
