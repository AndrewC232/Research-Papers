import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('default')
BASE = r'/Users/andrewsmac/CCIR/Traffic'
OUT = os.path.join(BASE, 'output')
CSV_FILE = os.path.join(OUT, 'st_gnn_sensitivity.csv')

df = pd.read_csv(CSV_FILE)
integers_df = df[df['Radius_Miles'].isin([1,2,3,4,5,6,7,8,9,10])].copy()

fig, ax1 = plt.subplots(figsize=(14, 7))
color_f1 = 'cyan'
color_edges = '#FF6B6B'

ax2 = ax1.twinx()
ax2.bar(integers_df['Radius_Miles'], integers_df['Edges'], alpha=0.15,
        color=color_edges, width=0.5, label='Edge Count', zorder=1)
ax2.set_ylabel('Number of Graph Edges', color=color_edges, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_edges)

ax1.plot(integers_df['Radius_Miles'], integers_df['F1_Score'], marker='o', color=color_f1,
         linewidth=2.5, markersize=10, label='F1 Score', zorder=3)
ax1.fill_between(integers_df['Radius_Miles'], integers_df['F1_Score'],
                 alpha=0.08, color=color_f1)
ax1.set_xlabel('Connection Radius (Miles)', fontsize=13)
ax1.set_ylabel('F1 Score', color=color_f1, fontsize=13)
ax1.tick_params(axis='y', labelcolor=color_f1)
ax1.set_ylim([0.665, 0.735])
ax1.set_xticks(integers_df['Radius_Miles'])
ax1.set_xticklabels([f'{int(r)} mi' for r in integers_df['Radius_Miles']], fontsize=11)
ax1.grid(True, alpha=0.12)

# Highlight global peak
best_idx = integers_df['F1_Score'].idxmax()
best_r = integers_df.loc[best_idx, 'Radius_Miles']
best_f1 = integers_df.loc[best_idx, 'F1_Score']

# FIXED: changed '★' to '*'
ax1.scatter(best_r, best_f1, color='red', s=200, zorder=5, marker='*')
ax1.annotate(f'Global Peak\n{int(best_r)} mi — F1={best_f1:.4f}',
             xy=(best_r, best_f1),
             xytext=(best_r - 2.5, best_f1 + 0.006),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Annotate each point
for _, row in integers_df.iterrows():
    ax1.annotate(f"{row['F1_Score']:.4f}",
                 (row['Radius_Miles'], row['F1_Score']),
                 textcoords="offset points", xytext=(0, 12),
                 ha='center', fontsize=8.5, color=color_f1)

ax1.set_title('LSTM-GNN F1 Score vs. Spatial Connection Radius (1–10 Miles)\n25 Epochs | 325 Bay Area PeMS Sensors', fontsize=14, pad=15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_integer_curve.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved st_gnn_integer_curve.png")
