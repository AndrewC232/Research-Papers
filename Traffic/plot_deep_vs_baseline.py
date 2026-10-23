import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('default')
BASE = r'/Users/andrewsmac/CCIR/Traffic'
OUT = os.path.join(BASE, 'output')
CSV_BASELINE = os.path.join(OUT, 'st_gnn_sensitivity.csv')
CSV_DEEP = os.path.join(OUT, 'deep_sweep_results.csv')

# Load Baseline (25 Epochs)
df_base = pd.read_csv(CSV_BASELINE)
df_base = df_base[df_base['Radius_Miles'].isin([1, 2, 3, 4, 4.5, 5, 6, 7, 8, 9, 10])].copy()
df_base.sort_values('Radius_Miles', inplace=True)

# Load Deep (Up to 200 Epochs)
df_deep = pd.read_csv(CSV_DEEP)

# We manually add the 2.0, 4.5, and 8.0 deep runs since they were run in separate scripts
manual_deep_runs = pd.DataFrame([
    {'Radius_Miles': 2.0, 'Best_F1': 0.7228},
    {'Radius_Miles': 4.5, 'Best_F1': 0.7234},
    {'Radius_Miles': 8.0, 'Best_F1': 0.7197}
])

df_deep_all = pd.concat([df_deep[['Radius_Miles', 'Best_F1']], manual_deep_runs])
df_deep_all.sort_values('Radius_Miles', inplace=True)

# Plotting
fig, ax1 = plt.subplots(figsize=(14, 7))

# Baseline Curve
ax1.plot(df_base['Radius_Miles'], df_base['F1_Score'], marker='o', color='gray',
         linewidth=2, markersize=7, label='Baseline (25 Epochs)', alpha=0.7)

# Deep Curve
ax1.plot(df_deep_all['Radius_Miles'], df_deep_all['Best_F1'], marker='s', color='blue',
         linewidth=3, markersize=9, label='Deep Training (Up to 200 Epochs)', zorder=4)

ax1.fill_between(df_deep_all['Radius_Miles'], df_base['F1_Score'], df_deep_all['Best_F1'],
                 color='blue', alpha=0.15, label='Improvement Delta')

# Annotations
ax1.set_xlabel('Connection Radius (Miles)', fontsize=13)
ax1.set_ylabel('F1 Score', color='black', fontsize=13)
ax1.set_ylim([0.66, 0.735])
ax1.grid(True, alpha=0.15)
ax1.set_xticks([1, 2, 3, 4, 4.5, 5, 6, 7, 8, 9, 10])
ax1.set_xticklabels(['1mi', '2mi', '3mi', '4mi', '4.5mi\n(Local Peak)', '5mi', '6mi', '7mi', '8mi\n(Global Peak)', '9mi', '10mi'])

# Highlight best absolute F1
best_deep_idx = df_deep_all['Best_F1'].idxmax()
best_r = df_deep_all.loc[best_deep_idx, 'Radius_Miles']
best_f1 = df_deep_all.loc[best_deep_idx, 'Best_F1']

ax1.scatter(best_r, best_f1, color='red', s=250, marker='*', zorder=5)
ax1.annotate(f'Absolute Ceiling\n{best_r}mi — F1={best_f1:.4f}',
             xy=(best_r, best_f1),
             xytext=(best_r - 1.5, best_f1 + 0.005),
             fontsize=11, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax1.set_title('Spatio-Temporal Performance Ceiling: Baseline vs Deep Convergence', fontsize=15, pad=15)
ax1.legend(loc='lower left', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_deep_vs_baseline.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("Plot saved to output/st_gnn_deep_vs_baseline.png")
