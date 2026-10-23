import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = r'/Users/andrewsmac/CCIR/Traffic'
CSV_OUT = os.path.join(BASE, 'output', 'concentrated_25epochs_results.csv')
OUT = os.path.join(BASE, 'output')

df = pd.read_csv(CSV_OUT)
radii = df['Radius_Miles'].values
results_f1 = df['Best_F1'].values
results_epochs = df['Epochs_Run'].values

plt.style.use('default')
plt.figure(figsize=(12,6))
bars = plt.bar([f'{r:g} mi' for r in radii], results_f1, color='blue', alpha=0.8)
for bar, ep in zip(bars, results_epochs):
    y = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, y+0.005, f'{y:.4f}\n({int(ep)}ep)', ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')
plt.title('LSTM‑GNN Concentrated Transition – 25 Epochs per Radius', fontsize=13, pad=15, color='black')
plt.xlabel('Spatial Connection Radius (mi)', fontsize=12, color='black')
plt.ylabel('Masked F1 Score', fontsize=12, color='black')
plt.ylim(0, max(results_f1)+0.1)
plt.grid(True, axis='y', alpha=0.1)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_concentrated_25epochs.png'), dpi=200, facecolor='white')
plt.close()

print('Done plotting st_gnn_concentrated_25epochs.png with explicit facecolor=white')
