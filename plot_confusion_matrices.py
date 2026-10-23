import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
OUT = os.path.join(BASE, 'output')

data = {
    1.0: {'TN': 198968, 'FP': 15064, 'FN': 12482, 'TP': 34786},
    3.0: {'TN': 198827, 'FP': 15205, 'FN': 12136, 'TP': 35132},
    4.0: {'TN': 203377, 'FP': 10655, 'FN': 14176, 'TP': 33092},
    5.0: {'TN': 202543, 'FP': 11489, 'FN': 13694, 'TP': 33574},
    6.0: {'TN': 198762, 'FP': 15270, 'FN': 12325, 'TP': 34943},
    7.0: {'TN': 193732, 'FP': 20300, 'FN': 9134,  'TP': 38134},
    9.0: {'TN': 200628, 'FP': 13404, 'FN': 13135, 'TP': 34133},
    10.0: {'TN': 201448, 'FP': 12584, 'FN': 13237, 'TP': 34031}
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Common kwargs for heatmap
cmap = sns.dark_palette("#00FFFF", as_cmap=True)

for i, (radius, metrics) in enumerate(data.items()):
    ax = axes[i]
    # Matrix format:
    # [[TN, FP],
    #  [FN, TP]]
    cm = np.array([[metrics['TN'], metrics['FP']],
                   [metrics['FN'], metrics['TP']]])
    
    # Calculate percentages for annotations
    total = np.sum(cm)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=['Normal', 'Breakdown'],
                yticklabels=['Normal', 'Breakdown'])
    
    ax.set_title(f'Radius: {radius} mi', fontsize=16, pad=10, color='gold')
    if i >= 4:
        ax.set_xlabel('Predicted Status', fontsize=12)
    if i % 4 == 0:
        ax.set_ylabel('True Status', fontsize=12)

fig.suptitle('Confusion Matrices by Connection Radius (Regular Dataset, 200 Epochs)\nLSTM-GNN Model', 
             fontsize=22, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, 'confusion_matrices_grid.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# Also create a grouped bar chart for easier comparison
fig, ax = plt.subplots(figsize=(14, 7))
radii = list(data.keys())
x = np.arange(len(radii))
width = 0.2

tps = [data[r]['TP'] for r in radii]
fns = [data[r]['FN'] for r in radii]
fps = [data[r]['FP'] for r in radii]

ax.bar(x - width, tps, width, label='True Positives (Correct Breakdown)', color='#00FF00')
ax.bar(x, fns, width, label='False Negatives (Missed Breakdown)', color='#FF6B6B')
ax.bar(x + width, fps, width, label='False Positives (False Alarm)', color='gold')

ax.set_ylabel('Number of Predictions', fontsize=12)
ax.set_xlabel('Connection Radius (Miles)', fontsize=12)
ax.set_title('Prediction Errors vs Correct Breakdown Detections across Radii', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f'{r} mi' for r in radii])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.1, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'confusion_metrics_comparison.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("Created confusion_matrices_grid.png")
print("Created confusion_metrics_comparison.png")
