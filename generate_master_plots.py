import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
OUT = os.path.join(BASE, 'output')

# 1. Master Baseline & GNN Comparison
# Updated F1 values — 200-epoch optimized runs (PEMS-BAY regular dataset)
fig, ax = plt.subplots(figsize=(16, 6))

models_global = ['XGBoost\n(Tabular)', 'Random\nForest', 'Simple\nRule', 'Decision\nTree', 'Logistic\nReg', 'KNN\n(k=5)', 'MLP\n(Neural Net)', 'Naive\nBayes', 'LSTM-SAGE\n(GNN)', 'GAT-LSTM\n(Attention)', 'Naive\nGraphSAGE']
scores_global = [0.8913, 0.8891, 0.8806, 0.8746, 0.8663, 0.8650, 0.8632, 0.6103, 0.7273, 0.6848, 0.6729]
colors_global = ['coral', 'coral', 'coral', 'coral', 'coral', 'coral', 'coral', 'coral', 'cyan', 'lime', 'gray']

bars = ax.bar(models_global, scores_global, color=colors_global)
ax.set_title('Global Steady-State Prediction: Tabular Models vs GNN Architectures', fontsize=15)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_ylim([0.5, 1.0])

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'master_global_comparison.png'), facecolor='white')
plt.close()

# 2. Master Transition Battle Comparison
# Updated F1 values — concentrated transition dataset (200-epoch, MPS GPU)
fig, ax = plt.subplots(figsize=(12, 6))

models_trans = ['GRU\n(Gated)', 'TCN\n(Dilated Conv)', 'LSTM\n(Temporal)', 'Transformer\n(Attention)', 'LightGBM\n(Leaf-wise)', 'XGBoost\n(Level-wise)', 'LSTM-SAGE\nGNN']
scores_trans = [0.6935, 0.6935, 0.6935, 0.6935, 0.6907, 0.6899, 0.7112]
colors_trans = ['gold', 'gold', 'cyan', 'cyan', 'coral', 'coral', 'lime']

bars = ax.bar(models_trans, scores_trans, color=colors_trans)
ax.set_title('Transition Event Prediction (Breakdown Onset): Model Battle', fontsize=14)
ax.set_ylabel('F1 Score')
ax.set_ylim([0.65, 0.73])

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'master_transition_comparison.png'), facecolor='white')
plt.close()

# 3. Combine Learning Curves (PIL)
img1_path = os.path.join(OUT, 'optimize_gru_curves.png')
img2_path = os.path.join(OUT, 'optimize_tcn_curves.png')

if os.path.exists(img1_path) and os.path.exists(img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Resize to match heights if needed
    h1, w1 = img1.size[1], img1.size[0]
    h2, w2 = img2.size[1], img2.size[0]
    
    # Let's assume they are the same size for now, they were generated similarly
    total_width = w1 + w2
    max_height = max(h1, h2)
    
    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste(img1, (0, 0))
    new_im.paste(img2, (w1, 0))
    
    new_im.save(os.path.join(OUT, 'combined_learning_curves.png'))

print("Master plots generated successfully.")
