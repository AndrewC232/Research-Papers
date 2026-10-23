import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import warnings
from device_utils import get_device


warnings.filterwarnings('ignore')
plt.style.use('default')

device = get_device()

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')

# 1. Load Data
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:10000]
CONG_THRESH = 60.0
scaler = MinMaxScaler()
norm_speeds = scaler.fit_transform(df.values)

loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns:
    loc_df.columns = ['sensor_id', 'latitude', 'longitude']
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)
valid_sensors = df.columns.tolist()
loc_dict = loc_df.set_index('sensor_id')[['latitude', 'longitude']].to_dict('index')
coords = np.array([[loc_dict.get(s, {'latitude': 0})['latitude'],
                    loc_dict.get(s, {'longitude': 0})['longitude']] for s in valid_sensors])
coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])
dist_matrix = haversine_distances(coords_rad, coords_rad) * 3959.0

# 2. Extract Concentrated Transition Sequences
seq_len, horizon = 12, 3
X_seq, y_lab, mask_list = [], [], []

print("Extracting Concentrated Transition Snapshots...")
speeds = df.values 

for t in range(seq_len, len(norm_speeds) - horizon):
    is_transition = (speeds[t-1, :] >= CONG_THRESH) & (speeds[t, :] < CONG_THRESH)
    
    if np.any(is_transition):
        feat = norm_speeds[t-seq_len:t].T.reshape(len(valid_sensors), seq_len, 1)
        target = (speeds[t + horizon] < CONG_THRESH).astype(np.float32)
        
        X_seq.append(torch.tensor(feat, dtype=torch.float32))
        y_lab.append(torch.tensor(target, dtype=torch.float32))
        mask_list.append(torch.tensor(is_transition, dtype=torch.bool))

split = int(len(X_seq) * 0.8)
test_X = X_seq[split:]
test_y = y_lab[split:]
test_mask = mask_list[split:]

# 3. Model
class LSTM_TrafficGNN(nn.Module):
    def __init__(self, in_dim=1, lstm_hidden=16, gnn_hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, lstm_hidden, batch_first=True)
        self.conv1 = SAGEConv(lstm_hidden, gnn_hidden)
        self.conv2 = SAGEConv(gnn_hidden, gnn_hidden)
        self.out = nn.Linear(gnn_hidden, 1)

    def forward(self, x, edge_idx):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = F.relu(self.conv1(x, edge_idx))
        x = F.relu(self.conv2(x, edge_idx))
        return self.out(x)

# 4. Evaluate all deep models
RADII = [1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
results_f1 = []
valid_radii = []

for r in RADII:
    # Handle the fact that some were saved as best_gnn_deep_{r}mi.pth and some as best_gnn_{r}mi.pth
    path1 = os.path.join(OUT, f'best_gnn_deep_{r}mi.pth')
    path2 = os.path.join(OUT, f'best_gnn_{r}mi.pth')
    
    model_path = None
    if os.path.exists(path1): model_path = path1
    elif os.path.exists(path2): model_path = path2
    
    if not model_path:
        print(f"Skipping {r}mi (No weights found)")
        continue
        
    print(f"\nEvaluating {r}mi on Concentrated Transitions...")
    
    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < r:
                edge_index.append([i, j])
    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device) if edge_index else torch.empty((2, 0), dtype=torch.long).to(device)

    model = LSTM_TrafficGNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for i in range(len(test_X)):
            mask = test_mask[i].to(device)
            if not mask.any(): continue
            
            out = model(test_X[i].to(device), e_tensor).squeeze()
            preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            
            out_masked = preds[mask.cpu().numpy()]
            y_masked = test_y[i].numpy()[mask.cpu().numpy()]
            
            all_preds.extend(out_masked)
            all_targets.extend(y_masked)

    f1 = f1_score(all_targets, all_preds)
    print(f"--> Masked F1: {f1:.4f}")
    results_f1.append(f1)
    valid_radii.append(r)

# 5. Plotting
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(valid_radii, results_f1, marker='s', color='#FF6B6B', linewidth=3, markersize=9, label='Deep ST-GNN on Concentrated Transitions')

# Add GRU baseline line for reference
gru_f1 = 0.7039
ax1.axhline(y=gru_f1, color='blue', linestyle='--', linewidth=2, label=f'Deep Sequential GRU (F1={gru_f1:.4f})')

ax1.set_xlabel('Spatial Connection Radius (Miles)', fontsize=13)
ax1.set_ylabel('Masked F1 Score', color='black', fontsize=13)
ax1.set_ylim([0.60, 0.72])
ax1.grid(True, alpha=0.15)
ax1.set_xticks(valid_radii)
ax1.set_xticklabels([f'{r}mi' for r in valid_radii])

best_idx = np.argmax(results_f1)
best_r = valid_radii[best_idx]
best_f = results_f1[best_idx]

ax1.scatter(best_r, best_f, color='red', s=250, marker='*', zorder=5)
ax1.annotate(f'GNN Ceiling\n{best_r}mi — F1={best_f:.4f}',
             xy=(best_r, best_f), xytext=(best_r - 1.5, best_f + 0.01),
             fontsize=11, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax1.set_title('Early-Warning Accuracy: Deep ST-GNNs vs. Pure GRU\nEvaluating strictly on isolated breakdown transitions', fontsize=15, pad=15)
ax1.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'deep_concentrated_performance.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("\nPlot saved to output/deep_concentrated_performance.png")
