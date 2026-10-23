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
print("Loading data...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:2016] # 1 week sweep
CONG_THRESH = 60.0

scaler = MinMaxScaler()
norm_speeds = scaler.fit_transform(df.values)
valid_sensors = df.columns.tolist()

loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns: loc_df.columns = ['sensor_id', 'latitude', 'longitude']
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)
loc_dict = loc_df.set_index('sensor_id')[['latitude', 'longitude']].to_dict('index')
coords = np.array([[loc_dict.get(s, {'latitude':0})['latitude'], loc_dict.get(s, {'longitude':0})['longitude']] for s in valid_sensors])
coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])
dist_matrix = haversine_distances(coords_rad, coords_rad) * 3959.0

# 2. Sequence Generation
seq_len, horizon = 12, 3
X_seq, y_lab = [], []
for t in range(seq_len, len(norm_speeds) - horizon):
    X_seq.append(torch.tensor(norm_speeds[t-seq_len:t].T.reshape(len(valid_sensors), seq_len, 1), dtype=torch.float32))
    y_lab.append(torch.tensor((df.values[t+horizon] < CONG_THRESH).astype(np.float32), dtype=torch.float32))

split = int(len(X_seq) * 0.8)
tr_X, te_X = X_seq[:split], X_seq[split:]
tr_y, te_y = y_lab[:split], y_lab[split:]

# 3. Model
class LSTM_SAGE(nn.Module):
    def __init__(self):
        super(LSTM_SAGE, self).__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.conv = SAGEConv(16, 32)
        self.out = nn.Linear(32, 1)
    def forward(self, x, edge_index):
        _, (h, _) = self.lstm(x)
        x = F.relu(self.conv(h[-1], edge_index))
        return self.out(x)

def evaluate_radius(radius):
    print(f"\n--- Testing Radius: {radius} miles ---")
    edges = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < radius:
                edges.append([i, j])
    
    edge_idx = torch.tensor(edges, dtype=torch.long).t().to(device)
    print(f"Edges: {edge_idx.size(1)}")
    
    model = LSTM_SAGE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    crit = nn.BCEWithLogitsLoss()
    
    model.train()
    for _ in range(10): # Short epochs for sweep
        for idx in range(0, len(tr_X), 64):
            batch = tr_X[idx:idx+64]
            batch_y = tr_y[idx:idx+64]
            for b_idx in range(len(batch)):
                opt.zero_grad()
                loss = crit(model(batch[b_idx].to(device), edge_idx).squeeze(), batch_y[b_idx].to(device))
                loss.backward()
                opt.step()

    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for i in range(len(te_X)):
            out = model(te_X[i].to(device), edge_idx).squeeze()
            all_p.extend((torch.sigmoid(out) > 0.5).cpu().numpy().astype(int))
            all_t.extend(te_y[i].numpy())
    
    f1 = f1_score(all_t, all_p)
    print(f"F1 Score: {f1:.4f}")
    return f1

radii = [1.0, 2.0, 5.0, 10.0]
results = [evaluate_radius(r) for r in radii]

plt.figure(figsize=(8, 5))
plt.plot(radii, results, marker='o', color='blue', linewidth=2)
plt.title('GNN Spatial Range Sensitivity Analysis')
plt.xlabel('Connection Radius (Miles)')
plt.ylabel('F1 Score (Breakdown Prediction)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUT, 'gnn_range_sensitivity.png'), facecolor='white')
plt.close()
print("\nSensitivity plot saved to output/gnn_range_sensitivity.png")
