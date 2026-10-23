import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from math import radians
from sklearn.metrics.pairwise import haversine_distances
import warnings

warnings.filterwarnings('ignore')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')

# 1. Load Data
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:4032]
scaler = MinMaxScaler()
norm_speeds = scaler.fit_transform(df.values)
valid_sensors = df.columns.tolist()

loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns:
    loc_df.columns = ['sensor_id', 'latitude', 'longitude']
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)
loc_dict = loc_df.set_index('sensor_id')[['latitude', 'longitude']].to_dict('index')

coords = np.array([[loc_dict.get(s, {'latitude': 0})['latitude'],
                    loc_dict.get(s, {'longitude': 0})['longitude']] for s in valid_sensors])
coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])
dist_matrix = haversine_distances(coords_rad, coords_rad) * 3959.0

# 2. Build Dataset (Regular 4032 timesteps)
seq_len, horizon = 12, 3
CONG_THRESH = 60.0

X, y = [], []
for i in range(len(norm_speeds) - seq_len - horizon):
    feat = norm_speeds[i:i+seq_len].T.reshape(len(valid_sensors), seq_len, 1)
    target = (df.values[i+seq_len+horizon] < CONG_THRESH).astype(np.float32)
    X.append(feat)
    y.append(target)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

split = int(len(X) * 0.8)
X_val, y_val = X[split:], y[split:]

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

radii = [1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0]
results = {}

for radius in radii:
    model_path = os.path.join(OUT, f'best_gnn_deep_{radius}mi.pth')
    if not os.path.exists(model_path):
        continue

    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < radius:
                edge_index.append([i, j])
    e_tensor = torch.tensor(edge_index, dtype=torch.long).t()

    model = LSTM_TrafficGNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for i in range(len(X_val)):
            out = model(X_val[i], e_tensor).squeeze()
            preds = (torch.sigmoid(out) > 0.5).numpy().astype(int)
            all_preds.extend(preds)
            all_targets.extend(y_val[i].numpy())

    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    results[radius] = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

for r, cm in results.items():
    print(f"Radius {r}mi:")
    print(f"  TN (True Negative - Correct Normal Traffic): {cm['TN']}")
    print(f"  FP (False Positive - False Alarm): {cm['FP']}")
    print(f"  FN (False Negative - Missed Breakdown): {cm['FN']}")
    print(f"  TP (True Positive - Correct Breakdown): {cm['TP']}")
    print()
