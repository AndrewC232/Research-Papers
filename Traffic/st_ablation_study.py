import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import warnings
from device_utils import get_device

warnings.filterwarnings('ignore')

device = get_device()

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')

# 1. Load Data
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:4032]
CONG_THRESH = 60.0
scaler = MinMaxScaler()
norm_speeds = scaler.fit_transform(df.values)

loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns: loc_df.columns = ['sensor_id', 'latitude', 'longitude']
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)
valid_sensors = df.columns.tolist()
loc_dict = loc_df.set_index('sensor_id')[['latitude', 'longitude']].to_dict('index')
coords = np.array([[loc_dict.get(s, {'latitude':0})['latitude'], loc_dict.get(s, {'longitude':0})['longitude']] for s in valid_sensors])

# 2. Edges
from sklearn.metrics.pairwise import haversine_distances
from math import radians
coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])
dist_matrix = haversine_distances(coords_rad, coords_rad) * 3959.0

edge_index = []
for i in range(len(valid_sensors)):
    for j in range(len(valid_sensors)):
        if i != j and dist_matrix[i, j] < 2.0:
            edge_index.append([i, j])
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device)
empty_edge_index = torch.empty((2, 0), dtype=torch.long).to(device) # For Spatial Ablation

# 3. Sequence
seq_len = 12
horizon = 3
X_seq, y_lab = [], []
for t in range(seq_len, len(norm_speeds) - horizon):
    feat = norm_speeds[t-seq_len:t].T.reshape(len(valid_sensors), seq_len, 1)
    target = (df.values[t + horizon] < CONG_THRESH).astype(np.float32)
    X_seq.append(torch.tensor(feat, dtype=torch.float32))
    y_lab.append(torch.tensor(target, dtype=torch.float32))

split = int(len(X_seq) * 0.8)
train_X, test_X = X_seq[:split], X_seq[split:]
train_y, test_y = y_lab[:split], y_lab[split:]

# 4. Model
class LSTM_TrafficGNN(nn.Module):
    def __init__(self, in_dim, lstm_hidden, gnn_hidden):
        super(LSTM_TrafficGNN, self).__init__()
        self.lstm = nn.LSTM(in_dim, lstm_hidden, batch_first=True)
        self.conv1 = SAGEConv(lstm_hidden, gnn_hidden)
        self.conv2 = SAGEConv(gnn_hidden, gnn_hidden)
        self.out = nn.Linear(gnn_hidden, 1)

    def forward(self, x, edge_index):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x)

model = LSTM_TrafficGNN(1, 16, 32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()

# Train quick model (10 epochs)
print("Training Baseline LSTM-GNN...")
batch_size = 64
model.train()
for epoch in range(15):
    indices = np.random.permutation(len(train_X))
    for i in range(0, len(train_X), batch_size):
        batch_indices = indices[i:i+batch_size]
        optimizer.zero_grad()
        for idx in batch_indices:
            out = model(train_X[idx].to(device), edge_index_tensor).squeeze()
            loss = criterion(out, train_y[idx].to(device))
            loss.backward()
        optimizer.step()

# Evaluation Function
def evaluate(x_data, edge_data):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i in range(len(x_data)):
            out = model(x_data[i].to(device), edge_data).squeeze()
            preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_targets.extend(test_y[i].numpy())
    return f1_score(all_targets, all_preds)

print("\n--- ABLATION RESULTS ---")
# 1. Baseline
base_f1 = evaluate(test_X, edge_index_tensor)
print(f"Baseline Spatio-Temporal F1: {base_f1:.4f}")

# 2. Spatial Ablation (Remove Graph Edges)
spatial_f1 = evaluate(test_X, empty_edge_index)
print(f"No Spatial Data (Isolated Nodes): {spatial_f1:.4f} (Drop: {base_f1 - spatial_f1:.4f})")

# 3. Temporal Ablation (Scramble Time Sequence)
# We shuffle the sequence axis (dim=1) for each sample
scrambled_X = []
for x in test_X:
    scrambled = x.clone()
    # Shuffle time dimension for all nodes
    perm = torch.randperm(seq_len)
    scrambled = scrambled[:, perm, :]
    scrambled_X.append(scrambled)

temporal_f1 = evaluate(scrambled_X, edge_index_tensor)
print(f"No Temporal Data (Scrambled Sequence): {temporal_f1:.4f} (Drop: {base_f1 - temporal_f1:.4f})")

# Save results
import matplotlib.pyplot as plt
plt.style.use('default')
labels = ['Baseline', 'Scrambled Time\n(Temporal Ablation)', 'No Neighbors\n(Spatial Ablation)']
scores = [base_f1, temporal_f1, spatial_f1]
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, scores, color=['cyan', 'coral', 'gold'])
plt.title('Spatio-Temporal Model Feature Importance (Ablation)')
plt.ylabel('F1 Score')
plt.ylim([0.4, 0.8])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_ablation_study.png'), facecolor='white')
print("Saved st_ablation_study.png")
