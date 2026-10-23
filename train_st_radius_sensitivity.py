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
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:4032] # 2 weeks for fast iteration
CONG_THRESH = 60.0

scaler = MinMaxScaler()
norm_speeds = scaler.fit_transform(df.values)

loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns: loc_df.columns = ['sensor_id', 'latitude', 'longitude']
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)
valid_sensors = df.columns.tolist()
loc_dict = loc_df.set_index('sensor_id')[['latitude', 'longitude']].to_dict('index')
coords = np.array([[loc_dict.get(s, {'latitude':0})['latitude'], loc_dict.get(s, {'longitude':0})['longitude']] for s in valid_sensors])

coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])
dist_matrix = haversine_distances(coords_rad, coords_rad) * 3959.0

# 2. Sequence Generation
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

# 3. Model Architecture
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

# 4. Sensitivity Testing Loop
radii = [1.0, 2.0, 3.5, 5.0, 7.5, 10.0]
results_f1 = []
results_edges = []

for r in radii:
    # Build Edges
    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < r:
                edge_index.append([i, j])
    
    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device)
    num_edges = e_tensor.size(1) if len(edge_index) > 0 else 0
    results_edges.append(num_edges)
    
    print(f"\n--- Testing Radius: {r} miles ({num_edges} edges) ---")
    
    model = LSTM_TrafficGNN(1, 16, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    
    batch_size = 64
    epochs = 25  # Extended training time!
    
    model.train()
    for epoch in range(epochs):
        indices = np.random.permutation(len(train_X))
        for i in range(0, len(train_X), batch_size):
            batch_indices = indices[i:i+batch_size]
            optimizer.zero_grad()
            for idx in batch_indices:
                out = model(train_X[idx].to(device), e_tensor).squeeze()
                loss = criterion(out, train_y[idx].to(device))
                loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i in range(len(test_X)):
            out = model(test_X[i].to(device), e_tensor).squeeze()
            preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_targets.extend(test_y[i].numpy())
            
    f1 = f1_score(all_targets, all_preds)
    print(f"Radius {r}m F1 Score: {f1:.4f}")
    results_f1.append(f1)

# 5. Plotting
plt.figure(figsize=(10, 6))
plt.plot(radii, results_f1, marker='o', color='blue', linewidth=2, markersize=8)
plt.title('Spatio-Temporal GNN Performance vs. Spatial Radius', fontsize=14)
plt.xlabel('Connection Radius (Miles)', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.grid(True, alpha=0.2)

for i, txt in enumerate(results_f1):
    plt.annotate(f"{txt:.4f}\n({results_edges[i]} edges)", 
                 (radii[i], results_f1[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_range_sensitivity.png'), facecolor='white')
plt.close()

# Save results
pd.DataFrame({'Radius_Miles': radii, 'Edges': results_edges, 'F1_Score': results_f1}).to_csv(os.path.join(OUT, 'st_gnn_sensitivity.csv'), index=False)
print(f"\nFinished! Results saved to {OUT}/st_gnn_range_sensitivity.png")
