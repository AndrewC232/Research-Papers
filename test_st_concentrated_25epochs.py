import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
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
CSV_OUT = os.path.join(OUT, 'concentrated_25epochs_results.csv')

MAX_EPOCHS = 25
PATIENCE = 15

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

print('Extracting Concentrated Transition Snapshots...')
speeds = df.values
for t in range(seq_len, len(norm_speeds) - horizon):
    is_transition = (speeds[t-1, :] >= CONG_THRESH) & (speeds[t, :] < CONG_THRESH)
    if np.any(is_transition):
        feat = norm_speeds[t-seq_len:t].T.reshape(len(valid_sensors), seq_len, 1)
        target = (speeds[t + horizon] < CONG_THRESH).astype(np.float32)
        X_seq.append(torch.tensor(feat, dtype=torch.float32))
        y_lab.append(torch.tensor(target, dtype=torch.float32))
        mask_list.append(torch.tensor(is_transition, dtype=torch.bool))

print(f'Found {len(X_seq)} Network Snapshots containing at least one transition event.')
print(f'Total individual sensor transition events to predict: {sum([m.sum().item() for m in mask_list])}')

split = int(len(X_seq) * 0.8)
train_X, test_X = X_seq[:split], X_seq[split:]
train_y, test_y = y_lab[:split], y_lab[split:]
train_mask, test_mask = mask_list[:split], mask_list[split:]

# 3. Model definition
class LSTM_TrafficGNN(nn.Module):
    def __init__(self, in_dim, lstm_hidden, gnn_hidden):
        super().__init__()
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

# 4. Train across all radii (1‑10 mi)
radii = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
results_f1 = []
results_epochs = []

# Initialise CSV
with open(CSV_OUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Radius_Miles', 'Edges', 'Best_F1', 'Epochs_Run'])

for r in radii:
    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < r:
                edge_index.append([i, j])
    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device) if edge_index else torch.empty((2,0), dtype=torch.long).to(device)
    num_edges = e_tensor.size(1)
    print(f'\n=== Training radius {r} mi ({num_edges} edges) ===')
    model = LSTM_TrafficGNN(1, 16, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    best_f1 = 0.0
    epochs_run = 0
    for epoch in range(1, MAX_EPOCHS+1):
        epochs_run = epoch
        model.train()
        indices = np.random.permutation(len(train_X))
        for i in range(0, len(train_X), 64):
            batch_idx = indices[i:i+64]
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_steps = 0
            for idx in batch_idx:
                mask = train_mask[idx].to(device)
                if not mask.any():
                    continue
                out = model(train_X[idx].to(device), e_tensor).squeeze()
                out_masked = out[mask]
                y_masked = train_y[idx].to(device)[mask]
                batch_loss += criterion(out_masked, y_masked)
                valid_steps += 1
            if valid_steps > 0:
                (batch_loss / valid_steps).backward()
                optimizer.step()
        # Evaluation after each epoch
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for i in range(len(test_X)):
                mask = test_mask[i].to(device)
                if not mask.any():
                    continue
                out = model(test_X[i].to(device), e_tensor).squeeze()
                preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
                all_preds.extend(preds[mask.cpu().numpy()])
                all_targets.extend(test_y[i].numpy()[mask.cpu().numpy()])
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        print(f'[{r}mi] Epoch {epoch:02d} | F1: {f1:.4f}')
        if f1 > best_f1:
            best_f1 = f1
    print(f'Finished {r}mi – Best F1: {best_f1:.4f} (run {epochs_run} epochs)')
    results_f1.append(best_f1)
    results_epochs.append(epochs_run)
    # Append to CSV
    with open(CSV_OUT, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([r, num_edges, best_f1, epochs_run])

# 5. Plotting results
plt.figure(figsize=(12,6))
bars = plt.bar([f'{r:g} mi' for r in radii], results_f1, color='blue', alpha=0.8)
for bar, ep in zip(bars, results_epochs):
    y = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, y+0.005, f'{y:.4f}\n({ep}ep)', ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')
plt.title('LSTM‑GNN Concentrated Transition – 25 Epochs per Radius', fontsize=13, pad=15)
plt.xlabel('Spatial Connection Radius (mi)', fontsize=12)
plt.ylabel('Masked F1 Score', fontsize=12)
plt.ylim(0, max(results_f1)+0.1)
plt.grid(True, axis='y', alpha=0.1)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_concentrated_25epochs.png'), dpi=200, facecolor='white')
plt.close()

print('\n=== SUMMARY (25‑epoch sweep) ===')
for r, f1, ep in zip(radii, results_f1, results_epochs):
    print(f'  {r:5.1f} mi | F1: {f1:.4f} | epochs: {ep}')
print('Plot saved as st_gnn_concentrated_25epochs.png')
print('Done!')
