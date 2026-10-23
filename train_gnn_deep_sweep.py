import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from math import radians
from sklearn.metrics.pairwise import haversine_distances
import warnings
import csv
from device_utils import get_device


warnings.filterwarnings('ignore')

device = get_device()

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)
CSV_OUT = os.path.join(OUT, 'deep_sweep_results.csv')

CONG_THRESH = 60.0
seq_len, horizon = 12, 3
MAX_EPOCHS = 200
PATIENCE = 10

# Initialize CSV
if not os.path.exists(CSV_OUT):
    with open(CSV_OUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Radius_Miles', 'Edges', 'Best_F1', 'Epochs_Run'])

# 1. Load Data
print("Loading data...")
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

# 2. Build Dataset
print("Building Dataset...")
X, y = [], []
for i in range(len(norm_speeds) - seq_len - horizon):
    feat = norm_speeds[i:i+seq_len].T.reshape(len(valid_sensors), seq_len, 1)
    target = (df.values[i+seq_len+horizon] < CONG_THRESH).astype(np.float32)
    X.append(feat)
    y.append(target)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

split = int(len(X) * 0.8)
X_tr, y_tr = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

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

# 3. Training Loop
RADII = [1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0]

for radius in RADII:
    print(f"\n{'='*50}\n TRAINING {radius} MILE RADIUS\n{'='*50}")
    
    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < radius:
                edge_index.append([i, j])
    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device)
    num_edges = e_tensor.size(1) if len(edge_index) > 0 else 0
    print(f"Edges: {num_edges}")

    model = LSTM_TrafficGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0
    epochs_run = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_run = epoch
        model.train()
        indices = np.random.permutation(len(X_tr))
        
        for i in range(0, len(X_tr), 64):
            batch_idx = indices[i:i+64]
            optimizer.zero_grad()
            batch_loss = 0
            for idx in batch_idx:
                out = model(X_tr[idx].to(device), e_tensor).squeeze()
                batch_loss += criterion(out, y_tr[idx].to(device))
            (batch_loss / len(batch_idx)).backward()
            optimizer.step()

        # Eval
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for i in range(len(X_val)):
                out = model(X_val[i].to(device), e_tensor).squeeze()
                val_loss += criterion(out, y_val[i].to(device)).item()
                preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
                all_preds.extend(preds)
                all_targets.extend(y_val[i].numpy())

        avg_val_loss = val_loss / len(X_val)
        val_f1 = f1_score(all_targets, all_preds)
        
        print(f"[{radius}mi] Ep {epoch:03d} | Val Loss: {avg_val_loss:.4f} | F1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUT, f'best_gnn_deep_{radius}mi.pth'))

        if patience_counter >= PATIENCE:
            print(f"  -> Early Stop triggered.")
            break

    # Save results to CSV immediately
    with open(CSV_OUT, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([radius, num_edges, best_val_f1, epochs_run])
    print(f"Finished {radius}mi. Best F1: {best_val_f1:.4f}")

print("\nAll deep sweeps complete!")
