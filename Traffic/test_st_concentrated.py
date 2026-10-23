import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time
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
print(f"Code explicitly optimized for: {device}")

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')
CSV_OUT = os.path.join(OUT, 'concentrated_sweep_results.csv')

MAX_EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 64

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
speeds = df.values  # [Time, Sensors]

for t in range(seq_len, len(norm_speeds) - horizon):
    is_transition = (speeds[t-1, :] >= CONG_THRESH) & (speeds[t, :] < CONG_THRESH)
    if np.any(is_transition):
        feat = norm_speeds[t-seq_len:t].T.reshape(len(valid_sensors), seq_len, 1)
        target = (speeds[t + horizon] < CONG_THRESH).astype(np.float32)
        X_seq.append(torch.tensor(feat, dtype=torch.float32))
        y_lab.append(torch.tensor(target, dtype=torch.float32))
        mask_list.append(torch.tensor(is_transition, dtype=torch.bool))

print(f"Found {len(X_seq)} Network Snapshots containing at least one transition event.")
total_transitions = sum([m.sum().item() for m in mask_list])
print(f"Total individual sensor transition events to predict: {total_transitions}")

split = int(len(X_seq) * 0.8)
train_X, test_X = X_seq[:split], X_seq[split:]
train_y, test_y = y_lab[:split], y_lab[split:]
train_mask, test_mask = mask_list[:split], mask_list[split:]

# GPU Optimization: Pre-load all data to device
print("Moving data to device for fast batched processing...")
train_X_t = torch.stack(train_X).to(device)
train_y_t = torch.stack(train_y).to(device)
train_mask_t = torch.stack(train_mask).to(device)
test_X_t = torch.stack(test_X).to(device)
test_y_t = torch.stack(test_y).to(device)
test_mask_t = torch.stack(test_mask).to(device)
num_sensors = len(valid_sensors)

def get_batched_edge_index(base_edge_index, num_nodes, batch_size, device):
    if base_edge_index.size(1) == 0:
        return torch.empty((2, 0), dtype=torch.long).to(device)
    batched_edges = []
    for b in range(batch_size):
        batched_edges.append(base_edge_index + b * num_nodes)
    return torch.cat(batched_edges, dim=1)

# 3. Model
class LSTM_TrafficGNN(nn.Module):
    def __init__(self, in_dim, lstm_hidden, gnn_hidden):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, lstm_hidden, batch_first=True)
        self.conv1 = SAGEConv(lstm_hidden, gnn_hidden)
        self.conv2 = SAGEConv(gnn_hidden, gnn_hidden)
        self.out = nn.Linear(gnn_hidden, 1)

    def forward(self, x, edge_index):
        # x shape: [B, N, seq_len, 1]
        B, N, L, D = x.shape
        x_reshaped = x.view(B * N, L, D)
        _, (h_n, _) = self.lstm(x_reshaped)
        x_g = h_n[-1] # shape: [B*N, lstm_hidden]
        x_g = F.relu(self.conv1(x_g, edge_index))
        x_g = F.relu(self.conv2(x_g, edge_index))
        out = self.out(x_g) # shape: [B*N, 1]
        return out.view(B, N)

# 4. Train on ALL radii (matching deep sweep)
radii = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
results_f1 = []
results_epochs = []

# Init CSV
with open(CSV_OUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Radius_Miles', 'Edges', 'Best_F1', 'Epochs_Run'])

for r in radii:
    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < r:
                edge_index.append([i, j])

    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device) \
        if edge_index else torch.empty((2, 0), dtype=torch.long).to(device)
    num_edges = e_tensor.size(1)

    print(f"\n{'='*50}")
    print(f" TRAINING {r} MILE RADIUS  ({num_edges} edges)")
    print(f"{'='*50}")

    model = LSTM_TrafficGNN(1, 16, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    epochs_run = 0

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_run = epoch

        # --- Train ---
        model.train()
        indices = torch.randperm(len(train_X_t), device=device)
        for i in range(0, len(train_X_t), BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            b_size = len(batch_indices)
            
            b_X = train_X_t[batch_indices]
            b_y = train_y_t[batch_indices]
            b_mask = train_mask_t[batch_indices]
            
            if not b_mask.any():
                continue
                
            optimizer.zero_grad()
            b_edge_index = get_batched_edge_index(e_tensor, num_sensors, b_size, device)
            
            out = model(b_X, b_edge_index) # [B, N]
            
            out_masked = out[b_mask]
            y_masked = b_y[b_mask]
            
            loss = criterion(out_masked, y_masked)
            loss.backward()
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for i in range(0, len(test_X_t), BATCH_SIZE):
                b_X = test_X_t[i:i+BATCH_SIZE]
                b_y = test_y_t[i:i+BATCH_SIZE]
                b_mask = test_mask_t[i:i+BATCH_SIZE]
                b_size = len(b_X)
                
                if not b_mask.any():
                    continue
                    
                b_edge_index = get_batched_edge_index(e_tensor, num_sensors, b_size, device)
                out = model(b_X, b_edge_index)
                
                out_masked = out[b_mask]
                y_masked = b_y[b_mask]
                
                if len(out_masked) > 0:
                    loss = criterion(out_masked, y_masked)
                    val_loss += loss.item() * b_size
                    
                    preds = (torch.sigmoid(out_masked) > 0.5).cpu().numpy().astype(int)
                    all_preds.extend(preds)
                    all_targets.extend(y_masked.cpu().numpy())

        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        avg_val_loss = val_loss / max(len(test_X_t), 1)

        print(f"[{r}mi] Ep {epoch:03d} | Val Loss: {avg_val_loss:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  -> Early Stop triggered at epoch {epoch}.")
            break

    elapsed = time.time() - start_time
    print(f"Finished {r}mi in {elapsed:.1f}s. Best F1: {best_f1:.4f} (ran {epochs_run} epochs)")
    results_f1.append(best_f1)
    results_epochs.append(epochs_run)

    # Save result to CSV immediately after each radius
    with open(CSV_OUT, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([r, num_edges, best_f1, epochs_run])

# 5. Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar([str(r) + ' mi' for r in radii], results_f1, color='blue', alpha=0.8, width=0.6)

for bar, ep in zip(bars, results_epochs):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.003,
             f"{yval:.4f}\n({ep}ep)", ha='center', va='bottom',
             fontsize=9, color='black', fontweight='bold')

best_idx = results_f1.index(max(results_f1))
bars[best_idx].set_color('gold')

plt.title('LSTM-GNN Early-Warning: Concentrated Transition Dataset\nAll Radii — Best F1 with Early Stopping', fontsize=13, pad=15)
plt.xlabel('Spatial Connection Radius', fontsize=12)
plt.ylabel('Masked F1 Score (Breakdown Events Only)', fontsize=12)
plt.ylim([0, max(results_f1) + 0.12])
plt.grid(True, alpha=0.1, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_concentrated_performance.png'), dpi=200, facecolor='white')
plt.close()

print("\n=== FINAL CONCENTRATED RESULTS ===")
for r, f1, ep in zip(radii, results_f1, results_epochs):
    print(f"  {r:5.1f} mi | F1: {f1:.4f} | Epochs: {ep}")
print(f"\nBest: {radii[best_idx]} mi — F1 {max(results_f1):.4f}")
print("\nSaved st_gnn_concentrated_performance.png")
print("Done!")
