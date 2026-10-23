import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import warnings
from device_utils import get_device

warnings.filterwarnings('ignore')

device = get_device()
print(f"Using device: {device}")

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')
CSV_FILE = os.path.join(OUT, 'gnn_10_radii_metrics.csv')

# 1. Load Data
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:4032]
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

# 2. Sequences
seq_len, horizon = 12, 3
X_seq, y_lab = [], []
for t in range(seq_len, len(norm_speeds) - horizon):
    feat = norm_speeds[t-seq_len:t].T.reshape(len(valid_sensors), seq_len, 1)
    target = (df.values[t + horizon] < CONG_THRESH).astype(np.float32)
    X_seq.append(torch.tensor(feat, dtype=torch.float32))
    y_lab.append(torch.tensor(target, dtype=torch.float32))

split = int(len(X_seq) * 0.8)
train_X, test_X = X_seq[:split], X_seq[split:]
train_y, test_y = y_lab[:split], y_lab[split:]

# 3. Model
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

# 4. Train All 10 Radii (up to 200 epochs with early stopping)
radii = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
results = []
MAX_EPOCHS = 200
PATIENCE = 10

for r in radii:
    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < r:
                edge_index.append([i, j])

    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device) \
        if edge_index else torch.empty((2, 0), dtype=torch.long).to(device)
    num_edges = e_tensor.size(1)
    print(f"\n=======================================================")
    print(f"--- Testing Radius: {r} miles ({num_edges} edges) ---")
    print(f"=======================================================")

    model = LSTM_TrafficGNN(1, 16, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        indices = np.random.permutation(len(train_X))
        for i in range(0, len(train_X), 64):
            batch_indices = indices[i:i+64]
            optimizer.zero_grad()
            for idx in batch_indices:
                out = model(train_X[idx].to(device), e_tensor).squeeze()
                loss = criterion(out, train_y[idx].to(device))
                loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(test_X)):
                out = model(test_X[i].to(device), e_tensor).squeeze()
                val_loss += criterion(out, test_y[i].to(device)).item()
                preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
                all_preds.append(preds)
                all_targets.append(test_y[i].numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        avg_val_loss = val_loss / len(test_X)
        
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        prec = precision_score(all_targets, all_preds, zero_division=0)
        rec = recall_score(all_targets, all_preds, zero_division=0)
        acc = accuracy_score(all_targets, all_preds)
        
        print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | Val Loss: {avg_val_loss:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Acc: {acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_acc = acc

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"-> Early Stopping triggered at Epoch {epoch}!")
            break

    print(f"\nFinal Best metrics for {r}mi:")
    print(f"F1: {best_f1:.4f} | Prec: {best_prec:.4f} | Rec: {best_rec:.4f} | Acc: {best_acc:.4f}")
    
    results.append({
        'Radius_Miles': float(r),
        'Edges': num_edges,
        'Epochs_Ran': epoch,
        'Accuracy': best_acc,
        'Precision': best_prec,
        'Recall': best_rec,
        'F1_Score': best_f1
    })

df_results = pd.DataFrame(results)
df_results.to_csv(CSV_FILE, index=False)

print("\n\nAll 10 Radii Completed. Results saved to:")
print(CSV_FILE)
print("\nFinal Results Table:")
print(df_results.to_string(index=False))
