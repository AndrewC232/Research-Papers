import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
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
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:4032] # 2 weeks
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

# Build Graph (Using 2.0 miles baseline)
edges = []
for i in range(len(valid_sensors)):
    for j in range(len(valid_sensors)):
        if i != j and dist_matrix[i, j] < 2.0:
            edges.append([i, j])
edge_idx = torch.tensor(edges, dtype=torch.long).t().to(device)

# 2. Sequence Generation
seq_len, horizon = 12, 3
X_seq, y_lab = [], []
for t in range(seq_len, len(norm_speeds) - horizon):
    X_seq.append(torch.tensor(norm_speeds[t-seq_len:t].T.reshape(len(valid_sensors), seq_len, 1), dtype=torch.float32))
    y_lab.append(torch.tensor((df.values[t+horizon] < CONG_THRESH).astype(np.float32), dtype=torch.float32))

split = int(len(X_seq) * 0.8)
tr_X, te_X = X_seq[:split], X_seq[split:]
tr_y, te_y = y_lab[:split], y_lab[split:]

# 3. Model: LSTM + GAT (Graph Attention Network)
class LSTM_GAT(nn.Module):
    def __init__(self, in_dim, lstm_h, gat_h):
        super(LSTM_GAT, self).__init__()
        self.lstm = nn.LSTM(in_dim, lstm_h, batch_first=True)
        # GAT with multi-head attention (2 heads)
        self.gat1 = GATConv(lstm_h, gat_h, heads=2, concat=True)
        self.gat2 = GATConv(gat_h * 2, gat_h, heads=1, concat=False)
        self.out = nn.Linear(gat_h, 1)

    def forward(self, x, edge_index):
        # x: (N, L, 1)
        _, (h, _) = self.lstm(x)
        x = h[-1] # (N, lstm_h)
        
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return self.out(x)

model = LSTM_GAT(1, 16, 32).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
crit = nn.BCEWithLogitsLoss()

# 4. Train
print("Training GAT-LSTM model...")
epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss = 0
    indices = np.random.permutation(len(tr_X))
    for i in range(0, len(tr_X), 32):
        batch_idx = indices[i:i+32]
        for idx in batch_idx:
            opt.zero_grad()
            loss = crit(model(tr_X[idx].to(device), edge_idx).squeeze(), tr_y[idx].to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(tr_X):.4f}")

# 5. Evaluate
print("Evaluating GAT-LSTM...")
model.eval()
all_p, all_t = [], []
with torch.no_grad():
    for i in range(len(te_X)):
        out = model(te_X[i].to(device), edge_idx).squeeze()
        all_p.extend((torch.sigmoid(out) > 0.5).cpu().numpy().astype(int))
        all_t.extend(te_y[i].numpy())

gat_f1 = f1_score(all_t, all_p)
print(f"\nGAT-LSTM F1 Score: {gat_f1:.4f}")

# Comparison Plot
model_names = ['LSTM-SAGE (Prev)', 'GAT-LSTM (New)', 'XGBoost Baseline']
scores = [0.7099, gat_f1, 0.8913]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, scores, color=['gray', 'lime', 'coral'])
plt.title('Graph Attention (GAT) vs Standard SAGE')
plt.ylabel('F1 Score')
plt.ylim([0.6, 1.0])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', fontweight='bold')

plt.savefig(os.path.join(OUT, 'gat_model_comparison.png'), facecolor='white')
plt.close()
print("Comparison plot saved to output/gat_model_comparison.png")
