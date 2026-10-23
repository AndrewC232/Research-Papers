import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

device = get_device()

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')

# ---------------------------------------------------------
# 1. Load & Normalize Data
# ---------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:4032] # 2 weeks
CONG_THRESH = 60.0

# Normalize speed for LSTM stability
scaler = MinMaxScaler()
norm_speeds = scaler.fit_transform(df.values)
thresh_norm = scaler.transform([[CONG_THRESH] * df.shape[1]])[0, 0]

# Load locations for graph
loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns: loc_df.columns = ['sensor_id', 'latitude', 'longitude']
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)

valid_sensors = df.columns.tolist()
loc_dict = loc_df.set_index('sensor_id')[['latitude', 'longitude']].to_dict('index')
coords = np.array([[loc_dict.get(s, {'latitude':0})['latitude'], loc_dict.get(s, {'longitude':0})['longitude']] for s in valid_sensors])

# ---------------------------------------------------------
# 2. Build Adjacency Matrix
# ---------------------------------------------------------
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from device_utils import get_device

coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])
dist_matrix = haversine_distances(coords_rad, coords_rad) * 3959.0

edge_index = []
for i in range(len(valid_sensors)):
    for j in range(len(valid_sensors)):
        if i != j and dist_matrix[i, j] < 2.0:
            edge_index.append([i, j])

edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device)
print(f"Graph: {len(valid_sensors)} nodes, {edge_index_tensor.size(1)} edges.")

# ---------------------------------------------------------
# 3. Sequence Generation
# ---------------------------------------------------------
print("Generating sequences...")
seq_len = 12 # 60 minutes
horizon = 3 # 15 minutes ahead

X_seq = []
y_lab = []

for t in range(seq_len, len(norm_speeds) - horizon):
    # Node features: sequence of normalized speeds
    # Shape: (N, seq_len, 1)
    feat = norm_speeds[t-seq_len:t].T.reshape(len(valid_sensors), seq_len, 1)
    target = (df.values[t + horizon] < CONG_THRESH).astype(np.float32)
    
    X_seq.append(torch.tensor(feat, dtype=torch.float32))
    y_lab.append(torch.tensor(target, dtype=torch.float32))

# Train/Test Split
split = int(len(X_seq) * 0.8)
train_X, test_X = X_seq[:split], X_seq[split:]
train_y, test_y = y_lab[:split], y_lab[split:]

# ---------------------------------------------------------
# 4. Model Architecture: LSTM + GNN
# ---------------------------------------------------------
class LSTM_TrafficGNN(nn.Module):
    def __init__(self, in_dim, lstm_hidden, gnn_hidden):
        super(LSTM_TrafficGNN, self).__init__()
        self.lstm = nn.LSTM(in_dim, lstm_hidden, batch_first=True)
        self.conv1 = SAGEConv(lstm_hidden, gnn_hidden)
        self.conv2 = SAGEConv(gnn_hidden, gnn_hidden)
        self.out = nn.Linear(gnn_hidden, 1)

    def forward(self, x, edge_index):
        # x shape: (N, seq_len, 1)
        # LSTM needs (N, seq_len, 1) if we treat each node independently
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1] # Final hidden state: (N, lstm_hidden)
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x)

model = LSTM_TrafficGNN(1, 16, 32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# ---------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------
print("Training Temporal-Spatial GNN...")
batch_size = 32
model.train()
epochs = 50

for epoch in range(epochs):
    epoch_loss = 0
    # Process in small batches of snapshots
    indices = np.random.permutation(len(train_X))
    for i in range(0, len(train_X), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_loss = 0
        optimizer.zero_grad()
        
        for idx in batch_indices:
            out = model(train_X[idx].to(device), edge_index_tensor).squeeze()
            loss = criterion(out, train_y[idx].to(device))
            loss.backward()
            batch_loss += loss.item()
            
        optimizer.step()
        epoch_loss += batch_loss
        
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/len(train_X):.4f}")

# ---------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------
print("Evaluating...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for i in range(len(test_X)):
        out = model(test_X[i].to(device), edge_index_tensor).squeeze()
        preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
        all_preds.extend(preds)
        all_targets.extend(test_y[i].numpy())

lstm_gnn_f1 = f1_score(all_targets, all_preds)
print(f"\nLSTM-GNN F1 Score: {lstm_gnn_f1:.4f}")

# Plot Comparison
models_names = ['Naive GNN', 'LSTM-GNN', 'XGBoost Baseline']
scores = [0.6729, lstm_gnn_f1, 0.8913] # Hardcoded previous scores

plt.figure(figsize=(8, 5))
bars = plt.bar(models_names, scores, color=['gray', 'cyan', 'coral'])
plt.title('GNN Evolution vs Baseline (F1 Score)')
plt.ylabel('F1 Score')
plt.ylim([0.6, 1.0])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', fontweight='bold')

plt.savefig(os.path.join(OUT, 'q18_lstm_gnn_comparison.png'), facecolor='white')
plt.close()
print("Results saved to output/q18_lstm_gnn_comparison.png")
