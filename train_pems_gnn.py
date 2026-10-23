import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("Loading time-series data...")
# Load only 2 weeks of data (approx 4000 timestamps) to make training feasible in this session
# 14 days * 24 hours * 12 readings/hr = 4032 readings
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:4032]
CONG_THRESH = 60.0

print("Loading locations...")
loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns and len(loc_df.columns) >= 3:
    loc_df.columns = ['sensor_id', 'latitude', 'longitude']
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)

valid_sensors = df.columns.tolist()
# Create mapping for coordinates
loc_dict = loc_df.set_index('sensor_id')[['latitude', 'longitude']].to_dict('index')

# Fallback for missing locations
for s in valid_sensors:
    if s not in loc_dict:
        loc_dict[s] = {'latitude': 0, 'longitude': 0}

coords = np.array([[loc_dict[s]['latitude'], loc_dict[s]['longitude']] for s in valid_sensors])

# ---------------------------------------------------------
# 2. Build Adjacency Matrix (Highway Connections)
# ---------------------------------------------------------
print("Building geographic adjacency matrix...")
from sklearn.metrics.pairwise import haversine_distances
from math import radians

# Compute distances in miles
coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])
dist_matrix = haversine_distances(coords_rad, coords_rad) * 3959.0

# Create edge index for sensors within 2 miles
edge_index = []
edge_weights = []
for i in range(len(valid_sensors)):
    for j in range(len(valid_sensors)):
        if i != j and dist_matrix[i, j] < 2.0 and dist_matrix[i, j] > 0:
            edge_index.append([i, j])
            # Inverse distance weighting
            edge_weights.append(1.0 / dist_matrix[i, j])

edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

print(f"Graph has {len(valid_sensors)} nodes and {edge_index_tensor.size(1)} edges.")

# ---------------------------------------------------------
# 3. Build Spatio-Temporal Graph Snapshots
# ---------------------------------------------------------
print("Building spatio-temporal features...")
# For each timestamp t, the graph has node features: speed at t, t-1, t-3, hour
# Target is speed at t+3 < 60

X_list = []
y_list = []

speed_values = df.values # shape: (T, N)
hours = df.index.hour.values # shape: (T,)

horizon = 3 # predict 15m ahead
lags = [0, 1, 3] # current, -5m, -15m

for t in range(max(lags), len(df) - horizon):
    # Node features for this timestamp shape: (N, num_features)
    # Features: [speed_t, speed_t-1, speed_t-3, hour_sin, hour_cos]
    feat_t = []
    
    speed_curr = speed_values[t]
    speed_lag1 = speed_values[t-1]
    speed_lag3 = speed_values[t-3]
    h_sin = np.full(len(valid_sensors), np.sin(2 * np.pi * hours[t] / 24))
    h_cos = np.full(len(valid_sensors), np.cos(2 * np.pi * hours[t] / 24))
    
    feat = np.stack([speed_curr, speed_lag1, speed_lag3, h_sin, h_cos], axis=1)
    
    # Target shape: (N,)
    target = (speed_values[t + horizon] < CONG_THRESH).astype(np.float32)
    
    X_list.append(torch.tensor(feat, dtype=torch.float32))
    y_list.append(torch.tensor(target, dtype=torch.float32))

print(f"Generated {len(X_list)} graph snapshots.")

# Train/Test Split
split = int(len(X_list) * 0.8)
train_X = X_list[:split]
train_y = y_list[:split]
test_X = X_list[split:]
test_y = y_list[split:]

# ---------------------------------------------------------
# 4. Define GraphSAGE Model
# ---------------------------------------------------------
class TrafficGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(TrafficGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.out(x)
        return x

model = TrafficGNN(in_channels=5, hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCEWithLogitsLoss()

# ---------------------------------------------------------
# 5. Train Model
# ---------------------------------------------------------
print("Training GNN...")
model.train()
epochs = 30 # Keep it short for execution speed

for epoch in range(epochs):
    epoch_loss = 0
    # Batch processing is conceptually simple here since graph structure is static
    for t_idx in range(len(train_X)):
        optimizer.zero_grad()
        out = model(train_X[t_idx], edge_index_tensor).squeeze()
        loss = criterion(out, train_y[t_idx])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_X):.4f}")

# ---------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------
print("Evaluating on Test Set...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for t_idx in range(len(test_X)):
        out = model(test_X[t_idx], edge_index_tensor).squeeze()
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).long().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(test_y[t_idx].numpy())

gnn_f1 = f1_score(all_targets, all_preds)
print(f"\nGNN F1 Score: {gnn_f1:.4f}")

# Compare with previous models
# Recreate flat feature matrix for the test set to quickly evaluate Simple Rule
flat_test_speeds = np.concatenate([X[:, 0].numpy() for X in test_X])
flat_test_targets = np.array(all_targets)
simple_rule_preds = (flat_test_speeds < 60.0).astype(int)
simple_f1 = f1_score(flat_test_targets, simple_rule_preds)

print(f"Simple Rule F1 (on same 2-week subset): {simple_f1:.4f}")

# XGBoost baseline from full run was 0.8913, but let's record it here for the chart
models = ['Simple Rule', 'Baseline XGBoost', 'GNN (GraphSAGE)']
scores = [simple_f1, 0.8913, gnn_f1]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores, color=['gray', 'coral', 'cyan'])
plt.title('Traffic Prediction F1 Score Comparison (15m Horizon)')
plt.ylabel('F1 Score')
plt.ylim([0.80, 1.0])
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

plt.savefig(os.path.join(OUT, 'q16_gnn_comparison.png'), facecolor='white')
plt.close()

print("Comparison plot saved to output/q16_gnn_comparison.png")
print("Script completed successfully.")
