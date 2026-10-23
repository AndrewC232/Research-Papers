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
EXISTING_CSV = os.path.join(OUT, 'st_gnn_sensitivity.csv')

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

# 4. New radii only
new_radii = [2.5, 3.0, 4.0, 4.5, 15.0, 20.0]
new_results = []

for r in new_radii:
    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < r:
                edge_index.append([i, j])

    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device) if edge_index else torch.empty((2, 0), dtype=torch.long).to(device)
    num_edges = e_tensor.size(1)
    print(f"\n--- Testing Radius: {r} miles ({num_edges} edges) ---")

    model = LSTM_TrafficGNN(1, 16, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(25):
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
    with torch.no_grad():
        for i in range(len(test_X)):
            out = model(test_X[i].to(device), e_tensor).squeeze()
            preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_targets.extend(test_y[i].numpy())

    f1 = f1_score(all_targets, all_preds)
    print(f"Radius {r}m F1 Score: {f1:.4f}")
    new_results.append({'Radius_Miles': r, 'Edges': num_edges, 'F1_Score': f1})

# 5. Merge with existing results and sort
existing_df = pd.read_csv(EXISTING_CSV)
new_df = pd.DataFrame(new_results)
combined_df = pd.concat([existing_df, new_df], ignore_index=True).sort_values('Radius_Miles').reset_index(drop=True)
combined_df.to_csv(EXISTING_CSV, index=False)
print("\n--- Combined Results ---")
print(combined_df.to_string(index=False))

# 6. Plot
fig, ax1 = plt.subplots(figsize=(13, 7))

color_f1 = 'cyan'
color_edges = '#FF6B6B'

ax1.plot(combined_df['Radius_Miles'], combined_df['F1_Score'], marker='o', color=color_f1, linewidth=2.5, markersize=9, label='F1 Score', zorder=3)
ax1.set_xlabel('Connection Radius (Miles)', fontsize=13)
ax1.set_ylabel('F1 Score', color=color_f1, fontsize=13)
ax1.tick_params(axis='y', labelcolor=color_f1)
ax1.set_ylim([0.65, 0.74])
ax1.grid(True, alpha=0.15)

# Highlight best point
best_idx = combined_df['F1_Score'].idxmax()
best_r = combined_df.loc[best_idx, 'Radius_Miles']
best_f1 = combined_df.loc[best_idx, 'F1_Score']
ax1.annotate(f'Peak: {best_f1:.4f}\n@ {best_r} mi',
             xy=(best_r, best_f1),
             xytext=(best_r + 0.5, best_f1 + 0.005),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

# Annotate all points with F1 values
for _, row in combined_df.iterrows():
    ax1.annotate(f"{row['F1_Score']:.4f}",
                 (row['Radius_Miles'], row['F1_Score']),
                 textcoords="offset points", xytext=(0, 10),
                 ha='center', fontsize=8, color=color_f1)

# Secondary axis for edge count
ax2 = ax1.twinx()
ax2.bar(combined_df['Radius_Miles'], combined_df['Edges'], alpha=0.2, color=color_edges, width=0.5, label='Edge Count', zorder=1)
ax2.set_ylabel('Number of Graph Edges', color=color_edges, fontsize=13)
ax2.tick_params(axis='y', labelcolor=color_edges)

ax1.set_title('Spatio-Temporal GNN Performance vs. Spatial Connection Radius\n(LSTM-GNN, 25 Epochs, 2-Week Dataset)', fontsize=14, pad=15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_range_sensitivity.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nUpdated plot saved to {OUT}/st_gnn_range_sensitivity.png")
