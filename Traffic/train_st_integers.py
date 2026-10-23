import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
CSV_FILE = os.path.join(OUT, 'st_gnn_sensitivity.csv')

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

# 4. Only train the 4 missing integer radii
new_radii = [6, 7, 8, 9]
new_results = []

for r in new_radii:
    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < r:
                edge_index.append([i, j])

    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device) \
        if edge_index else torch.empty((2, 0), dtype=torch.long).to(device)
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
    print(f"Radius {r}mi F1 Score: {f1:.4f}")
    new_results.append({'Radius_Miles': float(r), 'Edges': num_edges, 'F1_Score': f1})

# 5. Merge, keep only integer radii 1-10 + new ones, sort
existing_df = pd.read_csv(CSV_FILE)
new_df = pd.DataFrame(new_results)
combined_df = pd.concat([existing_df, new_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset='Radius_Miles', keep='last')
combined_df = combined_df.sort_values('Radius_Miles').reset_index(drop=True)

# Filter to clean integer 1-10 range for the primary plot (+ keep 15, 20 in CSV)
integers_df = combined_df[combined_df['Radius_Miles'].isin([1,2,3,4,5,6,7,8,9,10])].copy()

combined_df.to_csv(CSV_FILE, index=False)
print("\n--- Integer 1–10 Results ---")
print(integers_df[['Radius_Miles','Edges','F1_Score']].to_string(index=False))

# 6. Performance curve (integers 1-10 only for clean plot)
fig, ax1 = plt.subplots(figsize=(14, 7))
color_f1 = 'cyan'
color_edges = '#FF6B6B'

ax2 = ax1.twinx()
ax2.bar(integers_df['Radius_Miles'], integers_df['Edges'], alpha=0.15,
        color=color_edges, width=0.5, label='Edge Count', zorder=1)
ax2.set_ylabel('Number of Graph Edges', color=color_edges, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_edges)

ax1.plot(integers_df['Radius_Miles'], integers_df['F1_Score'], marker='o', color=color_f1,
         linewidth=2.5, markersize=10, label='F1 Score', zorder=3)
ax1.fill_between(integers_df['Radius_Miles'], integers_df['F1_Score'],
                 alpha=0.08, color=color_f1)
ax1.set_xlabel('Connection Radius (Miles)', fontsize=13)
ax1.set_ylabel('F1 Score', color=color_f1, fontsize=13)
ax1.tick_params(axis='y', labelcolor=color_f1)
ax1.set_ylim([0.665, 0.735])
ax1.set_xticks(integers_df['Radius_Miles'])
ax1.set_xticklabels([f'{int(r)} mi' for r in integers_df['Radius_Miles']], fontsize=11)
ax1.grid(True, alpha=0.12)

# Highlight global peak
best_idx = integers_df['F1_Score'].idxmax()
best_r = integers_df.loc[best_idx, 'Radius_Miles']
best_f1 = integers_df.loc[best_idx, 'F1_Score']
ax1.scatter(best_r, best_f1, color='red', s=200, zorder=5, marker='★')
ax1.annotate(f'Global Peak\n{int(best_r)} mi — F1={best_f1:.4f}',
             xy=(best_r, best_f1),
             xytext=(best_r - 2.5, best_f1 + 0.006),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Annotate each point
for _, row in integers_df.iterrows():
    ax1.annotate(f"{row['F1_Score']:.4f}",
                 (row['Radius_Miles'], row['F1_Score']),
                 textcoords="offset points", xytext=(0, 12),
                 ha='center', fontsize=8.5, color=color_f1)

ax1.set_title('LSTM-GNN F1 Score vs. Spatial Connection Radius (1–10 Miles)\n25 Epochs | 325 Bay Area PeMS Sensors', fontsize=14, pad=15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_integer_curve.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("\nSaved st_gnn_integer_curve.png")
print("Done!")
