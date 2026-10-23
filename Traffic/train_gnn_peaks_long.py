import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from math import radians
from sklearn.metrics.pairwise import haversine_distances
import matplotlib.pyplot as plt
import warnings
from device_utils import get_device


warnings.filterwarnings('ignore')
plt.style.use('default')

device = get_device()

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

CONG_THRESH = 60.0
seq_len, horizon = 12, 3
MAX_EPOCHS = 200
PATIENCE = 10

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
print("Building Continuous Spatio-Temporal Dataset...")
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
print(f"Train samples: {len(X_tr)}, Val samples: {len(X_val)}")

# 3. Model Definition
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

# 4. Train Each Peak Radius
RADII = [2.0, 4.5]
all_histories = {}

for radius in RADII:
    print(f"\n{'='*60}")
    print(f"  TRAINING {radius} MILE RADIUS")
    print(f"{'='*60}")
    
    # Build graph
    edge_index = []
    for i in range(len(valid_sensors)):
        for j in range(len(valid_sensors)):
            if i != j and dist_matrix[i, j] < radius:
                edge_index.append([i, j])
    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device)
    print(f"Graph: {e_tensor.size(1)} edges")

    model = LSTM_TrafficGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_prec = 0.0
    best_val_rec = 0.0
    patience_counter = 0
    save_path = os.path.join(OUT, f'best_gnn_{radius}mi.pth')

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        indices = np.random.permutation(len(X_tr))
        epoch_train_loss = 0

        for i in range(0, len(X_tr), 64):
            batch_idx = indices[i:i+64]
            optimizer.zero_grad()
            batch_loss = 0
            for idx in batch_idx:
                out = model(X_tr[idx].to(device), e_tensor).squeeze()
                batch_loss += criterion(out, y_tr[idx].to(device))
            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss.item()

        avg_train_loss = epoch_train_loss / (len(X_tr) / 64)

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for i in range(len(X_val)):
                out = model(X_val[i].to(device), e_tensor).squeeze()
                val_loss += criterion(out, y_val[i].to(device)).item()
                preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
                all_preds.append(preds)
                all_targets.append(y_val[i].numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        avg_val_loss = val_loss / len(X_val)
        val_f1 = f1_score(all_targets, all_preds)
        val_prec = precision_score(all_targets, all_preds, zero_division=0)
        val_rec = recall_score(all_targets, all_preds, zero_division=0)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Periodic Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_prec = val_prec
            best_val_rec = val_rec
        if epoch % 10 == 0:
            if val_f1 >= best_val_f1:
                torch.save(model.state_dict(), save_path)
                print(f"  --> Epoch {epoch}: Checkpoint saved! (Best F1: {best_val_f1:.4f})")

        if patience_counter >= PATIENCE:
            print(f"\nEarly Stopping at Epoch {epoch}! No improvement for {PATIENCE} epochs.")
            break

    all_histories[radius] = history
    print(f"\n{radius}mi DONE — Best F1: {best_val_f1:.4f} | Prec: {best_val_prec:.4f} | Rec: {best_val_rec:.4f} over {len(history['val_f1'])} epochs")

# 5. Combined Learning Curves Plot (all 3 radii including 8.0mi)
# Load the 8.0mi results we already have for comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor('white')

radii_plot = [2.0, 4.5]
colors_radii = ['#00FFFF', '#FFD700']

for ax, radius, color in zip(axes[:2], radii_plot, colors_radii):
    h = all_histories[radius]
    epochs_range = range(1, len(h['val_f1']) + 1)

    ax2 = ax.twinx()
    ax.plot(epochs_range, h['train_loss'], color='coral', linewidth=1.5, alpha=0.7, label='Train Loss')
    ax.plot(epochs_range, h['val_loss'], color='orange', linewidth=1.5, linestyle='--', label='Val Loss')
    ax2.plot(epochs_range, h['val_f1'], color=color, linewidth=2.5, label='Val F1')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', color='coral', fontsize=11)
    ax2.set_ylabel('F1 Score', color=color, fontsize=11)
    ax.tick_params(axis='y', labelcolor='coral')
    ax2.tick_params(axis='y', labelcolor=color)
    ax.grid(True, alpha=0.1)

    best_f1 = max(h['val_f1'])
    best_ep = h['val_f1'].index(best_f1) + 1
    ax.set_title(f'{radius}mi Radius\nBest F1: {best_f1:.4f} @ Epoch {best_ep}', fontsize=12, color='black')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)

# Third panel: placeholder message pointing to existing 8.0mi plot
axes[2].text(0.5, 0.5, '8.0 mi Radius\nSee gnn_8mi_learning_curves.png\n\nBest F1: 0.7197 @ Epoch 69\nEarly Stop: Epoch 81',
             transform=axes[2].transAxes, ha='center', va='center', fontsize=13, color='#FF6B6B',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF6B6B'))
axes[2].set_facecolor('white')
axes[2].set_axis_off()

fig.suptitle('LSTM-GNN 200-Epoch Training: Learning Curves by Spatial Radius', fontsize=15, color='black', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'gnn_peaks_learning_curves.png'), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print("\nSaved gnn_peaks_learning_curves.png")
print("ALL DONE!")
