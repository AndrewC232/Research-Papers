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

# Configuration
device = get_device()

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(OUT, 'best_gnn_8mi.pth')

CONG_THRESH = 60.0
seq_len, horizon = 12, 3
MAX_EPOCHS = 200
PATIENCE = 10
RADIUS_MILES = 8.0

# 1. Load Data
print("Loading data...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:4032] # 2-week slice
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
    # [325 sensors, 12 timesteps, 1 feature]
    feat = norm_speeds[i:i+seq_len].T.reshape(len(valid_sensors), seq_len, 1)
    # [325 sensors] binary labels
    target = (df.values[i+seq_len+horizon] < CONG_THRESH).astype(np.float32)
    X.append(feat)
    y.append(target)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

split = int(len(X) * 0.8)
X_tr, y_tr = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]
print(f"Train samples: {len(X_tr)}, Val samples: {len(X_val)}")

# 3. Build Graph Structure (8.0 Miles)
edge_index = []
for i in range(len(valid_sensors)):
    for j in range(len(valid_sensors)):
        if i != j and dist_matrix[i, j] < RADIUS_MILES:
            edge_index.append([i, j])

e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device)
print(f"Graph initialized with 8.0 mile radius: {e_tensor.size(1)} edges.")

# 4. Model Definition
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

model = LSTM_TrafficGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()

# 5. Training Loop
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
best_val_loss = float('inf')
best_val_f1 = 0.0
best_val_prec = 0.0
best_val_rec = 0.0
patience_counter = 0

print("\nStarting Training...")
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
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i in range(len(X_val)):
            out = model(X_val[i].to(device), e_tensor).squeeze()
            val_loss += criterion(out, y_val[i].to(device)).item()
            preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_targets.extend(y_val[i].numpy())
            
    avg_val_loss = val_loss / len(X_val)
    val_f1 = f1_score(all_targets, all_preds)
    val_prec = precision_score(all_targets, all_preds, zero_division=0)
    val_rec = recall_score(all_targets, all_preds, zero_division=0)
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_f1'].append(val_f1)
    
    print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    # Early Stopping Check (on Val Loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    # Periodic Model Saving Check (on Val F1)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_val_prec = val_prec
        best_val_rec = val_rec
    if epoch % 10 == 0:
        if val_f1 >= best_val_f1:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> Epoch {epoch}: Checkpoint saved! (Best F1: {best_val_f1:.4f})")
            
    if patience_counter >= PATIENCE:
        print(f"\nEarly Stopping triggered at Epoch {epoch}! No improvement in Val Loss for {PATIENCE} epochs.")
        break

# 6. Plot Learning Curves
fig, ax1 = plt.subplots(figsize=(12, 6))

ax2 = ax1.twinx()
ax1.plot(history['train_loss'], label='Train Loss', color='coral', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2, linestyle='--')
ax2.plot(history['val_f1'], label='Val F1 Score', color='blue', linewidth=2.5)

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Binary Cross-Entropy Loss', color='coral', fontsize=12)
ax2.set_ylabel('F1 Score', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='coral')
ax2.tick_params(axis='y', labelcolor='blue')

ax1.set_title(f'LSTM-GNN (8.0mi) Learning Curves over {len(history["val_f1"])} Epochs', fontsize=14)
ax1.grid(True, alpha=0.15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'gnn_8mi_learning_curves.png'), dpi=200, facecolor='white')
plt.close()

print("\nSaved learning curves to gnn_8mi_learning_curves.png")
print(f"Training run complete! Best F1: {best_val_f1:.4f} | Prec: {best_val_prec:.4f} | Rec: {best_val_rec:.4f}")
