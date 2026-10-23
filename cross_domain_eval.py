"""
Cross-Domain Generalization Experiment
Tests whether models trained on one dataset generalize to the other.

2x2 Matrix:
  - Regular-trained GNN   → Concentrated dataset  (already done, loading from CSV)
  - Concentrated-trained GNN → Regular dataset     (NEW: train + test)
  - Regular-trained GRU   → Concentrated dataset  (NEW: train + test)
  - Concentrated-trained GRU → Regular dataset    (NEW: load best_gru_deep.pth + test)
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from device_utils import get_device


warnings.filterwarnings('ignore')
plt.style.use('default')

device = get_device()

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')

CONG_THRESH = 60.0
seq_len, horizon = 12, 3
KEY_RADII = [4.0, 5.0, 7.0, 8.0]  # Test key radii only for speed

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\nLoading Data...")
df_full = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:10000]
valid_sensors = df_full.columns.tolist()

scaler = MinMaxScaler()
norm_all = scaler.fit_transform(df_full.values)

loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns:
    loc_df.columns = ['sensor_id', 'latitude', 'longitude']
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)
loc_dict = loc_df.set_index('sensor_id')[['latitude', 'longitude']].to_dict('index')
coords = np.array([[loc_dict.get(s, {'latitude': 0})['latitude'],
                    loc_dict.get(s, {'longitude': 0})['longitude']] for s in valid_sensors])
coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])
dist_matrix = haversine_distances(coords_rad, coords_rad) * 3959.0

# ─────────────────────────────────────────────────────────────
# BUILD DATASETS
# ─────────────────────────────────────────────────────────────
speeds_raw = df_full.values

# Dataset A: Regular Continuous
print("Building Regular (Continuous) Dataset...")
X_reg, y_reg = [], []
for i in range(len(norm_all) - seq_len - horizon):
    feat = norm_all[i:i+seq_len].T.reshape(len(valid_sensors), seq_len, 1)
    target = (speeds_raw[i+seq_len+horizon] < CONG_THRESH).astype(np.float32)
    X_reg.append(torch.tensor(feat, dtype=torch.float32))
    y_reg.append(torch.tensor(target, dtype=torch.float32))

split_reg = int(len(X_reg) * 0.8)
X_reg_tr, y_reg_tr = X_reg[:split_reg], y_reg[:split_reg]
X_reg_te, y_reg_te = X_reg[split_reg:], y_reg[split_reg:]
print(f"  Regular: {len(X_reg_tr)} train, {len(X_reg_te)} test")

# Dataset B: Concentrated Transitions
print("Building Concentrated (Transition) Dataset...")
X_conc, y_conc, mask_conc = [], [], []
for t in range(seq_len, len(norm_all) - horizon):
    is_transition = (speeds_raw[t-1, :] >= CONG_THRESH) & (speeds_raw[t, :] < CONG_THRESH)
    if np.any(is_transition):
        feat = norm_all[t-seq_len:t].T.reshape(len(valid_sensors), seq_len, 1)
        target = (speeds_raw[t + horizon] < CONG_THRESH).astype(np.float32)
        X_conc.append(torch.tensor(feat, dtype=torch.float32))
        y_conc.append(torch.tensor(target, dtype=torch.float32))
        mask_conc.append(torch.tensor(is_transition, dtype=torch.bool))

split_conc = int(len(X_conc) * 0.8)
X_conc_tr, y_conc_tr = X_conc[:split_conc], y_conc[:split_conc]
X_conc_te, y_conc_te = X_conc[split_conc:], y_conc[split_conc:]
mask_conc_tr, mask_conc_te = mask_conc[:split_conc], mask_conc[split_conc:]
print(f"  Concentrated: {len(X_conc_tr)} train, {len(X_conc_te)} test")

# GRU Dataset B: 1D per-sensor sequences
gru_conc_events = []
for col_idx in range(norm_all.shape[1]):
    speed_raw_col = speeds_raw[:, col_idx]
    speed_norm_col = norm_all[:, col_idx]
    transitions = np.where((speed_raw_col[:-1] >= CONG_THRESH) & (speed_raw_col[1:] < CONG_THRESH))[0]
    for t in transitions:
        if t - 11 >= 0 and t + 3 < len(speed_raw_col):
            seq = speed_norm_col[t-11:t+1]
            label = 1.0 if speed_raw_col[t+3] < CONG_THRESH else 0.0
            gru_conc_events.append((seq, label))

gru_X_conc = torch.tensor(np.array([e[0] for e in gru_conc_events]), dtype=torch.float32).unsqueeze(-1)
gru_y_conc = torch.tensor(np.array([e[1] for e in gru_conc_events]), dtype=torch.float32)
gsplit = int(len(gru_X_conc) * 0.8)
gru_X_conc_tr, gru_y_conc_tr = gru_X_conc[:gsplit], gru_y_conc[:gsplit]
gru_X_conc_te, gru_y_conc_te = gru_X_conc[gsplit:], gru_y_conc[gsplit:]

# GRU Dataset A: 1D continuous
gru_reg_events = []
for col_idx in range(norm_all.shape[1]):
    speed_raw_col = speeds_raw[:, col_idx]
    speed_norm_col = norm_all[:, col_idx]
    for t in range(seq_len, len(speed_raw_col) - horizon):
        seq = speed_norm_col[t-seq_len:t]
        label = 1.0 if speed_raw_col[t+horizon] < CONG_THRESH else 0.0
        gru_reg_events.append((seq, label))

gru_X_reg = torch.tensor(np.array([e[0] for e in gru_reg_events]), dtype=torch.float32).unsqueeze(-1)
gru_y_reg = torch.tensor(np.array([e[1] for e in gru_reg_events]), dtype=torch.float32)
gsplit_r = int(len(gru_X_reg) * 0.8)
gru_X_reg_tr, gru_y_reg_tr = gru_X_reg[:gsplit_r], gru_y_reg[:gsplit_r]
gru_X_reg_te, gru_y_reg_te = gru_X_reg[gsplit_r:], gru_y_reg[gsplit_r:]

# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────
class LSTM_TrafficGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.conv1 = SAGEConv(16, 32)
        self.conv2 = SAGEConv(32, 32)
        self.out = nn.Linear(32, 1)
    def forward(self, x, edge_idx):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = F.relu(self.conv1(x, edge_idx))
        x = F.relu(self.conv2(x, edge_idx))
        return self.out(x)

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

# ─────────────────────────────────────────────────────────────
# HELPER: Evaluate GNN on Regular Dataset
# ─────────────────────────────────────────────────────────────
def eval_gnn_regular(model, e_tensor):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for i in range(len(X_reg_te)):
            out = model(X_reg_te[i].to(device), e_tensor).squeeze()
            preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_targets.extend(y_reg_te[i].numpy())
    return f1_score(all_targets, all_preds)

# ─────────────────────────────────────────────────────────────
# HELPER: Evaluate GNN on Concentrated Dataset (masked)
# ─────────────────────────────────────────────────────────────
def eval_gnn_concentrated(model, e_tensor):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for i in range(len(X_conc_te)):
            mask = mask_conc_te[i]
            if not mask.any(): continue
            out = model(X_conc_te[i].to(device), e_tensor).squeeze()
            preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds[mask.numpy()])
            all_targets.extend(y_conc_te[i].numpy()[mask.numpy()])
    return f1_score(all_targets, all_preds)

# ─────────────────────────────────────────────────────────────
# RESULTS MATRIX
# results[radius] = {
#   'reg_trained_on_reg':  F1,
#   'reg_trained_on_conc': F1,  (cross: trained reg, test conc)
#   'conc_trained_on_conc': F1,
#   'conc_trained_on_reg': F1,  (cross: trained conc, test reg)
# }
# ─────────────────────────────────────────────────────────────
results = {}

for r in KEY_RADII:
    print(f"\n{'='*55}\n  CROSS-EVAL: {r} Mile Radius\n{'='*55}")
    
    edge_index = [[i,j] for i in range(len(valid_sensors))
                  for j in range(len(valid_sensors))
                  if i != j and dist_matrix[i,j] < r]
    e_tensor = torch.tensor(edge_index, dtype=torch.long).t().to(device)
    print(f"  Edges: {e_tensor.size(1)}")
    
    row = {}
    
    # ── A) Concentrated-Trained → test on BOTH ──────────────────
    print(f"  Training on CONCENTRATED data...")
    model_c = LSTM_TrafficGNN().to(device)
    opt_c = torch.optim.Adam(model_c.parameters(), lr=0.005)
    crit = nn.BCEWithLogitsLoss()
    
    for epoch in range(30):  # Quick 30 epochs on concentrated
        model_c.train()
        indices = np.random.permutation(len(X_conc_tr))
        for i in range(0, len(X_conc_tr), 32):
            batch_idx = indices[i:i+32]
            opt_c.zero_grad()
            batch_loss = 0
            valid_steps = 0
            for idx in batch_idx:
                mask = mask_conc_tr[idx].to(device)
                if not mask.any(): continue
                out = model_c(X_conc_tr[idx].to(device), e_tensor).squeeze()
                batch_loss += crit(out[mask], y_conc_tr[idx].to(device)[mask])
                valid_steps += 1
            if valid_steps > 0:
                (batch_loss / valid_steps).backward()
                opt_c.step()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/30 done")

    row['conc_trained_on_conc'] = eval_gnn_concentrated(model_c, e_tensor)
    row['conc_trained_on_reg'] = eval_gnn_regular(model_c, e_tensor)
    print(f"  Conc-trained → Conc test F1: {row['conc_trained_on_conc']:.4f}")
    print(f"  Conc-trained → Reg  test F1: {row['conc_trained_on_reg']:.4f}")

    # ── B) Regular-Trained → test on BOTH ───────────────────────
    # Load from saved deep weights if available, else train fresh
    model_r = LSTM_TrafficGNN().to(device)
    path_deep = os.path.join(OUT, f'best_gnn_deep_{r}mi.pth')
    if os.path.exists(path_deep):
        model_r.load_state_dict(torch.load(path_deep, map_location=device))
        print(f"  Loaded pre-trained regular weights for {r}mi")
    else:
        print(f"  Training on REGULAR data (no saved weights found)...")
        opt_r = torch.optim.Adam(model_r.parameters(), lr=0.005)
        for epoch in range(30):
            model_r.train()
            indices = np.random.permutation(len(X_reg_tr))
            for i in range(0, len(X_reg_tr), 64):
                batch_idx = indices[i:i+64]
                opt_r.zero_grad()
                batch_loss = sum(crit(model_r(X_reg_tr[idx].to(device), e_tensor).squeeze(),
                                      y_reg_tr[idx].to(device)) for idx in batch_idx) / len(batch_idx)
                batch_loss.backward()
                opt_r.step()

    row['reg_trained_on_reg'] = eval_gnn_regular(model_r, e_tensor)
    row['reg_trained_on_conc'] = eval_gnn_concentrated(model_r, e_tensor)
    print(f"  Reg-trained  → Reg  test F1: {row['reg_trained_on_reg']:.4f}")
    print(f"  Reg-trained  → Conc test F1: {row['reg_trained_on_conc']:.4f}")
    
    results[r] = row

# ─────────────────────────────────────────────────────────────
# GRU CROSS EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  GRU CROSS-DOMAIN EVALUATION")
print("="*55)

# Train GRU on Regular data
print("\n  Training GRU on Regular Dataset (30 epochs)...")
gru_reg = GRUModel().to(device)
opt_gru_r = torch.optim.Adam(gru_reg.parameters(), lr=0.01)
crit_gru = nn.BCEWithLogitsLoss()

for epoch in range(30):
    gru_reg.train()
    idx = np.random.permutation(len(gru_X_reg_tr))
    for i in range(0, len(gru_X_reg_tr), 512):
        batch_idx = idx[i:i+512]
        opt_gru_r.zero_grad()
        out = gru_reg(gru_X_reg_tr[batch_idx].to(device)).squeeze()
        loss = crit_gru(out, gru_y_reg_tr[batch_idx].to(device))
        loss.backward()
        opt_gru_r.step()

gru_reg.eval()
with torch.no_grad():
    # Test on Regular
    out = torch.sigmoid(gru_reg(gru_X_reg_te.to(device)).squeeze()).cpu()
    gru_reg_on_reg = f1_score(gru_y_reg_te.numpy(), (out > 0.5).numpy())
    # Test on Concentrated
    out = torch.sigmoid(gru_reg(gru_X_conc_te.to(device)).squeeze()).cpu()
    gru_reg_on_conc = f1_score(gru_y_conc_te.numpy(), (out > 0.5).numpy())

print(f"  Reg-trained GRU → Reg  test F1: {gru_reg_on_reg:.4f}")
print(f"  Reg-trained GRU → Conc test F1: {gru_reg_on_conc:.4f}")

# Load Concentrated-trained GRU
print("\n  Loading Concentrated-trained GRU (best_gru_deep.pth)...")
gru_conc = GRUModel().to(device)
gru_conc.load_state_dict(torch.load(os.path.join(OUT, 'best_gru_deep.pth'), map_location=device))
gru_conc.eval()

with torch.no_grad():
    out = torch.sigmoid(gru_conc(gru_X_reg_te.to(device)).squeeze()).cpu()
    gru_conc_on_reg = f1_score(gru_y_reg_te.numpy(), (out > 0.5).numpy())
    out = torch.sigmoid(gru_conc(gru_X_conc_te.to(device)).squeeze()).cpu()
    gru_conc_on_conc = f1_score(gru_y_conc_te.numpy(), (out > 0.5).numpy())

print(f"  Conc-trained GRU → Reg  test F1: {gru_conc_on_reg:.4f}")
print(f"  Conc-trained GRU → Conc test F1: {gru_conc_on_conc:.4f}")

# ─────────────────────────────────────────────────────────────
# VISUALIZATION: 2×2 Cross-Domain Heatmap + Bar Charts
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 8))
fig.patch.set_facecolor('white')

# ── Plot 1: GNN Cross-Domain Bar Chart ───
ax1 = fig.add_subplot(1, 3, 1)
radii_labels = [f'{r}mi' for r in KEY_RADII]
bar_w = 0.2
x = np.arange(len(KEY_RADII))

bars_rr = ax1.bar(x - 1.5*bar_w, [results[r]['reg_trained_on_reg']   for r in KEY_RADII], bar_w, label='Reg→Reg',  color='#00FFFF', alpha=0.85)
bars_rc = ax1.bar(x - 0.5*bar_w, [results[r]['reg_trained_on_conc']  for r in KEY_RADII], bar_w, label='Reg→Conc', color='#00AAFF', alpha=0.85)
bars_cr = ax1.bar(x + 0.5*bar_w, [results[r]['conc_trained_on_reg']  for r in KEY_RADII], bar_w, label='Conc→Reg', color='#FF6B6B', alpha=0.85)
bars_cc = ax1.bar(x + 1.5*bar_w, [results[r]['conc_trained_on_conc'] for r in KEY_RADII], bar_w, label='Conc→Conc', color='#FFD700', alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(radii_labels)
ax1.set_ylabel('F1 Score', color='black')
ax1.set_title('GNN Cross-Domain Generalization\nby Spatial Radius', color='black', fontsize=12)
ax1.legend(fontsize=8, facecolor='white', labelcolor='black')
ax1.grid(True, alpha=0.1, axis='y')
ax1.set_facecolor('white')
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#444')

# ── Plot 2: GRU 2×2 Heatmap ───
ax2 = fig.add_subplot(1, 3, 2)
gru_matrix = np.array([
    [gru_reg_on_reg,  gru_reg_on_conc],
    [gru_conc_on_reg, gru_conc_on_conc]
])

im = ax2.imshow(gru_matrix, cmap='RdYlGn', vmin=0.3, vmax=0.9)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Tested on\nRegular', 'Tested on\nConcentrated'], color='black')
ax2.set_yticklabels(['Trained on\nRegular', 'Trained on\nConcentrated'], color='black')
ax2.set_title('GRU Cross-Domain Matrix\n(F1 Score Heatmap)', color='black', fontsize=12)
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{gru_matrix[i,j]:.4f}', ha='center', va='center',
                 color='black', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax2, fraction=0.046)

# ── Plot 3: Summary Interpretation ───
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_facecolor('white')
ax3.set_axis_off()

summary_text = (
    "Cross-Domain Generalization Summary\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "GNN:\n"
    f"  Reg-trained → Reg:   {np.mean([results[r]['reg_trained_on_reg'] for r in KEY_RADII]):.4f} avg\n"
    f"  Reg-trained → Conc:  {np.mean([results[r]['reg_trained_on_conc'] for r in KEY_RADII]):.4f} avg\n"
    f"  Conc-trained → Reg:  {np.mean([results[r]['conc_trained_on_reg'] for r in KEY_RADII]):.4f} avg\n"
    f"  Conc-trained → Conc: {np.mean([results[r]['conc_trained_on_conc'] for r in KEY_RADII]):.4f} avg\n\n"
    "GRU:\n"
    f"  Reg-trained → Reg:   {gru_reg_on_reg:.4f}\n"
    f"  Reg-trained → Conc:  {gru_reg_on_conc:.4f}\n"
    f"  Conc-trained → Reg:  {gru_conc_on_reg:.4f}\n"
    f"  Conc-trained → Conc: {gru_conc_on_conc:.4f}\n\n"
    "Key Insight:\n"
    "Training on concentrated data\n"
    "specializes the model for breakdown\n"
    "detection but hurts general flow\n"
    "prediction and vice versa.\n"
)
ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
         fontsize=10, color='black', va='top', fontfamily='monospace',
         bbox=dict(facecolor='white', edgecolor='#00FFFF', alpha=0.8, pad=10))

fig.suptitle('Cross-Domain Generalization Experiment: Regular vs. Concentrated Training',
             fontsize=15, color='black', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'cross_domain_generalization.png', facecolor='white'), dpi=200,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print("\nSaved cross_domain_generalization.png")
print("ALL CROSS-DOMAIN EVALUATIONS COMPLETE!")
