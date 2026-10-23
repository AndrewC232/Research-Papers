"""
Train GRU, TCN, vanilla LSTM, and Transformer to 200 epochs (early stopping)
on the concentrated transition dataset, then update the master comparison charts.
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')

# ── Device ──────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# ── 1. Build concentrated transition dataset ─────────────────────────────────
print("Loading data and identifying transitions...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:20000]
CONG_THRESH = 60.0

event_data = []
for col in df.columns:
    speed = df[col].values
    transitions = np.where((speed[:-1] >= CONG_THRESH) & (speed[1:] < CONG_THRESH))[0]
    for t in transitions:
        if t - 12 >= 0 and t + 3 < len(speed):
            event_data.append({
                'seq': speed[t-12+1:t+1],
                'target': 1 if speed[t+3] < CONG_THRESH else 0
            })

X = np.array([e['seq'] for e in event_data])
y = np.array([e['target'] for e in event_data])

n = len(X)
tr_end, val_end = int(n * 0.7), int(n * 0.85)
print(f"Dataset: {n} events | Train: {tr_end} | Val: {val_end - tr_end} | Test: {n - val_end}")

# GRU / LSTM / Transformer expect (B, T, F)  → unsqueeze last dim
X_tr_gru  = torch.tensor(X[:tr_end],       dtype=torch.float32).unsqueeze(-1).to(device)
y_tr_gru  = torch.tensor(y[:tr_end],       dtype=torch.float32).unsqueeze(-1).to(device)
X_val_gru = torch.tensor(X[tr_end:val_end], dtype=torch.float32).unsqueeze(-1).to(device)
y_val_gru = torch.tensor(y[tr_end:val_end], dtype=torch.float32).unsqueeze(-1).to(device)
X_te_gru  = torch.tensor(X[val_end:],       dtype=torch.float32).unsqueeze(-1).to(device)
y_te_gru  = torch.tensor(y[val_end:],       dtype=torch.float32).unsqueeze(-1).to(device)

# TCN expects (B, F, T) → unsqueeze dim 1
X_tr_tcn  = torch.tensor(X[:tr_end],       dtype=torch.float32).unsqueeze(1).to(device)
y_tr_tcn  = torch.tensor(y[:tr_end],       dtype=torch.float32).unsqueeze(-1).to(device)
X_val_tcn = torch.tensor(X[tr_end:val_end], dtype=torch.float32).unsqueeze(1).to(device)
y_val_tcn = torch.tensor(y[tr_end:val_end], dtype=torch.float32).unsqueeze(-1).to(device)
X_te_tcn  = torch.tensor(X[val_end:],       dtype=torch.float32).unsqueeze(1).to(device)
y_te_tcn  = torch.tensor(y[val_end:],       dtype=torch.float32).unsqueeze(-1).to(device)


# ── 2. Model definitions ─────────────────────────────────────────────────────

class OptimizedGRU(nn.Module):
    def __init__(self, in_dim=1, hidden=64, layers=2, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=drop if layers > 1 else 0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class OptimizedLSTM(nn.Module):
    def __init__(self, in_dim=1, hidden=64, layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=drop if layers > 1 else 0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation, kernel=5):
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=kernel, padding=padding, dilation=dilation)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=1)
        self.res   = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        out = self.relu(self.conv1(x))
        out = self.drop(out)
        out = self.conv2(out)
        if out.shape[-1] != res.shape[-1]:
            out = out[:, :, :res.shape[-1]]
        return self.relu(out + res)


class OptimizedTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(1,  32, 1),
            TCNBlock(32, 64, 2),
            TCNBlock(64, 128, 4),
            TCNBlock(128, 128, 8),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x)
        return self.fc(self.pool(x).squeeze(-1))


class TransformerClassifier(nn.Module):
    def __init__(self, in_dim=1, d_model=64, nhead=4, num_layers=2, drop=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                     dim_feedforward=128, dropout=drop,
                                                     batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))


# ── 3. Generic training loop ─────────────────────────────────────────────────

def train_model(name, model, X_tr, y_tr, X_val, y_val, X_te, y_te,
                max_epochs=500, patience=50, lr=0.0005):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Step scheduler based on F1 (mode='max') instead of loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1      = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    val_f1_scores = []

    print(f"\n{'='*50}")
    print(f" TRAINING {name}  (max {max_epochs} epochs)")
    print(f"{'='*50}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_out  = model(X_val)
            v_loss = criterion(v_out, y_val)
            preds  = (torch.sigmoid(v_out) > 0.5).cpu().numpy().astype(int)
            val_f1 = f1_score(y_val.cpu().numpy(), preds, zero_division=0)

        train_losses.append(loss.item())
        val_losses.append(v_loss.item())
        val_f1_scores.append(val_f1)
        scheduler.step(val_f1)

        # Early stopping based on F1 Score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save the best model state
            torch.save(model.state_dict(), os.path.join(OUT, f'best_seq_{name}.pth'))
        else:
            patience_counter += 1

        if epoch % 20 == 0 or patience_counter == patience:
            print(f"  Ep {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {v_loss.item():.4f} | Val F1: {val_f1:.4f} (Best: {best_val_f1:.4f})")

        if patience_counter >= patience:
            print(f"  -> Early stop triggered at epoch {epoch} (No F1 improvement for {patience} epochs)")
            break

    # Final test evaluation using the best saved model state
    model.load_state_dict(torch.load(os.path.join(OUT, f'best_seq_{name}.pth')))
    model.eval()
    with torch.no_grad():
        te_probs = torch.sigmoid(model(X_te))
        te_preds = (te_probs > 0.5).cpu().numpy().astype(int)
        y_true = y_te.cpu().numpy()
        test_f1  = f1_score(y_true, te_preds, zero_division=0)
        test_prec = precision_score(y_true, te_preds, zero_division=0)
        test_rec = recall_score(y_true, te_preds, zero_division=0)

    print(f"  >> {name} — Best Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | Epochs: {epoch}")
    return test_f1, test_prec, test_rec, train_losses, val_losses, epoch


# ── 4. Run all models ─────────────────────────────────────────────────────────

models_to_train = [
    ('GRU',         OptimizedGRU(),            X_tr_gru, y_tr_gru, X_val_gru, y_val_gru, X_te_gru, y_te_gru),
    ('LSTM',        OptimizedLSTM(),           X_tr_gru, y_tr_gru, X_val_gru, y_val_gru, X_te_gru, y_te_gru),
    ('TCN',         OptimizedTCN(),            X_tr_tcn, y_tr_tcn, X_val_tcn, y_val_tcn, X_te_tcn, y_te_tcn),
    ('Transformer', TransformerClassifier(),   X_tr_gru, y_tr_gru, X_val_gru, y_val_gru, X_te_gru, y_te_gru),
]

results = {}
all_curves = {}

for name, model, Xtr, ytr, Xval, yval, Xte, yte in models_to_train:
    f1, prec, rec, tr_losses, vl_losses, epochs_run = train_model(
        name, model, Xtr, ytr, Xval, yval, Xte, yte,
        max_epochs=500, patience=50
    )
    results[name] = {'f1': f1, 'prec': prec, 'rec': rec, 'epochs': epochs_run}
    all_curves[name] = {'train': tr_losses, 'val': vl_losses}

print("\n\n=== FINAL RESULTS ===")
for name, res in results.items():
    print(f"  {name:15s}: F1 = {res['f1']:.4f} | Prec = {res['prec']:.4f} | Rec = {res['rec']:.4f}  (ran {res['epochs']} epochs)")


# ── 5. Learning curves plot ───────────────────────────────────────────────────
curve_colors = {'GRU': 'gold', 'LSTM': 'cyan', 'TCN': '#FF6B6B', 'Transformer': '#BB86FC'}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor('white')
for ax, (name, curves) in zip(axes.flatten(), all_curves.items()):
    ax.set_facecolor('white')
    color = curve_colors[name]
    ax.plot(curves['train'], label='Train Loss', color=color, alpha=0.7, linewidth=1.5)
    ax.plot(curves['val'],   label='Val Loss',   color='black', linewidth=1.5, linestyle='--')
    ax.set_title(f'{name}  (F1 = {results[name]["f1"]:.4f}, {results[name]["epochs"]}ep)',
                 color='black', fontsize=12)
    ax.set_xlabel('Epoch', color='black')
    ax.set_ylabel('BCE Loss', color='black')
    ax.tick_params(colors='black')
    ax.legend(fontsize=9, facecolor='white', labelcolor='black')
    ax.grid(True, alpha=0.1)
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray')

fig.suptitle('Sequential Model Learning Curves — 200 Epoch Run (Transition Dataset)',
             fontsize=14, color='black', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, 'sequential_learning_curves_200ep.png'), facecolor='white', dpi=200,
            bbox_inches='tight')
plt.close()
print("\nSaved sequential_learning_curves_200ep.png")


# ── 6. Update master comparison charts ───────────────────────────────────────

gru_f1   = results['GRU']['f1']
lstm_f1  = results['LSTM']['f1']
tcn_f1   = results['TCN']['f1']
trans_f1 = results['Transformer']['f1']

# --- Chart A: Global steady-state (tabular vs GNN) ---  no change to tabular values
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

models_global = [
    'XGBoost\n(Tabular)', 'Random\nForest', 'Simple\nRule', 'Decision\nTree',
    'Logistic\nReg', 'KNN\n(k=5)', 'MLP\n(Neural Net)', 'Naive\nBayes',
    'LSTM-SAGE\n(GNN)', 'GAT-LSTM\n(Attention)', 'Naive\nGraphSAGE'
]
scores_global = [0.8913, 0.8891, 0.8806, 0.8746, 0.8663, 0.8650, 0.8632, 0.6103,
                 0.7273, 0.6848, 0.6729]
colors_global = ['coral','coral','coral','coral','coral','coral','coral','coral',
                 'cyan','lime','gray']

bars = ax.bar(models_global, scores_global, color=colors_global, edgecolor='black', linewidth=0.4)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.008,
            f'{yval:.4f}', ha='center', fontweight='bold', fontsize=9, color='black')

ax.set_title('Global Steady-State Prediction: Tabular Models vs GNN Architectures', fontsize=15, color='black', pad=12)
ax.set_ylabel('F1 Score', fontsize=12, color='black')
ax.set_ylim([0.5, 1.0])
ax.tick_params(colors='black')
ax.grid(True, alpha=0.08, axis='y', linestyle='--')
for spine in ax.spines.values():
    spine.set_edgecolor('lightgray')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'master_global_comparison.png'), facecolor='white', dpi=200,
            bbox_inches='tight')
plt.close()
print("Saved master_global_comparison.png")

# --- Chart B: Transition battle with updated sequential scores ---
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

models_trans = [
    f'GRU\n(Gated)\n{gru_f1:.4f}',
    f'TCN\n(Dilated Conv)\n{tcn_f1:.4f}',
    f'LSTM\n(Temporal)\n{lstm_f1:.4f}',
    f'Transformer\n(Attention)\n{trans_f1:.4f}',
    'LightGBM\n(Leaf-wise)\n0.6944',
    'XGBoost\n(Level-wise)\n0.6849',
    'LSTM-SAGE\nGNN (6mi)\n0.7112',
]
scores_trans = [gru_f1, tcn_f1, lstm_f1, trans_f1, 0.6944, 0.6849, 0.7112]
colors_trans = ['gold', 'gold', 'cyan', 'cyan', 'coral', 'coral', 'lime']

bars = ax.bar(models_trans, scores_trans, color=colors_trans, edgecolor='black', linewidth=0.4)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.001,
            f'{yval:.4f}', ha='center', fontweight='bold', fontsize=10, color='black')

all_scores = scores_trans
ax.set_ylim([min(all_scores) - 0.015, max(all_scores) + 0.022])
ax.set_title('Transition Event Prediction (Breakdown Onset): Model Battle\n200 Epochs with Early Stopping',
             fontsize=14, color='black', pad=12)
ax.set_ylabel('F1 Score', fontsize=12, color='black')
ax.tick_params(colors='black')
ax.grid(True, alpha=0.08, axis='y', linestyle='--')
for spine in ax.spines.values():
    spine.set_edgecolor('lightgray')

legend_patches = [
    mpatches.Patch(facecolor='gold',   label='Sequential (GRU / TCN)'),
    mpatches.Patch(facecolor='blue',   label='Sequential (LSTM / Transformer)'),
    mpatches.Patch(facecolor='coral',  label='Tabular Boosting'),
    mpatches.Patch(facecolor='lime',   label='Spatio-Temporal GNN'),
]
ax.legend(handles=legend_patches, fontsize=10, facecolor='white',
          edgecolor='#444', labelcolor='black', framealpha=0.9, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'master_transition_comparison.png'), facecolor='white', dpi=200,
            bbox_inches='tight')
plt.close()
print("Saved master_transition_comparison.png")
print("\nAll done!")
