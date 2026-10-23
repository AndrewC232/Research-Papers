"""
Temporal Ablation Study: Sequence Length & Enriched Features
Tests how changing the amount of information (Speed + Hour + DOW) and the
retention time (6, 12, 24, 36 steps) affects the breakdown prediction ceiling.
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
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

# ── 1. Model Definition ──────────────────────────────────────────────────────
class OptimizedGRU(nn.Module):
    def __init__(self, in_dim=3, hidden=64, layers=2, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True, dropout=drop if layers > 1 else 0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

# ── 2. Training Function ─────────────────────────────────────────────────────
def train_gru_ablation(window_size, X_tr, y_tr, X_val, y_val, X_te, y_te):
    model = OptimizedGRU(in_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    patience_counter = 0
    max_epochs = 500
    patience = 50
    best_state = None

    print(f"  Training {window_size}-step model...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_out = model(X_val)
            v_loss = criterion(v_out, y_val)
            preds = (torch.sigmoid(v_out) > 0.5).cpu().numpy().astype(int)
            val_f1 = f1_score(y_val.cpu().numpy(), preds, zero_division=0)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Evaluate on test set using the best state
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    model.eval()
    with torch.no_grad():
        te_probs = torch.sigmoid(model(X_te))
        te_preds = (te_probs > 0.5).cpu().numpy().astype(int)
        test_f1 = f1_score(y_te.cpu().numpy(), te_preds, zero_division=0)

    print(f"  -> Window {window_size} | Best Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f} (stopped at ep {epoch})")
    return test_f1

# ── 3. Data Processing & Experiment Execution ──────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:20000]
CONG_THRESH = 60.0

windows = [6, 12, 24, 36]
results = {}

for w in windows:
    print(f"\n{'='*50}\n ABLATION: Window = {w} steps ({(w*5)/60:.1f} hours)\n{'='*50}")
    
    # Rebuild dataset dynamically based on window size
    event_data = []
    
    # We will compute cyclic features for time of day and day of week
    hours = df.index.hour.values
    dows = df.index.dayofweek.values
    
    # Normalize features roughly so the GRU gradients are stable
    norm_hour = hours / 23.0
    norm_dow = dows / 6.0
    
    for col in df.columns:
        speed = df[col].values
        norm_speed = speed / 80.0 # Normalize speed roughly to [0, 1] range
        
        transitions = np.where((speed[:-1] >= CONG_THRESH) & (speed[1:] < CONG_THRESH))[0]
        for t in transitions:
            if t - w >= 0 and t + 3 < len(speed):
                # 3 features: speed, hour, dow
                seq = np.stack([
                    norm_speed[t-w+1:t+1],
                    norm_hour[t-w+1:t+1],
                    norm_dow[t-w+1:t+1]
                ], axis=1)
                
                target = 1 if speed[t+3] < CONG_THRESH else 0
                event_data.append({'seq': seq, 'target': target})
                
    X = np.array([e['seq'] for e in event_data])
    y = np.array([e['target'] for e in event_data])
    
    n = len(X)
    tr_end, val_end = int(n * 0.7), int(n * 0.85)
    
    X_tr  = torch.tensor(X[:tr_end], dtype=torch.float32).to(device)
    y_tr  = torch.tensor(y[:tr_end], dtype=torch.float32).unsqueeze(-1).to(device)
    X_val = torch.tensor(X[tr_end:val_end], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[tr_end:val_end], dtype=torch.float32).unsqueeze(-1).to(device)
    X_te  = torch.tensor(X[val_end:], dtype=torch.float32).to(device)
    y_te  = torch.tensor(y[val_end:], dtype=torch.float32).unsqueeze(-1).to(device)
    
    f1 = train_gru_ablation(w, X_tr, y_tr, X_val, y_val, X_te, y_te)
    results[w] = f1

# ── 4. Plot Results ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

labels = [f'{w} steps\n({(w*5)//60}h {(w*5)%60}m)' for w in windows]
scores = [results[w] for w in windows]

# Baseline we want to compare against is the old 12-step single-feature model which hit 0.6935
baseline_score = 0.6935

bars = ax.bar(labels, scores, color='blue', alpha=0.8, edgecolor='black', width=0.5)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', 
            ha='center', va='bottom', fontweight='bold', color='black')

# Draw baseline
ax.axhline(baseline_score, color='gold', linestyle='--', linewidth=2, label=f'Baseline (12 steps, Speed Only) = {baseline_score}')

ax.set_ylim([min(scores + [baseline_score]) - 0.02, max(scores + [baseline_score]) + 0.02])
ax.set_title('Temporal Ablation Study: Enriched Features (Speed+Time+DOW) & Varying Lookbacks', pad=15)
ax.set_ylabel('Test F1 Score')
ax.legend(facecolor='white', edgecolor='black')

for spine in ax.spines.values():
    spine.set_edgecolor('#333')
ax.grid(axis='y', alpha=0.1)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'temporal_ablation_results.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("\nSaved output/temporal_ablation_results.png")
print("Done!")
