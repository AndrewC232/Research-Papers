import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')

# 1. Dataset Preparation
print("Loading data and identifying transitions...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:10000] # smaller subset for speed
CONG_THRESH = 60.0

event_data = []
for col in df.columns:
    speed = df[col].values
    transitions = np.where((speed[:-1] >= CONG_THRESH) & (speed[1:] < CONG_THRESH))[0]
    for t in transitions:
        if t - 12 >= 0 and t + 3 < len(speed):
            event_data.append({'seq': speed[t-12+1:t+1], 'target': 1 if speed[t+3] < CONG_THRESH else 0})

X = np.array([e['seq'] for e in event_data])
y = np.array([e['target'] for e in event_data])

n = len(X)
tr_end = int(n * 0.8)

X_tr_t = torch.tensor(X[:tr_end], dtype=torch.float32).unsqueeze(-1)
y_tr_t = torch.tensor(y[:tr_end], dtype=torch.float32).unsqueeze(-1)
X_te_t = torch.tensor(X[tr_end:], dtype=torch.float32).unsqueeze(-1)
y_te_t = torch.tensor(y[tr_end:], dtype=torch.float32).unsqueeze(-1)

# 2. Model
class OptimizedGRU(nn.Module):
    def __init__(self, in_dim=1, hidden=32, layers=1): # Simple fast GRU
        super(OptimizedGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

model = OptimizedGRU()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()

print(f"Training fast GRU on {len(X_tr_t)} events for permutation testing...")
for epoch in range(15):
    model.train()
    optimizer.zero_grad()
    out = model(X_tr_t)
    loss = criterion(out, y_tr_t)
    loss.backward()
    optimizer.step()

# Baseline Eval
model.eval()
with torch.no_grad():
    base_probs = torch.sigmoid(model(X_te_t))
    base_preds = (base_probs > 0.5).numpy().astype(int)
    base_f1 = f1_score(y_te_t.numpy(), base_preds)

print(f"\nBaseline F1 Score: {base_f1:.4f}")

# 3. Permutation Feature Importance (Temporal Ablation)
# We have 12 timesteps (T-60 to T-5)
importances = []
labels = []

for step in range(12):
    # Copy test set
    X_ablated = X_te_t.clone()
    
    # Scramble this specific timestep across all samples
    scrambled_indices = torch.randperm(X_ablated.size(0))
    X_ablated[:, step, :] = X_ablated[scrambled_indices, step, :]
    
    with torch.no_grad():
        probs = torch.sigmoid(model(X_ablated))
        preds = (probs > 0.5).numpy().astype(int)
        f1 = f1_score(y_te_t.numpy(), preds)
        
    drop = base_f1 - f1
    minutes_ago = (12 - step) * 5
    label = f"T-{minutes_ago} min"
    
    importances.append(drop)
    labels.append(label)
    
    print(f"Scrambling {label}: New F1 = {f1:.4f} (Drop: {drop:+.4f})")

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, importances, color='blue')
plt.title('Temporal Feature Importance (Permutation F1 Drop)')
plt.ylabel('Drop in F1 Score')
plt.xlabel('Timestep Scrambled')
plt.axhline(0, color='gray', linewidth=0.5)
plt.xticks(rotation=45)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (0.001 if yval > 0 else -0.003), f'{yval:+.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'temporal_importance.png'), facecolor='white')
print(f"\nSaved plot to {OUT}/temporal_importance.png")
