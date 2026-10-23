import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')

# ---------------------------------------------------------
# 1. Identify Breakdown Events (Transitions)
# ---------------------------------------------------------
print("Loading data and identifying transitions...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:10000] # Use first 10k timestamps for speed
CONG_THRESH = 60.0

event_data = []
# Window: 12 steps before (1hr), 6 steps after (30m)
window_before = 12
window_after = 6

for col in df.columns:
    speed = df[col].values
    # Detect transitions: t was >=60, t+1 was <60
    transitions = np.where((speed[:-1] >= CONG_THRESH) & (speed[1:] < CONG_THRESH))[0]
    
    for t in transitions:
        if t - window_before >= 0 and t + window_after + 3 < len(speed):
            # Feature window (1hr before the breakdown)
            # Target is 15m (3 steps) after the CURRENT time t
            window = speed[t - window_before + 1 : t + 1]
            target_val = speed[t + 3]
            
            event_data.append({
                'sensor': col,
                'timestamp': df.index[t],
                'sequence': window,
                'target': 1 if target_val < CONG_THRESH else 0
            })

print(f"Identified {len(event_data)} breakdown transition events.")

# Convert to ML format
X = np.array([e['sequence'] for e in event_data])
y = np.array([e['target'] for e in event_data])

# Split
split = int(len(X) * 0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# ---------------------------------------------------------
# 2. Train XGBoost (Tabular Baseline on Transitions)
# ---------------------------------------------------------
print("Training Baseline (HistGBM) on focused transitions...")
gbm = HistGradientBoostingClassifier(random_state=42)
gbm.fit(X_tr, y_tr)
gbm_preds = gbm.predict(X_te)
gbm_f1 = f1_score(y_te, gbm_preds)
print(f"Focused GBM F1 Score: {gbm_f1:.4f}")

# ---------------------------------------------------------
# 3. Train LSTM (Temporal Model on Transitions)
# ---------------------------------------------------------
print("Training LSTM on focused transitions...")
X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
X_te_t = torch.tensor(X_te, dtype=torch.float32).unsqueeze(-1)
y_te_t = torch.tensor(y_te, dtype=torch.float32)

class SimpleLSTM(nn.Module):
    def __init__(self, in_dim, hidden):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

lstm = SimpleLSTM(1, 32)
opt = torch.optim.Adam(lstm.parameters(), lr=0.01)
crit = nn.BCEWithLogitsLoss()

for epoch in range(50):
    lstm.train()
    opt.zero_grad()
    out = lstm(X_tr_t).squeeze()
    loss = crit(out, y_tr_t)
    loss.backward()
    opt.step()

lstm.eval()
with torch.no_grad():
    l_out = torch.sigmoid(lstm(X_te_t).squeeze())
    l_preds = (l_out > 0.5).numpy().astype(int)
    lstm_f1 = f1_score(y_te, l_preds)

print(f"Focused LSTM F1 Score: {lstm_f1:.4f}")

# ---------------------------------------------------------
# 4. Results & Plot
# ---------------------------------------------------------
# Compare these scores with the previous "Global" scores
models = ['Global XGBoost', 'Focused XGBoost', 'Global LSTM-GNN', 'Focused LSTM']
# Global scores hardcoded from previous runs
scores = [0.8913, gbm_f1, 0.7099, lstm_f1]

plt.figure(figsize=(10, 6))
colors = ['coral', 'red', 'cyan', 'blue']
bars = plt.bar(models, scores, color=colors)
plt.title('Performance Drop on Congestion Transitions (Breakdown Prediction)')
plt.ylabel('F1 Score')
plt.ylim([0, 1.0])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', fontweight='bold')

plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random Chance')
plt.savefig(os.path.join(OUT, 'q19_transition_comparison.png'), facecolor='white')
plt.close()

print("Results saved to output/q19_transition_comparison.png")
