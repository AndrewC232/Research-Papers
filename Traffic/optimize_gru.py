import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')

# 1. Dataset Preparation
print("Loading data and identifying transitions...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:20000]
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

# Train/Val/Test Split (70/15/15)
n = len(X)
tr_end, val_end = int(n * 0.7), int(n * 0.85)

X_tr_t = torch.tensor(X[:tr_end], dtype=torch.float32).unsqueeze(-1)
y_tr_t = torch.tensor(y[:tr_end], dtype=torch.float32).unsqueeze(-1)
X_val_t = torch.tensor(X[tr_end:val_end], dtype=torch.float32).unsqueeze(-1)
y_val_t = torch.tensor(y[tr_end:val_end], dtype=torch.float32).unsqueeze(-1)
X_te_t = torch.tensor(X[val_end:], dtype=torch.float32).unsqueeze(-1)
y_te_t = torch.tensor(y[val_end:], dtype=torch.float32).unsqueeze(-1)

# 2. Optimized GRU Architecture
class OptimizedGRU(nn.Module):
    def __init__(self, in_dim=1, hidden=64, layers=2, drop=0.3):
        super(OptimizedGRU, self).__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True, dropout=drop if layers > 1 else 0)
        self.fc = nn.Linear(hidden, 1)
        
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

# 3. Training Loop with Early Stopping
model = OptimizedGRU()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
criterion = nn.BCEWithLogitsLoss()

train_losses, val_losses = [], []
best_val_loss = float('inf')
patience_counter = 0

print(f"Training Optimized GRU on {len(X_tr_t)} events...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(X_tr_t)
    loss = criterion(out, y_tr_t)
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        v_out = model(X_val_t)
        v_loss = criterion(v_out, y_val_t)
    
    train_losses.append(loss.item())
    val_losses.append(v_loss.item())
    scheduler.step(v_loss)
    
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        patience_counter = 0
        # torch.save(model.state_dict(), 'best_gru.pth') # Optional: save best model
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Val Loss: {v_loss.item():.4f}")
    
    if patience_counter >= 10:
        print("Early stopping triggered.")
        break

# 4. Evaluation
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_te_t))
    preds = (probs > 0.5).numpy().astype(int)
    f1 = f1_score(y_te_t.numpy(), preds)

print(f"\nOptimized GRU F1 Score: {f1:.4f}")

# Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Val Loss', color='orange')
plt.title('Optimized GRU Learning Curves')
plt.legend()
plt.savefig(os.path.join(OUT, 'optimize_gru_curves.png'), facecolor='white')
plt.close()
