import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
from device_utils import get_device


warnings.filterwarnings('ignore')
plt.style.use('default')

device = get_device()

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)
SAVE_PATH = os.path.join(OUT, 'best_gru_deep.pth')

CONG_THRESH = 60.0
MAX_EPOCHS = 200
PATIENCE = 10

# 1. Dataset Preparation (Focusing on Transitions)
print("Loading data and identifying transitions...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:20000]

# Normalize
scaler = MinMaxScaler()
norm_speeds = scaler.fit_transform(df.values)

event_data = []
for col_idx in range(norm_speeds.shape[1]):
    speed_raw = df.iloc[:, col_idx].values
    speed_norm = norm_speeds[:, col_idx]
    
    # Identify transition: speed was >= 60, next time step is < 60
    transitions = np.where((speed_raw[:-1] >= CONG_THRESH) & (speed_raw[1:] < CONG_THRESH))[0]
    
    for t in transitions:
        if t - 11 >= 0 and t + 3 < len(speed_raw):
            # 12 steps up to and including t
            seq = speed_norm[t-11:t+1]
            target = 1.0 if speed_raw[t+3] < CONG_THRESH else 0.0
            event_data.append({'seq': seq, 'target': target})

X = np.array([e['seq'] for e in event_data])
y = np.array([e['target'] for e in event_data])

split = int(len(X) * 0.8)
X_tr = torch.tensor(X[:split], dtype=torch.float32).unsqueeze(-1)
y_tr = torch.tensor(y[:split], dtype=torch.float32)
X_val = torch.tensor(X[split:], dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y[split:], dtype=torch.float32)

print(f"Deep GRU Dataset -> Train events: {len(X_tr)}, Val events: {len(X_val)}")

# 2. GRU Architecture
class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

model = GRUModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# 3. Training Loop with Early Stopping
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
best_val_loss = float('inf')
best_val_f1 = 0.0
patience_counter = 0

print("\nStarting Deep GRU Training...")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    indices = np.random.permutation(len(X_tr))
    epoch_train_loss = 0
    
    # Mini-batch training for stability
    for i in range(0, len(X_tr), 128):
        batch_idx = indices[i:i+128]
        optimizer.zero_grad()
        output = model(X_tr[batch_idx].to(device)).squeeze()
        
        loss = criterion(output, y_tr[batch_idx].to(device))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * len(batch_idx)
        
    avg_train_loss = epoch_train_loss / len(X_tr)
    
    # Validation
    model.eval()
    with torch.no_grad():
        out_val = model(X_val.to(device)).squeeze()
        val_loss = criterion(out_val, y_val.to(device)).item()
        
        probs = torch.sigmoid(out_val).cpu()
        preds = (probs > 0.5).numpy().astype(int)
        val_f1 = f1_score(y_val.numpy(), preds)
        
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    
    print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), SAVE_PATH)
        
    if patience_counter >= PATIENCE:
        print(f"\nEarly Stopping triggered at Epoch {epoch}!")
        break

print(f"\nGRU Deep Training Complete. Best F1 Score: {best_val_f1:.4f}")

# 4. Plot Learning Curves
fig, ax1 = plt.subplots(figsize=(12, 6))

ax2 = ax1.twinx()
ax1.plot(history['train_loss'], label='Train Loss', color='coral', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2, linestyle='--')
ax2.plot(history['val_f1'], label='Val F1 Score', color='blue', linewidth=2.5)

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color='coral', fontsize=12)
ax2.set_ylabel('F1 Score', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='coral')
ax2.tick_params(axis='y', labelcolor='blue')

ax1.set_title(f'Deep Sequential GRU Learning Curves\nConcentrated Transition Dataset | Best F1: {best_val_f1:.4f}', fontsize=14)
ax1.grid(True, alpha=0.15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'gru_deep_learning_curves.png'), dpi=200, facecolor='white')
plt.close()
print("Saved gru_deep_learning_curves.png")
