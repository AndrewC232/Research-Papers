import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import time
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')

# 1. Dataset Preparation (Focusing on Transitions)
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
split = int(len(X) * 0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
X_te_t = torch.tensor(X_te, dtype=torch.float32).unsqueeze(-1)
y_te_t = torch.tensor(y_te, dtype=torch.float32)

# 2. 1D-CNN Architecture
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x: (B, L, 1) -> (B, 1, L)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

# 3. Training
print(f"Training 1D-CNN on {len(X_tr)} events...")
model = CNN1D()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    output = model(X_tr_t).squeeze()
    loss = criterion(output, y_tr_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

# 4. Evaluation
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_te_t).squeeze())
    preds = (probs > 0.5).numpy().astype(int)
    f1 = f1_score(y_te, preds)

print(f"\n1D-CNN F1 Score (Transitions): {f1:.4f}")
