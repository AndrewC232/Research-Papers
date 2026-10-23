import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')

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
X_tr_t = torch.tensor(X[:split], dtype=torch.float32).unsqueeze(-1)
y_tr_t = torch.tensor(y[:split], dtype=torch.float32)
X_te_t = torch.tensor(X[split:], dtype=torch.float32).unsqueeze(-1)
y_te_t = torch.tensor(y[split:], dtype=torch.float32)

# 2. LSTM Architecture
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# 3. Training
print(f"Training LSTM on {len(X_tr_t)} events...")
model = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
    f1 = f1_score(y_te_t.numpy(), preds)

print(f"\nLSTM F1 Score (Transitions): {f1:.4f}")
