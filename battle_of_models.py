import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
import time
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')

# ---------------------------------------------------------
# 1. Prepare Transition Dataset
# ---------------------------------------------------------
print("Loading data and identifying transitions...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True).iloc[:10000]
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

# ---------------------------------------------------------
# 2. Architectures
# ---------------------------------------------------------

# 1. 1D-CNN
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# 2. Simple Transformer (With Embedding)
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(1, 32)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Global average pooling over time
        return self.fc(x)

# 3. LSTM (from previous step)
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ---------------------------------------------------------
# 3. Training Function
# ---------------------------------------------------------
def train_model(model, name):
    print(f"Training {name}...")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.BCEWithLogitsLoss()
    start = time.time()
    for _ in range(30):
        model.train()
        opt.zero_grad()
        out = model(X_tr_t).squeeze()
        loss = crit(out, y_tr_t)
        loss.backward()
        opt.step()
    
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_te_t).squeeze()) > 0.5).numpy().astype(int)
    return f1_score(y_te, preds), time.time() - start

# ---------------------------------------------------------
# 4. The Model Battle
# ---------------------------------------------------------
results = {}

# GBM (Non-DL)
print("Training HistGBM...")
gbm = HistGradientBoostingClassifier(random_state=42)
start = time.time()
gbm.fit(X_tr, y_tr)
results['HistGBM'] = (f1_score(y_te, gbm.predict(X_te)), time.time() - start)

# DL Models
results['LSTM'] = train_model(LSTMModel(), "LSTM")
results['CNN-1D'] = train_model(CNN1D(), "CNN-1D")
results['Transformer'] = train_model(SimpleTransformer(), "Transformer")

# ---------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------
df_res = pd.DataFrame(results, index=['F1 Score', 'Time']).T
print("\nModel Battle Leaderboard:")
print(df_res)

plt.figure(figsize=(10, 6))
df_res['F1 Score'].plot(kind='bar', color='blue')
plt.title('Battle of Models: Predicting Traffic Breakdown')
plt.ylabel('F1 Score')
plt.ylim([0.6, 0.75])
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUT, 'model_battle_comparison.png'), facecolor='white')
plt.close()
print(f"Leaderboard plot saved to output/model_battle_comparison.png")
