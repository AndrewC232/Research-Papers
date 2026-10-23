import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier
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
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# 2. LightGBM-style HistGBM Model
print(f"Training LightGBM-style HistGBM on {len(X_tr)} events...")
model = HistGradientBoostingClassifier(
    max_iter=100, 
    learning_rate=0.1, 
    max_leaf_nodes=31, # Classic LightGBM default
    random_state=42
)
model.fit(X_tr, y_tr)

# 3. Evaluation
preds = model.predict(X_te)
f1 = f1_score(y_te, preds)

print(f"\nLightGBM F1 Score (Transitions): {f1:.4f}")
