import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
import os
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA)
df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

CONG_THRESH = 60.0
FREE_FLOW = df.quantile(0.95)
cong_freq = (df < CONG_THRESH).mean() * 100
top50 = cong_freq.sort_values(ascending=False).head(50).index.tolist()

def get_features(thresh=60.0):
    all_features = []
    for i, sid in enumerate(top50):
        s = df[sid]
        f = pd.DataFrame(index=s.index)
        
        # Temporal
        f['hour'] = s.index.hour
        f['dow'] = s.index.dayofweek
        f['is_weekend'] = (f['dow'] >= 5).astype(int)
        f['is_rush_am'] = ((f['hour'] >= 7) & (f['hour'] <= 9)).astype(int)
        f['is_rush_pm'] = ((f['hour'] >= 16) & (f['hour'] <= 18)).astype(int)
        f['hour_sin'] = np.sin(2 * np.pi * f['hour'] / 24)
        f['hour_cos'] = np.cos(2 * np.pi * f['hour'] / 24)
        f['dow_sin'] = np.sin(2 * np.pi * f['dow'] / 7)
        f['dow_cos'] = np.cos(2 * np.pi * f['dow'] / 7)
        
        # Speed
        f['speed'] = s.values
        f['speed_ratio'] = s.values / FREE_FLOW[sid]
        
        # Lags
        for lag in [1, 2, 3, 6, 12]:
            f[f'lag_{lag}'] = s.shift(lag).values
            
        # Rolling
        f['roll_mean_6'] = s.rolling(6).mean().values
        f['roll_mean_12'] = s.rolling(12).mean().values
        f['roll_std_6'] = s.rolling(6).std().values
        f['roll_std_12'] = s.rolling(12).std().values
        f['roll_min_12'] = s.rolling(12).min().values
        
        # Diff
        f['diff_1'] = s.diff(1).values
        f['diff_3'] = s.diff(3).values
        f['diff_6'] = s.diff(6).values
        
        # Target (3 steps = 15m ahead)
        f['target'] = (s.shift(-3) < thresh).astype(int)
        f['sensor_idx'] = i
        
        f.dropna(inplace=True)
        all_features.append(f)
        
    feat_df = pd.concat(all_features)
    feat_df.sort_index(inplace=True)
    feature_cols = [c for c in feat_df.columns if c != 'target']
    
    split = int(len(feat_df) * 0.8)
    X_train = feat_df[feature_cols].iloc[:split].values
    X_test = feat_df[feature_cols].iloc[split:].values
    y_train = feat_df['target'].iloc[:split].values
    y_test = feat_df['target'].iloc[split:].values
    
    return X_train, X_test, y_train, y_test

def train_xgb(X_tr, y_tr, X_te, y_te):
    model = HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return f1_score(y_te, preds)

print("\n--- Q7: Detailed Threshold Sensitivity (55 to 65 mph) ---")
thresholds = np.arange(55.0, 66.0, 1.0)
results = []

for t in thresholds:
    class_balance = (df < t).mean().mean() * 100
    X_tr, X_te, y_tr, y_te = get_features(t)
    if y_tr.sum() == 0 or y_te.sum() == 0:
        f1_t = 0.0
    else:
        f1_t = train_xgb(X_tr, y_tr, X_te, y_te)
    print(f"Threshold {t} mph: F1={f1_t:.4f}, Class Balance (PosRate)={class_balance:.2f}%")
    results.append({'threshold': t, 'f1': f1_t, 'balance': class_balance})

res_df = pd.DataFrame(results)

# Create Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'cyan'
ax1.set_xlabel('Speed Threshold (mph)')
ax1.set_ylabel('F1 Score', color=color)
ax1.plot(res_df['threshold'], res_df['f1'], marker='o', color=color, linewidth=2, label='F1 Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, 1.0])
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()  
color = 'coral'
ax2.set_ylabel('Class Balance (%)', color=color)  
ax2.plot(res_df['threshold'], res_df['balance'], marker='s', color=color, linewidth=2, linestyle='--', label='Class Balance')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, 100])

plt.title('Threshold Sensitivity: F1 Score and Class Balance (55 - 65 mph)')
fig.tight_layout()  
plt.savefig(os.path.join(OUT, 'q7_thresholds_detailed.png'), facecolor='white')
plt.close()

print("\nSaved detailed plot to output/q7_thresholds_detailed.png")
