import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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

# Define feature extraction
def get_features(horizon_steps=3, thresh=60.0, subset_sensors=None):
    sensors_to_use = subset_sensors if subset_sensors else top50
    all_features = []
    for i, sid in enumerate(sensors_to_use):
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
        
        # Target
        f['target'] = (s.shift(-horizon_steps) < thresh).astype(int)
        f['sensor_idx'] = i
        
        f.dropna(inplace=True)
        all_features.append(f)
        
    feat_df = pd.concat(all_features)
    feat_df.sort_index(inplace=True) # Sort by timestamp to ensure proper temporal split!
    feature_cols = [c for c in feat_df.columns if c != 'target']
    
    split = int(len(feat_df) * 0.8)
    X_train = feat_df[feature_cols].iloc[:split].values
    X_test = feat_df[feature_cols].iloc[split:].values
    y_train = feat_df['target'].iloc[:split].values
    y_test = feat_df['target'].iloc[split:].values
    
    return X_train, X_test, y_train, y_test, feature_cols, feat_df

def train_xgb(X_tr, y_tr, X_te, y_te):
    # We use HistGradientBoostingClassifier to emulate XGBoost
    model = HistGradientBoostingClassifier(
        max_iter=100, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return f1_score(y_te, preds)

# Baseline
print("\n--- Baseline Model (15m, 60mph) ---")
X_tr_base, X_te_base, y_tr_base, y_te_base, fcols, fdf_base = get_features(3, 60.0)
f1_base = train_xgb(X_tr_base, y_tr_base, X_te_base, y_te_base)
print(f"Baseline F1: {f1_base:.4f}")

# Q6. Prediction Horizon
print("\n--- Q6: Prediction Horizon ---")
horizons = [3, 6, 12] # 15m, 30m, 60m
f1_horizons = []
for h in horizons:
    if h == 3:
        f1_horizons.append(f1_base)
    else:
        X_tr, X_te, y_tr, y_te, _, _ = get_features(h, 60.0)
        f1_h = train_xgb(X_tr, y_tr, X_te, y_te)
        f1_horizons.append(f1_h)
        print(f"Horizon {h*5}m F1: {f1_h:.4f}")

plt.figure(figsize=(8, 5))
plt.plot([15, 30, 60], f1_horizons, marker='o', color='blue')
plt.title('F1 Score vs Prediction Horizon')
plt.xlabel('Prediction Horizon (minutes)')
plt.ylabel('F1 Score')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUT, 'q6_horizon.png'), facecolor='white')
plt.close()

# Q7. Threshold Sensitivity
print("\n--- Q7: Threshold Sensitivity ---")
threshs = [55.0, 60.0, 65.0]
for t in threshs:
    class_balance = (df < t).mean().mean() * 100
    if t == 60.0:
        f1_t = f1_base
    else:
        X_tr, X_te, y_tr, y_te, _, _ = get_features(3, t)
        f1_t = train_xgb(X_tr, y_tr, X_te, y_te)
    print(f"Threshold {t} mph: F1={f1_t:.4f}, Class Balance (PosRate)={class_balance:.1f}%")

# Q8. Feature Ablation
print("\n--- Q8: Feature Ablation ---")
groups = {
    'Temporal': ['hour', 'dow', 'is_weekend', 'is_rush_am', 'is_rush_pm', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
    'Lag': ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12'],
    'Rolling': ['roll_mean_6', 'roll_mean_12', 'roll_std_6', 'roll_std_12', 'roll_min_12'],
    'RateOfChange': ['diff_1', 'diff_3', 'diff_6']
}

ablation_results = {}
for gname, gcols in groups.items():
    keep_cols = [c for c in fcols if c not in gcols]
    k_idx = [fcols.index(c) for c in keep_cols]
    
    X_tr_ab = X_tr_base[:, k_idx]
    X_te_ab = X_te_base[:, k_idx]
    
    f1_ab = train_xgb(X_tr_ab, y_tr_base, X_te_ab, y_te_base)
    drop = f1_base - f1_ab
    ablation_results[gname] = drop
    print(f"Removed {gname} -> F1: {f1_ab:.4f} (Drop: {drop:.4f})")

# Q9. Simpler Models
print("\n--- Q9: Simpler Models ---")
speed_idx = fcols.index('speed')
simple_preds = (X_te_base[:, speed_idx] < 60.0).astype(int)
f1_rule = f1_score(y_te_base, simple_preds)
print(f"Simple Rule F1: {f1_rule:.4f}")

sample_idx = np.random.choice(len(X_tr_base), 100000, replace=False)
lr = LogisticRegression(max_iter=500, n_jobs=-1)
lr.fit(X_tr_base[sample_idx], y_tr_base[sample_idx])
f1_lr = f1_score(y_te_base, lr.predict(X_te_base))
print(f"Logistic Regression F1: {f1_lr:.4f}")

rf = RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_tr_base[sample_idx], y_tr_base[sample_idx])
f1_rf = f1_score(y_te_base, rf.predict(X_te_base))
print(f"Random Forest F1: {f1_rf:.4f}")

# Q10. Per-Sensor Models
print("\n--- Q10: Per-Sensor Models ---")
global_model = HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.1).fit(X_tr_base, y_tr_base)
global_preds = global_model.predict(X_te_base)
sensor_idx_col = fcols.index('sensor_idx')

per_sensor_f1_global = []
per_sensor_f1_local = []

print("Training individual models for top 10 sensors...")
for i in range(10):
    tr_mask = X_tr_base[:, sensor_idx_col] == i
    te_mask = X_te_base[:, sensor_idx_col] == i
    
    X_tr_s = X_tr_base[tr_mask]
    y_tr_s = y_tr_base[tr_mask]
    X_te_s = X_te_base[te_mask]
    y_te_s = y_te_base[te_mask]
    
    if y_te_s.sum() > 0:
        glob_f1 = f1_score(y_te_s, global_preds[te_mask])
    else:
        glob_f1 = 0.0
    per_sensor_f1_global.append(glob_f1)
    
    if len(np.unique(y_tr_s)) > 1:
        loc_model = HistGradientBoostingClassifier(max_iter=50, max_depth=5, learning_rate=0.1)
        loc_model.fit(X_tr_s, y_tr_s)
        loc_f1 = f1_score(y_te_s, loc_model.predict(X_te_s))
    else:
        loc_f1 = 0.0
    per_sensor_f1_local.append(loc_f1)
    print(f"Sensor {top50[i]} - Global F1: {glob_f1:.4f}, Local F1: {loc_f1:.4f}")

print("\nMean Global F1 (Top 10):", np.mean(per_sensor_f1_global))
print("Mean Local F1 (Top 10):", np.mean(per_sensor_f1_local))
print("\nScript completed.")
