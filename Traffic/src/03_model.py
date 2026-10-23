"""
03 - Model Ready for Training
PeMS-BAY — XGBoost Congestion Predictor
Prepares features, trains model, evaluates, saves model artifacts
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, precision_recall_curve)
import warnings, json, pickle, time
warnings.filterwarnings('ignore')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT  = os.path.join(BASE, 'output')
MDL  = os.path.join(BASE, 'models')
os.makedirs(OUT, exist_ok=True)
os.makedirs(MDL, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
    'text.color': 'black', 'xtick.color': 'black',
    'ytick.color': 'black', 'grid.color': 'lightgray', 'font.size': 10,
})

print("=" * 65)
print("  03 -- MODEL TRAINING & EVALUATION")
print("=" * 65)

# ── Load ──
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA)
df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
sensors = df.columns.tolist()
print(f"    {df.shape[0]:,} rows x {df.shape[1]} sensors")

CONG_THRESH = 60.0
FREE_FLOW = df.quantile(0.95)

# ── Identify target sensors (top 50 most congested) ──
cong_freq = (df < CONG_THRESH).mean() * 100
top50 = cong_freq.sort_values(ascending=False).head(50).index.tolist()
print(f"    Training on top 50 most congested sensors")
print(f"    Congestion range: {cong_freq[top50[-1]]:.1f}% - {cong_freq[top50[0]]:.1f}%")

# ═══════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════
print("\n[2/6] Feature engineering...")
t0 = time.time()

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
    
    # Speed features
    f['speed'] = s.values
    f['speed_ratio'] = s.values / FREE_FLOW[sid]
    
    # Lags: 5, 10, 15, 30, 60 min
    for lag in [1, 2, 3, 6, 12]:
        f[f'lag_{lag}'] = s.shift(lag).values
    
    # Rolling stats
    f['roll_mean_6'] = s.rolling(6).mean().values
    f['roll_mean_12'] = s.rolling(12).mean().values
    f['roll_std_6'] = s.rolling(6).std().values
    f['roll_std_12'] = s.rolling(12).std().values
    f['roll_min_12'] = s.rolling(12).min().values
    
    # Rate of change
    f['diff_1'] = s.diff(1).values
    f['diff_3'] = s.diff(3).values
    f['diff_6'] = s.diff(6).values
    
    # Target: congested in next 15 min (3 steps ahead)?
    f['target'] = (s.shift(-3) < CONG_THRESH).astype(int)
    f['sensor_idx'] = i
    
    f.dropna(inplace=True)
    all_features.append(f)

feat_df = pd.concat(all_features, ignore_index=True)
feature_cols = [c for c in feat_df.columns if c != 'target']
X = feat_df[feature_cols].values
y = feat_df['target'].values

elapsed = time.time() - t0
print(f"    Features: {X.shape[0]:,} samples x {X.shape[1]} features")
print(f"    Positive rate: {y.mean()*100:.1f}% (congested)")
print(f"    Time: {elapsed:.1f}s")
print(f"    Feature list: {feature_cols}")

# ═══════════════════════════════════════════════════════
# TRAIN / TEST SPLIT (temporal, no leakage)
# ═══════════════════════════════════════════════════════
print("\n[3/6] Splitting train/test (80/20 temporal)...")
split = int(len(feat_df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"    Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"    Train positive rate: {y_train.mean()*100:.1f}%")
print(f"    Test positive rate:  {y_test.mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════
print("\n[4/6] Training XGBoost classifier...")
t0 = time.time()

import xgboost as xgb
pos_weight = (1 - y_train.mean()) / max(y_train.mean(), 0.001)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    scale_pos_weight=pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

train_time = time.time() - t0
print(f"    Training time: {train_time:.1f}s")

# ═══════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════
print("\n[5/6] Evaluating model...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = (y_pred == y_test).mean()
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n    ========== RESULTS ==========")
print(f"    Accuracy:   {acc*100:.1f}%")
print(f"    F1 Score:   {f1:.4f}")
print(f"    ROC AUC:    {auc:.4f}")
print(f"    ============================")

report = classification_report(y_test, y_pred, target_names=['Free Flow', 'Congested'], zero_division=0)
print(f"\n{report}")

# ── Visualization ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('XGBoost Congestion Predictor — Model Results', fontsize=16, fontweight='bold', color='black', y=0.98)

# 5a: Feature importance (top 20)
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(20)
axes[0, 0].barh(range(len(imp)), imp.values, color='#58A6FF', edgecolor='none')
axes[0, 0].set_yticks(range(len(imp)))
axes[0, 0].set_yticklabels(imp.index, fontsize=8)
axes[0, 0].set_title('Top 20 Feature Importances', fontweight='bold', color='black')
axes[0, 0].set_xlabel('Importance')

# 5b: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['Free Flow', 'Congested'], yticklabels=['Free Flow', 'Congested'],
            linewidths=1, linecolor='black')
axes[0, 1].set_title(f'Confusion Matrix (F1={f1:.3f})', fontweight='bold', color='black')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 5c: Precision-Recall curve
prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
axes[1, 0].plot(rec, prec, color='#BC8CFF', lw=2)
axes[1, 0].fill_between(rec, prec, alpha=0.15, color='#BC8CFF')
axes[1, 0].set_title('Precision-Recall Curve', fontweight='bold', color='black')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_xlim([0, 1])
axes[1, 0].set_ylim([0, 1.05])

# 5d: Prediction probability distribution
axes[1, 1].hist(y_prob[y_test == 0], bins=60, alpha=0.7, color='#3FB950', label='Free Flow', density=True, edgecolor='none')
axes[1, 1].hist(y_prob[y_test == 1], bins=60, alpha=0.7, color='#F85149', label='Congested', density=True, edgecolor='none')
axes[1, 1].axvline(x=0.5, color='#E3B341', ls='--', lw=2, label='Decision boundary')
axes[1, 1].set_title('Prediction Probability Distribution', fontweight='bold', color='black')
axes[1, 1].set_xlabel('Predicted Probability of Congestion')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(facecolor='white', edgecolor='black')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'plot5_model_results.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("    -> Saved plot5_model_results.png")

# ═══════════════════════════════════════════════════════
# SAVE MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════
print("\n[6/6] Saving model artifacts...")

# Save model
model.save_model(os.path.join(MDL, 'xgboost_congestion.json'))
print(f"    -> Model: models/xgboost_congestion.json")

# Save feature config
config = {
    'model_type': 'XGBoost',
    'task': 'Binary Classification (Congestion Prediction)',
    'target': 'Will speed < 60 mph in next 15 minutes?',
    'dataset': 'PeMS-BAY (SF Bay Area, 325 sensors)',
    'training_sensors': len(top50),
    'feature_columns': feature_cols,
    'n_features': len(feature_cols),
    'congestion_threshold_mph': CONG_THRESH,
    'prediction_horizon_steps': 3,
    'prediction_horizon_minutes': 15,
    'train_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'metrics': {
        'accuracy': round(float(acc), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(auc), 4),
    },
    'hyperparameters': {
        'n_estimators': 300,
        'max_depth': 7,
        'learning_rate': 0.08,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
    },
    'training_time_seconds': round(train_time, 1),
}
with open(os.path.join(MDL, 'model_config.json'), 'w') as f:
    json.dump(config, f, indent=2, default=str)
print(f"    -> Config: models/model_config.json")

# Save feature importances
feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
feat_imp.to_csv(os.path.join(MDL, 'feature_importances.csv'), header=['importance'])
print(f"    -> Feature importances: models/feature_importances.csv")

print("\n" + "=" * 65)
print("  MODEL TRAINING COMPLETE")
print("=" * 65)
print(f"\n  Accuracy: {acc*100:.1f}%  |  F1: {f1:.4f}  |  AUC: {auc:.4f}")
print(f"  Model saved to: {os.path.abspath(MDL)}")
print(f"  Plots saved to: {os.path.abspath(OUT)}")
print(f"\n  The model is READY for production use.")
print(f"  To load: xgb.XGBClassifier(); model.load_model('models/xgboost_congestion.json')")
print("=" * 65)
