import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------
# 1. Prepare Data
# ---------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True)
CONG_THRESH = 60.0

# Subsetting to 5 most congested sensors to make KNN/MLP computationally viable in 1 minute
cong_freq = (df < CONG_THRESH).mean()
top_sensors = cong_freq.nlargest(5).index.tolist()

all_features = []
print("Extracting features for top 5 congested sensors...")
for sid in top_sensors:
    s = df[sid]
    f = pd.DataFrame(index=s.index)
    f['hour'] = s.index.hour
    f['dow'] = s.index.dayofweek
    f['speed_current'] = s.values
    f['speed_lag_1'] = s.shift(1).values
    f['speed_lag_3'] = s.shift(3).values
    f['speed_lag_6'] = s.shift(6).values
    f['rolling_mean_12'] = s.rolling(12).mean()
    f['rolling_std_12'] = s.rolling(12).std()
    
    # Target: 15m ahead (shift -3)
    target = s.shift(-3)
    f['target'] = (target < CONG_THRESH).astype(int)
    
    f.dropna(inplace=True)
    all_features.append(f)

feat_df = pd.concat(all_features).sort_index()

# Split 80/20 temporally
split = int(len(feat_df) * 0.8)
feature_cols = ['hour', 'dow', 'speed_current', 'speed_lag_1', 'speed_lag_3', 'speed_lag_6', 'rolling_mean_12', 'rolling_std_12']

X_tr = feat_df[feature_cols].iloc[:split].values
X_te = feat_df[feature_cols].iloc[split:].values
y_tr = feat_df['target'].iloc[:split].values
y_te = feat_df['target'].iloc[split:].values

print(f"Training samples: {len(X_tr)}, Test samples: {len(X_te)}")

# ---------------------------------------------------------
# 2. Train & Compare Models
# ---------------------------------------------------------
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
    'HistGBM': HistGradientBoostingClassifier(random_state=42),
    'Logistic Reg': LogisticRegression(max_iter=500),
    'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=50, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Naive Bayes': GaussianNB()
}

results = []

print("\nEvaluating Models:")
for name, model in models.items():
    start_t = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start_t
    
    preds = model.predict(X_te)
    f1 = f1_score(y_te, preds)
    
    results.append({'Model': name, 'F1 Score': f1, 'Training Time (s)': train_time})
    print(f" - {name:15s}: F1 = {f1:.4f} | Time = {train_time:.2f}s")

res_df = pd.DataFrame(results)

# Plot Comparison
fig, ax1 = plt.subplots(figsize=(12, 6))

ax2 = ax1.twinx()
ax1.bar(res_df['Model'], res_df['F1 Score'], color='coral', alpha=0.8, label='F1 Score')
ax2.plot(res_df['Model'], res_df['Training Time (s)'], color='blue', marker='o', linewidth=2, label='Train Time')

ax1.set_ylabel('F1 Score', color='coral')
ax1.set_ylim([0.0, 1.0])
ax2.set_ylabel('Training Time (seconds)', color='blue')
ax1.set_title('Model Performance Comparison: Accuracy vs. Speed')
ax1.set_xticklabels(res_df['Model'], rotation=45, ha='right')

# Adding F1 value text on bars
for i, v in enumerate(res_df['F1 Score']):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', color='black', fontweight='bold')

fig.tight_layout()
plt.savefig(os.path.join(OUT, 'q17_model_comparison.png'), facecolor='white')
plt.close()

# ---------------------------------------------------------
# 3. Feature Importance & "When" Traffic Breaks Down
# ---------------------------------------------------------
print("\nExtracting Feature Importances from Random Forest...")
rf_model = models['Random Forest']
importances = rf_model.feature_importances_

plt.figure(figsize=(10, 5))
indices = np.argsort(importances)[::-1]
sorted_features = [feature_cols[i] for i in indices]

plt.bar(range(len(importances)), importances[indices], color='mediumspringgreen')
plt.xticks(range(len(importances)), sorted_features, rotation=45, ha='right')
plt.title('What Drives Congestion? (Feature Importance)')
plt.ylabel('Relative Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'q17_feature_importance.png'), facecolor='white')
plt.close()

# ---------------------------------------------------------
# 4. Temporal Analysis: "When" is congestion most likely?
# ---------------------------------------------------------
print("\nAnalyzing temporal congestion probabilities...")
# Predict probabilities for the entire test set using HistGBM (usually best)
best_model = models['HistGBM']
test_df = feat_df.iloc[split:].copy()
test_df['pred_prob'] = best_model.predict_proba(X_te)[:, 1]

# Average predicted probability of congestion by Hour of Day
hourly_prob = test_df.groupby('hour')['pred_prob'].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_prob.index, hourly_prob.values, color='red', marker='s', linewidth=2)
plt.title('Predicted Probability of Congestion by Hour of Day')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Probability of Congestion (Speed < 60 in 15m)')
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.axvspan(6, 9, color='red', alpha=0.1, label='AM Rush')
plt.axvspan(15, 18, color='orange', alpha=0.1, label='PM Rush')
plt.legend()
plt.savefig(os.path.join(OUT, 'q17_temporal_probability.png'), facecolor='white')
plt.close()

print("All tasks completed successfully. Plots saved to output/")
