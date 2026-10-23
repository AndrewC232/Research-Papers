import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from community import community_louvain
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True)
CONG_THRESH = 60.0

# ---------------------------------------------------------
# Q11. Congestion Propagation
# ---------------------------------------------------------
print("\n--- Q11: Congestion Propagation ---")
# Pick a highly congested sensor
cong_freq = (df < CONG_THRESH).mean()
target_sensor = cong_freq.idxmax()
print(f"Using heavily congested target sensor: {target_sensor}")

# Find a geographically close sensor (we'll just use high correlation as proxy for proximity)
corrs = df.corr()[target_sensor]
neighbor = corrs.drop(target_sensor).idxmax()
print(f"Highly correlated neighbor sensor: {neighbor}")

# Cross-correlation with time shifts
target_speed = df[target_sensor]
neighbor_speed = df[neighbor]

shifts = range(-6, 7) # -30 mins to +30 mins (5 min intervals)
cross_corrs = []
for s in shifts:
    shifted_target = target_speed.shift(s)
    corr = neighbor_speed.corr(shifted_target)
    cross_corrs.append(corr)

plt.figure(figsize=(8, 5))
plt.plot([s*5 for s in shifts], cross_corrs, marker='o', color='coral')
plt.axvline(0, color='gray', linestyle='--')
plt.title(f'Cross-Correlation between {target_sensor} and {neighbor}')
plt.xlabel('Time Shift of Target Sensor (minutes)')
plt.ylabel('Pearson Correlation')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUT, 'q11_propagation.png'), facecolor='white')
plt.close()
max_shift = shifts[np.argmax(cross_corrs)] * 5
print(f"Maximum correlation occurs when target is shifted by {max_shift} minutes.")


# ---------------------------------------------------------
# Q12. Anomaly Detection
# ---------------------------------------------------------
print("\n--- Q12: Anomaly Detection ---")
# Daily network-wide congestion rate
daily_cong = (df < CONG_THRESH).mean(axis=1).resample('D').mean() * 100

z_scores = (daily_cong - daily_cong.mean()) / daily_cong.std()
anomalies = daily_cong[abs(z_scores) > 2.0]

plt.figure(figsize=(12, 5))
plt.plot(daily_cong.index, daily_cong.values, color='blue', alpha=0.7)
plt.scatter(anomalies.index, anomalies.values, color='red', label='Anomalies (|Z| > 2)')
plt.title('Daily Network-Wide Congestion Rate (Anomaly Detection)')
plt.ylabel('Congestion Rate (%)')
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig(os.path.join(OUT, 'q12_anomalies.png'), facecolor='white')
plt.close()

print("Anomaly Dates Detected:")
for date, rate in anomalies.items():
    print(f" - {date.strftime('%Y-%m-%d')}: {rate:.2f}% congestion (Z={z_scores[date]:.2f})")

# ---------------------------------------------------------
# Q13 & Q14 Data Prep
# ---------------------------------------------------------
print("\nPreparing features for Q13 and Q14 (Subset of 10 sensors to save memory/time)...")
top10 = cong_freq.nlargest(10).index.tolist()
FREE_FLOW = df.quantile(0.95)

all_features = []
for i, sid in enumerate(top10):
    s = df[sid]
    f = pd.DataFrame(index=s.index)
    f['hour'] = s.index.hour
    f['dow'] = s.index.dayofweek
    f['speed'] = s.values
    
    # Lags
    f['lag_1'] = s.shift(1).values
    f['lag_3'] = s.shift(3).values
    
    # Target (15m ahead)
    target_speed_val = s.shift(-3)
    f['target_binary'] = (target_speed_val < CONG_THRESH).astype(int)
    
    # Multi-class target: Free Flow (>65), Moderate (60-65), Congested (45-60), Severe (<45)
    conditions = [
        target_speed_val > 65.0,
        (target_speed_val <= 65.0) & (target_speed_val > 60.0),
        (target_speed_val <= 60.0) & (target_speed_val > 45.0),
        target_speed_val <= 45.0
    ]
    choices = [0, 1, 2, 3] # 0=FreeFlow, 1=Moderate, 2=Congested, 3=Severe
    f['target_multi'] = np.select(conditions, choices, default=np.nan)
    
    f.dropna(inplace=True)
    all_features.append(f)

feat_df = pd.concat(all_features)
feat_df.sort_index(inplace=True)

split = int(len(feat_df) * 0.8)
feature_cols = ['hour', 'dow', 'speed', 'lag_1', 'lag_3']
X_tr = feat_df[feature_cols].iloc[:split].values
X_te = feat_df[feature_cols].iloc[split:].values
y_tr_bin = feat_df['target_binary'].iloc[:split].values
y_te_bin = feat_df['target_binary'].iloc[split:].values

# ---------------------------------------------------------
# Q13. Class Imbalance Strategies
# ---------------------------------------------------------
print("\n--- Q13: Class Imbalance Strategies ---")
print(f"Training Class Balance: {y_tr_bin.mean()*100:.1f}% Positive")

def eval_model(model, X_train, y_train, threshold=0.5):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= threshold).astype(int)
    return f1_score(y_te_bin, preds)

base_model = HistGradientBoostingClassifier(random_state=42)

# a) Baseline
f1_base = eval_model(base_model, X_tr, y_tr_bin)
print(f"Baseline F1: {f1_base:.4f}")

# b) SMOTE (Note: SMOTE can be memory intensive, we sample down X_tr if it's too large)
print("Running SMOTE...")
if len(X_tr) > 200000:
    idx = np.random.choice(len(X_tr), 200000, replace=False)
    X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_tr[idx], y_tr_bin[idx])
else:
    X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_tr, y_tr_bin)
f1_smote = eval_model(base_model, X_sm, y_sm)
print(f"SMOTE F1: {f1_smote:.4f}")

# c) Undersampling
print("Running Undersampling...")
rus = RandomUnderSampler(random_state=42)
X_us, y_us = rus.fit_resample(X_tr, y_tr_bin)
f1_us = eval_model(base_model, X_us, y_us)
print(f"Undersampling F1: {f1_us:.4f}")

# d) Threshold Tuning
base_model.fit(X_tr, y_tr_bin) # Fit once for threshold tuning
probs = base_model.predict_proba(X_te)[:, 1]

f1_04 = f1_score(y_te_bin, (probs >= 0.4).astype(int))
print(f"Threshold 0.4 F1: {f1_04:.4f}")

f1_03 = f1_score(y_te_bin, (probs >= 0.3).astype(int))
print(f"Threshold 0.3 F1: {f1_03:.4f}")


# ---------------------------------------------------------
# Q14. Multi-Class Severity
# ---------------------------------------------------------
print("\n--- Q14: Multi-Class Severity ---")
y_tr_multi = feat_df['target_multi'].iloc[:split].values
y_te_multi = feat_df['target_multi'].iloc[split:].values

multi_model = HistGradientBoostingClassifier(random_state=42)
multi_model.fit(X_tr, y_tr_multi)
multi_preds = multi_model.predict(X_te)

print("Classification Report:")
target_names = ['Free Flow (>65)', 'Moderate (60-65)', 'Congested (45-60)', 'Severe (<45)']
print(classification_report(y_te_multi, multi_preds, target_names=target_names))

# ---------------------------------------------------------
# Q15. Graph-Based Analysis (Community Detection)
# ---------------------------------------------------------
print("\n--- Q15: Graph-Based Analysis ---")
# Use a subset of 100 sensors to make the graph readable
subset_df = df[df.columns[:100]]
corr_matrix = subset_df.corr()

G = nx.Graph()
for col in corr_matrix.columns:
    G.add_node(col)

# Add edges for high correlation
threshold = 0.6
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        n1 = corr_matrix.columns[i]
        n2 = corr_matrix.columns[j]
        c = corr_matrix.iloc[i, j]
        if c > threshold:
            G.add_edge(n1, n2, weight=c)

# Remove isolated nodes
G.remove_nodes_from(list(nx.isolates(G)))

# Louvain Community Detection
partition = community_louvain.best_partition(G)
num_communities = len(set(partition.values()))
print(f"Found {num_communities} communities based on correlation > {threshold}.")

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
cmap = plt.get_cmap('tab10')

for com in set(partition.values()):
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size=50, node_color=[cmap(com)]*len(list_nodes))
    
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='black')
plt.title('Sensor Communities (Louvain Detection on Correlation Graph)')
plt.axis('off')
plt.savefig(os.path.join(OUT, 'q15_communities.png'), facecolor='white')
plt.close()

print("Script completed.")
