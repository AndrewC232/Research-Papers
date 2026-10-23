import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT  = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

# Set style
plt.style.use('default')

print("Loading data...")
df = pd.read_csv(DATA)
df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

CONG_THRESH = 60.0
cong_mask = df < CONG_THRESH

# Q1. Seasonal Trends
print("\n--- Q1: Seasonal Trends ---")
monthly_cong = cong_mask.resample('ME').mean() * 100
network_monthly = monthly_cong.mean(axis=1)

plt.figure(figsize=(10, 5))
plt.plot(network_monthly.index, network_monthly.values, marker='o', color='coral')
plt.title('Monthly Average Congestion Rate (Jan - Jun 2017)')
plt.ylabel('% Time Congested')
plt.xlabel('Month')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUT, 'q1_seasonal_trends.png'), facecolor='white')
plt.close()

for m, val in network_monthly.items():
    print(f"{m.strftime('%Y-%B')}: {val:.2f}%")

# Q2. Sensor Correlation
print("\n--- Q2: Sensor Correlation ---")
congestion_freq = cong_mask.mean() * 100
top5_sensors = congestion_freq.sort_values(ascending=False).head(5).index.tolist()
print("Top 5 congested sensors:", top5_sensors)

top5_df = df[top5_sensors]
corr_matrix = top5_df.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Pearson Correlation of Top 5 Congested Sensors')
plt.savefig(os.path.join(OUT, 'q2_correlation.png'), facecolor='white')
plt.close()
print("Correlation matrix:\n", corr_matrix)

# Q3. Congestion Duration
print("\n--- Q3: Congestion Duration ---")
# Pick the top congested sensor as an example, or all top 5
durations_am = []
durations_pm = []

# To compute duration, we find contiguous blocks of True in cong_mask
for sensor in top5_sensors:
    s_cong = cong_mask[sensor]
    # Identify blocks
    blocks = s_cong.ne(s_cong.shift()).cumsum()
    # Group by blocks
    cong_blocks = s_cong.groupby(blocks).agg(['sum', lambda x: x.index[0].hour])
    cong_blocks.columns = ['length', 'start_hour']
    # Filter only blocks that are congested (length of True values)
    # Actually 'sum' works since True=1, False=0. If the block is False, sum is 0.
    cong_blocks = cong_blocks[cong_blocks['length'] > 0]
    
    # 5 min per tick -> length * 5 = duration in minutes
    cong_blocks['duration_min'] = cong_blocks['length'] * 5
    
    # AM: 7-9 (includes blocks starting 6-9 to be safe)
    # PM: 16-18
    am_blocks = cong_blocks[(cong_blocks['start_hour'] >= 6) & (cong_blocks['start_hour'] <= 9)]
    pm_blocks = cong_blocks[(cong_blocks['start_hour'] >= 15) & (cong_blocks['start_hour'] <= 18)]
    
    durations_am.extend(am_blocks['duration_min'].tolist())
    durations_pm.extend(pm_blocks['duration_min'].tolist())

print(f"AM average duration: {np.mean(durations_am):.1f} min")
print(f"PM average duration: {np.mean(durations_pm):.1f} min")

plt.figure(figsize=(10, 5))
plt.hist(durations_am, bins=50, alpha=0.5, label='AM Rush', range=(0, 300), density=True)
plt.hist(durations_pm, bins=50, alpha=0.5, label='PM Rush', range=(0, 300), density=True)
plt.title('Distribution of Congestion Episode Durations (Top 5 Sensors)')
plt.xlabel('Duration (minutes)')
plt.ylabel('Density')
plt.legend()
plt.savefig(os.path.join(OUT, 'q3_durations.png'), facecolor='white')
plt.close()

# Q4. Speed Variability
print("\n--- Q4: Speed Variability ---")
std_speed = df.std()
top5_std = std_speed.sort_values(ascending=False).head(5)
print("Top 5 sensors by standard deviation:")
for s, val in top5_std.items():
    freq = congestion_freq[s]
    print(f"Sensor {s}: StdDev={val:.2f}, CongestionFreq={freq:.2f}%")

# Q5. 0-Speed Mystery
print("\n--- Q5: 0-Speed Mystery ---")
zero_mask = (df == 0.0)
zero_counts_per_sensor = zero_mask.sum()
sensors_with_zeros = zero_counts_per_sensor[zero_counts_per_sensor > 0].sort_values(ascending=False)
print("Top sensors with 0-speed readings:")
print(sensors_with_zeros.head(5))

# When do they happen?
zero_hours = []
for col in df.columns:
    z_idx = df[df[col] == 0].index
    zero_hours.extend(z_idx.hour.tolist())

plt.figure(figsize=(10, 5))
plt.hist(zero_hours, bins=24, range=(0, 24), color='red', alpha=0.7)
plt.title('Hourly Distribution of 0-Speed Readings')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.savefig(os.path.join(OUT, 'q5_zeros.png'), facecolor='white')
plt.close()

print("Script completed.")
