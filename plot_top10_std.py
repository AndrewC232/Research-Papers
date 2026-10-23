import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

plt.style.use('default')

print("Loading data...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True)

print("Calculating standard deviations...")
std_speed = df.std()
top10_std = std_speed.sort_values(ascending=False).head(10)

print("Top 10 Sensors by Standard Deviation:")
print(top10_std)

plt.figure(figsize=(10, 6))
# Create a bar chart
bars = plt.bar(top10_std.index.astype(str), top10_std.values, color='coral', edgecolor='none')

plt.title('Top 10 Sensors with Highest Speed Variability (Standard Deviation)', fontsize=14, fontweight='bold', color='black')
plt.xlabel('Sensor ID', fontsize=12, color='black')
plt.ylabel('Standard Deviation (mph)', fontsize=12, color='black')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
out_path = os.path.join(OUT, 'q4_top10_std.png')
plt.savefig(out_path, facecolor='white')
plt.close()

print(f"Plot saved to {out_path}")
