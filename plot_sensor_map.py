import pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request

BASE = r'/Users/andrewsmac/CCIR/Traffic'
URL = 'https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations_bay.csv'
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
BOT_FILE = os.path.join(BASE, 'output', 'bottleneck_rankings.csv')
OUT = os.path.join(BASE, 'output')

# Download the file if not exists
if not os.path.exists(LOC_FILE):
    print("Downloading sensor locations from GitHub...")
    urllib.request.urlretrieve(URL, LOC_FILE)

# Load data
loc_df = pd.read_csv(LOC_FILE)
# Ensure columns are named correctly, usually it's sensor_id, latitude, longitude
if 'sensor_id' not in loc_df.columns and len(loc_df.columns) >= 3:
    loc_df.columns = ['sensor_id', 'latitude', 'longitude']
# Cast sensor_id to string to match bottleneck rankings
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)

bot_df = pd.read_csv(BOT_FILE)
bot_df['sensor_id'] = bot_df['sensor_id'].astype(str)

# Merge
df = pd.merge(loc_df, bot_df, on='sensor_id', how='inner')
print(f"Merged {len(df)} sensors with location data.")

# Set up the plot
plt.style.use('default')
plt.figure(figsize=(12, 10))

# Scatter plot: x=longitude, y=latitude, color=stress_score
sc = plt.scatter(
    df['longitude'], 
    df['latitude'], 
    c=df['stress_score'], 
    cmap='hot_r', 
    s=df['stress_score']*2 + 20, # Size proportional to stress
    alpha=0.8,
    edgecolors='white',
    linewidths=0.5
)

# Highlight top 5 bottlenecks
top5 = df.nlargest(5, 'stress_score')
for i, row in top5.iterrows():
    plt.annotate(
        row['sensor_id'], 
        (row['longitude'], row['latitude']),
        textcoords="offset points", 
        xytext=(0,10), 
        ha='center',
        fontsize=9,
        color='blue',
        fontweight='bold'
    )

plt.colorbar(sc, label='Stress Score (0-100)')
plt.title('Geographic Distribution of Traffic Bottlenecks (PeMS-BAY)', fontsize=15, fontweight='bold', color='black')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.2)

# Save
out_path = os.path.join(OUT, 'q11_geographic_bottlenecks.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved geographic plot to {out_path}")

# Quick analysis: correlation of congestion between geographically close sensors
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import numpy as np

coords = df[['latitude', 'longitude']].apply(lambda col: col.map(radians)).values
dist_matrix = haversine_distances(coords, coords) * 3959 # miles

# For each sensor, find average stress of neighbors within 2 miles
neighbor_stress = []
for i in range(len(df)):
    # indices of neighbors within 2 miles (excluding self)
    neighbors = np.where((dist_matrix[i] < 2.0) & (dist_matrix[i] > 0))[0]
    if len(neighbors) > 0:
        avg_stress = df.iloc[neighbors]['stress_score'].mean()
    else:
        avg_stress = np.nan
    neighbor_stress.append(avg_stress)

df['neighbor_avg_stress'] = neighbor_stress
valid = df.dropna()
correlation = valid['stress_score'].corr(valid['neighbor_avg_stress'])

print(f"\nPearson correlation between a sensor's stress and its 2-mile neighbors' average stress: {correlation:.3f}")
if correlation > 0.3:
    print("Finding: Strong spatial autocorrelation. Congested sensors tend to be clustered together physically.")
else:
    print("Finding: Weak spatial autocorrelation. Bottlenecks may be isolated points rather than long corridors.")
