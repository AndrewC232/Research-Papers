import pandas as pd
import numpy as np

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = f'{BASE}/data/PEMS-BAY.csv'

print("Loading data...")
df = pd.read_csv(DATA)
df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Find all 0.0 readings
# Iterate over all sensors to find zero readings
zero_surroundings = []
sensor_averages = df.mean()

for col in df.columns:
    zero_indices = df.index[df[col] == 0.0].tolist()
    for idx in zero_indices:
        idx_pos = df.index.get_loc(idx)
        
        # Get t-1 and t+1, making sure we don't go out of bounds
        val_before = df[col].iloc[idx_pos - 1] if idx_pos > 0 else np.nan
        val_after = df[col].iloc[idx_pos + 1] if idx_pos < len(df) - 1 else np.nan
        
        # Also let's get the sensor's average speed for comparison
        sensor_avg = sensor_averages[col]
        
        zero_surroundings.append({
            'sensor': col,
            'timestamp': idx,
            'speed_before': val_before,
            'speed_zero': 0.0,
            'speed_after': val_after,
            'sensor_avg': sensor_avg
        })

z_df = pd.DataFrame(zero_surroundings)
print(f"Total 0-speed readings found: {len(z_df)}")

# Calculate the mean of the surrounding readings
mean_before = z_df['speed_before'].mean()
mean_after = z_df['speed_after'].mean()
mean_sensor_avg = z_df['sensor_avg'].mean()
network_avg = df.mean().mean()

print(f"\nAverage speed 5 mins BEFORE the 0-reading: {mean_before:.2f} mph")
print(f"Average speed 5 mins AFTER the 0-reading:  {mean_after:.2f} mph")
print(f"Average speed of those specific sensors:   {mean_sensor_avg:.2f} mph")
print(f"Overall network average speed:             {network_avg:.2f} mph")

# How many are surrounded by free-flow (>60 mph) vs congestion (<60 mph)?
free_flow_before = (z_df['speed_before'] >= 60).mean() * 100
free_flow_after = (z_df['speed_after'] >= 60).mean() * 100

print(f"\n% of 0-readings with free-flow (>60mph) right BEFORE: {free_flow_before:.1f}%")
print(f"% of 0-readings with free-flow (>60mph) right AFTER:  {free_flow_after:.1f}%")

print("\nSample of 0-readings and their surroundings:")
print(z_df.sample(min(10, len(z_df))).to_string())
