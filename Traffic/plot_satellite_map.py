import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import os

BASE = r'/Users/andrewsmac/CCIR/Traffic'
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')

# Load coordinates
loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns and len(loc_df.columns) >= 3:
    loc_df.columns = ['sensor_id', 'latitude', 'longitude']

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    loc_df, 
    geometry=gpd.points_from_xy(loc_df.longitude, loc_df.latitude),
    crs="EPSG:4326"
)

# Convert to Web Mercator for contextily
gdf = gdf.to_crs(epsg=3857)

# Plot
fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(ax=ax, color='blue', markersize=40, alpha=0.8, edgecolor='black', linewidth=0.8)

# Add satellite basemap
try:
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
except Exception as e:
    print(f"Error fetching tile: {e}")
    # Fallback to default OpenStreetMap if Esri fails
    ctx.add_basemap(ax)

ax.set_axis_off()
plt.title('PeMS-BAY Sensor Network Overview', fontsize=20, color='black', pad=20)
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'sensor_satellite_map.png'), facecolor=fig.get_facecolor(), dpi=300, bbox_inches='tight')
plt.close()
print("Satellite map generated successfully.")
