import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import contextily as ctx
import os
from sklearn.metrics.pairwise import haversine_distances
from math import radians

plt.style.use('default')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
CSV_FILE = os.path.join(BASE, 'output', 'st_gnn_sensitivity.csv')
OUT = os.path.join(BASE, 'output')

# ─────────────────────────────────────────────
# PLOT 1: Performance vs Radius (Enhanced)
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_FILE).sort_values('Radius_Miles')

fig, ax1 = plt.subplots(figsize=(14, 7))
color_f1 = '#1a6eb5'
color_edges = '#FF6B6B'

ax2 = ax1.twinx()
ax2.bar(df['Radius_Miles'], df['Edges'], alpha=0.15, color=color_edges,
        width=0.4, label='Edge Count', zorder=1)
ax2.set_ylabel('Number of Graph Edges', color=color_edges, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_edges)

ax1.plot(df['Radius_Miles'], df['F1_Score'], marker='o', color=color_f1,
         linewidth=2.5, markersize=9, label='F1 Score', zorder=3)
ax1.set_xlabel('Connection Radius (Miles)', fontsize=13)
ax1.set_ylabel('F1 Score', color=color_f1, fontsize=13)
ax1.tick_params(axis='y', labelcolor=color_f1)
ax1.set_ylim([0.670, 0.730])
ax1.grid(True, alpha=0.12)
ax1.set_xticks(df['Radius_Miles'])
ax1.set_xticklabels([f'{r}mi' for r in df['Radius_Miles']], rotation=30)

# Highlight the two peaks
peaks = {4.5: 'Local Peak\n4.5mi', 10.0: 'Global Peak\n10.0mi'}
for r, label in peaks.items():
    row = df[df['Radius_Miles'] == r].iloc[0]
    ax1.scatter(row['Radius_Miles'], row['F1_Score'], color='red', s=150, zorder=5)
    ax1.annotate(f"{label}\nF1={row['F1_Score']:.4f}",
                 xy=(row['Radius_Miles'], row['F1_Score']),
                 xytext=(row['Radius_Miles'] + 0.2, row['F1_Score'] + 0.006),
                 fontsize=9, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

# Annotate all F1 values
for _, row in df.iterrows():
    ax1.annotate(f"{row['F1_Score']:.4f}",
                 (row['Radius_Miles'], row['F1_Score']),
                 textcoords="offset points", xytext=(0, 11),
                 ha='center', fontsize=7.5, color=color_f1)

ax1.set_title('Spatio-Temporal GNN (LSTM-GNN) Performance vs. Spatial Connection Radius\n'
              '25 Epochs | 2-Week Dataset | 325 Sensors', fontsize=13, pad=15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'st_gnn_performance_curve.png'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved st_gnn_performance_curve.png")

# ─────────────────────────────────────────────
# PLOT 2 & 3: Geographic Radius Visualization — split into two 2-panel maps
# ─────────────────────────────────────────────
loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns:
    loc_df.columns = ['sensor_id', 'latitude', 'longitude']

# Pick a representative central sensor (sensor closest to the centroid)
center_lat = loc_df['latitude'].mean()
center_lon = loc_df['longitude'].mean()
dists = np.sqrt((loc_df['latitude'] - center_lat)**2 + (loc_df['longitude'] - center_lon)**2)

coords = loc_df[['latitude', 'longitude']].values
coords_rad = np.array([[radians(c[0]), radians(c[1])] for c in coords])
center_idx = dists.idxmin()
center_rad = coords_rad[center_idx]

# Haversine distance from center sensor to all others
center_dists = haversine_distances([center_rad], coords_rad)[0] * 3959.0  # miles

radii_to_show = [1.0, 2.0, 4.5, 10.0]
colors = ['#00AA00', '#0088CC', '#FFD700', '#FF6B6B']
titles = [
    '1.0 Mile Radius\n(F1 = 0.7040)',
    '2.0 Mile Radius\n(F1 = 0.7051)',
    '4.5 Mile Radius\n(F1 = 0.7055 — Local Peak)',
    '10.0 Mile Radius\n(F1 = 0.7179 — Global Peak)'
]

# Pre-build GeoDataFrame once
gdf_all = gpd.GeoDataFrame(
    loc_df,
    geometry=gpd.points_from_xy(loc_df['longitude'], loc_df['latitude']),
    crs="EPSG:4326"
).to_crs(epsg=3857)


def make_geo_panel(ax, r, color, title):
    """Render a single radius panel onto ax."""
    connected = center_dists <= r
    not_connected = ~connected

    if not_connected.any():
        gdf_all[not_connected].plot(ax=ax, color='gray', markersize=12, alpha=0.4, zorder=3)
    gdf_all[connected].plot(ax=ax, color=color, markersize=40, alpha=0.9,
                            edgecolor='white', linewidth=0.5, zorder=4)
    gdf_all.iloc[[center_idx]].plot(ax=ax, color='white', markersize=100, marker='*', zorder=5)

    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=10)
    except Exception:
        ctx.add_basemap(ax, zoom=10)

    center_pt = gdf_all.iloc[center_idx].geometry
    radius_m = r * 1609.34
    circle = center_pt.buffer(radius_m)
    circle_gdf = gpd.GeoDataFrame(geometry=[circle], crs="EPSG:3857")
    circle_gdf.boundary.plot(ax=ax, color=color, linewidth=2.5, linestyle='--', alpha=0.9, zorder=4)

    pad_m = 12.0 * 1609.34
    x_c, y_c = center_pt.x, center_pt.y
    ax.set_xlim(x_c - pad_m, x_c + pad_m)
    ax.set_ylim(y_c - pad_m, y_c + pad_m)

    n_connected = connected.sum()
    n_edges = n_connected * 2
    ax.set_axis_off()
    ax.set_title(f'{title}\n{n_connected} neighbors | ~{n_edges} edges from center',
                 fontsize=11, color='black', pad=10)


# ── Map A: 1 mi and 2 mi (small radii) ──────────────────────────────────────
fig_a, axes_a = plt.subplots(1, 2, figsize=(14, 7))
fig_a.patch.set_facecolor('white')

for ax, r, color, title in zip(axes_a, radii_to_show[:2], colors[:2], titles[:2]):
    make_geo_panel(ax, r, color, title)

legend_a = [
    mpatches.Patch(facecolor='white', edgecolor='black', label='★ Reference Sensor'),
    mpatches.Patch(facecolor=colors[0], label='Connected (1mi)'),
    mpatches.Patch(facecolor=colors[1], label='Connected (2mi)'),
    mpatches.Patch(facecolor='gray', alpha=0.5, label='Not Connected'),
]
fig_a.legend(handles=legend_a, loc='lower center', ncol=4, fontsize=10,
             facecolor='white', edgecolor='gray', labelcolor='black', framealpha=0.9)
fig_a.suptitle('Sensor Connection Radius — Bay Area Highway Network (Small Radii: 1mi & 2mi)',
               fontsize=13, color='black', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'radius_geographic_small.png'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved radius_geographic_small.png")

# ── Map B: 4.5 mi and 10 mi (peak radii) ─────────────────────────────────────
fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 7))
fig_b.patch.set_facecolor('white')

for ax, r, color, title in zip(axes_b, radii_to_show[2:], colors[2:], titles[2:]):
    make_geo_panel(ax, r, color, title)

legend_b = [
    mpatches.Patch(facecolor='white', edgecolor='black', label='★ Reference Sensor'),
    mpatches.Patch(facecolor=colors[2], label='Connected (4.5mi — Local Peak)'),
    mpatches.Patch(facecolor=colors[3], label='Connected (10mi — Global Peak)'),
    mpatches.Patch(facecolor='gray', alpha=0.5, label='Not Connected'),
]
fig_b.legend(handles=legend_b, loc='lower center', ncol=4, fontsize=10,
             facecolor='white', edgecolor='gray', labelcolor='black', framealpha=0.9)
fig_b.suptitle('Sensor Connection Radius — Bay Area Highway Network (Peak Radii: 4.5mi & 10mi)',
               fontsize=13, color='black', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'radius_geographic_peaks.png'), dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved radius_geographic_peaks.png")
print("Done!")
