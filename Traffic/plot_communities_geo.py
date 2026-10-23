import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
import os

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
LOC_FILE = os.path.join(BASE, 'data', 'sensor_locations.csv')
OUT = os.path.join(BASE, 'output')

print("Loading time-series data...")
df = pd.read_csv(DATA, index_col=0, parse_dates=True)

print("Loading locations...")
loc_df = pd.read_csv(LOC_FILE)
if 'sensor_id' not in loc_df.columns and len(loc_df.columns) >= 3:
    loc_df.columns = ['sensor_id', 'latitude', 'longitude']
loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)

print("Computing correlation matrix for all sensors...")
# To save memory, we can compute correlation on a downsampled dataset or just full
corr_matrix = df.corr()

print("Building graph...")
G = nx.Graph()
threshold = 0.6

# Create a mapping of sensor ID to coordinates
loc_dict = loc_df.set_index('sensor_id')[['longitude', 'latitude']].to_dict('index')

valid_nodes = set(corr_matrix.columns).intersection(set(loc_dict.keys()))

for col in valid_nodes:
    G.add_node(col, pos=(loc_dict[col]['longitude'], loc_dict[col]['latitude']))

for i in range(len(corr_matrix.columns)):
    n1 = corr_matrix.columns[i]
    if n1 not in valid_nodes: continue
    for j in range(i+1, len(corr_matrix.columns)):
        n2 = corr_matrix.columns[j]
        if n2 not in valid_nodes: continue
        
        c = corr_matrix.iloc[i, j]
        if c > threshold:
            G.add_edge(n1, n2, weight=c)

# Remove isolates to clean up the plot
G.remove_nodes_from(list(nx.isolates(G)))
print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

print("Running Louvain community detection...")
partition = community_louvain.best_partition(G, weight='weight', random_state=42)
num_communities = len(set(partition.values()))
print(f"Found {num_communities} communities.")

# Plot Geographic Communities
plt.style.use('default')
plt.figure(figsize=(14, 12))

# Get positions for all nodes still in G
pos = nx.get_node_attributes(G, 'pos')

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')

# Draw nodes colored by community
cmap = plt.get_cmap('tab20')
for com in set(partition.values()):
    list_nodes = [node for node in partition.keys() if partition[node] == com]
    # We only draw nodes that exist in 'pos' (which is all nodes in G)
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size=60, 
                           node_color=[cmap(com)]*len(list_nodes), 
                           label=f'Community {com}')

plt.title(f'Geographic Sensor Communities (Louvain, Corr > {threshold})', fontsize=16, fontweight='bold', color='black')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.2)
# Remove the legend if there are too many communities, or keep it if reasonable
if num_communities <= 20:
    plt.legend(loc='best', fontsize=8)

out_path = os.path.join(OUT, 'q15_communities_geo.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Geographic community plot saved to {out_path}")
