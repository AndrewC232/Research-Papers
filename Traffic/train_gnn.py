import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report

ARTIFACTS_DIR = "/Users/andrewsmac/.gemini/antigravity/brain/6f588973-1e4f-4c94-b293-3e963ca3c2d1/artifacts"
CSV_PATH = os.path.join(ARTIFACTS_DIR, "traffic_data_with_maps.csv")
PLOT_PATH = os.path.join(ARTIFACTS_DIR, "gnn_learning_curve.png")

# 1. Download Graph for San Francisco
print("Downloading San Francisco road network (this may take a minute)...")
G = ox.graph_from_place("San Francisco, California, USA", network_type="drive", simplify=True)
print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")

# 2. Load Traffic Data and Map to Nodes
print("Loading traffic data and mapping to nodes...")
df = pd.read_csv(CSV_PATH)
# Filter specifically for SF bounds roughly
sf_df = df[
    (df['coord_lat'] > 37.7) & (df['coord_lat'] < 37.85) &
    (df['coord_lon'] > -122.52) & (df['coord_lon'] < -122.35)
].copy()

# Find nearest nodes for each incident
nearest_nodes = ox.nearest_nodes(G, sf_df['coord_lon'].values, sf_df['coord_lat'].values)
incident_nodes = set(nearest_nodes)
print(f"Mapped incidents to {len(incident_nodes)} unique nodes.")

# 3. Construct PyTorch Geometric Data Object
nodes = list(G.nodes(data=True))
node_mapping = {node_id: idx for idx, (node_id, _) in enumerate(nodes)}

# Node Features: [lat, lon, degree]
x_features = []
y_labels = []

for node_id, data in nodes:
    lat = data['y']
    lon = data['x']
    deg = G.degree(node_id)
    x_features.append([lat, lon, deg])
    
    # Binary Label: 1 if incident fell on this node, else 0
    if node_id in incident_nodes:
        y_labels.append(1)
    else:
        y_labels.append(0)

# Normalize features
x_tensor = torch.tensor(x_features, dtype=torch.float)
x_tensor = (x_tensor - x_tensor.mean(dim=0)) / x_tensor.std(dim=0)

y_tensor = torch.tensor(y_labels, dtype=torch.long)

# Edges
edge_index = []
for u, v, _ in G.edges(data=True):
    edge_index.append([node_mapping[u], node_mapping[v]])
    edge_index.append([node_mapping[v], node_mapping[u]]) # Undirected for message passing

edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

data_pyg = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)
print(f"PyG Data: {data_pyg}")

# Class imbalance weights
num_pos = y_tensor.sum().item()
num_neg = len(y_tensor) - num_pos
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float)

# Masks for Train/Test
indices = torch.randperm(data_pyg.num_nodes)
train_split = int(0.8 * data_pyg.num_nodes)
train_mask = torch.zeros(data_pyg.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data_pyg.num_nodes, dtype=torch.bool)
train_mask[indices[:train_split]] = True
test_mask[indices[train_split:]] = True

data_pyg.train_mask = train_mask
data_pyg.test_mask = test_mask

# 4. Define GNN Architecture
class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(data_pyg.num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.out(x)
        return x

model = GraphSAGE(hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 5. Training Loop
print("Training GNN...")
model.train()
losses = []

for epoch in range(100):
    optimizer.zero_grad()
    out = model(data_pyg.x, data_pyg.edge_index).squeeze()
    loss = criterion(out[data_pyg.train_mask], data_pyg.y[data_pyg.train_mask].float())
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# 6. Evaluation
model.eval()
with torch.no_grad():
    out = model(data_pyg.x, data_pyg.edge_index).squeeze()
    pred = (torch.sigmoid(out) > 0.5).long()
    test_y = data_pyg.y[data_pyg.test_mask]
    test_pred = pred[data_pyg.test_mask]
    
    print("\n--- Test Set Evaluation ---")
    print(classification_report(test_y.cpu().numpy(), test_pred.cpu().numpy(), target_names=['Clear (0)', 'Incident (1)']))

# 7. Plot Learning Curve
plt.figure(figsize=(8, 5))
plt.plot(losses, label="BCE Loss", color='blue')
plt.title("GNN Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=120)
print(f"Learning curve saved to: {PLOT_PATH}")
