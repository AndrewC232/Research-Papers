# Category D: Advanced Modeling & Critical Thinking

## Q16. Graph Neural Network (GNN) Traffic Prediction

We built a **Spatio-Temporal Graph Neural Network** to predict highway congestion (15 minutes into the future) by explicitly using the physical geography of the highway system.

### GNN Architecture & Construction
1.  **Nodes:** All 325 PeMS-BAY sensors.
2.  **Edges (Highway Connections):** We mapped the physical latitude/longitude of all sensors and computed the Haversine distance between every pair. We drew an edge between any two sensors that are within **2 miles** of each other, explicitly modeling physical proximity.
3.  **Graph Snapshots:** The time-series data was chunked into discrete graphs. For each timestamp, each node's features included the current speed, past speeds (5 mins ago, 15 mins ago), and the time of day.
4.  **Model:** A PyTorch Geometric `GraphSAGE` model. Rather than treating each sensor as an independent row (like XGBoost does), GraphSAGE *aggregates* the speed features of the physically adjacent sensors before predicting whether the target sensor will become congested.

### Performance Comparison

We compared the GNN's F1 Score against the baseline models we established in Category B:

![GNN Comparison](/Users/andrewsmac/CCIR/Traffic/output/q16_gnn_comparison.png)

| Model | F1 Score | Type |
| :--- | :--- | :--- |
| **Baseline XGBoost** | `0.8913` | Tabular (Tree) |
| **Simple Rule** | `0.8628` | Heuristic |
| **GNN (GraphSAGE)** | `0.6729` | Deep Learning Graph |

### Critical Analysis: Why did the GNN underperform?

It is often assumed that more complex models (like deep learning GNNs) will inherently beat simpler models. Here, the GNN drastically underperformed the Gradient Boosted Tree (0.67 vs 0.89). Why?

1.  **Tabular Supremacy:** Traffic speed prediction (when framed as a flat tabular row with lag features) is fundamentally a tabular data problem. Gradient Boosted Trees (like XGBoost) are notoriously the absolute best-in-class algorithms for tabular data. They can make hard, precise splits on exact speed thresholds.
2.  **Weak Spatial Correlation:** In Category C, we mathematically proved that spatial autocorrelation between 2-mile neighbors is very weak (`0.217`). Bottlenecks are highly localized. Therefore, a GNN that inherently *smoothes* and aggregates data from neighboring nodes actually **dilutes** the signal of the target node! The GNN gets confused by free-flowing neighbors when trying to predict a highly localized choke point.
3.  **Temporal Evolution: Adding LSTM Layers**
    To address the temporal naivety of the first model, we upgraded the architecture to include an **LSTM layer** before the Graph Convolution. 
    - **Naive GNN F1:** `0.6729`
    - **LSTM-GNN (15 Epochs):** `0.7099` (Peak performance)
    - **LSTM-GNN (50 Epochs):** `0.6904` (Degradation due to overfitting)
    
**Conclusion:** Increasing the training cycles actually **hurt** the model's performance. While the training loss continued to drop, the model began overfitting to the specific 2-week training subset, losing its ability to generalize. This highlights a major advantage of the **XGBoost** baseline: it is far more stable and less prone to the catastrophic overfitting often seen in deep learning when working with limited time-series data.
