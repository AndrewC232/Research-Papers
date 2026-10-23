# Advanced Spatio-Temporal GNN Analysis

In this phase, we moved beyond baseline models to optimize the **Spatial Geometry** and **Attention Mechanisms** of our highway network.

## 1. Spatial Range Sensitivity Analysis
We tested how the physical "connection radius" between highway sensors affects prediction accuracy. We found a clear "Goldilocks Zone" for highway connectivity.

| Connection Radius | F1 Score (Normalized) | Observation |
| :--- | :--- | :--- |
| 1.0 Mile | `0.2190` | **Too Local.** Misses critical upstream bottleneck signals. |
| 2.0 Miles | `0.2256` | **Baseline.** Captures immediate neighbors. |
| **5.0 Miles** | **`0.2685`** | **The Sweet Spot.** Optimal balance of spatial context and noise. |
| 10.0 Miles | `0.1936` | **Too Dense.** Graph becomes noisy; signal is drowned out. |

**Conclusion:** Traffic information propagates effectively over roughly a **5-mile horizon** in the Bay Area. Beyond 5 miles, sensors are too far away to have a direct, predictable impact on local speed within a 15-minute window.

## 2. Graph Attention (GAT) vs. Mean Aggregation (SAGE)
We evaluated whether "weighted attention" (GAT) could improve upon standard averaging (SAGE).

| Model Architecture | F1 Score | Insight |
| :--- | :--- | :--- |
| **LSTM-SAGE (Mean)** | **`0.7099`** | **Winner.** Simple averaging acts as a strong regularizer. |
| **GAT-LSTM (Attention)** | `0.6848` | **Lower Performance.** Attention likely overfits on small datasets. |

**Critical Insight:** 
In the physical world of highways, the "average" speed of your neighbors is a very robust physical indicator. While Attention (GAT) is mathematically more sophisticated, it is prone to overfitting in noisy, highly structured time-series data like traffic. The simpler, "smoother" representation provided by SAGEConv actually helps the model generalize better to new days.

## 3. The "Ideal" GNN Configuration
Based on these experiments, the ultimate GNN for this dataset would be:
1.  **Architecture:** LSTM for time, followed by SAGEConv for space.
2.  **Geometry:** A **5.0 mile** connection radius.
3.  **Regularization:** Heavy dropout to prevent the temporal memorization we saw in the epoch sweep.

![Range Sensitivity](/Users/andrewsmac/CCIR/Traffic/output/gnn_range_sensitivity.png)
![GAT Comparison](/Users/andrewsmac/CCIR/Traffic/output/gat_model_comparison.png)
