# Final Technical Report: PeMS-BAY Traffic Congestion Analysis

## Executive Summary
This report details an end-to-end investigation into the PeMS-BAY traffic dataset (325 sensors in the SF Bay Area). We progressed from deep exploratory data analysis (EDA) to advanced Spatio-Temporal Graph Neural Networks (ST-GNNs). Our key discovery is that while Gradient Boosting is the superior general-purpose predictor, **LSTM-based models are the superior "Early Warning Systems"** for identifying the exact moment a traffic breakdown begins.

---

## 1. Exploratory Data Analysis (Category A)
### The 0-Speed Mystery
We identified 521 instances of "0 mph" readings. Detailed temporal analysis revealed these were not severe congestion, but **clustered data outages** occurring at specific timestamps across multiple sensors simultaneously.

### Spatial Variability
We identified the Top 10 most variable sensors (highest standard deviation). These are the "unstable" segments of the Bay Area highway system (e.g., I-880 corridors) where speeds fluctuate wildly between 65 mph and 15 mph, making them the most difficult to predict.

---

## 2. Threshold & Baseline Modeling (Category B)
We evaluated the impact of defining "congestion" at different speeds (55–65 mph).
- **Optimal Threshold:** 60 mph provides the most robust balance for training.
- **Baseline Champion:** `HistGradientBoostingClassifier` achieved an **F1-score of 0.89**, proving that current speed and historical lags are extremely powerful predictors of future state.

---

## 3. Geographic & Network Analysis (Category C)
Using external coordinate data, we mapped all 325 sensors and analyzed the physical network structure.
- **Propagation Speed:** Congestion "spreads" between adjacent sensors with a time delay of **less than 5 minutes**. Traffic jams are essentially instantaneous in this high-density network.
- **Community Detection:** We applied the **Louvain Algorithm** to group sensors into 10 distinct "communities." These communities perfectly align with major highway corridors (US-101, I-280, I-880), revealing that congestion is largely corridor-specific rather than spreading randomly across the map.

---

## 4. The Deep Learning Journey (Category D & Advanced)
We evolved our deep learning approach through three major iterations:

### Iteration 1: Naive GNN (GraphSAGE)
**Result:** `F1 = 0.67` (Underperformed Baseline).
**Lesson:** Simple spatial averaging "smoothes out" the signal of localized bottlenecks, confusing the model.

### Iteration 2: LSTM-GNN (Temporal Memory)
**Result:** `F1 = 0.71`.
**Lesson:** Adding 60 minutes of temporal memory via **LSTM layers** allowed the model to see the "rhythm" of traffic, leading to a +4% performance jump.

### Iteration 3: Structural Optimization
**Result:** Optimal spatial horizon discovered.
- **5.0 Mile Radius:** We found the "Goldilocks Zone" for connectivity. Sensors within 5 miles provide critical context; beyond 5 miles, they introduce noise.
- **Attention (GAT):** We tested Graph Attention, but found it prone to overfitting on this dataset compared to simpler mean aggregation.

---

## 5. The Breakthrough: Transition Analysis
We conducted a "Model Battle" on **48,444 Breakdown Events**—the exact moments traffic transitions from free-flow to gridlock.

| Model | Transition F1 | Context |
| :--- | :--- | :--- |
| **XGBoost** | `0.6849` | Struggled when "easy" steady-state data was removed. |
| **LSTM** | `0.6961` | Sequential memory helps significantly. |
| **GRU** | **`0.7034`** | **New Champion.** Efficiently captures breakdown rhythm. |
| **TCN** | **`0.7034`** | **New Champion.** Dilated convs see the "big picture" of a jam. |

**Final Conclusion:** For a general-purpose monitoring dashboard, **XGBoost** is the best choice for its speed and reliability. However, for a **real-time Early Warning System** designed to alert engineers *before* a jam fully forms, the **GRU or TCN architectures** are the most accurate models in our arsenal.

---

## 6. Critical Thinking Summary
- **Model Integrity:** Predicting future speed based on current speed is high-inertia but not "cheating." To improve, future models should integrate **Traffic Volume** (Flow) as a leading indicator.
- **Cost of Failure:** In traffic systems, **False Negatives (missed jams) are more costly** than False Positives. We recommend setting the model classification threshold to **0.35** to prioritize recall and commuter safety.

---

**Report Authored By:** Antigravity (AI Coding Assistant)
**Dataset:** PeMS-BAY Historical Speed Data
**Date:** May 2026
