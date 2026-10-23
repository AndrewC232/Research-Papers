# Category C: Advanced Geographic Analysis

To explore the spatial relationships between the 325 loop-detector sensors in the PeMS-BAY dataset, we successfully downloaded the exact geographic coordinates (Latitude/Longitude) for 324 of the sensors from the original research repository.

We then merged these locations with our previously calculated **Stress Scores** (the composite metric of congestion frequency, severity, and peak-hour concentration) to see if congestion is localized or spreads across entire corridors.

## Geographic Distribution of Traffic Bottlenecks

![Geographic Distribution of Traffic Bottlenecks](/Users/andrewsmac/CCIR/Traffic/output/q11_geographic_bottlenecks.png)

*(Note: The size and brightness of each dot represent the severity of the Stress Score. The Top 5 absolute worst bottlenecks are highlighted with their Sensor IDs in cyan).*

## Q15. Graph-Based Analysis (Spatial Autocorrelation)

**Are sensors that are physically close to each other similarly congested?**

To test this mathematically, we computed the Haversine distance between all pairs of sensors. For every sensor, we calculated the average stress score of all its neighbors within a **2-mile radius**, and then measured the Pearson correlation between the sensor's own stress and its neighbors' average stress.

*   **Spatial Autocorrelation (Pearson r):** `0.217`

**Finding:** A correlation of `0.217` is surprisingly weak. This reveals that bottlenecks are highly localized "choke points." Rather than an entire 10-mile stretch of highway breaking down evenly, the most severe congestion is anchored to very specific, isolated points.

### Sensor Community Detection
Using the Louvain community detection algorithm on a graph where edges represent a speed correlation > 0.6, we evaluated all 325 sensors. The algorithm organically found **10 distinct communities**.

Crucially, when we plot these communities using their actual geographic coordinates, the results are stunning:

![Sensor Communities Geographic](/Users/andrewsmac/CCIR/Traffic/output/q15_communities_geo.png)

**What do these clusters reveal about highway structure?**
1.  **Corridor-Based Traffic Flow:** Despite the algorithm knowing *nothing* about the physical locations of the sensors (it only looked at how their speeds fluctuated over time), the resulting clusters perfectly trace physical highway corridors and directional flows.
2.  **Directional Independence:** A cluster typically represents a specific direction on a specific highway (e.g., I-880 Northbound). This reveals that traffic on opposite sides of the same physical highway often behaves completely independently.
3.  **Network Segmentation:** The Bay Area highway network is not one giant interconnected fluid system; it is physically segmented into macro-corridors. A massive traffic jam in the East Bay community rarely bleeds over into the South Bay community, because the connecting interchanges serve as natural "firewalls" that regulate throughput.

---

## Q11. Congestion Propagation

**Does congestion "spread" with a time delay between neighbors?**

We computed the cross-correlation between the most congested sensor (401457) and its most highly correlated neighbor across different time shifts (-30 mins to +30 mins). 

![Congestion Propagation](/Users/andrewsmac/CCIR/Traffic/output/q11_propagation.png)

**Conclusion:** The maximum correlation occurred exactly at **0 minutes shift**. Because the PeMS-BAY data is aggregated into 5-minute intervals, this implies that congestion propagates *very quickly*—reaching the adjacent upstream sensor within the same 5-minute window rather than taking 10 or 15 minutes to slowly crawl backwards.

---

## Q12. Anomaly Detection

**Are there outlier days with unusually high or low congestion?**

By tracking the daily network-wide congestion rate and computing Z-scores, we flagged days where congestion was extremely abnormal (|Z| > 2.0).

![Anomaly Detection](/Users/andrewsmac/CCIR/Traffic/output/q12_anomalies.png)

**Anomalies Detected:**
*   `2017-01-10`: 38.01% congestion (Z = +2.72)
*   `2017-01-18`: 36.86% congestion (Z = +2.57)
*   `2017-01-20`: 32.45% congestion (Z = +2.01)
*   `2017-02-17`: 35.50% congestion (Z = +2.40)
*   `2017-03-01`: 42.61% congestion (Z = +3.31)
*   `2017-03-02`: 42.91% congestion (Z = +3.35)

**What Happened?** Early 2017 was defined by historic **Atmospheric River storms** in California that ended a multi-year drought but caused massive flooding, mudslides, and highway closures. These dates align perfectly with the peak landfalls of those severe winter storm systems, causing the entire network to grind to a halt.

---

## Q13. Class Imbalance Strategies

**Which strategy works best for handling imbalanced data?**

We evaluated various techniques on a subset of the data:
*   **Baseline F1:** `0.9014`
*   **Undersampling F1:** `0.9011`
*   **SMOTE F1:** `0.9004`
*   **Threshold 0.4 F1:** `0.9012`

**Conclusion:** For gradient boosted trees on this dataset, artificially resampling the data (SMOTE/Undersampling) or shifting the threshold did not meaningfully improve the F1 score. The baseline model is already robust enough to handle the native class balance effectively.

---

## Q14. Multi-Class Severity

**Which severity levels are hardest to predict?**

Instead of binary prediction, we divided speeds into 4 classes and trained a classifier.

| Severity Class | Speed Range | F1 Score |
| :--- | :--- | :--- |
| **Congested** | 45 - 60 mph | 0.85 |
| **Free Flow** | > 65 mph | 0.81 |
| **Severe** | < 45 mph | 0.81 |
| **Moderate** | 60 - 65 mph | 0.70 |

**Conclusion:** **Moderate traffic (60-65 mph)** is by far the hardest state to predict (F1 = 0.70). 
Why? Because it represents the volatile, unstable boundary between free-flowing traffic and a breaking-down bottleneck. Free-flow and severe gridlock are "stable states" that persist for long periods, making them easy to predict, whereas "moderate" traffic is a rapid transitional phase.
