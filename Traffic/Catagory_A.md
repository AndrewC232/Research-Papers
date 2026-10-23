# Category A: Exploratory Data Analysis

Based on historical data from the PeMS-BAY dataset (January–June 2017) and supplementary web research, the following findings answer the exploratory data analysis questions.

## Q1. Seasonal Trends

**Does congestion get worse or better over the 6-month period (January → June)?**

| Month | Network Congestion Rate (< 60 mph) |
| :--- | :--- |
| January 2017 | 17.72% |
| February 2017 | 16.84% |
| March 2017 | 17.05% |
| April 2017 | 13.74% |
| May 2017 | 16.47% |
| June 2017 | 18.51% |

![Monthly Congestion](/Users/andrewsmac/CCIR/Traffic/output/q1_seasonal_trends.png)

**Conclusion:** Traffic congestion is not perfectly consistent; it generally worsens heading into the summer months, with June peaking at 18.51%. There is a notable dip in April (13.74%), possibly due to spring breaks or localized factors, but the overall trend aligns with Bay Area macro trends. Typically, traffic volume in the Bay Area increases by early summer due to warmer weather and increased seasonal travel.

---

## Q2. Sensor Correlation

**Are nearby bottlenecks correlated?**

We selected the Top 5 most congested sensors (401457, 400970, 407373, 400461, 407335) and computed their pairwise Pearson correlations.

| | 401457 | 400970 | 407373 | 400461 | 407335 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **401457** | 1.000 | 0.259 | 0.235 | 0.027 | 0.041 |
| **400970** | 0.259 | 1.000 | **0.495** | 0.059 | 0.044 |
| **407373** | 0.235 | **0.495** | 1.000 | 0.008 | 0.031 |
| **400461** | 0.027 | 0.059 | 0.008 | 1.000 | **0.472** |
| **407335** | 0.041 | 0.044 | 0.031 | **0.472** | 1.000 |

![Correlation Matrix](/Users/andrewsmac/CCIR/Traffic/output/q2_correlation.png)

**Conclusion:** 
Yes, specific bottlenecks show strong correlations. For instance, **400970** and **407373** exhibit a high correlation (~0.495), as do **400461** and **407335** (~0.472). 
**Implication:** High correlation implies that these sensors are physically adjacent or reside on the same critical travel corridor. When traffic builds up at one sensor, the congestion wave propagates to the other.

---

## Q3. Congestion Duration

**How long does it typically stay congested?**

By isolating contiguous blocks of time where the speed dropped below 60 mph for the top bottleneck sensors, we measured the duration of congestion events:

*   **AM Rush Average Duration:** 98.2 minutes
*   **PM Rush Average Duration:** 47.6 minutes

![Duration Distribution](/Users/andrewsmac/CCIR/Traffic/output/q3_durations.png)

**Conclusion:** AM congestion events for these specific extreme bottlenecks tend to be longer and more sustained (averaging over an hour and a half), whereas PM rush events—though potentially more severe in speed drops—are broken into shorter, fragmented continuous periods (under an hour). 

---

## Q4. Speed Variability

**Which sensors have the highest standard deviation in speed?**

| Sensor | Std Dev (mph) |
| :--- | :--- |
| 400069 | 18.11 |
| 400895 | 18.03 |
| 400586 | 17.27 |
| 400172 | 16.34 |
| 400372 | 16.31 |
| 400394 | 16.28 |
| 402365 | 16.23 |
| 403414 | 15.91 |
| 400209 | 15.88 |
| 400973 | 15.14 |

![Top 10 Sensors Std Dev](/Users/andrewsmac/CCIR/Traffic/output/q4_top10_std.png)

**Conclusion:** High-variability sensors are **different** from the absolute highest-congestion sensors. The most congested sensors (e.g., 401457 at 77.5% frequency) are chronically slow, meaning their standard deviation is lower because they remain clustered at lower speeds. 
**Implication:** Sensors with high variability but moderate congestion frequency experience extreme "stop-and-go" conditions or severe but localized rush hour spikes, rapidly swinging between 65+ mph free-flow and < 20 mph gridlock. 

---

## Q5. The 0-Speed Mystery

**Do the 521 readings of 0.0 mph represent actual gridlock, zero traffic volume, or sensor malfunctions?**

At first glance, one might wonder if 0 mph means there were simply *zero cars* driving on the highway at that exact time (especially since some of these happen late at night). However, a deeper dive reveals the truth:

*   **Average speed 5 mins BEFORE a 0-reading:** 66.03 mph
*   **Average speed 5 mins AFTER a 0-reading:** 66.33 mph
*   **Timestamp Clustering:** The 521 zero readings do not happen randomly. They are clustered into exactly **four** distinct 5-minute intervals across the entire 6 months:
    *   `2017-04-20 01:40:00` (91 sensors reported 0.0)
    *   `2017-05-05 19:55:00` (81 sensors reported 0.0)
    *   `2017-05-09 20:35:00` (132 sensors reported 0.0)
    *   `2017-06-15 01:00:00` (217 sensors reported 0.0)

![Zero-Speed Distribution](/Users/andrewsmac/CCIR/Traffic/output/q5_zeros.png)

**Conclusion:** These are definitively **system-wide data outages or aggregation failures** at the Caltrans PeMS server level. 
It is physically impossible for 81 highway sensors in the Bay Area to have exactly zero cars passing over them at 7:55 PM on a Friday (`2017-05-05 19:55:00`). Likewise, traffic cannot go from 66 mph to 0 mph and back to 66 mph in 15 minutes across 217 sensors simultaneously. 

**How to handle them:** In a machine learning model, these should be handled via **imputation** (e.g., forward-filling from the previous 5-minute step or using a localized rolling median) so the model doesn't falsely interpret a 0.0 reading as extreme physical traffic severity or a completely empty road.
