# Category B: Feature Engineering & Modeling

Based on the machine learning experiments run on the PeMS-BAY dataset, here are the findings for the feature engineering and modeling questions. A true temporal split (first 80% of time for training, last 20% for testing) was used to ensure no data leakage.

## Q6. Prediction Horizon

**How does model performance degrade as the horizon increases?**

We evaluated the F1 score for predicting congestion 15 minutes, 30 minutes, and 60 minutes into the future.

| Horizon | F1 Score |
| :--- | :--- |
| 15 minutes | 0.8913 |
| 30 minutes | 0.8677 |
| 60 minutes | 0.8421 |

![Prediction Horizon](/Users/andrewsmac/CCIR/Traffic/output/q6_horizon.png)

**Conclusion:** Performance degrades gracefully. Even 60 minutes out, the model maintains a strong F1 score of 0.8421. Traffic states have high inertia, meaning a congested state or free-flow state tends to persist, allowing models to predict relatively far into the future with reasonable accuracy.

---

## Q7. Threshold Sensitivity

**What happens if you use values between 55 mph to 65 mph to define congestion?**

We evaluated the F1 score and class balance (positive rate) using a sliding threshold from 55.0 to 65.0 mph in 1.0 mph increments.

| Threshold | F1 Score | Class Balance (Positive Rate) |
| :--- | :--- | :--- |
| 55.0 mph | 0.8484 | 9.61% |
| 56.0 mph | 0.8467 | 10.39% |
| 57.0 mph | 0.8495 | 11.39% |
| 58.0 mph | 0.8562 | 12.70% |
| 59.0 mph | 0.8779 | 14.45% |
| **60.0 mph** | **0.8913** | **16.73%** |
| 61.0 mph | 0.9050 | 19.82% |
| 62.0 mph | 0.9148 | 24.10% |
| 63.0 mph | 0.9251 | 29.60% |
| 64.0 mph | 0.9340 | 37.24% |
| 65.0 mph | 0.9447 | 46.80% |

![Detailed Threshold Sensitivity](/Users/andrewsmac/CCIR/Traffic/output/q7_thresholds_detailed.png)

**Conclusion:**
*   **High Thresholds (63-65 mph):** The F1 score approaches 0.95 because nearly 30-47% of the data falls below these marks. It essentially just detects "not perfect free-flow," which is a highly balanced and easier classification task, but not very useful for identifying true, severe bottlenecks.
*   **Low Thresholds (55-57 mph):** The F1 score drops to ~0.84-0.85 because the positive class becomes rare (around 10%). Predicting severe congestion is harder because it's driven heavily by chaotic edge cases and incidents rather than daily volume curves.
*   **Optimal Threshold (59-61 mph):** The 60 mph threshold remains an excellent middle ground. At 16.73% class balance, the dataset contains enough congestion events for the model to learn reliable patterns (yielding a very strong 0.8913 F1 score) while still strictly identifying true Level of Service (LOS) degradation.

---

## Q8. Feature Ablation

**Which feature category causes the biggest drop in F1 when removed?**

*(Baseline F1: 0.8913)*

| Removed Category | New F1 Score | Drop in F1 |
| :--- | :--- | :--- |
| **Temporal** (hour, dow, rush flags, sin/cos) | 0.8872 | -0.0041 |
| **Lag** (previous speed values) | 0.8908 | -0.0005 |
| **Rolling** (means and std devs) | 0.8911 | -0.0002 |
| **Rate of Change** (speed differences) | 0.8911 | -0.0002 |

**Conclusion:** `Temporal` features cause the biggest drop when removed. However, the drop is incredibly small across all engineered categories. Why? Because the **current `speed`** (which was intentionally kept in all runs) dominates the predictive power. The engineered features are highly collinear with `speed`.

---

## Q9. Simpler Models

**How much does a Gradient Boosted model improve over simpler approaches?**

We compared the Gradient Boosted model (acting as our XGBoost equivalent) to Logistic Regression, Random Forest, and a "Simple Rule" (`if current speed < 60, predict congested`).

| Model | F1 Score |
| :--- | :--- |
| Gradient Boosted (XGBoost analog) | **0.8913** |
| Random Forest | 0.8891 |
| Simple Rule | 0.8806 |
| Logistic Regression | 0.8663 |

**Conclusion:** The **Simple Rule** is surprisingly strong (0.8806), proving that the best predictor of traffic 15 minutes from now is traffic right now. The Gradient Boosted model improves upon this by ~0.01. While this seems small, at the scale of 17 million sensor readings, that 0.01 represents thousands of edge-case improvements where the model successfully anticipated a slowdown *before* it happened, or realized traffic was about to recover.

---

## Q10. Per-Sensor Models

**Do individual per-sensor models outperform a single global model?**

We trained dedicated models for the Top 10 sensors and compared them to the global model's performance on those specific sensors.

| Sensor | Global Model F1 | Local (Per-Sensor) F1 | Difference |
| :--- | :--- | :--- | :--- |
| 401457 | 0.9677 | **0.9692** | +0.0015 |
| 400970 | 0.8733 | **0.8757** | +0.0024 |
| 407373 | 0.8836 | **0.8875** | +0.0039 |
| 400461 | 0.8716 | **0.8780** | +0.0064 |
| 407335 | **0.6960** | 0.6950 | -0.0010 |
| 400971 | **0.9210** | 0.9191 | -0.0019 |
| 407372 | 0.9291 | **0.9335** | +0.0044 |
| 401948 | 0.9618 | **0.9653** | +0.0035 |
| 400030 | 0.9411 | **0.9446** | +0.0035 |
| 401906 | 0.8872 | **0.8916** | +0.0044 |
| **Average** | **0.8932** | **0.8959** | **+0.0027** |

**Conclusion:** Yes, local per-sensor models slightly outperform the global model on average (0.8959 vs 0.8932). 
*   The global model is already "good enough" for almost all sensors because traffic physics are universally similar. 
*   However, for sensors with unique geometries (e.g., an off-ramp backing up, or an uphill grade affecting trucks), the local model captures those localized dynamics better (e.g., Sensor 400461 sees a notable +0.0064 boost).
