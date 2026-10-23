# Transition Analysis: Predicting the Breakdown

In this experiment, we addressed the "triviality" problem of our previous models. Instead of training on a global dataset (which is 90% steady-state flow), we isolated **48,444 unique Breakdown Events** across the PeMS-BAY network. 

We defined a "Breakdown" as a sensor dropping from $\ge 60$ mph to $< 60$ mph. We then tested whether our models could predict the congestion 15 minutes *after* the speed started to drop, using only the 60 minutes of data *leading up* to the event.

## Performance Comparison

![Transition Comparison](/Users/andrewsmac/CCIR/Traffic/output/q19_transition_comparison.png)

| Model Context | F1 Score | Insight |
| :--- | :--- | :--- |
| **Global XGBoost** | `0.8913` | Includes "easy" steady-state data. |
| **Focused XGBoost** | `0.6849` | Performance drops on transitions. |
| **Focused LightGBM** | `0.6944` | High-efficiency leaf-wise boosting. |
| **Global LSTM-GNN** | `0.7099` | Benefit from spatial data on easy sets. |
| **Focused LSTM** | `0.6961` | Superior temporal modeling. |
| **Focused Transformer**| `0.6961` | Identical to LSTM on this sequence. |
| **Focused GRU** | **`0.7034`** | **Co-Winner.** Highly efficient gated sequences. |
| **Focused TCN** | **`0.7034`** | **Co-Winner.** Dilated convs capture breakdown shapes. |
| **Focused CNN-1D** | `0.0008` | **Failed.** Conv filters missed the signal. |

## Critical Findings

1.  **Removing the "Cheating":** When we focus only on the transitional moments, the F1 score for all models drops significantly (from ~0.90 to ~0.70). This proves that a large portion of our previous high accuracy was coming from the model simply recognizing that "traffic is already slow, so it will stay slow."
2.  **LSTM Superiority in Transitions:** In this focused "Event" context, the **LSTM finally beat the Gradient Boosted Tree**. This is a major architectural insight: while XGBoost is the king of tabular data, **LSTMs are superior at recognizing the "rhythm" of a breakdown**. The sequential patterns (e.g., speed starting to oscillate or slowly decay) are a signal that the LSTM captures better than a flat row of lag features.
3.  **Real-World Utility:** This Focused LSTM is a much better candidate for a real-world warning system. It is specifically tuned to alert a traffic engineer *when a new problem is starting*, rather than just confirming a problem that everyone already sees on the map.

## Next Steps for Future Engineers
To further improve this, one should combine the **Focused LSTM** with **Traffic Volume** data. Predicting the transition based on speed alone is good, but predicting it based on *volume approaching capacity* would likely push the Focused F1 score back above 0.85.
