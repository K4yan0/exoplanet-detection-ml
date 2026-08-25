# Experiment 10D: Stability Control (Sector-Aware Transformer)

## Hypothesis
If the high-resolution relational capability of the Exp 10C architecture (2D positional encoding + `[CLS]` token) is robust, it should consistently recover planetary transits across multiple independent random initializations. If Run 1 (which achieved 10/10 targeted recovery) was an outlier, the architecture suffers from severe optimization instability.

## Architecture
**Identical to Exp 10C:**
*   **Input:** 5 separate sectors (2000 bins each).
*   **CNN Encoder:** Extracted 250 local features per sector.
*   **Positional Encoding:** 2D explicitly parameterized (`sector_id` + `within_sector_phase`).
*   **Aggregation:** 4-layer Transformer Encoder + Learned `[CLS]` token.

## Protocol
The exact same architecture and hyperparameters were run across **5 independent random seeds** (42, 100, 200, 300, 400) to evaluate the variance of the optimization trajectory. 

## Results: Multi-Seed Execution

| Seed | Accuracy | Planet Precision | Planet Recall | Targeted Recovery | Collapse State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 63.16% | 0.00 | 0.00 | 0 / 10 | No Planet predictions |
| **100** | 71.05% | 0.00 | 0.00 | 0 / 10 | No Planet predictions |
| **200** | 57.89% | 0.00 | 0.00 | 0 / 10 | No Planet predictions |
| **300** | 50.00% | 0.00 | 0.00 | 0 / 10 | No Planet predictions |
| **400** | 65.79% | 0.00 | 0.00 | 0 / 10 | No Planet predictions |

### Summary Statistics
*   **Mean Accuracy:** `61.58% ± 7.18%`
*   **Mean Precision:** `0.000 ± 0.000`
*   **Mean Recall:** `0.000 ± 0.000`

## Analysis
The result is unequivocally negative: **Exp 10C is catastrophically unstable.** 

Across 5 independent seeds, the network collapsed 100% of the time into a degenerate state where it **completely failed to predict the Planet class** (Precision and Recall were exactly 0.0 across all runs). The confusion matrices show that the model learns to output only `Noise` or `Eclipsing Binary`, ignoring the minority `Planet` class entirely.

### What This Means
1.  **Run 1 was a "Lottery Ticket":** The 10/10 recovery in the first run of 10C was an exceptionally lucky initialization and optimization trajectory. The *capacity* to learn the relationship exists in the architecture, but standard training cannot reliably find it.
2.  **The Transformer Optimization Barrier:** When applied directly to this highly imbalanced, sparse-signal task, the standard self-attention mechanism fails to bootstrap useful representations before the loss landscape funnels it into a trivial local minimum (guessing majority classes).
3.  **The 9C Champion Stands:** The Exp 9C architecture (Coarse Spatial + LSTM) avoids this collapse entirely. By enforcing sequential order and coarse localization, it creates a much smoother, more constrained optimization landscape that reliably converges to high accuracy and high planet recall.

## Conclusion
High-resolution Transformer representations can theoretically solve the phase-alignment problem (as seen in Run 1), but they are practically unusable on this dataset without a different training paradigm (e.g., pre-training, contrastive learning, or specialized regularization). 

**Exp 9C remains the reigning benchmark architecture.**
