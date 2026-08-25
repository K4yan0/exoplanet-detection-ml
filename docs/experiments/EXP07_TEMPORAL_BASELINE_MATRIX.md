# Experiment 7: Temporal Baseline Robustness (1-Sector vs 5-Sector)

## 1. Experimental Setup
To rigorously evaluate the impact of extended temporal baselines on model performance, uncertainty, and calibration, we generated a strictly paired cohort of **58 test targets** (29 Noise, 13 Planet, 16 EB) that successfully passed all data-integrity and NaN checks. 

Every target was evaluated twice:
- **1-Sector Input**: A flattened, folded array derived exclusively from the target's *first available* TESS observation sector.
- **5-Sector Input**: A flattened, folded array derived from concatenating up to 5 consecutive/available TESS observation sectors for the exact same target.

We trained two identical CNN architectures: one on exclusively 1-Sector data (`1-Sector CNN`), and one on exclusively 5-Sector data (`5-Sector CNN`). 

## 2. 2x2 Summary Matrix

| Evaluation Metric | 1-Sector Input (Inference) | 5-Sector Input (Inference) |
| :--- | :--- | :--- |
| **1-Sector CNN (Accuracy)** | 0.6724 (Native) | **0.7586** |
| **5-Sector CNN (Accuracy)** | 0.5862 | 0.6552 (Native) |
| | | |
| **1-Sector CNN (ROC-AUC)** | 0.7594 (Native) | **0.8524** |
| **5-Sector CNN (ROC-AUC)** | 0.7647 | 0.8208 (Native) |
| | | |
| **1-Sector CNN (ECE)** | 0.1716 (Native) | **0.0799** |
| **5-Sector CNN (ECE)** | 0.1591 | 0.1096 (Native) |
| | | |
| **1-Sector CNN (MC-Var)** | 0.0406 (Native) | 0.0403 |
| **5-Sector CNN (MC-Var)** | 0.0128 | **0.0121** (Native) |

## 3. Scientific Analysis

These results isolate the physical effect of having a longer temporal baseline and demonstrate a strong dependence of classification behavior on temporal baseline.

### Finding A: The Input-Side Effect (Better Discrimination & Calibration)
If you look at the columns of the matrix, the **5-Sector Input** is associated with substantially better discrimination and calibration than one-sector inputs, even when processed by the same `1-Sector-trained CNN`. 
When the `1-Sector CNN` is fed 5-sector input during inference, its Accuracy jumps from 0.6724 to 0.7586, its ROC-AUC jumps from 0.7594 to 0.8524, and its Expected Calibration Error (ECE) improves massively (0.1716 to 0.0799).
**Conclusion:** The improvement is consistent with the hypothesis that additional temporal coverage provides a cleaner or more informative representation of the transit signal, without requiring the model itself to have been trained on five-sector data.

### Finding B: The Training-Condition Effect (Uncertainty Reduction)
If you look at the rows of the matrix, specifically at MC-Variance, we see a strong training-condition effect.
The `1-Sector CNN` operates with a high predictive uncertainty (MC-Var ~0.040) no matter what data it sees. 
**Conclusion:** The CNN trained on five-sector inputs exhibited substantially lower MC-Dropout predictive variance (~0.012) than the CNN trained on one-sector inputs, across both evaluation conditions. 

### Finding C: The Conservative Shift (Precision vs Recall)
The `5-Sector CNN` has slightly lower overall *accuracy* than the `1-Sector CNN`. The confusion matrices reveal this is due to a dramatic shift in "Planet" recall. In this small, strictly paired test cohort (13 planets total):
* `1-Sector CNN` correctly guessed 7/13 planets.
* `5-Sector CNN` correctly guessed 2/13 planets (Precision = 1.00).

**Conclusion:** The 5-Sector-trained model adopted a more conservative decision behavior, characterized by substantially lower planetary recall but higher planetary precision in this cohort. The present experiment does not yet establish the physical or representational mechanism responsible for this conservative shift.

### Final Exp 7 Conclusion
Exp 7 demonstrates that extending the observational baseline from one to five TESS sectors can substantially alter classifier behavior within a strictly paired cohort. Five-sector inputs improved discrimination and calibration for both tested CNNs, while the model trained on five-sector data exhibited substantially lower MC-Dropout variance across both input conditions. However, the five-sector-trained model also became markedly more conservative toward planetary classifications, trading planetary recall for higher precision. The present experiment therefore demonstrates a strong dependence of classification behavior on temporal baseline, but does not yet establish the physical or representational mechanism responsible for this conservative shift.
