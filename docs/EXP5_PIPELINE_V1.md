# EXP5_PIPELINE_V1: The Reference Contract

This document defines the strict, reproducible preprocessing and evaluation contract for **Experiment 5: The Robust Pipeline**. 

Exp 5 is not an experiment testing a new variable; it is the synthesis of the empirically validated baseline into a research-grade reference platform. Every future experiment (TTVs, Multi-Sector, alternative architectures) will be evaluated against this exact contract.

## 1. Preprocessing Configuration

| Step | Specification | Justification |
| :--- | :--- | :--- |
| **Data Source** | TESS light curves | Primary dataset for the current scope. |
| **Temporal Coverage** | 1 sector (27 days) | Multi-sector introduces unresolved data-engineering imbalances (Exp 3). |
| **Filtering** | Savitzky-Golay (window=101) | SG401 (Exp 1) degraded long-period EB calibration. |
| **Outlier Removal** | None | Asymmetric clipping (Exp 4) skewed the stochastic noise distribution. |
| **Scaling** | None (prior to binning) | MAD scaling (Exp 2) harmed the neural representation. |
| **Phase Folding** | V1 Standard Method | Simple strict periodicity folding. |
| **Binning** | 2000 points | Standardized input vector for the 1D CNN. |
| **Normalization** | Z-score (per-sample) | $z = \frac{x - \mu}{\sigma}$. Applied strictly post-binning. |

## 2. Model Architecture
* **Architecture:** 1D Convolutional Neural Network (CNN).
* **Weights:** Independently trained fresh model (not loaded from prior experiments).
* **Uncertainty Component:** Monte Carlo (MC) Dropout layers activated during inference.

## 3. Training & Evaluation Protocol
* **Data Splitting:** Strict, mathematically isolated training/validation cohorts (`random_state=42`).
* **Reproducibility:** A JSON manifest must be generated containing dataset hashes, model hashes, exact Python package versions, and hyperparameter logs.
* **Evaluation Metrics:**
  * Global Accuracy, ROC-AUC, Brier Score, ECE.
  * **Per-Class Metrics:** Precision, Recall, F1 for Planet, EB, and Noise.
  * Confusion Matrix.
* **Uncertainty Evaluation:** Predictions must be accompanied by MC-Dropout variance to distinguish between high-confidence and low-confidence decisions.

## 4. Explainable AI (XAI) Suite
The pipeline will routinely subject representative test cases to a tri-method XAI evaluation:
1. **Grad-CAM**
2. **Integrated Gradients**
3. **SHAP**

Beyond visualizations, XAI outputs should be quantified where possible (e.g., attribution concentration around the transit phase) to mathematically evaluate if the pipeline is consistently basing predictions on physically meaningful regions.

---
*Contract frozen on: [Date of Execution]*
