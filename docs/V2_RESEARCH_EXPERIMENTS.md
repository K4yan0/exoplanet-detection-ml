# Exoplanet AI: V2 Research Experiments

This document defines the research framework for upgrading the Exoplanet AI pipeline from V1 to V2. The objective is to empirically determine which preprocessing strategy produces the most accurate, calibrated, uncertainty-aware, and scientifically interpretable classifier.

## The Contractual Baseline (V1)
Model V1 is strictly frozen to the exact dataset distribution it was trained on. 
* **Model Checkpoint**: `exoplanet_cnn_v1.keras`
* **Sectors**: 1 (No multi-sector stitching)
* **Outlier Removal**: OFF
* **Savitzky-Golay Filter**: Window Length = 101
* **Mathematical Scaling**: Z-Score Normalization
* **Clipping**: OFF

## V2 Experimental Matrix
We will evaluate the effects of "Robust" preprocessing methods. The architecture and training protocol of the CNN will remain **strictly constant** across all experiments to ensure that any variation in performance metrics is directly attributable to the data preprocessing changes.

| Experiment | Description | Sectors | SG Window | Outlier Removal | Scaling | Clipping |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | The V1 training specification | 1 | 101 | OFF | Z-Score | OFF |
| **Exp 1** | Window Length Test | 1 | 401 | OFF | Z-Score | OFF |
| **Exp 2** | Scaling Method Test | 1 | 101 | OFF | MAD | OFF |
| **Exp 3** | Multi-Sector Stability Test | 5 | 101 | OFF | Z-Score | OFF |
| **Exp 4** | Temporal Outlier Removal Test | 1 | 101 | 4σ | Z-Score | OFF |
| **Exp 5** | Full Robust Pipeline | 5 | 401 | 4σ | MAD | OFF |

*(Note: Hard positive flux clipping (e.g., `+3.0 MAD`) is strictly forbidden across all experiments due to the required preservation of ellipsoidal tidal variations in Eclipsing Binaries.)*

## Evaluation Metrics
Each experiment will be subjected to the following evaluation criteria on a dedicated test holdout set:
1. **Classifier Performance**: Precision, Recall, F1 Score (evaluated separately for Noise, Planet, and Eclipsing Binary classes).
2. **Global Metrics**: ROC-AUC, PR-AUC.
3. **Calibration & Reliability**: Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams.
4. **Epistemic Uncertainty**: Monte Carlo Dropout uncertainty thresholds.
5. **False Positive Rates**: Performance specifically on known instrumental artifact edge-cases.
6. **XAI Stability**: Verification via Grad-CAM / SHAP / Integrated Gradients to ensure the model focuses on physical transit features rather than edge-artifacts.
7. **Robustness Regimes**: Stratified performance across different Signal-to-Noise Ratios (SNR) and transit depths.
