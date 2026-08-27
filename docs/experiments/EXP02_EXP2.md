# V2_EXP2_001

## Configuration
* **Model**: `exoplanet_cnn_v1.keras` (Frozen V1 Reference)
* **Dataset**: `tess_dataset_ternary.npz` (Test Split: 20%, Seed 42, Stratified)
* **Preprocessing Contract**: TERNARY_V1
  * Sectors: 1
  * Outlier Removal: OFF
  * SG Window: 101
  * **Scaling: MAD (Median/Median Absolute Deviation)**
  * Clipping: OFF

## Global Metrics
* **ROC-AUC (OVR)**: 0.8902
* **Expected Calibration Error (ECE)**: 0.0681
* **Brier Score**: 0.1211
* **Mean MC Dropout Epistemic Uncertainty**: 0.0885

## Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Noise)** | 0.6000 | 0.7368 | 0.6614 | 57.0 |
| **1 (Planet)** | 0.7797 | 0.7667 | 0.7731 | 60.0 |
| **2 (EB)** | 0.9130 | 0.7241 | 0.8077 | 58.0 |
| **Accuracy** | - | - | **0.7429** | 175.0 |

## Artifacts
![Confusion Matrix](/docs/assets/v2_exp2_cm.png)

*(Note: Advanced dynamic metrics like SNR and Transit Depth will be mapped incrementally in future experimental scripts. This establishes the numerical floor for Exps 1-5).*
