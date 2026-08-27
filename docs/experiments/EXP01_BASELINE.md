# V2_BASELINE_001

## Configuration
* **Model**: `exoplanet_cnn_v1.keras` (Frozen V1 Reference)
* **Dataset**: `tess_dataset_ternary.npz` (Test Split: 20%, Seed 42, Stratified)
* **Preprocessing Contract**: TERNARY_V1
  * Sectors: 1
  * Outlier Removal: OFF
  * SG Window: 101
  * Scaling: Z-Score (Mean/Std)
  * Clipping: OFF

## Global Metrics
* **ROC-AUC (OVR)**: 0.9089
* **Expected Calibration Error (ECE)**: 0.0509
* **Brier Score**: 0.1063
* **Mean MC Dropout Epistemic Uncertainty**: 0.0679

## Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Noise)** | 0.6515 | 0.7544 | 0.6992 | 57.0 |
| **1 (Planet)** | 0.7966 | 0.7833 | 0.7899 | 60.0 |
| **2 (EB)** | 0.9200 | 0.7931 | 0.8519 | 58.0 |
| **Accuracy** | - | - | **0.7771** | 175.0 |

## Artifacts
![Confusion Matrix](/docs/assets/v2_baseline_cm.png)

*(Note: Advanced dynamic metrics like SNR and Transit Depth will be mapped incrementally in future experimental scripts. This establishes the numerical floor for Exps 1-5).*
