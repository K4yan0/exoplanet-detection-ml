# Experiment 6: Native Normalization Comparison

## 1. Hypothesis
Training the identical CNN architecture natively with MAD-normalized inputs will produce an internally consistent representation different from the Z-score-trained model. This will potentially alter predictive performance, statistical calibration, MC-Dropout uncertainty, and XAI attribution stability.

We hypothesize that evaluating both models symmetrically on familiar and alien noise distributions (a 2x2 matrix) will definitively prove whether performance degradation is caused by the *algorithmic quality* of the normalization, or simply a *representation shift*.

## 2. Experimental Controls
To guarantee a rigorous comparison against the Exp 5 Reference Model, all variables except the mathematical normalization remain frozen.

| Component | V1 Baseline (Exp 5) | Exp 6 Intervention |
| :--- | :--- | :--- |
| **Dataset cohort** | `tess_dataset_full.npz` | Same |
| **Train/Validation Split** | 80/20, Seed=42, Stratified | Same |
| **Architecture** | 1D CNN (Ternary, MC-Dropout active) | Same |
| **Optimizer & Loss** | Adam (lr=1e-4), Sparse Categorical | Same |
| **Epochs** | 50 (with identical EarlyStopping callbacks) | Same |
| **SG filter** | 101 | Same |
| **Clipping** | None | Same |
| **Independent variable** | Z-Score Normalization | **MAD Normalization** |

## 3. The 2x2 Evaluation Matrix
Once trained, we will generate a comparative 2x2 matrix evaluating both models against both data distributions:

| | Z-Score Input | MAD Input |
| :--- | :--- | :--- |
| **Z-Score Trained Model** | Exp 5 (Baseline) | Exp 2 (Representation Shift) |
| **MAD Trained Model** | Exp 6 (Cross-Test) | Exp 6 (Native MAD) |

## 4. Explainable AI (XAI) Sub-Experiment
For identical targets (Planet, Eclipsing Binary, Noise), we will compute the Grad-CAM attribution heatmaps between the **Native Z-Score CNN** and the **Native MAD CNN** when fed their native input distributions. By mathematically quantifying the overlap (MSE, Pearson Correlation, fractional transit attribution), this will reveal whether the two models exhibit substantially different attribution patterns under the two preprocessing regimes.
