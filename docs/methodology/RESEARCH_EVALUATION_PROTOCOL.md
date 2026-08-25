# Research Evaluation Protocol

To ensure methodological consistency and prevent the "Exp 3 Methodological Failure" (where experimental changes obscured underlying dataset imbalances), every future experiment in Phase III, IV, and V must strictly report against this centralized evaluation matrix.

> **Prime Directive: No experiment is considered successful merely because one metric (e.g., Accuracy) improves.** 

An experiment must be evaluated holistically across discrimination, calibration, uncertainty, explainability, and data integrity.

## 1. Data Integrity
Before any neural network is evaluated, the experimental state of the dataset must be proven.
*   **Cohort Size:** Total number of raw targets considered.
*   **Dropped Samples:** Explicit count and reason for any targets dropped during preprocessing (e.g., NaN values, length mismatches).
*   **Class Distribution:** Final exact counts of Noise (0), Planet (1), and Eclipsing Binary (2) in the test cohort.
*   **Independent Variable:** A strict definition of what specifically was altered in this experiment.

## 2. Prediction (Discrimination)
Standard classification metrics evaluated on the strictly sequestered test cohort.
*   **Accuracy:** Overall classification correctness.
*   **ROC-AUC (OVR, Weighted):** Multiclass Receiver Operating Characteristic Area Under the Curve to measure class-separation ability.
*   **Per-Class Precision, Recall, and F1-Score:** To ensure rare classes (Planets) are not being sacrificed for dominant classes.
*   **Confusion Matrix:** Exact integer matrix of true vs predicted labels.

## 3. Calibration
Measures whether the model's predicted softmax probabilities reflect true statistical likelihoods.
*   **Expected Calibration Error (ECE):** The weighted average of the absolute difference between accuracy and confidence across 10 probability bins.
*   **Multiclass Brier Score:** The mean squared error between predicted probabilities and one-hot true labels.

## 4. Epistemic Uncertainty
Measures the model's structural confidence in its internal representation.
*   **MC-Dropout Variance:** The mean variance across 50 Monte Carlo Dropout forward passes (`training=True`) for every target in the test set.
*   **Uncertainty vs Correctness:** Comparative variance mapping between Correctly classified targets and Incorrectly classified targets.

## 5. Explainability (XAI)
Audits the physical mechanism the CNN uses to achieve its prediction.
*   **Grad-CAM:** Generating attribution heatmaps for a controlled subset of identically seeded targets.
*   **Attribution Stability:** Quantitative mapping (MSE, Pearson Correlation, Fractional Transit Focus) if comparing two identically-shaped networks (e.g., Native Z vs Native MAD). 
