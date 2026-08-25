# Protocol: Experiment 7 (Controlled Multi-Sector Robustness)

## 1. Objective
To determine if adding additional temporal observations (5 TESS sectors vs 1 TESS sector) improves the classification performance, calibration, and uncertainty of the pipeline without introducing selection bias. 

Crucially, this experiment separates two distinct questions using a 2x2 matrix:
1. **Representation Robustness:** Does additional temporal information improve performance when fed into a frozen model trained only on 1 sector?
2. **Native Performance:** Does having five sectors available during training allow the network to learn a fundamentally better representation?

## 2. Experimental Matrix
Both models will be independently trained on their respective native datasets, then evaluated cross-condition on a strictly paired test cohort.

| Training | 1-Sector Input | 5-Sector Input |
| :--- | :--- | :--- |
| **1-Sector CNN** | Baseline (A) | Inference Shift (B) |
| **5-Sector CNN** | Inference Shift (A) | Native (B) |

## 3. Strict Paired Cohort Definition
To prevent the methodological failure of Exp 3, the evaluation cohort must consist of a strict paired cohort of targets for which five **usable** sectors can be constructed under the predefined data-quality rules.

If a target does not have 5 usable sectors, it is dropped from the experiment completely. The 1-Sector representation (Dataset A) will use *only* the first chronological sector of these exact same targets.

## 4. The 5-Sector Stitching Contract
Adding multiple sectors changes the underlying data representation. Before generating the dataset, the exact handling of multi-sector data is governed by the following rules:

1. **Gap Handling:** Observational gaps between sectors are ignored during phase-folding. The time arrays are simply concatenated.
2. **Flux Normalization:** Because TESS points slightly differently in each sector, raw flux offsets will occur. The pipeline will apply local median normalization to each sector *before* concatenating them.
3. **Phase-Folding:** The fully stitched 5-sector light curve will be phase-folded jointly using a single uniform period and epoch across all data points.
4. **Binning:** The resulting folded curve will be binned into exactly 2000 bins using the standard median-binning approach to ensure the input shape (2000, 1) remains perfectly identical to the 1-sector representation.
5. **Standardization:** After binning, the entire 2000-length array will undergo standard Z-score normalization.

## 5. Evaluation Protocol
The experiment will be evaluated strictly using the `ResearchEvaluator` class, producing reports for:
*   Prediction (Accuracy, ROC-AUC, F1, Confusion Matrix)
*   Calibration (ECE, Brier)
*   Uncertainty (Global, Correct, Incorrect MC-Dropout variance)
