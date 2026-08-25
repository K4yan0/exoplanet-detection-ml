# V2_EXP4_001: Outlier Removal

## Hypothesis
Removing outliers (stellar flares, instrumental spikes) *before* folding and binning will reduce noise and improve the CNN's ability to learn transit morphology.

## Experimental Controls & Methodological Caveats
This experiment utilized infinite MAST retries to process raw, unbinned TESS light curves. The exact same non-destructive outlier removal algorithm was applied directly to the raw data (interpolating over points > +3σ and < -10σ) before phase-folding. 

**Methodological Note:** Due to the infinite retries weathering MAST outages, the script succeeded in fetching 876 targets (recovering 34 targets that had silently timed out during the V1 Baseline run). Because the acquisition history differs from V1 (842 targets), target-level identity is not perfectly aligned. The test set grew from 175 (V1) to 176 (Exp 4). While the cohort is overwhelmingly identical, this slight variation must be documented.

## Metrics Comparison (Exp 4 vs V1 Baseline)
| Metric | V1 Baseline (Raw Data, No Outlier Clip) | Exp 4 (Raw Data, Outlier Clipped) | Delta |
|--------|-----------------------------------------|-----------------------------------|-------|
| **Accuracy** | 0.7771 | **0.7216** | -0.0555 (Worse) |
| **Macro ROC-AUC** | 0.9089 | **0.8665** | -0.0424 (Worse) |
| **Planet F1** | 0.7899 | **0.7200** | -0.0699 (Worse) |
| **EB F1** | 0.8519 | **0.7900** | -0.0619 (Worse) |
| **Noise F1** | 0.6992 | **0.6700** | -0.0292 (Worse) |
| **Brier Score** | 0.1063 | **0.1367** | +0.0304 (Worse) |
| **ECE** | 0.0509 | **0.0543** | +0.0034 (Worse) |

## Classification Report (Exp 4)
```text
              precision    recall  f1-score   support

   Noise (0)       0.59      0.77      0.67        57
  Planet (1)       0.78      0.67      0.72        60
      EB (2)       0.86      0.73      0.79        59

    accuracy                           0.72       176
   macro avg       0.74      0.72      0.73       176
weighted avg       0.75      0.72      0.73       176
```

## Scientific Conclusion: REJECTED
By running the pipeline as intended (applying clipping to the raw, unbinned light curves), we confirmed the previous preliminary finding: **non-destructive outlier removal damages the astrophysical signal**. 

**Why did this happen?**
1. **Clipping Deep Eclipses**: True transits and eclipses are fundamentally *negative outliers*. By applying a -10σ lower clip to the raw unbinned data, we truncated the deepest Eclipsing Binaries. Truncating an EB's sharp "V-shape" makes it look artificially shallow and flat-bottomed—which exactly mimics a U-shaped Planetary Transit. This provides a plausible mechanistic explanation for the observed EB degradation and should be validated with attribution/morphology analysis.
2. **Stripping Context**: Positive outliers (stellar flares) are real astrophysical features of active stars. Stripping them away before phase-folding removes contextual variance that the CNN was likely using to separate "quiet" planet-hosting stars from "noisy" active stars.

**Verdict**: Outlier removal destroys physical signal. The V1 Baseline strategy (Savitzky-Golay filtering + Z-score normalization, while retaining *all* morphological outliers) is the superior, scientifically sound methodology for this architecture.
