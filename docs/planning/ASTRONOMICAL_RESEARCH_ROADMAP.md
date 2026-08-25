# Exoplanet ML Pipeline: Astronomical Research Roadmap

This roadmap structures the scientific evolution of the 1D CNN pipeline for TESS transit classification. It moves from auditing the fundamental preprocessing architecture to testing robustness under observational degradation, and finally to advanced astrophysical structure modeling (TTVs) and cross-instrument generalization.

## PHASE I: PIPELINE AUDIT (Closed)
Establish a strict, reproducible baseline for the existing CNN architecture.
- [x] **V1 Contract:** Exact training/inference preprocessing contract established.
- [x] **Reproducibility:** Model, dataset, environment, and XAI artifacts frozen and preserved.
- [x] **XAI Baseline:** Grad-CAM evaluation framework built.

## PHASE II: PREPROCESSING EXPERIMENTS (Closed)
Test the fragility of the neural representation against algorithmic preprocessing transformations.
- [x] **Exp 1 (Filter Window):** SG101 vs SG401. Window choice substantially changes model behavior; SG401 creates important class/calibration trade-offs.
- [x] **Exp 1A (XAI Morphology):** Attribution analysis investigated the SG401 failure mechanism.
- [x] **Exp 2 (MAD Drop-in):** MAD degrades the frozen Z-trained model.
- [x] **Exp 2A (Sanity Check):** Supported amplitude/representation mismatch.
- [x] **Exp 3 (Multi-sector v1):** Methodological failure. Cohort/data-availability and dropout made the original comparison invalid.
- [x] **Exp 4 (Outlier Removal):** Asymmetric clipping degraded the pipeline.
- [x] **Exp 5 (Reference Pipeline):** Clean independently trained Z-score reference model.
- [x] **Exp 6 (Native MAD Retraining):** Retraining on MAD did not recover Z-score performance.
- [x] **Exp 6A (XAI Statistical Validation):** Attribution behavior differs; matched-target correlation was statistically indistinguishable from the random-pair null.
*Conclusion: Preprocessing transformations fundamentally alter the statistical representation available to the CNN, affecting discrimination, calibration, and attribution.*

## PHASE III: OBSERVATIONAL ROBUSTNESS (Next)
Can the validated pipeline remain reliable when observational conditions become more difficult or incomplete?
- [ ] **Multi-sector (Proper):** 1-sector vs multi-sector for targets that genuinely possess both.
- [ ] **Missing Observations:** Systematically introducing observational gaps.
- [ ] **Noise Injection:** Degrading data quality (stellar noise, instrumental noise).
- [ ] **Temporal Coverage:** Reduced baseline observation windows.

## PHASE IV: ASTROPHYSICAL STRUCTURE
Handling physical mechanisms that violate the 1D strict periodic phase-folding constraint.
- [ ] **Transit Timing Variations (TTVs):** Modeling gravitational interactions.
- [ ] **Hierarchical Architecture:** Global (phase-folded) + Local (individual un-folded transits) network.
- [ ] **Individual Transit Morphology:** Analyzing single dip characteristics.

## PHASE V: EXTERNAL VALIDATION & GENERALIZATION
Testing the network against fundamental distribution shifts and different instrumentation.
- [ ] **Distribution Shift:** Training under idealized conditions, testing on degraded conditions.
- [ ] **Cross-Condition Validation:** Evaluating performance decay curves.
- [ ] **Cross-Mission (Photometric):** Generalizing the TESS-trained model to K2/Kepler light curves (same modality, different noise/cadence/pipeline).
- [ ] **Cross-Mission (Spectroscopic):** Investigating if representations can translate to ESO/HARPS (radial-velocity), a fundamentally different observable.

---
*Note: All future experiments from Phase III onwards must strictly adhere to the formalized `RESEARCH_EVALUATION_PROTOCOL.md` to ensure validity.*
