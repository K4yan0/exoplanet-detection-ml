# V2_EXP3_001: Multi-Sector Stacking vs Single-Sector Baseline
**STATUS: INVALIDATED** (Methodological Failure / Cohort Attrition)

## 1. Hypothesis
Multi-sector observations improve the robustness of the V1 classifier by increasing the signal-to-noise ratio (SNR) of transits and providing more transit events for the CNN to learn from. 

## 2. Experimental Attempt
The experiment evaluated the fixed V1 CNN architecture and training protocol (SG101 + Z-score + no outlier removal) on a dataset constructed by stitching and folding all available sectors for the cohort, compared against the V1 baseline which used only 1 sector.

## 3. Failure Modes
A rigorous cohort reconciliation pass revealed two decisive failures in the experimental setup:

### Failure A: Insufficient Multi-Sector Coverage (The Independent Variable Did Not Change)
Out of 591 physical targets evaluated, the sector distribution retrieved from MAST was:
* **1 Sector: 450 targets (76%)**
* 2 Sectors: 15 targets
* 3 Sectors: 20 targets
* 4 Sectors: 18 targets
* 5 Sectors: 88 targets (15%)

Because TESS only observed the majority of these star fields for a single 27-day window, 76% of the test cohort was evaluated on the exact same 1-sector data as the V1 baseline. The experiment was not a clean `1-sector vs 5-sector` comparison, but rather `1-sector vs a heterogeneous mixture of 1-5 sectors`.

### Failure B: NaN-Induced Sample Attrition (Selection Bias)
The dataset construction script identified exactly 44 dropped samples from the V1 cohort:
* Noise: 24 dropped
* Planet: 15 dropped
* EB: 5 dropped

These samples failed with `generate_sample failed (likely NaN or length)`. Stitching heterogeneous multi-sector observations with variable baseline fluxes and multi-month temporal gaps resulted in un-interpolatable NaN holes in the folded arrays. 
By silently dropping these arrays, the pipeline artificially filtered out the most difficult, gap-ridden Planet and Noise samples, leaving only the "cleanest" arrays behind. 

## 4. Observed Artifact
The experiment yielded an apparent massive performance jump:
* ROC-AUC: 0.9089 -> 0.9696 (+0.0607)
* Planet F1: 0.7899 -> 0.9204 (+0.1305)
* Noise F1: 0.6992 -> 0.8496 (+0.1504)

Because the largest apparent gains were precisely in the classes with the highest attrition (Planet and Noise), this improvement is a statistical artifact manufactured by non-random sample exclusion. 

## 5. Scientific Conclusion
**No conclusion about the effect of multi-sector observations can be drawn from this experiment.** The +13-point F1 improvement is discarded as evidence of model capability.

## 6. Future Work
The engineering failures in this experiment expose a significant open research question:
*How should heterogeneous, discontinuous multi-sector TESS observations be transformed into a scientifically valid representation for machine-learning exoplanet detection?*

Future multi-sector research must address:
* Sector-to-sector flux normalization
* Temporal gaps and missing bins
* Interpolation and uncertainty propagation
* Variable sector counts and weighting
* Whether mathematical stitching alters transit morphology
* XAI behavior around artificial gap boundaries

## Artifacts
![Confusion Matrix](/docs/assets/v2_exp3_cm.png)
