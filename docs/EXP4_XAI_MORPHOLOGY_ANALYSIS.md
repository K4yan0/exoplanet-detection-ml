# Experiment 4: The Impact of Asymmetric Outlier Removal on Representation Learning

Through a strict validation split and Grad-CAM interpretability diagnostics, Exp 4 demonstrates that the tested asymmetric outlier-removal strategy substantially degrades classification performance compared to the V1 baseline (Accuracy dropping from 79% to 70%).

## Methodology & Experimental Design

A crucial distinction in this experiment is that the XAI diagnostics were generated using **two distinct, independently trained models** (`aligned_v1.keras` and `aligned_exp4.keras`), trained on the exact same `random_state=42` split. 

Therefore, this XAI comparison is answering: *How did the learned representations differ after changing the training data?* It is not answering how the exact same frozen model reacts to a preprocessing transformation.

| Component | V1 | Exp 4 |
| :--- | :--- | :--- |
| **Outlier treatment** | None | +3σ / -10σ |
| **Outlier replacement** | N/A | NaN → linear interpolation |
| **CNN weights** | Independently trained | Independently trained |
| **XAI comparison** | V1 representation | Exp 4 representation |
| **Transit points clipped in representative cases** | 0 | 0 |
| **Positive excursions removed** | No | Yes |
| **Conclusion** | Baseline Confirmed | Exp 4 Rejected |

## Representative Case Studies

We examined three specific casualties to understand how the models' learned representations diverged.

### 1. The Planet Case (TIC 375654303)
* **Observation:** The V1 model correctly fixates heavily on the transit dip, whereas the Exp 4 model's attention is entirely chaotic and dispersed, misclassifying it as Noise. 
* **Numerical Verification:** Numerical inspection confirmed that **zero** points were clipped from the bottom of the transit (the -10σ threshold was much deeper than the transit minima). The algorithm only clipped and interpolated 31 upper positive excursions. 

### 2. The Eclipsing Binary Case (TIC 0349911034)
* **Observation:** The V1 model correctly focuses on the deep primary eclipse and the secondary eclipses. The Exp 4 model misclassifies the EB as a Planet, with its Grad-CAM attention shifting differently across the phase.
* **Numerical Verification:** Again, **zero** points were clipped from the bottom of the deep primary eclipse. The algorithm clipped and interpolated 13 upper positive excursions.

### 3. The Noise Case (TIC 389900760)
* **Observation:** The data contains a standard observational downlink gap. In both V1 and Exp 4, `generate_sample()` interpolated across this empty phase space, creating a V-shaped artifact. However, the V1 model correctly ignored this artifact (predicting Noise), while the Exp 4 model fixated on it brilliantly and misclassified the star as a Planet.

## Conclusion

The observed degradation cannot be attributed to direct truncation of the transit morphology, as the transit minima were completely untouched in the representative cases examined. 

Instead, the results are consistent with a **representation shift caused by asymmetric modification of the stochastic component of the light curves.** The procedure removed positive excursions above +3σ while retaining negative excursions below -10σ, producing a measurable change in the statistical distribution and skewness of the light curves. Deprived of the natural, symmetric high-frequency noise floor, the independently trained Exp 4 model assigned substantially different importance to existing downward structures, such as interpolation regions associated with observational gaps.

We therefore reject the tested asymmetric outlier-removal configuration for the current architecture. The precise mechanism by which the altered noise distribution affects learned representations remains a hypothesis requiring further controlled experimentation.
