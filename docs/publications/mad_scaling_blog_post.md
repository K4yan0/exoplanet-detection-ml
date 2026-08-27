---
title: "When Robust Scaling Becomes a Representation Shift: An Empirical Study of Z-Score and MAD Normalization for TESS Transit Classification"
description: "Why mathematically superior preprocessing techniques can fatally degrade an astrophysics pipeline if they alter the statistical distribution learned by the neural network."
pubDate: 2026-08-16
tags: ["Astrophysics", "Machine Learning", "TESS", "Data Normalization", "Representation Shift"]
coverImage: "/portfolio/images/TESS.jpg"
---

*?? Source Code: [github.com/K4yan0/exoplanet-detection-ml](https://github.com/K4yan0/exoplanet-detection-ml)*

## 1. Abstract
In machine learning, robust scaling is universally prescribed as the antidote to datasets plagued by extreme outliers. In the realm of exoplanet transit detection, where massive stellar flares frequently distort the pristine light curves of distant stars, replacing standard **Z-Score normalization** with the robust **Median Absolute Deviation (MAD)** seems like an obvious upgrade. 

However, in this empirical case study, we demonstrate that a preprocessing transformation is not merely an input formatting choice. By conducting a controlled 2x2 cross-normalization experiment with independently trained Convolutional Neural Networks (CNNs), we show that changing the mathematical normalization alters the statistical representation available to the CNN. This shift measurably impacts the model's discrimination, probabilistic calibration, epistemic uncertainty, and XAI attribution patterns.

## 2. Why Normalization Matters for Transit Classification
The Transiting Exoplanet Survey Satellite (TESS) provides continuous light curves covering vast swaths of the sky. To prepare these raw photon counts for deep learning, they must be normalized.

### The Z-Score Baseline
Our reference architecture utilized standard Z-score standardization:
$$ z = \frac{x - \mu}{\sigma} $$
Where $\mu$ is the mean flux and $\sigma$ is the standard deviation. Because standard deviation is highly sensitive to outliers, a single massive stellar flare artificially inflates $\sigma$, structurally compressing the microscopic dip of a planetary transit into statistical noise.

### The MAD Hypothesis
To insulate the pipeline against stellar flares, we hypothesized that **Robust Scaling** would preserve the signal. We implemented scaling based on the Median Absolute Deviation (MAD):
$$ \text{MAD} = \text{median}(|x_i - \text{median}(X)|) $$
$$ x_{\text{robust}} = \frac{x - \text{median}(X)}{\text{MAD} \times 1.4826} $$
Because MAD ignores the numerical magnitude of outliers, the transit depth would remain mathematically un-squashed.

## 3. The Drop-In Replacement Hypothesis (Exp 2)
In applied machine learning, a common assumption is that preprocessing is modular. We first tested whether MAD scaling could act as a drop-in upgrade for an existing, frozen Z-score-trained CNN.

The experimental result contradicted the assumption. Feeding MAD-scaled data systematically degraded the performance of the frozen model.

| Metric | Z-Score Input | MAD Input | Impact |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.7771 | 0.7429 | -0.0342 |
| **ROC-AUC** | 0.9089 | 0.8902 | -0.0187 |

This proved that models learn highly rigid amplitude distributions. Post-training modifications induce a representation mismatch. But this raised a far more important question: **Is MAD intrinsically a worse representation for this architecture, or did it just fail because the CNN was trained on Z-Score?**

## 4. The 2x2 Native Normalization Matrix (Exp 6)
To answer this, we designed a perfectly symmetrical control experiment. We trained a brand new, identically parameterized CNN natively on MAD-scaled data, isolating the scaling method as the single independent variable. 

We then evaluated both models against both data distributions to create a 2x2 evaluation matrix:

| Training | Z-Score Input | MAD Input |
| :--- | :--- | :--- |
| **Native Z-Score CNN** | **0.7771** | 0.7257 |
| **Native MAD CNN** | 0.6971 | 0.7086 |

There are two very clear scientific findings here:
1. **For this experimental setup, Native Z-score performs better than Native MAD.** Training the CNN natively on MAD (70.86%) does not recover the baseline performance of the Z-score pipeline (77.71%). Z-score performed better than MAD under the controlled conditions of this experiment.
2. **There is a measurable cross-normalization penalty.** The Z-score model loses 5.14 percentage points when given alien MAD data, and the MAD model loses 1.15 percentage points when given alien Z-score data.

## 5. Calibration and Uncertainty Deficits
Beyond accuracy, the MAD model showed lower discrimination, worse calibration, and higher MC-Dropout variance. We deployed Monte Carlo Dropout (MC-Dropout) to quantify epistemic uncertainty during inference:

| Calibration Metric | Native Z-Score | Native MAD |
| :--- | :--- | :--- |
| **ROC-AUC (OVR)** | 0.9089 | 0.8805 |
| **Brier Score (Lower=Better)** | 0.3109 | 0.3511 |
| **Expected Calibration Error (ECE)** | 0.0262 | 0.0444 |
| **MC-Dropout Variance** | 0.00624 | 0.01360 |

The MAD model is not merely losing accuracy; it is associated with lower discrimination, worse probabilistic calibration, and approximately **2.18x greater MC-Dropout variance**.

![Predictive Uncertainty: Z-Score vs MAD](/docs/assets/mad_uncertainty_plot.png)
*Caption: Mean predictive variance via MC-Dropout increases notably under MAD scaling.*

## 6. XAI Representation Analysis (Statistical Validation)
To investigate *how* these native pipelines differed, we ran a quantitative Explainable AI (XAI) analysis using Grad-CAM on all 60 planetary transits in the test cohort. We compared the attribution heatmaps of the Native Z-Score CNN and the Native MAD CNN on the exact same physical targets.

To ensure the statistical validity of this comparison, we normalized the maps and conducted a 10,000-shuffle permutation test to calculate the difference between the matched-target correlation and a randomly paired null distribution.

**Quantitative Results:**
- **Matched-Target Pearson Correlation:** 0.0007
- **Null Distribution Mean (10,000 shuffles):** 0.0019 (95% CI: [-0.0118, 0.0153])
- **Permutation p-value:** 0.9183

Under this Grad-CAM comparison, the attribution correlation for matched targets was not detectably greater than that expected under random pairing. There is no evidence that the two models' attribution maps become more similar merely because they are looking at the exact same physical planet.

![Grad-CAM Comparison: Z-Score vs MAD](/docs/assets/exp6_native_xai_comparison.png)

## 7. Conclusion
We initially hypothesized that robust scaling would smoothly improve the pipeline by insulating it from outliers. By running the identical architecture natively on both scaling techniques, we rejected that hypothesis. 

These results provide strong evidence that the CNN's learned representation is measurably sensitive to the statistical normalization of its input. For this dataset, preprocessing pipeline, CNN architecture, and evaluation protocol, native Z-score normalization produced superior classification, calibration, and uncertainty characteristics compared with native MAD normalization, while the two preprocessing regimes produced markedly different Grad-CAM attribution patterns.

---
*Next up: Breaking the 1D phase-folding constraint to hunt for Transit Timing Variations (TTVs) using a hierarchical Global/Local CNN architecture.*
![MAD vs Grad-CAM](/docs/assets/mad_gradcam_comparison.png)
