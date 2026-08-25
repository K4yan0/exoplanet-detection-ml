---
title: "An XAI-Driven, Uncertainty-Aware Ternary Triage Engine for Exoplanet Detection"
description: "Overcoming BLS algorithmic failures and false positives on TESS data using a calibrated Ternary CNN powered by Explainable AI and Monte Carlo Dropout."
pubDate: 2026-08-11
tags: ["Explainable AI", "Astrophysics", "Machine Learning", "TESS", "Model Calibration"]
coverImage: "/portfolio/images/hero.jpg"
---

*🔗 Source Code: [github.com/K4yan0/exoplanet-detection-ml](https://github.com/K4yan0/exoplanet-detection-ml)*

## 1. Abstract
Traditional exoplanet detection pipelines rely heavily on Box-Least Squares (BLS) algorithms to identify periodic planetary transits. However, as observation baselines extend across multiple years, these mathematical models become increasingly vulnerable to multi-sector data gaps and astrophysical false positives—most notably, Eclipsing Binaries (EBs) mimicking planetary transits. 

This paper presents an automated AI Triage Engine utilizing a 1D Convolutional Neural Network (CNN) upgraded to a Ternary architecture (Planet vs. Eclipsing Binary vs. Noise). By integrating Monte Carlo Dropout for epistemic uncertainty and Temperature Scaling for statistical calibration, the model provides rigorous, trustworthy probabilities. Furthermore, we leverage Explainable AI (XAI) consensus and counterfactual ablation to provide strong evidence that the network independently learns deep astrophysical features (such as ellipsoidal tidal variations), allowing it to successfully veto mathematical artifacts that traditionally deceive classical BLS algorithms.

## 2. Data Acquisition & Preprocessing Methodology
The pipeline connects directly to the NASA MAST API via `lightkurve` to process un-detrended TESS (Transiting Exoplanet Survey Satellite) Simple Prerelease Object Catalog (SPOC) data. To expand the detection baseline beyond a single 27-day sector, the pipeline dynamically stitches up to five sectors. 

**Preprocessing constraints include:**
1. **Corrupted Sector Filtering:** Individual sectors exhibiting a negative median background flux are systematically dropped prior to stitching to prevent transit-inversion during normalization.
2. **High-Pass Spline Detrending:** Low-frequency stellar variability (e.g., rotating starspots on M-dwarfs) is decoupled from high-frequency transit signals using a wide-window Savitzky-Golay filter (`window_length=401`). This ensures 4-to-5-hour transits are largely preserved while massive sinusoidal stellar waves are flattened.
3. **Robust Normalization:** The pipeline applies Robust Scaling using Median Absolute Deviation (MAD) rather than Standard Deviation, preventing massive stellar flares from squashing transit depths. Positive anomalies are hard-clipped at `+3.0` MAD, while deep negative transits remain unclipped to preserve physical geometry.

![Raw vs Detrended Light Curve](/assets/raw_vs_detrended.png)

## 3. The Ternary CNN Upgrade (Solving the EB Problem)
A persistent flaw in binary classifiers (Planet vs. Noise) is that an Eclipsing Binary's primary eclipse is mathematically nearly identical to a deep planetary transit, flooding the pipeline with false positives.

To resolve this, the network architecture was expanded into a **Ternary Classifier** utilizing `sparse_categorical_crossentropy` and a `softmax` output layer. The model was trained on an augmented dataset specifically injected with confirmed Eclipsing Binaries from the Villanova Catalog. Rather than just looking for a U-shaped dip, the network learns to detect secondary eclipses and the continuous V-shaped slopes characteristic of binary star systems.

## 4. Epistemic Uncertainty & Model Calibration
In rigorous scientific pipelines, a static 99% probability is insufficient; researchers require a measurable margin of error and statistical reliability. 

### Monte Carlo Dropout
To prevent the model from blindly guessing on borderline signals, we implemented **Monte Carlo Dropout**. By keeping the dropout layers active during inference, the pipeline executes 50 distinct probabilistic forward passes. The variance across these passes generates the **Standard Deviation (Epistemic Uncertainty)**. A high variance (e.g., `± 20%`) indicates the network is guessing, while a tight grouping (e.g., `± 1.2%`) indicates high structural confidence.

### Expected Calibration Error (ECE) & Temperature Scaling
Neural networks are notorious for being overconfident. To ensure mathematical rigor, we calculated the model's **Expected Calibration Error (ECE)**. Our Ternary model natively achieved a highly accurate Uncalibrated ECE of just **2.35%**. 

To further optimize this, we applied **Temperature Scaling** via a SciPy optimizer, yielding an optimal calibration constant of $T=1.0853$. By dividing the CNN's raw logits by $T$ prior to the softmax activation, we align the model so that when the pipeline claims "90% confidence", it is statistically highly likely to be correct roughly 9 out of 10 times.

![Reliability Diagram and Temperature Scaling](/assets/reliability_diagram_ternary.png)

## 5. Explainable AI & Counterfactual Ablation
To evaluate whether the CNN relies on physical astrophysics rather than dataset artifacts, we established an **XAI Consensus** utilizing Grad-CAM, SHAP (Game Theory), and Integrated Gradients. 

![XAI Consensus Heatmaps for Eclipsing Binary](docs/assets/xai_grid_tic185259483.png)

We then performed an **Ablation Analysis**, systematically masking out physical regions of the phase-folded light curve to observe the drop (or increase) in the model's confidence.

**The Eclipsing Binary Paradox:**
During ablation testing on **TIC 185259483** (a confirmed Eclipsing Binary), we discovered a fascinating behavior. When we mathematically zeroed out the primary transit (the deep dip in the center), the AI's confidence that the star was an EB actually *increased*. 

Why? Because the primary eclipse is the only feature an EB shares with a standard exoplanet. By masking the primary eclipse, we deleted the "planet-like" ambiguity, leaving behind purely binary astrophysics: secondary eclipses and ellipsoidal tidal variations (continuous gravity waves). The AI correctly interpreted the remaining out-of-eclipse waveform as binary in nature. This provides strong empirical evidence that the CNN's convolutional layers have successfully learned the macroscopic gravitational physics of dual-star systems, rather than simply hunting for deep holes in the data.

## 6. Empirical Case Studies (The AI Veto Engine)

### Overcoming the Multi-Sector Blind Spot (TOI-700)
Standard single-sector pipelines are blind to orbits longer than ~13 days. By dynamically stitching 5 sectors and explicitly expanding the BLS calculation grid to evaluate 100,000 continuous period combinations, our pipeline successfully processed **TOI-700 (TIC 150428135)**. The engine extracted the exact **16.0512-day** orbital period corresponding to the sub-Neptune exoplanet **TOI-700 c**.

### Vetoing BLS Gap Artifacts (TOI-1231)
When stitching data separated by multi-month observation gaps (e.g., Sector 11 to Sector 27), traditional BLS algorithms suffer mathematical aliasing. 

When evaluating **TOI-1231 (TIC 447061717)** across 5 non-consecutive sectors, the massive observation gap tricked the classical BLS algorithm into identifying an artifact peak at **40.8093 Days**. However, when this 40.8-day phase-folded signal was passed to the AI Triage Engine, the CNN analyzed the physical morphology, recognized it as a gap-induced mathematical anomaly rather than a U-shaped transit, and successfully **VETOED** the signal. This demonstrates the critical utility of using AI as a Triage Engine to cross-verify mathematically derived classical outputs on multi-sector data.

## 7. Future Work
While the Ternary classifier and Calibration engine have vastly reduced false positives, a major astrophysical edge case remains: **Transit Timing Variations (TTVs)**. 

The current pipeline assumes a perfectly periodic orbit. In multi-planet systems with strong gravitational interactions, planets transit slightly early or late. Strict phase-folding smears these offset transits into undetectable noise. Future iterations of this pipeline will investigate routing the *unfolded* continuous time-series through a 1D ResNet or Self-Attention mechanism (Transformer) to spot individual, a-periodic transit events natively.
