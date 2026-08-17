# Exoplanet Detection Pipeline Architecture

This document maps the entire end-to-end data flow and processing architecture of the exoplanet detection system. It traces the lifecycle of a target star from raw NASA API ingestion to final calibrated probabilistic output and XAI validation.

---

## 1. Data Ingestion & Astrophysics Preprocessing
The pipeline begins by fetching raw photometric time-series data from the NASA MAST archive.

1. **Multi-Sector Stitching:** The system queries the `lightkurve` API and downloads up to 5 non-consecutive TESS (Transiting Exoplanet Survey Satellite) observation sectors. These sectors are seamlessly stitched together to maximize the observational baseline, enabling the detection of long-period planets.
2. **Corrupt Sector Filtering:** The pipeline analyzes the raw flux of every sector before stitching. Any sector with a negative median background flux is automatically dropped to prevent mathematical normalization errors that invert transits into artificial solar flares.
3. **High-Pass Detrending:** The stitched light curve is passed through a Savitzky-Golay filter (window length 401). This acts as a high-pass filter, flattening out low-frequency stellar variability (like rotating starspots) while largely preserving high-frequency planetary transit dips.
4. **MAD Normalization & Clipping:** The flux is normalized using the robust Median Absolute Deviation (MAD). Positive outliers are hard-clipped at `+3.0` to crush cosmic rays and massive stellar flares, while negative outliers (the actual planetary transits) are left completely unclipped to preserve their true physical depth.

## 2. Phase-Folding & Period Finding
To allow the Neural Network to detect extremely faint signals, the pipeline must align and stack multiple transit events on top of each other.

1. **Box-Least Squares (BLS):** The pipeline runs a high-resolution Box-Least Squares algorithm over the detrended light curve. To prevent memory overflow and calculation crashes (the "14 million points" bug), the algorithm is explicitly constrained to evaluate a strict grid of 20,000 period combinations and 5 transit durations.
2. **Astronomical Metrics Extraction:** The BLS algorithm extracts the physical properties of the system:
   * **Orbital Period:** The time it takes the object to orbit the star.
   * **Epoch (T0):** The exact timestamp of the first transit.
   * **Transit Depth:** The amount of light blocked (measured in parts per thousand, ppt).
   * **Transit Duration:** The length of the transit event in hours.
3. **Folding & Binning:** Using the optimal Period and Epoch, the light curve is phase-folded (stacked) and then interpolated into exactly 2000 fixed bins. This converts the variable-length time-series into a standardized 1D tensor suitable for Neural Network ingestion.

## 3. Inference & Uncertainty Estimation
The standardized 1D tensor is fed into a 1D Convolutional Neural Network (CNN).

1. **Forward Pass:** The CNN extracts spatial morphological features (U-shapes vs V-shapes) to output raw logits.
2. **Monte Carlo Dropout (Uncertainty):** Rather than outputting a single static prediction, the pipeline forces the CNN's Dropout layers to remain active during inference. The pipeline runs **50 distinct forward passes** for the same star. The outputs are averaged to calculate the final predicted class, and the variance across the 50 passes is calculated as the **Standard Deviation**. This provides true *epistemic uncertainty* (e.g., `Confidence: 99.0% ± 1.2%`).

## 4. Model Calibration
Neural Networks are inherently overconfident. To ensure mathematical rigor, the pipeline subjects the model to strict statistical calibration.

1. **Expected Calibration Error (ECE):** The pipeline measures the difference between the model's predicted confidence and its actual empirical accuracy across multiple confidence bins. 
2. **Reliability Diagram:** These bins are plotted on a Reliability Diagram, visualizing whether the model is overconfident or underconfident compared to a perfect 1-to-1 diagonal baseline.
3. **Temperature Scaling:** The pipeline uses a SciPy optimizer to find an optimal mathematical constant, Temperature ($T$). During inference, the raw logits of the CNN are divided by $T$ prior to the Softmax activation. This calibration process aligns the output so that a "90% confidence" score is statistically close to a 90% probability of being correct.

## 5. Explainable AI (XAI) Suite
To prevent the CNN from functioning as a "black box", the pipeline calculates attribution maps to visualize exactly which data points influenced the prediction.

1. **Grad-CAM (Gradient-weighted Class Activation Mapping):** Highlights macroscopic spatial structures. The pipeline extracts heatmaps from both `Conv1` (broad structural shapes like out-of-eclipse gravity waves) and `Conv3` (high-frequency edges like the sharp ingress/egress of a transit).
2. **Integrated Gradients (IG):** Integrates the gradients along a path from a flat baseline (a light curve with zero transits) to the actual input, providing highly precise, pixel-level attribution.
3. **SHAP (SHapley Additive exPlanations):** Uses Game Theory to calculate the exact marginal contribution of every single phase bin to the final prediction, utilizing a `GradientExplainer`.

## 6. XAI Validation (Ablation Analysis)
A heatmap is meaningless if it doesn't actually impact the model's decision. The pipeline includes an interactive Ablation Engine to quantitatively validate the XAI explanations.

The engine mathematically zeros out specific segments of the light curve, re-runs the model, and calculates the exact drop (or increase) in confidence. The four ablation scenarios are:
1. **Transit Region (Physics):** Masks the central transit region (phase 0.5). Tests if the model relies on the actual physical transit.
2. **XAI Highlighted Region:** Masks the top 30% "hottest" pixels identified by the currently selected XAI method. Tests whether the regions the XAI claims are important are *actually* important.
3. **Pre-Transit (Baseline):** Masks a known flat background region prior to the transit. Acts as a control group.
4. **Random Background:** Masks a randomly selected chunk of background noise. Acts as a secondary control group.

## 7. Model Evaluation Metrics
During training and validation, the pipeline evaluates the CNN using comprehensive ML metrics rather than relying purely on Accuracy:

1. **Accuracy:** The percentage of correctly classified targets.
2. **Precision:** The ratio of true positive predictions to total predicted positives (minimizing false alarms).
3. **Recall:** The ratio of true positive predictions to actual real positives (minimizing missed planets).
4. **F1 Score:** The harmonic mean of Precision and Recall, providing a balanced metric for imbalanced astronomical datasets.
5. **Confusion Matrix:** A grid detailing the exact breakdown of True Positives, True Negatives, False Positives (e.g., Eclipsing Binaries classified as Planets), and False Negatives.
