# Exoplanet Detection: From Kaggle Failure to NASA API Success

![Project Status: Completed - High Accuracy](https://img.shields.io/badge/Status-Completed%20(90%25%20Accuracy)-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project documents a two-part investigation into exoplanet detection using Deep Learning. 
Part 1 documents an exhaustive investigation and eventual failure caused by a corrupted Kaggle dataset.
Part 2 documents the breakthrough success of abandoning the static dataset, building a custom pipeline that fetches live data from the NASA MAST archive, and training a 90% accurate Convolutional Neural Network (CNN) protected by heuristic engineering guardrails and deployed as a Flask Web Application featuring interactive Plotly visualization, Explainable AI (Grad-CAM), and Asynchronous Batch Processing.

---

## Table of Contents
- [Part 1: The Initial Investigation (Kaggle Dataset)](#part-1-the-initial-investigation-kaggle-dataset)
- [Part 2: The Breakthrough (Live NASA API & Deep Learning)](#part-2-the-breakthrough-live-nasa-api--deep-learning)
  - [1. The Pivot](#1-the-pivot)
  - [2. The Advanced Data Pipeline](#2-the-advanced-data-pipeline)
  - [3. The Deep Learning Core (CNN vs Random Forest)](#3-the-deep-learning-core-cnn-vs-random-forest)
  - [4. Engineering Guardrails (The Heuristic Veto)](#4-engineering-guardrails-the-heuristic-veto)
  - [5. The Web Application & XAI Dashboard](#5-the-web-application--xai-dashboard)
  - [6. Batch Discovery Engine](#6-batch-discovery-engine)
- [Installation & Usage](#installation--usage)
- [License](#license)

---

## Part 1: The Initial Investigation (Kaggle Dataset)

The initial hypothesis was that a planetary transit creates an identifiable morphological signature (a U-shaped dip in a light curve) that a machine learning model could learn to recognize. The initial task was binary classification on a heavily imbalanced Kaggle dataset.

### Methodology and Iterative Discoveries

What began as a simple baseline comparison evolved into a multi-step investigation to understand why the models were failing.

**Phase A: Baseline (Random Forest on Simple Features)**
* **Approach:** Feature engineering (mean, std, skew) on the light curve points.
* **Result:** Total failure. The PR-AUC score was equivalent to random guessing.

**Phase B: 1D CNN "End-to-End"**
* **Approach:** Feeding the raw sequential points directly into a CNN.
* **Result:** Total failure. The transit signal was too weak and drowned in background noise. The CNN could not converge.

**Phase C: 1D CNN on Phase-Folding (Lomb-Scargle)**
* **Approach:** Using a Lomb-Scargle periodogram to find the best period, then phase-folding the light curve to amplify the signal.
* **Result:** "Garbage In, Garbage Out". A sanity check proved the period-finding function was producing unusable data.

**Phase D: 1D CNN on Full Periodogram**
* **Approach:** Feeding the entire Lomb-Scargle power spectrum to the CNN to find power peaks.
* **Result:** Partial success but chaotic. The model found 6 out of 7 planets but generated over 360 False Positives, making it unusable.

**Phase E: Statistical Model on BLS**
* **Approach:** Abandoning the CNN. We used NASA's standard Box-Least Squares (BLS) algorithm to extract features for a Random Forest.
* **Result:** Definitive failure. The BLS algorithm proved conclusively that the 37 light curves labeled as "Planets" in the Kaggle dataset were statistically indistinguishable from background noise. The dataset was inherently flawed.

---

## Part 2: The Breakthrough (Live NASA API & Deep Learning)

### 1. The Pivot
After proving the Kaggle dataset was corrupted, we rebuilt the pipeline from scratch. Instead of relying on static CSVs, we connected directly to the NASA MAST API using the `lightkurve` library to download pristine, raw TESS (Transiting Exoplanet Survey Satellite) data.

### 2. The Advanced Data Pipeline
* **Data Ingestion:** Fetching highly rigorous SPOC-processed light curves directly from NASA.
* **Astrophysics Processing:** 
  * Flattening the light curve using a 1001-point rolling median. This wide window removes stellar rotation variations without accidentally falling into transits and creating artificial "horns" (over-filtering artifacts).
  * Using Box-Least Squares (BLS) with high-resolution period grids (100,000 points) to accurately find the orbital period and epoch, preventing aliasing and signal smearing.
  * Phase-folding the light curve to stack multiple transits and amplify the signal, then binning it into exactly 2000 points.
* **Robust Normalization (MAD):** To prevent massive positive stellar flares from inflating the standard deviation and squashing the transit depths, we implemented Robust Scaling using the Median and Median Absolute Deviation (MAD). 
* **One-Sided Clipping:** We clipped positive outliers at `+3.0` to crush cosmic rays and flares, while leaving the deep negative transits completely unclipped to preserve their true physical depth.

### 3. The Deep Learning Core (CNN vs Random Forest)
At the genesis of this project, we evaluated traditional ensemble methods (Random Forest) against Deep Learning (CNNs). While powerful for structured tabular data, Random Forests evaluate features independently, meaning they struggle to intrinsically understand the sequential, time-series "shape" of a light curve without massive feature engineering.

Instead, we trained a 1D Convolutional Neural Network (CNN) with a lightweight architecture (16 -> 32 -> 64 filters with Dropout) to prevent overfitting on the small but pristine dataset. Because CNNs evaluate local spatial coherence, the network inherently learned the morphological signature of a transit (the steep ingress, the flat bottom, and the egress).
* **Performance:** The model shattered previous ceilings, achieving **90.37% Accuracy** with an **AUC of 0.924** and a significantly lowered False Negative rate.

<p align="center">
  <img src="assets/confusion_matrix.png" width="45%" alt="Confusion Matrix"/>
  <img src="assets/roc_curve.png" width="45%" alt="ROC Curve"/>
</p>

### 4. Engineering Guardrails (The Heuristic Veto)
A known artifact of the one-sided clip is that highly variable stars can create an artificial flat ceiling at `+3.0`. The CNN occasionally mistook the natural downward swing from this artificial ceiling for a deep planetary transit (a Clever Hans effect). 

Instead of forcing the probabilistic Neural Network to learn this deterministic mathematical artifact, we implemented a **Heuristic Veto**. This engineering guardrail sits in front of the inference pipeline. If it detects more than 50 data points stuck to the `+3.0` ceiling, it recognizes the stellar variability artifact, bypasses the Neural Network entirely, and automatically rejects the target with a 0% confidence veto.

### 5. The Web Application & XAI Dashboard
The entire inference pipeline is wrapped in a modern, dark-mode Flask Web Application featuring glassmorphism UI design. Rather than relying on static images, the application generates **interactive Plotly.js visualizations**. 

Because the CNN physically learns the shape of the transit rather than just looking at isolated data points, we deployed **Explainable AI (XAI)** via a custom 1D Gradient-weighted Class Activation Mapping (Grad-CAM) algorithm. This acts as an "AI MRI", mapping exactly what the CNN is paying attention to onto the plotted light curve.
* Users can toggle between the **First Convolutional Layer** (which highlights the broad "W" shape of the transit) and the **Final Convolutional Layer** (which rigorously targets the ingress/egress edges).
* The dashboard dynamically extracts critical astrophysics telemetry, including **Orbital Period**, **Transit Depth**, and **Transit Duration**.

### 6. Batch Discovery Engine
Astronomers don't analyze one star at a time. The platform includes a dedicated Bulk Processing engine allowing users to input dozens of TIC IDs simultaneously. A background thread processes the targets asynchronously, updating the UI via long-polling with a live progress bar. Each processed star features an inline, expandable XAI mini-graph for rapid human verification.

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/K4yan0/exoplanet-detection-ml.git
   cd exoplanet-detection-ml
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Web App:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`. Test it out with a known exoplanet like `TIC 261136679` (Pi Mensae) or `TIC 34068865` (WASP-126)!

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.