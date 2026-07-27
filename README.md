# 🪐 Exoplanet Hunter AI

An advanced, end-to-end Machine Learning platform for discovering exoplanets using NASA's TESS (Transiting Exoplanet Survey Satellite) data. 

This project was built to explore the performance of different machine learning models—specifically comparing **Random Forest (RF)** and **Convolutional Neural Networks (CNN)**—in detecting the faint, periodic dips in starlight caused by orbiting exoplanets. The final deployment leverages a custom 1D CNN equipped with Explainable AI (Grad-CAM) to not only predict the presence of a planet, but to visually prove *why*.

---

## ✨ Standout Features

### 1. Interactive Scientific Dashboarding
Replaced static plots with dynamic, interactive **Plotly.js** visualizations. Users can zoom, pan, and hover over individual phase-binned flux data points (Z-scores) in real-time to manually inspect the light curve anomalies.
* Automatically extracts critical astrophysics telemetry: **Orbital Period**, **Transit Depth (ppt)**, and **Transit Duration (Hrs)**.

### 2. Explainable AI (XAI) with Grad-CAM toggling
Machine Learning in astrophysics cannot be a "black box". We implemented a custom 1D Gradient-weighted Class Activation Mapping (Grad-CAM) algorithm that traces the CNN's decision-making process.
* **Dual-Layer Inspection**: Toggle the UI to view the AI's attention at the **First Layer** (broad shape detection, identifying the macro "W" or "V" transit dip) or the **Final Layer** (rigorous edge detection). 

### 3. Scalable Batch Discovery Mode
Astronomers don't analyze one star at a time. The platform includes a dedicated Bulk Processing engine:
* Input dozens of TIC (TESS Input Catalog) IDs simultaneously.
* Background threading downloads and processes NASA MAST data asynchronously.
* An interactive progress bar tracks the pipeline, and results feature inline, expandable "Mini-Graphs" for rapid human verification.

---

## 🧠 Model Evaluation: CNN vs. Random Forest

At the genesis of this project, we evaluated traditional ensemble methods (Random Forest) against Deep Learning (CNNs). 
* **Random Forest (RF):** While powerful for structured tabular data, RFs evaluate features independently. They struggle to inherently understand the sequential, time-series nature of a light curve without massive feature engineering.
* **1D Convolutional Neural Network (CNN):** CNNs evaluate local spatial coherence. By using 1D convolutions, the network inherently learns the morphological signature of a transit (the steep ingress, the flat bottom, and the egress). 

Because the CNN physically learns the shape of the transit rather than just looking at isolated data points, we were able to deploy the **Grad-CAM** heatmaps—something impossible to visualize intuitively with a Random Forest.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* TensorFlow / Keras 3
* Lightkurve
* Flask

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask Server:
   ```bash
   python app.py
   ```
4. Open your browser to `http://127.0.0.1:5000`

### Example TIC IDs to try:
* `TIC 261136679` (Confirmed Planet)
* `TIC 34068865` (Noise / Eclipsing Binary)

---
*Built with Flask, TensorFlow/Keras, Lightkurve, and Plotly.*