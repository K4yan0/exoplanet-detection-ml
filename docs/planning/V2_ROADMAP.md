# 🚀 V2 Roadmap: Multi-Modal Exoplanet Discovery AI

This document serves as the architectural blueprint for "V2" of the Exoplanet Hunter AI. 

While V1 successfully implements a 1D Convolutional Neural Network (CNN) to analyze Transit Light Curves (yielding the *radius* of a planet), it remains susceptible to "imposters" such as Eclipsing Binaries. To achieve true scientific certainty and unlock the ability to determine planetary composition, V2 will cross-validate the Transit Method with **Radial Velocity (Doppler Spectroscopy)** data.

---

## 🌌 The Scientific Objective
By combining two disparate astronomical data streams, the V2 Neural Network will calculate the physical **Density** of the detected object:
1. **Transit Method (Light Curve):** Determines planetary **Radius** (Volume).
2. **Radial Velocity (Doppler Shift):** Determines planetary **Mass**.
3. **Synthesis:** `Density = Mass / Volume`. 

Knowing the density allows the AI to classify the exoplanet type (e.g., Rocky/Earth-like, Ocean World, Gas Giant) and definitively reject massive Eclipsing Binaries.

---

## 🏗️ Architectural Blueprint

### Phase 1: Data Acquisition & Alignment
The hardest part of this project will be building a dataset of stars that have *both* high-quality TESS light curves and high-resolution Radial Velocity spectra.
* **Transit Data:** Continue using NASA MAST / `lightkurve`.
* **Radial Velocity Data:** Query databases such as the NASA Exoplanet Archive, HARPS (High Accuracy Radial velocity Planet Searcher), or HIRES.
* **Challenge:** You must time-align the epoch of the RV data with the epoch of the transit data.

### Phase 2: The Multi-Input Neural Network
You will need to abandon the Keras `Sequential` API and use the `Functional API` to build a multi-branch network.

```python
from tensorflow.keras.layers import Input, Conv1D, Dense, Concatenate
from tensorflow.keras.models import Model

# Branch 1: Transit Light Curve
input_transit = Input(shape=(2000, 1), name="transit_input")
x = Conv1D(32, 5, activation='relu')(input_transit)
# ... flattening and pooling ...
branch_1_out = Dense(64, activation='relu')(x)

# Branch 2: Radial Velocity Spectrum
input_rv = Input(shape=(1000, 1), name="rv_input")
y = Conv1D(32, 5, activation='relu')(input_rv)
# ... flattening and pooling ...
branch_2_out = Dense(64, activation='relu')(y)

# The Synthesis (Merge)
merged = Concatenate()([branch_1_out, branch_2_out])
z = Dense(128, activation='relu')(merged)
final_output = Dense(1, activation='sigmoid', name="planet_confidence")(z)

# Multi-Modal Model
model_v2 = Model(inputs=[input_transit, input_rv], outputs=final_output)
```

### Phase 3: The Physics Layer
Once the AI predicts a planet, a secondary Python physics layer in the web app will extract the physical properties from the raw data:
* **Extract Mass** from the RV amplitude (K).
* **Extract Radius** from the Transit depth (d).
* **Calculate Density** (ρ) and map it to a classification:
  * `< 2 g/cm³`: Gas Giant
  * `2 - 4 g/cm³`: Ocean / Ice World
  * `> 4 g/cm³`: Rocky / Terrestrial

### Phase 4: Multi-Modal XAI (Grad-CAM 2.0)
You will need to upgrade the Grad-CAM algorithm to calculate heatmaps for *both* inputs simultaneously. The Web Dashboard will feature two graphs side-by-side:
1. The Light Curve, highlighting the Transit Dip.
2. The RV Spectrum, highlighting the Doppler Wobble.

---

## 🛠️ Step-by-Step Action Plan

### Step 1: Prove you can get the data (The "Hello World" of V2)
Don't try to download 10,000 stars yet. Pick one highly famous exoplanet that we know has both Transit and Radial Velocity data. A perfect example is **HD 209458** (also known as Osiris).
* Write a small Python script to download its Light Curve (using `lightkurve`).
* Write a script using `astroquery` to connect to the NASA Exoplanet Archive (or ESO/HARPS) and download its Radial Velocity time-series data.
* **Your Goal:** Plot them side-by-side using matplotlib or plotly. If you can successfully display the Transit dip and the RV wobble for the same star on your screen, you have proven the pipeline is possible!

### Step 2: Build the Cross-Matching Script
Once you can do it for one star, you need to automate it.
* Write a script using `astroquery.ipac.nexsci.nasa_exoplanet_archive`.
* Query the database for a list of confirmed exoplanets.
* Filter the list to only keep planets that have the discovery method flagged as both Transit AND Radial Velocity.
* Extract their TIC IDs (for TESS) and their HD/TOI IDs (for HARPS). Save this as a master `dataset_keys.csv`.

### Step 3: The Mass Download & Alignment (The Hard Part)
* Loop through your `dataset_keys.csv`.
* Download the Light Curve and the RV Curve for every single star.
* **The tricky part:** You have to normalize and align them so the neural network can understand them. You need to phase-fold both the Light Curve AND the Radial Velocity curve to the exact same Orbital Period and Epoch.

### How to begin right now:
If you want to start this evening, open a brand new Jupyter Notebook (or a scratch Python file), `pip install astroquery`, and see if you can successfully download the Radial Velocity curve for **HD 209458**. That is your very first milestone. Once you crack that, the rest is just scaling it up!

---

## 🏆 The End Goal
Completing this roadmap elevates the project from a highly polished Web App into a Master's or PhD-level computer science architecture capable of autonomous, rigorous exoplanet characterization.
