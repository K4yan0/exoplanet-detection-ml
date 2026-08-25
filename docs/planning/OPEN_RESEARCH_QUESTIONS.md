# Open Scientific Research Questions & Peer Review FAQ

This document serves as a living record of the astrophysical and machine learning research questions generated during the development of this pipeline. It is designed to act as a FAQ for peer reviewers and a roadmap for future scientific exploration.

---

## Part 1: Peer Review FAQ (Answered Questions)

**1. How do we know the neural network isn't just randomly guessing when it sees a noisy, borderline signal?**
* **Answer (Monte Carlo Dropout):** We use Monte Carlo Dropout during inference. If the model is guessing, the probabilistic forward passes will wildly disagree, and our uncertainty metric will spike (e.g., `Confidence: 55% ± 20%`). If the model truly recognizes the signal, the variance remains tightly grouped (e.g., `95% ± 1%`). We do not just output a guess; we output the *epistemic uncertainty* of the network.

**2. If your model says it's 90% confident it found a planet, is that statistically true, or is the neural network just overconfident like most AI?**
* **Answer (Model Calibration):** It is statistically true. We calculated the Expected Calibration Error (ECE) and mapped the confidence bins on a Reliability Diagram. Our model natively achieved an incredibly low Uncalibrated ECE of 2.35%. We then used Temperature Scaling optimization ($T=1.0853$) to further calibrate the softmax logits. When our model claims 90% confidence, it is statistically highly likely to be correct roughly 9 out of 10 times.

**3. Neural Networks are notorious "black boxes". How do we know it actually learned stellar astrophysics, and didn't just memorize random artifacts in your dataset?**
* **Answer (XAI Ablation):** We built an Explainable AI Ablation Engine and tested it on Eclipsing Binaries. When we mathematically zeroed out the primary transit (the only feature a binary shares with a planet), the Grad-CAM convolutional layers highlighted the remaining background light and actually *increased* their confidence that it was a binary. This indicates the AI likely learned to independently recognize "ellipsoidal tidal variations" (the continuous gravity waves between two stars), demonstrating strong physical feature alignment.

**4. Why use AI instead of classical mathematical algorithms like Box-Least Squares (BLS) or Lomb-Scargle?**
* **Answer (The AI Veto Engine):** Traditional algorithms mathematically assume continuous observation. When presented with a multi-month data gap (e.g., missing TESS sectors), BLS will inevitably output a "perfect" artifact signal that fits neatly inside the gap (as seen in our TOI-1231 test). Classical algorithms are incapable of realizing this is an artifact. Our CNN physically inspects the folded shape, recognizes the gap-induced anomaly is not a physical U-shaped transit, and intelligently vetos the classical math.

---

## Part 2: Open Research Questions (Unresolved)

**1. Transit Timing Variations (TTVs) in Multi-Planet Systems**
* **The Problem:** The entire data pipeline currently relies on Phase-Folding, which assumes a planet's orbit is a perfect, unchanging clock. However, in multi-planet systems (like Kepler-9), planets exert gravitational tugs on one another, causing them to transit a few minutes early or late (TTVs). A strict phase-fold will smear these offset transits into undetectable noise.
* **Current Status:** Unresolved.
* **Proposed Research Path:** Can we implement a "Global + Local" Attention mechanism (Transformer architecture)? Instead of only feeding the folded curve to the CNN, we pass the raw, unfolded time-series through an Attention layer to spot individual, slightly-offset transit events.

**2. Detection of Grazing Exoplanets**
* **The Problem:** If a planet only "grazes" the top edge of a star rather than crossing the center, it produces a shallow, V-shaped transit. Since we just trained our Ternary Classifier to associate V-shapes with Eclipsing Binaries, will the model immediately misclassify a grazing planet as a binary?
* **Current Status:** Unresolved.
* **Proposed Research Path:** We need to acquire confirmed data for grazing exoplanets, run them through the current Ternary pipeline, and observe the AI's confidence. If it misclassifies them, we may need to introduce a 4th class ("Grazing Planet") or rely on the absence of secondary eclipses to differentiate them from binaries.

**3. Optimal Background Masking for XAI Ablation**
* **The Problem:** In our Ablation Analysis, masking "Random Background" sets the flux exactly to `0.0` (the normalized median). While this cleans the signal and increases model confidence, it might inadvertently introduce sharp artificial "cliffs" at the edges of the mask, which a CNN might interpret as high-frequency information.
* **Current Status:** Under Investigation.
* **Proposed Research Path:** Instead of masking with absolute zero, should we mask using a local spline interpolation, or inject Gaussian noise with a standard deviation matching the local baseline? We need to test how different masking techniques affect the confidence drop to ensure the XAI validation remains robust and unbiased.
