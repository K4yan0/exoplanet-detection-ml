# Exoplanet Detection Pipeline: Astronomical Research Roadmap

This document outlines the strategic upgrade paths required to elevate the current Exoplanet Detection pipeline from a "Tier 1 Triage Engine" to a publication-grade, fully robust astronomical tool.

While the current pipeline successfully implements rigorous physics-informed preprocessing (BLS phase-folding, MAD normalization) and Explainable AI consensus (Grad-CAM, SHAP, Integrated Gradients), it must address several known astrophysical edge cases to minimize false positives and maximize detection range.

---

## Critique 1: The Eclipsing Binary Problem (False Positives)
**The Vulnerability:** The universe is flooded with Eclipsing Binaries (two stars orbiting each other). When one star passes in front of the other, it creates a massive dip in the light curve. The current model performs Binary Classification (Planet vs. Noise), meaning it will likely flag an Eclipsing Binary as a "Planet" with 99% confidence because it detects a significant dip.

**The Physics:**
* Planets generally create flat-bottomed, **U-shaped** transits.
* Eclipsing Binaries create sharp, **V-shaped** transits.
* Furthermore, binary systems often exhibit a "Secondary Eclipse" (a smaller dip occurring exactly half an orbit later when the smaller star passes behind the primary).

**The Upgrade Path:**
1. Upgrade the model architecture from a Binary Classifier to a **Ternary Classifier (Planet vs. Eclipsing Binary vs. Noise)**.
2. Fetch confirmed Eclipsing Binaries from the MAST/TESS catalog to retrain the CNN, explicitly teaching it to distinguish between U-shapes, V-shapes, and secondary eclipses.

---

## Critique 2: The Multi-Sector Blind Spot
**The Vulnerability:** The current pipeline's Box-Least Squares (BLS) algorithm is hardcoded to stop searching at 20 days. Additionally, the pipeline only downloads the first available TESS observation sector (`search_result[0]`), which corresponds to just 27 days of data. This renders the pipeline completely blind to any planet in a habitable zone (e.g., Earth takes 365 days).

**The Physics:** To find long-period planets, algorithms require long observational baselines. TESS frequently observes the same star across multiple sectors spanning several years.

**The Upgrade Path:**
1. Update the `astronomy.py` module to use `search_result.download_all()`.
2. Implement the `lightkurve.stitch()` method to seamlessly merge years of data, handling the inherent gaps between observational sectors.
3. Make the BLS algorithm **dynamic**: rather than hardcoding a 20-day limit, it should automatically search up to half the length of the stitched baseline (e.g., if 300 days of data are stitched, search up to 150 days).

**Resolution & Empirical Proof (TOI-700 Test):**
This upgrade was successfully implemented by downloading and stitching up to 5 TESS sectors, yielding ~135 days of continuous baseline. 
When tested on **TOI-700 (TIC 150428135)**, a multi-planet system, the dynamic BLS algorithm successfully pierced the noise and extracted an exact **16.0512 Day** orbital period with a depth of **0.83 ppt**. This corresponds precisely to the known sub-Neptune exoplanet **TOI-700 c**. The CNN correctly classified this stitched, long-period signal as a planet with **79.03% confidence**, confirming the pipeline is no longer blind to longer-period orbits.

---

## Critique 3: Stellar Variability & Starspots
**The Vulnerability:** The pipeline currently uses a robust +3.0 MAD (Median Absolute Deviation) ceiling to veto sharp solar flares. However, it does not account for low-frequency stellar variability, such as Starspots. Active stars (like M-dwarfs) rotate and possess massive spots that cause huge, slow, sinusoidal rolling waves in the light curve. Phase-folding this data will overlap the waves and bury the transit signal.

**The Physics:** Starspots cause low-frequency noise, while transits cause high-frequency, sharp signals. These must be decoupled.

**The Upgrade Path:**
1. Prior to phase-folding, implement **Spline Detrending** or a **Savitzky-Golay filter**.
2. This acts as a "High-Pass Filter", flattening out the slow, rolling stellar waves while largely preserving the sharp transit dips.

---

## Critique 4: Multi-Year Data Gaps & The AI VETO Engine
**The Vulnerability:** When stitching TESS data from sectors separated by months or years (e.g., Sector 11 to Sector 27), the massive empty gaps in the time-series create severe mathematical chaos for traditional periodogram algorithms.
**The Physics:** Box-Least Squares (BLS) and Lomb-Scargle algorithms mathematically assume continuous observation. When presented with a multi-month gap, they will inevitably find "perfect" mathematical artifact peaks at periods that neatly fit inside the gap (e.g., a massive 80-day gap will create a false 40-day signal). Traditional algorithms are fundamentally incapable of realizing this is an artifact.

**Resolution & Empirical Proof (The TOI-1231 VETO):**
This vulnerability was empirically tested by querying **TOI-1231 (TIC 447061717)** across 5 non-consecutive sectors. As predicted, the traditional BLS algorithm was tricked by a massive data gap and mistakenly output a **40.8093 Day** period artifact. 
However, the Exoplanet AI pipeline did not blindly trust the BLS math. The 1D CNN inspected the phase-folded 40.8-day signal, recognized the physical morphology was a gap-induced mathematical anomaly rather than a U-shaped transit, and successfully **VETOED** the signal, flagging it as an artifact. This demonstrates the potential utility of using AI as a Triage Engine to identify and overrule artifacts produced by classical algorithms.

---

## Critique 5: Transit Timing Variations (TTVs)
**The Vulnerability:** The entire pipeline relies on Phase-Folding. Phase-folding fundamentally assumes that the planet's orbit is a perfect, unchanging clock. However, in multi-planet systems, planets exert gravitational tugs on one another, causing them to transit a few minutes early or late (TTVs). A strict phase-fold will smear these offset transits into undetectable noise.

**The Physics:** Multi-planet systems (like Kepler-9) exhibit massive TTVs. Strict periodic folding destroys this data.

**The Upgrade Path:**
1. Implement a **Global + Local** neural network architecture.
2. Rather than only feeding the folded curve to the CNN, the pipeline will also pass the raw, unfolded time-series through a 1D ResNet or Self-Attention mechanism (Transformer) to spot individual transit events that do not align perfectly with a periodic clock.
