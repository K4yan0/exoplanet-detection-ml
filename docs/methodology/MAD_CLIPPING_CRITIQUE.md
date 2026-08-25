# Scientific Audit: Positive Flux MAD Clipping in the Ternary Model

## 1. The Core Question
**Should we apply a hard-clip at `+3.0` Median Absolute Deviation (MAD) on the positive normalized flux before feeding the data to the Ternary 1D-CNN?**

This decision fundamentally alters the data geometry presented to the network. In the original Binary Model (Planet vs. Noise), positive flux was largely considered irrelevant (or noise/flares) because a planetary transit is strictly a negative-flux event. However, upgrading to a **Ternary Model** (Planet vs. Eclipsing Binary vs. Noise) introduces complex macroscopic physics that exist *above* the median baseline.

---

## 2. Option A: IMPLEMENTING `MAD = +3.0` Clipping

In this scenario, any data point exceeding 3x the Median Absolute Deviation above the baseline is flattened to exactly `+3.0`.

### The Justification (Why we used to do it):
* **Stellar Flares & Cosmic Rays:** M-dwarf stars frequently emit massive solar flares that can spike the flux by 10x to 50x the normal variance.
* **Normalization Squashing:** If a massive flare survives into the folded 2000-bin array, global normalization (e.g., Min-Max scaling to `[0, 1]`) will heavily compress the negative side. A 1% planetary transit will be mathematically squashed into a microscopic, undetectable numeric change because the flare dominates the mathematical scale.
* **Network Stability:** Neural networks converge faster and more reliably when extreme outliers are suppressed.

### The Consequences (The Danger to the Ternary Model):
* **Destruction of Ellipsoidal Variations:** In close Eclipsing Binaries (EBs), the two stars physically stretch each other into egg shapes due to intense gravity. As they orbit, they present different surface areas to the telescope, creating continuous, rolling sinusoidal waves (Ellipsoidal Tidal Variations). 
* **Amputation of the Peaks:** The peaks of these waves occur at Phase 0.25 and 0.75 (when the stars are side-by-side). These peaks often exceed +3 MAD. By clipping them, we **amputate the rounded crests of the gravity waves**, turning them into artificial flat plateaus.
* **XAI Blindness:** Grad-CAM (Conv1 and Conv3) actively relies on these positive peaks to differentiate an EB from a Planet. If we artificially flatten them, the CNN loses the critical morphological feature it uses to confirm the physics of the binary system.

---

## 3. Option B: NOT IMPLEMENTING Clipping (Preserving all Positive Flux)

In this scenario, we allow the positive flux to remain completely unclipped, no matter how high it reaches.

### The Justification (Why the Ternary Model needs it):
* **Perfect Preservation of Binary Physics:** The ellipsoidal variations, the O'Connell effect (unequal peak heights), and reflection effects all occur in the positive flux domain. Leaving the data unclipped allows the CNN to see the true, unaltered gravitational physics.
* **XAI Validation:** As observed, the AI's attention mechanism natively hunts for these macro-structures. By leaving the positive domain intact, we allow the XAI to build high confidence off the actual astrophysical wave rather than guessing based on the deep primary eclipse alone.

### The Consequences (The Engineering Risks):
* **The Flare Vulnerability:** If a massive, un-flagged stellar flare survives into the folded array, the CNN's convolutional filters will encounter a massive numeric spike. 
* **False Positives for "Noise":** Because the flare drastically alters the global variance of the 2000-bin array, the network might become confused by the extreme numeric range and mistakenly classify a valid Eclipsing Binary or Planet as "Noise/Junk".

---

## 4. Final Verdict & Architectural Recommendation

We **CANNOT** use a blanket `+3.0` MAD hard-clip on the folded array in the Ternary Model. Doing so actively destroys the astrophysics the model is trying to learn. 

### The Solution: A Two-Stage Preprocessing Split
Instead of crippling the final array, the pipeline must handle outliers *temporally* before folding, rather than *statistically* after folding:

1. **Stage 1 (Time-Domain Outlier Removal):** 
   We must strictly use `lightkurve.remove_outliers(sigma_upper=4.0)` on the raw, un-folded time-series. This uses an iterative rolling window to specifically hunt down and delete transient spikes (stellar flares and cosmic rays) by replacing them with `NaN`s, which are then interpolated over. This neutralizes the "Normalization Squashing" threat.
2. **Stage 2 (Unclipped Phase Folding):**
   Once the temporal flares are removed, the remaining positive flux is purely astrophysical (stellar variability and ellipsoidal variations). When we phase-fold this cleaned data into the 2000-bin array, we **DO NOT CLIP IT**. The positive variations are preserved perfectly for the Ternary CNN and its Grad-CAM feature extractors.

**Conclusion:** The blog post was incorrect to state that we clip at `+3.0` MAD on the final array. For the Ternary model to function, we rely on temporal flare-removal, followed by completely unclipped, physics-preserving phase folding.
