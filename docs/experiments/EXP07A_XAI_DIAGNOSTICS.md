# Experiment 7A: Diagnostic Attribution Analysis (The "Smearing" Effect)

## 1. The Goal
In **Experiment 7**, we discovered a fascinating behavioral shift: training a CNN on 5 sectors of temporal data did not strictly improve its planetary recall. Instead, the model became vastly more statistically certain (dropping MC-variance by 70%) but adopted a highly conservative decision boundary—sacrificing recall to achieve a precision of 1.00 on the test cohort.

**Experiment 7A** was designed to investigate *why* this shift occurs. We isolated the 10 planets that the `1-Sector CNN` successfully identified, but which the `5-Sector CNN` aggressively rejected as Noise. We then applied **Grad-CAM (Gradient-weighted Class Activation Mapping)** to trace exactly *where* the models were looking in the light curves.

## 2. Visual Evidence: TIC 36724087

The most illuminating case study is **TIC 36724087**. 

![Grad-CAM for TIC 36724087](C:/Users/Admin/.gemini/antigravity-cli/brain/afbd7ad9-01de-4c5e-9ab9-cc1d18c908a6/exp7a_tic36724087.png)

> [!IMPORTANT]
> **Top Panel (1-Sector Model):** The 1-sector representation contains a deeper, more compact transit. The model correctly predicts "Planet" (P=0.709) and the Grad-CAM heatmap perfectly highlights the central transit.
> 
> **Bottom Panel (5-Sector Model):** The 5-sector representation is wider and shallower. The model predicts "Noise" (P=0.275), and the Grad-CAM heatmap completely abandons the central transit, shifting its attribution to random noise spikes.

This is a very compelling observational chain: Longer baseline -> altered folded morphology -> attribution moves away from the transit -> planetary classification is rejected.

## 3. Quantitative Evidence
To ensure this smearing phenomenon wasn't a visual artifact of a single target, we quantified the morphological shift across all 10 planets rejected by the 5-sector model. 

Comparing the central transit window in the 1-sector vs 5-sector representations, the rejected planetary subset exhibited a **~9.0% median reduction in transit depth** and a slight increase in apparent transit width. This confirms that the 5-sector representation introduces increased morphological smearing for this subset of targets.

## 4. The Representational Hypothesis
Why does the 5-sector representation smear the transit? 

To generate a 5-sector input, we take up to 150 days of sequential observations and "fold" them over each other using a single, fixed orbital period. The morphological smearing is therefore consistent with phase-folding misalignment caused by **transit-timing variations (TTVs)**, period/epoch inaccuracies, or other sources of transit-time inconsistency across the extended baseline.

The network appears to have learned a more restrictive morphological decision boundary that actively penalizes transit signals whose phase-folded morphology becomes smeared over the extended baseline. 

## 5. Bridging to Phase IV
This experiment gives us a clean, fundamental research question:
*Does the conservative behavior arise primarily from the CNN's learned decision boundary, or from information loss introduced by rigid phase folding before the CNN sees the data?*

This exposes a critical weakness in traditional rigid phase folding and perfectly motivates the next architectural question: Can a **Global + Local (TTV-aware) representation** that preserves individual transit timing information recover the planetary candidates that became morphologically degraded under long-baseline rigid folding?
