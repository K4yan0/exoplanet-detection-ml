# Experiment 10 Research Design

## 1. The Core Research Question
Phase IV systematically eliminated the standard CNN/LSTM design space, revealing a three-way architectural trade-off between spatial resolution, parameter efficiency, and the preservation of cross-sector positional information.

This tightly constrained the research problem. Our new research question is strictly:
> **Can we preserve high-resolution transit morphology while simultaneously allowing the model to learn relationships between individual observations across the temporal baseline?**

## 2. The Hypothesis and Mechanism
**Candidate Experiment 10:** Cross-sector relational fusion using high-resolution feature maps.

We hypothesize that an architecture explicitly designed to correlate high-resolution features across time—without compressing their spatial dimensions—can break the Catch-22 observed in Experiments 9A-9C. 

To test this, we will introduce **one** architectural change relative to the Exp 9 series control models: we will replace the spatial-compression + LSTM aggregation with a relational mechanism (such as Cross-Attention) that can compare high-resolution feature maps from different sectors directly against one another.

## 3. Experimental Controls
To ensure the experiment isolated the effect of the new architectural mechanism, we will freeze all other variables:
- **Cohort:** The same Exp 8 TESS dataset.
- **Representation:** 5 independent sector phase-folds (2,000 bins each).
- **Encoder:** The same shared 1D CNN used to extract morphology.
- **Evaluation:** Accuracy, Planet Precision/Recall, Noise Recall, ECE, Brier Score, and the 10 Targeted Difficult Planets.

We will **not** introduce any sophisticated "TTV-aware" mechanisms, handcrafted timing features, or physical models. This is a controlled test of high-resolution cross-sector interaction.
