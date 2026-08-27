# Exp 4A XAI Validation: Noise Morphology Degradation

This analysis specifically investigates the mechanistic cause of the false positive degradation observed in Experiment 4 (Outlier Removal). We isolated a Noise sample (`TIC 230127302`) that the V1 Baseline correctly classified as Noise, but Exp 4 incorrectly classified as an Eclipsing Binary (EB).

![Exp 4A Noise XAI](/docs/images/exp4a_xai_Noise.png)

### Key Observations
* **Morphological Scrambling**: Non-destructive outlier clipping aggressively modifies the high-frequency natural variance in the light curve. 
* **False V-Shapes**: When folded, the interpolated regions of clipped outliers form artificial V-shaped divots.
* **Grad-CAM Alignment**: The Grad-CAM attribution shows that Exp 4's model actively fixates on these artificial divots, mistaking them for secondary eclipses characteristic of Eclipsing Binaries, thereby explaining the model's confusion.
![Morphology Comparison](/docs/images/morphology_comparison_230127302.png)
