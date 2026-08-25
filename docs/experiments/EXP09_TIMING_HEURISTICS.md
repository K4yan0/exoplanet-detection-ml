# Methodological Checkpoint: The Failure of Handcrafted Timing Heuristics

## 1. The Objective
Before designing Experiment 9B (Recurrent Sector Fusion), we tested the hypothesis that we could explicitly calculate the timing consistency of planetary transits (TTV residuals) from the raw phase-folded arrays, and feed those directly into the neural network as metadata. 

We sought a low-dimensional explicit timing feature representing "cross-sector phase drift."

## 2. Independent Verification
We wrote independent mathematical routines to extract the phase drift of Sector 1 relative to Sectors 2-5, testing two classic signal-processing heuristics:
1. **Deepest-Point Phase (`argmin`):** Finding the minimum flux bin in each sector array to compute the phase offset.
2. **Cross-Correlation Lag:** Mathematically correlating the 2000-bin array of Sector 1 against the subsequent sectors to find the phase lag that maximizes alignment.

We tested these heuristics on known True Planets and known False Positives from the Exp 8 dataset.

## 3. The Results
The heuristics performed perfectly on high-SNR (Signal-to-Noise Ratio) targets.
For example, on the high-SNR True Planet `TIC 219852584_Positive`, cross-correlation successfully detected near-zero phase drift across all 5 sectors:
`Phase Drifts: ['0.000', '-0.001', '-0.001', '0.001']`

However, on shallow or noisy targets, the mathematical heuristics failed completely. 
For the shallow True Planet `TIC 230127302_Positive`, cross-correlation produced wildly erratic phase drifts:
`Phase Drifts: ['0.190', '0.018', '0.112', '-0.150']`

Because TESS data contains stellar variability, secondary eclipses, and instrumental noise spikes, standard heuristics latch onto noise instead of the true transit center when the transit is shallow.

## 4. Methodological Conclusion
Explicit, handcrafted timing features (like deepest-point phase or cross-correlation lag) are unreliable on shallow/noisy transits. 

If we manually calculate these offsets and feed them into the neural network as "explicit timing metadata," we will inject severe noise into the training process and degrade the model's performance on difficult planets. 

This methodological failure justifies the progression to **Experiment 9B**. Rather than manually calculating unreliable timing offsets, we must test whether the raw sequence of learned sector representations itself contains enough information for a learned sequence model (like an LSTM) to discover robust temporal relationships.
