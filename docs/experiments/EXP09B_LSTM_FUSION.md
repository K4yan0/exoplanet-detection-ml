# Experiment 9B: Recurrent Sector Fusion (LSTM)

## 1. The Goal
Experiment 9A revealed that flattening the sector embeddings caused severe overfitting (`accuracy: 1.000` vs `val_accuracy: 0.650`). Furthermore, our independent verification of handcrafted timing heuristics (Phase Offset, Cross-Correlation) proved that TESS data is too noisy for simple mathematical TTV extraction without full MCMC fitting.

This set up a highly constrained hypothesis for **Experiment 9B**:
> *"Can a parameter-efficient learned sequence model (LSTM) extract useful cross-sector temporal relationships from the preserved local morphologies without overfitting?"*

We replaced the `Flatten` aggregation layer with an `LSTM` sequence model. We also utilized `GlobalAveragePooling1D` inside the shared local CNN to maintain parameter efficiency and prevent the 2-million parameter explosion seen in Exp 9A.

## 2. The Results
The LSTM architecture successfully solved the overfitting crisis of Exp 9A, but completely failed to recover the lost Precision.

### Aggregate Performance
| Metric | Exp 9A (Flatten) | Exp 9B (LSTM) |
| :--- | :--- | :--- |
| **Test Accuracy** | 60.0% | **66.0%** |
| **Planet Recall** | 1.00 | **0.82** |
| **Planet Precision** | 0.48 | **0.41** |
| **Noise Recall** | 0.40 | **0.48** |
| **Expected Calibration Error** | 0.1743 | **0.0933** |

**Training Dynamics:**
Exp 9B successfully regularized the network. The final epoch showed:
`accuracy: 0.6013 - loss: 0.6215 - val_accuracy: 0.7250 - val_loss: 0.5435`
The massive overfitting of Exp 9A was entirely cured, as reflected by the greatly improved ECE (0.0933).

**Confusion Matrix (Exp 9B):**
```text
         Pred Noise  Pred Planet  Pred EB
Noise        12          13         0
Planet        1           9         1
EB            2           0        12
```
Despite curing the overfitting, the LSTM still produced **13 False Positives** (Noise classified as Planet). It failed to restore the noise-rejection capabilities of the Exp 7 baseline.

### Targeted Diagnostic Recovery
```text
TIC TIC 259377017_Positive: RECOVERED | P(Planet) = 0.533 ± 0.056
TIC TIC 36724087_Positive:  RECOVERED | P(Planet) = 0.488 ± 0.043
...
Total Recovered: 10/10
```
While all 10 difficult planets were classified as Planet, the network's confidence was entirely washed out ($P \approx 0.50$ for all targets). The model is essentially guessing.

## 3. Scientific Conclusion: The Architectural Catch-22
Experiment 9B reveals a fundamental bottleneck in applying standard CNNs to phase-folded transits.

To allow an LSTM to track cross-sector phase drift, the local CNN embedding must retain spatial phase information. However, compressing a 2000-bin sector into a low-dimensional embedding presents a Catch-22:
1. **Preserve Spatial Position (Exp 9A):** Using `Flatten()` retains phase position, but generates over 2,000,000 parameters per sector. This causes the classifier to overfit wildly, destroying generalization.
2. **Prevent Overfitting (Exp 9B):** Using `GlobalAveragePooling1D()` reduces the parameters to a highly efficient 8,192. However, this operation is translationally invariant. It averages the feature map across all bins, completely destroying the spatial location of the transit. The LSTM receives an embedding that says "A transit exists," but has no idea *where* it is, rendering it completely blind to cross-sector phase drift.

**Conclusion:**
We have exposed a significant structural limitation of this standard implicit spatial architecture. Under the tested CNN + pooling + LSTM design, the model could not recover useful cross-sector phase consistency without either losing spatial information or introducing severe overfitting.

This sets up a very clear final test for the Phase IV sequence. Before jumping to an advanced mechanism like Cross-Attention or explicitly calculated MCMC timing residuals, we must test a middle-ground architecture that compresses the 2000 bins into a coarse positional representation, directly attacking this architectural Catch-22.
