# Experiment 9A: Order-Preserving Fusion (The "V3" Architecture)

## 1. The Goal
Phase IV established a critical progression:
* **Exp 7:** Rigid global folding causes representation loss, destroying difficult planets.
* **Exp 8 (Local Morphology):** Processing the 5 sectors independently using a `TimeDistributed` CNN and then averaging them (`Mean Pooling`) successfully **recovered 10/10 difficult planets**, but completely lost its ability to reject Noise (Planet Precision plummeted to 0.48).

**Experiment 9A** tests a specific hypothesis regarding the lost precision: *Does Mean Pooling discard the cross-sector phase consistency needed to distinguish true planets from unaligned periodic noise?*

By changing the fusion mechanism from `Mean Pooling` (which treats sectors as an unordered bag) to `Flatten` (which preserves the 5 distinct sector identities in order), we test whether preserving sector order alone is sufficient to recover noise rejection.

## 2. The Architecture
We kept the dataset and preprocessing completely identical to Exp 8.
* **Input:** 5 independent phase-folded sectors.
* **Encoder:** `TimeDistributed(Shared CNN)` $\rightarrow$ produces (5, 128) embeddings.
* **Fusion:** `Flatten` $\rightarrow$ produces (640,) explicit cross-sector representation.
* **Classifier:** Dense layers.

## 3. The Results
The results of this minimal architectural change reveal a fundamental limitation in how neural networks learn temporal consistency.

### Aggregate Performance
| Metric | Exp 8 (Mean Pooling) | Exp 9A (Order-Preserving) |
| :--- | :--- | :--- |
| **Accuracy** | 66.0% | **60.0%** |
| **Planet Recall** | 1.00 | **1.00** |
| **Planet Precision** | 0.48 | **0.48** |
| **Noise Recall** | 0.52 | **0.40** |
| **Expected Calibration Error** | 0.1423 | **0.1743** |

**Confusion Matrix (Exp 9A):**
```text
         Pred Noise  Pred Planet  Pred EB
Noise        10          12         3
Planet        0          11         0
EB            5           0         9
```
The `Flatten` operation failed to restore Planet Precision. It produced the exact same 12 False Positives (Noise classified as Planet) as Exp 8.

Furthermore, analyzing the training logs reveals severe overfitting:
`accuracy: 1.0000 - loss: 0.0185 - val_accuracy: 0.6500 - val_loss: 0.9711`

By flattening the 5 local embeddings into a massive 640-dimensional vector, the Dense classifier simply memorized the training set (100% accuracy) but failed to generalize a robust cross-sector consistency rule (65% validation accuracy).

### Targeted Diagnostic Recovery
```text
TIC TIC 259377017_Positive: RECOVERED | P(Planet) = 0.724 ± 0.114
...
TIC TIC 307809773_Positive: RECOVERED | P(Planet) = 0.812 ± 0.104
TIC TIC 254113311_Positive: RECOVERED | P(Planet) = 0.966 ± 0.043

Total Recovered: 10/10
```
While the network retained the 10/10 recovery of difficult planets (proving that independent folds still protect morphology), the MC-Dropout uncertainty remains high for many targets.

## 4. Scientific Conclusion
Experiment 9A represents a highly successful scientific failure. 

It explicitly tests the hypothesis that simply preserving sector order (via `Flatten`) allows the Dense classifier to learn cross-sector phase alignment. Instead of extracting a robust rule to compare Sector 1 against Sector 5, the expanded parameter space of the `Flatten` layer caused the network to overfit.

This experiment demonstrates that this particular Flatten-based order-preserving fusion did not recover the lost precision under these experimental conditions. It reveals that preserving sector order in the representation is not sufficient, by itself, to make the classifier learn cross-sector temporal consistency.

These results directly motivate advancing to **Experiment 9B**, where we will test an explicitly defined timing representation (such as planetary transit timing offsets/residuals) rather than relying on implicit architectural ordering.
