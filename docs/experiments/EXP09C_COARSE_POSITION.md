# Experiment 9C: Position-Preserving Low-Dimensional Fusion

## 1. The Goal
Experiment 9B established a severe architectural Catch-22:
- `Flatten` preserves spatial position but causes a 2,000,000+ parameter explosion, leading to severe overfitting (Exp 9A).
- `GlobalAveragePooling1D` prevents overfitting (8,192 parameters) but destroys spatial position, rendering the LSTM blind to cross-sector phase drift (Exp 9B).

**Experiment 9C** tests a middle-ground architecture to break this Catch-22. We test whether compressing the 2,000-bin sector into a coarse positional representation allows the LSTM to recover precision without overfitting.

## 2. The Architecture
- **Input:** 5 independent phase-folded sectors.
- **Encoder:** Shared CNN utilizing heavy strided `MaxPooling1D` layers. The spatial dimension is reduced from 2,000 $\rightarrow$ 1,000 $\rightarrow$ 500 $\rightarrow$ 100 $\rightarrow$ **20 coarse positional bins**.
- **Compression:** We `Flatten` these 20 bins (20 bins $\times$ 64 channels = 1,280 features) and pass them through a 128-D Dense bottleneck.
- **Aggregation:** LSTM.

This provides the LSTM with a parameter-efficient representation that explicitly retains coarse spatial location (e.g., "The transit occurred in coarse bin 10").

## 3. The Results
The results confirm that the LSTM *can* learn temporal relationships when provided with positional information, but doing so via coarse spatial compression introduces a new trade-off.

### Aggregate Performance
| Metric | Exp 9A (Flatten) | Exp 9B (LSTM) | Exp 9C (Coarse LSTM) |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 60.0% | 66.0% | **84.0%** |
| **Planet Recall** | 1.00 | 0.82 | **0.91** |
| **Planet Precision** | 0.48 | 0.41 | **0.77** |
| **Noise Recall** | 0.40 | 0.48 | **0.84** |
| **Expected Calibration Error**| 0.1743 | 0.0933 | **0.0583** |

**Confusion Matrix (Exp 9C):**
```text
         Pred Noise  Pred Planet  Pred EB
Noise        21           3         1
Planet        1          10         0
EB            3           0        11
```
The architecture successfully slashed the False Positives from 13 (in Exp 9B) down to just **3**, causing Planet Precision to surge to 0.77. 
Furthermore, the Expected Calibration Error dropped to **0.0583**, the lowest we have achieved, indicating that the model is no longer wildly guessing.

### Targeted Diagnostic Recovery
```text
TIC TIC 259377017_Positive: RECOVERED | P(Planet) = 0.738 ± 0.100
TIC TIC 36724087_Positive:  RECOVERED | P(Planet) = 0.867 ± 0.103
TIC TIC 287328202_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.440 ± 0.109
TIC TIC 345143460_Positive: RECOVERED | P(Planet) = 0.817 ± 0.123
TIC TIC 234994474_Positive: RECOVERED | P(Planet) = 0.890 ± 0.122
TIC TIC 150030205_Positive: RECOVERED | P(Planet) = 0.564 ± 0.135
TIC TIC 262530407_Positive: RECOVERED | P(Planet) = 0.881 ± 0.102
TIC TIC 181804752_Positive: RECOVERED | P(Planet) = 0.875 ± 0.130
TIC TIC 307809773_Positive: RECOVERED | P(Planet) = 0.852 ± 0.113
TIC TIC 254113311_Positive: RECOVERED | P(Planet) = 0.477 ± 0.186

Total Recovered: 9/10
```
The coarse pooling introduced a new penalty: **Resolution Loss**. 
By compressing 2,000 bins into 20 coarse bins, each spatial pixel now represents 100 original data points. Shallow morphological features are blurred out. As a result, the network missed `TIC 287328202_Positive`, predicting it as Noise.

Furthermore, analyzing the MC-Dropout probabilities reveals an important distinction between *classification recovery* and *confident recovery*. For example, `TIC 254113311_Positive` is technically classified as a Planet (since its probability is the maximum among the 3 classes), but its mean probability is only $0.477$ with a massive uncertainty of $\pm 0.186$. Similarly, `TIC 150030205_Positive` ($0.564 \pm 0.135$) is substantially less decisive than `TIC 234994474_Positive` ($0.890 \pm 0.122$). This shows that the resolution loss severely impacted the model's confidence on borderline cases.

## 4. Scientific Conclusion
Experiment 9C demonstrates that a parameter-efficient sequence model can exploit cross-sector relationships when coarse positional information is retained. This substantially improves noise rejection and calibration relative to the high-resolution pooled and flattened alternatives tested in Experiments 9A and 9B. However, coarse spatial compression introduces a new limitation: shallow transit morphology can be lost through resolution reduction, as demonstrated by the missed TIC 287328202 case.

Across Experiments 9A–9C, the results reveal a three-way architectural trade-off between spatial resolution, parameter efficiency, and preservation of cross-sector positional information. High-resolution representations preserve morphology but can overfit; global pooling controls complexity but removes positional information; coarse positional representations recover useful cross-sector structure but can blur weak transit morphology.

The next experimental question is therefore whether high-resolution local morphology and cross-sector temporal relationships can be modeled simultaneously without relying on either extreme flattening or aggressive spatial compression.
