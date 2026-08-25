# Experiment 10: High-Resolution Self-Attention (Failure)

## 1. The Design
Following the conclusions of Phase IV, we designed **Experiment 10** to answer the question: *Can we preserve high-resolution transit morphology while simultaneously allowing the model to learn relationships between individual observations across the temporal baseline?*

We implemented a Cross-Attention architecture:
1. **Encoder:** Shared CNN pooling only down to 250 bins per sector (preserving high resolution).
2. **Unrolling:** Concatenated the 5 sectors into a single continuous sequence of 1250 bins.
3. **Positional Encoding:** Added 1D positional embeddings so the network knows exactly which sector and phase bin it is looking at.
4. **Relational Fusion:** A `TransformerBlock` (Multi-Head Self-Attention) allowed any high-resolution bin in Sector A to explicitly attend to any bin in Sector B.
5. **Aggregation:** `GlobalAveragePooling1D` followed by the dense classifier head.

## 2. The Results
The experiment failed catastrophically. The model completely lost the ability to detect planets.

### Aggregate Performance
| Metric | Exp 9C (Coarse LSTM) | Exp 10 (Transformer) |
| :--- | :--- | :--- |
| **Test Accuracy** | 84.0% | **54.0%** |
| **Planet Recall** | 0.91 | **0.00** |
| **Planet Precision** | 0.77 | **0.00** |
| **Noise Recall** | 0.84 | **0.68** |

**Confusion Matrix (Exp 10):**
```text
         Pred Noise  Pred Planet  Pred EB
Noise        17           4         4
Planet       11           0         0
EB            4           0        10
```
It classified every single planet in the test set as Noise.

### Targeted Diagnostic Recovery
```text
TIC TIC 259377017_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.442 ± 0.076
TIC TIC 36724087_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.434 ± 0.068
TIC TIC 287328202_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.430 ± 0.076
TIC TIC 345143460_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.422 ± 0.080
TIC TIC 234994474_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.450 ± 0.083
TIC TIC 150030205_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.425 ± 0.073
TIC TIC 262530407_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.441 ± 0.069
TIC TIC 181804752_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.447 ± 0.073
TIC TIC 307809773_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.437 ± 0.080
TIC TIC 254113311_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.408 ± 0.072

Total Recovered: 0/10
```

## 3. Analysis of the Failure
The architecture successfully modeled the cross-sector relationships without exploding parameters (only 178k params). However, it introduced a new, fatal flaw in the aggregation step.

Because we preserved 250 spatial bins per sector, the unrolled sequence was 1,250 bins long. A typical transit only occupies ~5–10 bins per sector (25–50 bins total). The `TransformerBlock` successfully contextualized these bins, but the final `GlobalAveragePooling1D` layer indiscriminately averaged all 1,250 output vectors together. The highly-activated transit features were completely washed out by the 1,200 bins of background noise, destroying the signal before it reached the classifier.

This reveals another critical architectural constraint:

**The results identify unweighted global aggregation as a plausible bottleneck for sparse transit detection in the tested high-resolution self-attention architecture. Because the aggregation mechanism was not independently controlled, Experiment 10 does not establish that pooling alone caused the failure. Experiment 10A will isolate this factor by replacing GlobalAveragePooling with a learned sequence-level aggregation mechanism while keeping the remainder of the architecture fixed.**
