# Experiment 8: Preserving Local Morphology (The TimeDistributed Architecture)

## 1. The Goal
In **Phase III (Experiments 7 & 7A)**, we discovered that forcing 5 sectors of long-baseline temporal data into a single, rigid phase-fold caused morphological smearing, which actively penalized dynamically complex planetary systems.

**Experiment 8** was designed as a surgical control to test a fundamental hypothesis: *Can we recover those lost planets simply by preserving the local morphological integrity of the individual sectors?*

Rather than jumping directly to a complex TTV-aware timing architecture, we tested a local-only baseline to isolate the cause of the failure.

## 2. Cohort Quality Control
Because Exp 8 required each of the five sector-level representations to independently pass strict Z-score and NaN quality-control checks (whereas Exp 7 could interpolate across concatenated gaps), 19 targets were excluded from the original Exp 7 cohort. The final Exp 8 cohort therefore contained 124 targets (248 samples). This stricter QC requirement differs from the Exp 7 generation procedure and is documented as a limitation when comparing aggregate performance.

## 3. The Architecture
We compared two representations:
* **Exp 7 (Rigid Global Fold):** 5 sectors $\rightarrow$ concatenate $\rightarrow$ one global phase fold $\rightarrow$ CNN
* **Exp 8 (Independent Sector Folds):** 5 sectors $\rightarrow$ fold independently $\rightarrow$ 5 morphologies $\rightarrow$ `TimeDistributed(Shared CNN)` $\rightarrow$ Mean Pooling $\rightarrow$ Classifier

In Exp 8, the identical CNN encoder from Exp 7 was wrapped in a `TimeDistributed` layer. The network processes the 5 sectors independently using shared weights—asking *"What does this sector's morphology look like?"*—before averaging the 5 independent embeddings into a single fused representation.

## 4. Targeted Diagnostic Recovery
The most critical test of Exp 8 was its performance on the specific 10 planets that the rigid 5-sector model confidently rejected as Noise in Exp 7A. 

When the Exp 8 `TimeDistributed` model evaluated those exact 10 targets:

```text
TIC TIC 259377017_Positive: P(Planet) = 0.513 ± 0.123
TIC TIC 36724087_Positive:  P(Planet) = 0.548 ± 0.138
TIC TIC 287328202_Positive: P(Planet) = 0.482 ± 0.115
TIC TIC 345143460_Positive: P(Planet) = 0.544 ± 0.126
TIC TIC 234994474_Positive: P(Planet) = 0.584 ± 0.131
TIC TIC 150030205_Positive: P(Planet) = 0.502 ± 0.110
TIC TIC 262530407_Positive: P(Planet) = 0.653 ± 0.138
TIC TIC 181804752_Positive: P(Planet) = 0.569 ± 0.120
TIC TIC 307809773_Positive: P(Planet) = 0.670 ± 0.124
TIC TIC 254113311_Positive: P(Planet) = 0.593 ± 0.132

Total Recovered: 10/10
```

The model achieved a 10/10 recovery rate for the targeted subset. Crucially, as the MC-Dropout standard deviations (± 0.11 to 0.14) indicate, these were not blindly confident predictions. The architecture exhibited appropriate uncertainty, suggesting: *"I can see enough planetary morphology here to classify this as Planet, but I am not certain."*

## 5. Aggregate Performance and Trade-offs
While the 10/10 targeted recovery is striking, a full evaluation matrix reveals that Exp 8 is **not globally superior** to Exp 7.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 66.0% |
| **Macro ROC-AUC** | 0.9220 |
| **Planet Precision** | 0.48 |
| **Planet Recall** | 1.00 |
| **Noise Recall** | 0.52 |
| **Expected Calibration Error** | 0.1423 |

**Confusion Matrix:**
```text
         Pred Noise  Pred Planet  Pred EB
Noise        13          12         0
Planet        0          11         0
EB            5           0         9
```

By removing the global timing constraints, the model traded away precision. While Planet Recall jumped to 1.00, it misclassified 12 Noise samples as Planets (Precision 0.48). This indicates that local morphology alone is highly sensitive to planets, but insufficient to confidently reject structured noise.

## 6. Scientific Conclusion
The 10/10 recovery of the Exp 7A missed-planet subset **strongly supports the hypothesis** that when a long observational baseline is rigidly collapsed into a single phase-folded representation, timing inconsistencies across the baseline can degrade the resulting transit morphology. Exp 8 provides strong causal evidence that this degradation can be mitigated by preserving sector-level representations before aggregation.

However, the severe drop in Planet Precision proves that independent local morphologies are not enough on their own.

## 7. Bridging to Experiment 9 (Global + Local)
Exp 8 successfully demonstrated that local morphology must be preserved to recover difficult planets, but the increase in False Positives shows the network now lacks the context to confidently reject noise. 

This leads directly into **Experiment 9**:
*Does explicitly injecting global timing information (to establish when those local morphologies occurred) allow the network to regain its Noise rejection precision while maintaining its pristine Planet recall?*
