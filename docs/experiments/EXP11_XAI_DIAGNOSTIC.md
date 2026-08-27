# Experiment 11 Diagnostic (XAI): 5x5 Cross-Sector Attention Matrix

## Hypothesis
If the Phase-Aligned Cross-Sector Attention mechanism (Exp 11) successfully mitigates representation collapse, we expect the Transformer to strongly attend across multiple sectors when a genuine periodic transit (like a Planet or Eclipsing Binary) is present. Conversely, for random noise or artifacts, the attention should remain localized or disperse without strong cross-sector consensus.

Most importantly, by comparing the exact same difficult target processed by a **Successful initialization (Seed 300)** versus a **Collapsed initialization (Seed 100)**, we can mechanistically verify whether the collapse corresponds to a failure to utilize the cross-sector attention prior.

## Protocol
- We extract the `(batch*250, 4, 6, 6)` attention weight tensor from the `cross_sector_transformer` MultiHeadAttention layer.
- We average the weights across the 4 attention heads and all 250 phase bins to obtain a macroscopic view of the `6x6` cross-sector attention map.
- The 6 tokens correspond to: `[CLS]`, `[Sector 1]`, `[Sector 2]`, `[Sector 3]`, `[Sector 4]`, `[Sector 5]`.
- We evaluate this on 4 key cases:
  1. A Clean Planet
  2. A Difficult Target Planet (TIC 259377017)
  3. An Eclipsing Binary (EB)
  4. A Noise False Positive

## Results: Seed 300 (Successful Initialization)

Seed 300 successfully recovered 10/10 difficult planets. Its attention matrix reveals a structured, functional macroscopic attention pattern:

**Difficult Planet (TIC 259377017)**
![Seed 300 Difficult Target](/docs/images/exp11_att_300_Difficult_Target_TIC_259377017.png)

**Clean Planet**
![Seed 300 Clean Planet](/docs/images/exp11_att_300_Clean_Planet.png)

**Eclipsing Binary**
![Seed 300 Eclipsing Binary](/docs/images/exp11_att_300_Eclipsing_Binary.png)

**Noise False Positive**
![Seed 300 Noise False Positive](/docs/images/exp11_att_300_Noise_False_Positive.png)


## Results: Seed 100 (Collapsed Initialization)

Seed 100 failed entirely (0/10 targeted recovery). Its attention matrix reveals a substantially more uniform attention distribution when averaged:

**Difficult Planet (TIC 259377017)**
![Seed 100 Difficult Target](/docs/images/exp11_att_100_Difficult_Target_TIC_259377017.png)

**Clean Planet**
![Seed 100 Clean Planet](/docs/images/exp11_att_100_Clean_Planet.png)

**Eclipsing Binary**
![Seed 100 Eclipsing Binary](/docs/images/exp11_att_100_Eclipsing_Binary.png)

**Noise False Positive**
![Seed 100 Noise False Positive](/docs/images/exp11_att_100_Noise_False_Positive.png)

## Analysis & Conclusion

The diagnostic provides strong correlational evidence linking successful training to structured attention, but falls short of causal proof. 

1. **The Nature of the Collapse**: The phase- and head-averaged attention matrix for Seed 100 is substantially more uniform than that of Seed 300. The collapsed solution is associated with this near-uniform, diffuse attention. However, because this matrix averages across 4 heads and 250 phase bins, it is possible that strong phase-specific structure exists but cancels out when aggregated. Finer-grained phase/head analysis is required to state whether the collapse is truly a state of perfect uniformity or an aggregation artifact.

2. **The Functional State**: In Seed 300, the model exhibits structured attention at the macroscopic level. Interestingly, the `[CLS]` token assigns high weight to the sector tokens (~0.18), and the sector tokens assign very high weight back to the `[CLS]` token (~0.19). Notably, the self-attention (S1 attending to S1) is slightly *lower* than cross-attention (S1 attending to S2). This provides strong evidence that the model places substantial attention on cross-sector interactions.

**Conclusion:** We observe that **successful and collapsed runs exhibit qualitatively different macroscopic attention patterns.** Successful runs display structured cross-sector interactions, while collapsed runs display diffuse, averaged attention. 

### Critical Next Steps: Causal Validation
Before freezing this diagnostic as mechanistic proof, we must move from correlation to causation. 
Attention weights tell us where the network allocates attention, but they do not guarantee that the prediction causally depends on it.

The immediate next step is an **Attention Ablation** experiment:
1. Take the successful Seed 300 model.
2. At inference time, artificially suppress cross-sector attention (forcing it to only attend within-sector and to CLS).
3. Measure the drop in performance on the validation set and targeted planets.
If cross-sector attention is functionally necessary, performance should collapse. Once this causal link is verified, we will proceed to **Exp 11A (Local Phase-Windowing)**.