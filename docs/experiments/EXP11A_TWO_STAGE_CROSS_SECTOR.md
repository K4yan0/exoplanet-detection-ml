# Experiment 11A: Two-Stage Cross-Sector Reasoning

## Hypothesis
In Experiment 11, we discovered via causal ablation that the original architecture was structurally unable to use the cross-sector representations (`S_i -> S_j`) it computed. Because the `[CLS]` token was extracted immediately after a single Transformer block, the mixed sector representations were discarded without ever being used in the forward pass.

By adding a **second Transformer block**, we establish a causal pathway:
1. **Block 1**: Sector tokens mix with each other (`S_i -> S_j`), updating their representations to reflect cross-sector consensus.
2. **Block 2**: The `[CLS]` token reads the *updated, mixed* sector representations and aggregates them for the classifier.

**Prediction**: If true cross-sector reasoning is required to identify difficult planets, the Two-Stage architecture should demonstrate stronger or more stable optimization. Most importantly, **ablating cross-sector attention in Block 1 should now measurably degrade the classifier's performance.**

## Protocol
- **Data**: 250 phase bins, identical to Exp 11.
- **Architecture**:
  - `TimeDistributed(CNN)` -> `(batch*250, 5, 64)`.
  - Add `[CLS]` token -> `(batch*250, 6, 64)`.
  - **Transformer Block 1**: Cross-sector mixing.
  - **Transformer Block 2**: Aggregation.
  - Extract `[CLS]` -> `(batch*250, 64)`.
  - **Global Phase Transformer**: Aggregates the 250 phases.
- **Evaluation**: 
  - Train standard models (e.g. 5 seeds).
  - Perform **Causal Attention Ablation** on Block 1 (masking out `S_i -> S_j` attention) and measure the impact on validation accuracy and targeted Planet Recall.

## Results
*(To be populated after execution)*

## Conclusion
*(To be populated after execution)*