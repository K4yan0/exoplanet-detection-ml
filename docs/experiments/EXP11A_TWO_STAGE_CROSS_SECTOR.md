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
We successfully trained the Two-Stage model and ran the rigorous causal ablation test as proposed.

*   **Intact Model (Standard):** 0.7000 Validation Accuracy, 0.1111 Planet Recall
*   **Block 1 Ablated (No Cross-Sector Mixing):** 0.5000 Validation Accuracy, 0.0000 Planet Recall
*   **Both Blocks Ablated:** 0.5000 Validation Accuracy, 0.0000 Planet Recall

## Conclusion
The experiment demonstrates a **causal dependence under the specific ablation procedure**.

By introducing the second Transformer block, we created a valid structural pathway for cross-sector interaction. The ablation test confirms that the trained model is now utilizing this pathway: **ablating cross-sector attention in Block 1 caused a substantial degradation in validation performance (dropping to the 0.50 majority-class baseline) and reduced Planet recall to zero, providing direct evidence that the trained model's predictions depend on the cross-sector information propagated through Block 1.** 

Furthermore, because ablating *both* blocks yielded the exact same collapsed performance as ablating *only Block 1*, we can conclude that the downstream classifier cannot rescue the representation once the initial cross-sector pathway is removed. 

*(Note: The difference between Exp 11 and Exp 11A is an architectural evolution, not a pure causal comparison. The valid causal test is entirely within 11A, comparing intact vs ablated.)*

This is a major conceptual milestone for the project, successfully moving from visual explanation (attention heatmaps) to causal verification. However, because this was tested on a single initialization (Seed 42), we cannot yet claim the architecture is robust.

**Next Step:** Experiment 11B (Multi-Seed Two-Stage Stability) to verify if this causal cross-sector dependence is reliably learned across initializations.