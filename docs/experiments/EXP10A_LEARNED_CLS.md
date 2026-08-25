# Experiment 10A: Learned CLS Aggregation

## 1. The Design
Experiment 10A was a surgical follow-up to Experiment 10. We kept the entire High-Resolution Self-Attention architecture identical, but replaced the final unweighted `GlobalAveragePooling1D` step with a learned `[CLS]` token.

The sequence length was increased from 1,250 to 1,251 by prepending a trainable `[CLS]` embedding vector. After the `TransformerBlock`, we extracted only the `[CLS]` vector and passed it to the dense classifier head. This allowed the network to learn a weighted summary of the sequence, explicitly testing the hypothesis that unweighted averaging was destroying sparse transit signals.

## 2. The Results
The addition of the learned aggregation token produced a dramatic recovery of planetary detection capabilities.

### Aggregate Performance
| Metric | Exp 10 (GlobalAvgPool) | Exp 10A (CLS Token) |
| :--- | :--- | :--- |
| **Test Accuracy** | 54.0% | **64.0%** |
| **Planet Recall** | 0.00 | **0.36** |
| **Planet Precision** | 0.00 | **0.36** |
| **Noise Recall** | 0.68 | **0.76** |

**Confusion Matrix (Exp 10A):**
```text
         Pred Noise  Pred Planet  Pred EB
Noise        19           5         1
Planet        7           4         0
EB            3           2         9
```

### Targeted Diagnostic Recovery
```text
TIC TIC 259377017_Positive: RECOVERED | P(Planet) = 0.883 ± 0.095
TIC TIC 36724087_Positive: RECOVERED | P(Planet) = 0.879 ± 0.124
TIC TIC 287328202_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.513 ± 0.214
TIC TIC 345143460_Positive: RECOVERED | P(Planet) = 0.927 ± 0.066
TIC TIC 234994474_Positive: RECOVERED | P(Planet) = 0.922 ± 0.087
TIC TIC 150030205_Positive: RECOVERED | P(Planet) = 0.898 ± 0.090
TIC TIC 262530407_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.059 ± 0.106
TIC TIC 181804752_Positive: RECOVERED | P(Planet) = 0.892 ± 0.106
TIC TIC 307809773_Positive: MISSED    | Predicted: 0 | P(Planet) = 0.015 ± 0.031
TIC TIC 254113311_Positive: RECOVERED | P(Planet) = 0.949 ± 0.062

Total Recovered: 7/10
```

## 3. Scientific Conclusions
The dramatic jump from 0 to 7 targeted planet recoveries confirms that **the 10A result demonstrates that the catastrophic failure of Exp 10 was substantially attributable to the aggregation mechanism, because replacing unweighted global averaging with a learned aggregation token recovered 7/10 targeted planets without changing the high-resolution Transformer representation.** Unweighted global averaging effectively erased the sparse, high-resolution planetary transit features.

By allowing the network to learn its own sequence summary query via the `[CLS]` token, the model successfully recovered the ability to identify planetary morphology across sectors.

However, the overall performance (64% accuracy, 36% Planet Precision) is still significantly weaker than our strongest baseline (Exp 9C: 84% accuracy, 77% Planet Precision). While the `[CLS]` token fixed the catastrophic signal loss, the network still struggles to achieve high precision and recall simultaneously using only self-attention over the raw positional sequence.
