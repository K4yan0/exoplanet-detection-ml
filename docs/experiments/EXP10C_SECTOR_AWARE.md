# Experiment 10C: Sector-Aware Transformer

## Hypothesis
In Experiment 10A, the `[CLS]` token demonstrated strong cross-sector phase attention, recovering 7/10 targeted planets but falling short of the Exp 9C benchmark (9/10). We hypothesized that the standard 1D positional encoding (1-1250) was forcing the Transformer to implicitly relearn the rigid 5-sector structure, hindering its ability to robustly align corresponding phases. By explicitly providing a 2D positional encoding (Sector ID + Phase ID), the Transformer should be able to cleanly reason about phase alignments across sectors.

## Architecture
- **Input**: 5 sectors of 250 bins (shape: `5, 250, 1`)
- **Backbone**: CNN feature extractor (output shape: `1250, 64`)
- **Positional Encoding**: `SectorPhaseEmbedding` layer replacing the standard 1D encoding.
  - Computes `sector_id = position // 250` (1 to 5)
  - Computes `phase_id = position % 250` (1 to 250)
  - Adds a learned 64D Sector Embedding and a learned 64D Phase Embedding to the CNN features.
- **Aggregation**: A learned `[CLS]` token concatenated to the sequence, processed by 1 Transformer block, followed by a classification head.

## Results

### Optimization Instability (Two Runs)
During testing, this architecture demonstrated significant **optimization instability**, heavily dependent on random initialization. We ran this identical architecture twice, producing two very different outcomes that highlight a classic failure mode for Transformers on this dataset.

#### Run 1: Successful Optimization (Epoch 93 Convergence)
In the first run, the model successfully optimized, reaching 70% accuracy on the test set.

**Classification Report:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Noise | 0.83 | 0.60 | 0.70 | 25 |
| Planet | 0.50 | 0.82 | 0.62 | 11 |
| EB | 0.79 | 0.79 | 0.79 | 14 |
| **Accuracy** | **0.70** | | | **50** |

**Confusion Matrix:**
```
[[15  7  3]   # Noise (7 False Positives as Planet)
 [ 2  9  0]   # Planet (9 True Positives)
 [ 1  2 11]]  # EB
```
**Diagnostic Recovery:** 10/10 Targeted Planets Recovered.

#### Run 2: Degenerate Collapse (Early Stopping at Epoch 32)
In the second run, the model failed to find a gradient path to separate the classes. It became trapped in a local minimum where it output a nearly flat probability distribution slightly biased towards the Planet class (`P(Planet) ≈ 0.47` for almost all inputs).

**Classification Report:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Noise | 0.00 | 0.00 | 0.00 | 25 |
| Planet | 0.29 | 0.91 | 0.44 | 11 |
| EB | 0.69 | 0.79 | 0.73 | 14 |
| **Accuracy** | **0.42** | | | **50** |

**Confusion Matrix:**
```
[[ 0 21  4]   # Noise (Guessed Planet 21 times)
 [ 0 10  1]   # Planet (Guessed Planet 10 times)
 [ 0  3 11]]  # EB
```
**Diagnostic Recovery:** "10/10" Recovered (Hollow victory, as it classified almost everything as a Planet).

## Attention Analysis (From Run 1)
To understand how the `[CLS]` token is recovering these planets when it successfully optimizes, we extracted the attention map for one of the recovered targets (TIC 259377017_Positive). 

![Exp 10C Attention](/docs/images/EXP10C_attention_plot.png)

This visualization shows the attention scores from the `[CLS]` token to all 1250 sequence positions (250 bins x 5 sectors). It clearly demonstrates that the network has learned to allocate periodic spikes of attention at matching relative phase bins across all 5 sectors. 

By explicitly feeding Sector (1-5) and Phase (1-250) IDs into the embedding layer, the Transformer no longer has to implicitly count indices. It simply attends to the precise phase coordinate where morphological transit features are occurring, cross-checking them against identical phase coordinates in other sectors.

## Conclusion

**Exp 10C confirms two critical properties of the Transformer for Exoplanet Detection:**

1. **Explicit Coordinates Enable Relational Reasoning:** By explicitly providing 2D structural awareness (Sector + Phase) rather than a flat 1D sequence index, the `[CLS]` token successfully recovered **10/10** of the targeted difficult planets (Run 1). This demonstrates that the Transformer can exploit cross-sector phase correspondence when sector identity and within-sector phase are explicitly represented, which is consistent with learning timing-related structure.
2. **Optimization Instability:** Unlike the LSTM in Exp 9C, the Transformer is highly sensitive to initialization and optimization dynamics (Run 2). It easily collapses into degenerate states (e.g., predicting the majority class or outputting flat distributions).

Ultimately, we have uncovered a new trade-off: **9C trades representational resolution for optimization stability, while 10C trades optimization stability for high-resolution relational modeling.**

While 10C has demonstrated greater representational potential for difficult targets, it has not surpassed the **Exp 9C (Coarse Position + LSTM)** benchmark. Exp 9C remains the superior architecture due to its empirical reliability, stable training dynamics, higher overall accuracy (84%), higher planet precision (0.77), and lower false-positive rate.
