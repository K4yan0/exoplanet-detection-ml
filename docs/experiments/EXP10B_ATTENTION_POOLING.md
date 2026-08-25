# Experiment 10B: Learned Attention Pooling

## 1. Hypothesis
Following the recovery of planetary signals in Experiment 10A using a `[CLS]` token, this experiment sought to explicitly test the aggregation mechanism. Can learned, sparse aggregation of high-resolution Transformer features outperform a single learned `[CLS]` summary without changing the underlying representation or attention mechanism?

By applying an explicit Attention Pooling mechanism across the sequence (where the network assigns a learned scalar weight to each of the 1,250 tokens and takes a weighted sum), we force the model to learn a signal-selection mechanism. If successful, the model should concentrate its weights on the transit regions, explicitly ignoring the background noise.

## 2. Architecture & Design
All parameters from Exp 10A were held perfectly constant except for the aggregation layer.

* **Input:** 5 independent sectors, 2,000 bins each
* **Shared CNN Encoder:** High-resolution spatial extraction (1D Conv layers reducing 2000 -> 250 spatial bins per sector)
* **Sequence Length:** 1,250 tokens (5 sectors * 250 bins)
* **Positional Encoding:** Standard learned embedding
* **Transformer Block:** 1 layer, 4 heads, `embed_dim=64`, `ff_dim=128`
* **NEW AGGREGATION:** Removed `[CLS]` token. Implemented `AttentionPooling1D`: applies `Dense(1)` and softmax over the sequence dimension to generate explicitly learned weights for each token, producing a weighted sum across the sequence.
* **Classifier:** Same MLP block `(Dropout -> Dense(32) -> Dropout -> Dense(3))`

## 3. Results
The experiment successfully trained but produced a surprising regression in planet recovery compared to Exp 10A:

* **Overall Accuracy:** 72.0% (Improved from 10A's 64%)
* **Macro ROC-AUC:** 0.8403
* **Planet Precision:** 0.50 (Improved from 10A's 0.36)
* **Planet Recall:** 0.09 (Fell drastically from 10A's 0.36)
* **Planet F1-Score:** 0.15

### Targeted Diagnostic Recovery
The most striking failure occurred on the targeted planet sample.

* TIC 259377017: MISSED | P(Planet) = 0.436
* TIC 36724087: MISSED | P(Planet) = 0.421
* TIC 287328202: **RECOVERED** | P(Planet) = 0.445
* TIC 345143460: MISSED | P(Planet) = 0.444
* TIC 234994474: MISSED | P(Planet) = 0.427
* TIC 150030205: MISSED | P(Planet) = 0.431
* TIC 262530407: MISSED | P(Planet) = 0.454
* TIC 181804752: MISSED | P(Planet) = 0.429
* TIC 307809773: MISSED | P(Planet) = 0.417
* TIC 254113311: MISSED | P(Planet) = 0.460

**Total Recovered: 1 / 10** (compared to 7/10 in Exp 10A)

## 4. Scientific Conclusions
The explicit attention-pooling mechanism failed to recover the difficult planets, pulling the targeted recovery rate down from 7/10 (with `[CLS]`) to 1/10.

While the model achieved higher overall accuracy (72%) and precision (0.50), it did so by becoming extremely conservative, identifying almost everything as Noise (Recall = 0.09). Therefore, 10B's higher accuracy is misleading and does not represent a better planetary detector than 10A.

**Exp 10B does not demonstrate that high-resolution self-attention is incapable of representing the transit signal. It demonstrates that a simple scalar attention-pooling head is inadequate for extracting the sparse planetary signal from the Transformer representation.**

This is a highly informative failure. The `[CLS]` token in Exp 10A has access to the entire sequence simultaneously through self-attention, allowing it to construct a relational representation (e.g., "This feature in Sector 1 corresponds to this feature in Sector 2, with consistent phase displacement"). In contrast, the attention pooling in 10B assigns an isolated scalar score to each token before averaging.

This suggests that the features produced by the Self-Attention block might not be separable enough for a simple linear scoring function to isolate transits, or that the pooling loses sector identity, or that the weighted sum destroys relationships between multiple transit locations. The useful information is likely relational, which scalar pooling cannot capture as effectively as a `[CLS]` aggregator.
