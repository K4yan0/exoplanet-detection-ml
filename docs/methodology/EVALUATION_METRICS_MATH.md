# Evaluation Metrics Math (Exp 5 Reference Pipeline)

This document formally defines the mathematical calculations used in the Exp 5 Reference Pipeline evaluation suite. It is designed to ensure strict, transparent comparability between the V1 reference model and all future experimental branches.

## 1. Multiclass Brier Score

The Brier Score measures the accuracy of probabilistic predictions. For a multiclass problem (where the number of classes $K = 3$ for Noise, Planet, EB), the Brier Score is defined as the mean squared difference between the predicted probability assigned to each class and the actual outcome (one-hot encoded).

### Mathematical Definition
For $N$ total samples:
$$ BS = \frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} (f_{nk} - o_{nk})^2 $$

Where:
*   $f_{nk}$ is the predicted probability that sample $n$ belongs to class $k$.
*   $o_{nk}$ is $1$ if sample $n$ belongs to class $k$, and $0$ otherwise (one-hot encoding).
*   $K = 3$ (Noise=0, Planet=1, EB=2).

### Interpretation Constraint
**Crucial Note:** A Brier score calculated for a binary classification task ($K=2$) cannot be directly compared to a Brier score calculated for a ternary classification task ($K=3$). In Exp 5, the model splits probability across three classes, which structurally inflates the sum of squared errors compared to a binary model. The Exp 5 multiclass Brier score (e.g., $0.3109$) is the new foundational baseline for all future $K=3$ models.

---

## 2. Expected Calibration Error (ECE)

ECE measures how well a model's predicted confidence aligns with its actual accuracy. A model is perfectly calibrated if, for all predictions where it is $90\%$ confident, it is correct exactly $90\%$ of the time.

### Mathematical Definition
To calculate ECE, we partition the $N$ predictions into $M$ equally spaced bins (we use $M=10$ bins: $[0.0, 0.1), [0.1, 0.2), \dots, [0.9, 1.0]$) based on the model's **maximum predicted probability** (its confidence).

$$ ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right| $$

Where:
*   $B_m$ is the set of indices of samples whose predicted confidence falls into bin $m$.
*   $|B_m|$ is the number of samples in bin $m$.
*   $\text{acc}(B_m)$ is the actual accuracy of the samples in bin $m$ (number of correct predictions divided by $|B_m|$).
*   $\text{conf}(B_m)$ is the average predicted confidence for the samples in bin $m$.

### Multiclass Implementation Note
In our ternary setup, the "confidence" is always defined as $\max(p_{\text{Noise}}, p_{\text{Planet}}, p_{\text{EB}})$. The accuracy is calculated by comparing `argmax(predictions)` against the true label. The low ECE achieved in Exp 5 ($0.0262$, or $2.62\%$) indicates that the CNN is spectacularly well-calibrated out of the box, without requiring temperature scaling.

---

## 3. Monte Carlo (MC) Dropout Uncertainty

To quantify epistemic uncertainty (how "sure" the model is of its own internal representation), we keep the dropout layers active during inference and pass the same sample through the network $T$ times (where $T=50$).

### Mathematical Definition
For a single input $x$, we obtain $T$ probability vectors: $p_1, p_2, \dots, p_T$.

The final **Predicted Probability** for class $k$ is the mean:
$$ P(y=k | x) = \frac{1}{T} \sum_{t=1}^{T} p_{t, k} $$

The **Predictive Uncertainty** (Variance) is calculated as the mean variance across all classes:
$$ \text{Uncertainty}(x) = \frac{1}{K} \sum_{k=1}^{K} \text{Var}(p_{1, k}, \dots, p_{T, k}) $$

Where $\text{Var}()$ is the standard variance formula. High variance (e.g., $>0.005$) indicates the model is struggling to map the input to a known representation (high epistemic uncertainty), while a variance near $0.0000$ indicates strong structural familiarity.
