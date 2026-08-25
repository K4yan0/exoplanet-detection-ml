# Experiment 10E: Multi-Seed Stability Control for Sector-Aware Transformer

## 1. Motivation
Experiment 10C demonstrated that explicitly factorizing 1D positional encodings into 2D (Sector, Phase) coordinates allowed a single-layer Transformer `[CLS]` token to achieve **10/10 targeted transit recovery**. However, subsequent re-evaluations raised severe concerns about the stability of this architecture. 

Experiment 10E was designed as a strict methodological control: run the exact 10C architecture (identical dataset split, identical 50-sample test cohort, identical parameters) across 5 different random initializations to measure the reliability of the optimization landscape.

## 2. Experimental Design
* **Architecture:** Identical to 10C (Shared High-Res CNN $\rightarrow$ 1250-length sequence $\rightarrow$ 2D Sector/Phase Embedding $\rightarrow$ `[CLS]` Token $\rightarrow$ 1 Transformer Block $\rightarrow$ Classifier).
* **Training Protocol:** Identical to 10C, but iterating over 5 random seeds: `[42, 100, 200, 300, 400]`.
* **Class Weights:** Explicitly included to heavily penalize majority-class collapse.

## 3. Results

The multi-seed evaluation reveals extreme variance dependent on weight initialization:

| Seed | Epochs | Accuracy | Planet Precision | Planet Recall | Targeted Recovery |
|------|--------|----------|------------------|---------------|-------------------|
| 42   | 21     | 64.00%   | 0.0000           | 0.0000        | 0/10              |
| 100  | 93     | 56.00%   | 0.3548           | **1.0000**    | **10/10**         |
| 200  | 81     | 70.00%   | 0.4000           | 0.5455        | 7/10              |
| 300  | 37     | 64.00%   | 0.0000           | 0.0000        | 0/10              |
| 400  | 26     | 62.00%   | 0.0000           | 0.0000        | 0/10              |

**Summary Statistics:**
* **Accuracy:**  `0.6320 ± 0.0449`
* **Planet Precision:** `0.1510 ± 0.1854`
* **Planet Recall:**    `0.3091 ± 0.4049`

## 4. Scientific Conclusions

1. **Representational capacity was demonstrated, but reliable optimization was not.** 
   Seed 100 serves as an existence proof: the 10C architecture possesses the structural capacity to perfectly identify transits (`1.00` Planet Recall, `10/10` targeted recovery). The attention mechanism *can* successfully learn cross-sector phase-alignment reasoning.
   
2. **The optimization landscape is highly brittle.**
   Despite class weights, 3 out of 5 seeds suffered complete degenerate collapse (predicting strictly Noise/Eclipsing Binaries and entirely ignoring Planets). The massive standard deviation in Planet Recall (`±0.4049`) confirms that finding the global optimum is a matter of "lucky" initialization.

3. **Comparison to Exp 9C (LSTM):**
   Experiment 9C converges highly reliably across runs. We hypothesize that 9C's sequential architecture inherently provides a smoother optimization landscape by forcing corresponding phases to be processed synchronously across sectors. In contrast, the Transformer must learn to sift through 1,250 tokens from scratch to find relevant cross-sector pairs, leaving it highly vulnerable to local minima.

## 5. Next Steps
To make the high-resolution Transformer viable, we must explicitly inject the inductive bias that makes 9C stable. Instead of forcing global attention to learn Phase 137's relation to Phase 137 from scratch, we should structure the attention mechanism (e.g., via hard-coded attention masks or cross-sector attention reshaping) to explicitly prioritize cross-sector correspondence at identical phase coordinates.
