# Research Experimental Record

This document records the exact progression of experiments conducted during Phase I and Phase II of the Exoplanet Detection ML project. 

The value of this progression lies not just in the final pipeline architecture, but in the established experimental framework: generating hypotheses, testing them quantitatively, detecting methodological failures, and actively correcting conclusions.

| Experiment | Status | Question | Result |
| :--- | :--- | :--- | :--- |
| **V1** | Frozen | What is the reference pipeline? | SG101 + Z-score + no clipping established as reference. |
| **Exp 1** | Closed | Does SG window choice matter? | SG401 substantially changed performance and calibration. |
| **Exp 1A** | Closed | Why did SG401 change behavior? | Attribution/morphology analysis identified substantial EB representation changes. |
| **Exp 2** | Closed | Can MAD replace Z-score without retraining? | No; frozen Z-score model degraded. |
| **Exp 2A** | Closed | Is the degradation consistent with representation mismatch? | XAI sanity analysis supported the mismatch interpretation. |
| **Exp 3** | Invalidated | Does multi-sector data improve robustness? | Original comparison invalid because sector availability and sample dropout changed the cohort. |
| **Exp 4** | Closed | Does asymmetric outlier removal help? | Asymmetric +3σ/10σ clipping degraded performance; subsequent validation rejected the initial hypothesis that transit/eclipse depths were directly clipped. |
| **Exp 4A** | Closed | Why did Exp 4 fail? | Initial morphology explanation rejected; asymmetric noise-distribution alteration became the defensible hypothesis. |
| **Exp 5** | Frozen | What is the clean reference model? | Independently trained Z-score reference established. |
| **Exp 6** | Closed | Is MAD intrinsically inferior or merely incompatible with Z-trained CNN? | Native MAD did not recover Z-score performance. |
| **Exp 6A** | Frozen | Are native models learning similar attribution patterns? | Matched-target Grad-CAM correlation was not above permutation null. |

### Note on Methodological Rigor
This record deliberately preserves invalidated experiments (Exp 3) and rejected hypotheses (Exp 4A). 
In applied machine learning, hiding failures obscures the actual sensitivity of the network. A model's failure mode is just as important as its success mode, and documenting the scientific trajectory (Hypothesis -> Result -> Tempting Explanation -> Verification -> Explanation Rejected -> Narrower Explanation Retained) ensures that subsequent architectural decisions are based on verified mechanisms rather than appealing assumptions.
