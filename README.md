# Weaponizing Transparency

### Defending Explainable Fake News Detectors Against Attribution-Guided LLM Attacks

**FactLens** · Graduate Capstone Project · Florida International University

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.44-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20progress-orange.svg)]()

> *"When the flashlight becomes the burglar's map."*

---

## Overview

Explainable AI was built to make models trustworthy — to let researchers audit what a model actually pays attention to. But the same explanations that reveal a model's reasoning can reveal its weaknesses to an adversary.

This project tests that hypothesis on fake news detection. We use **Integrated Gradients** to extract the exact features our detector relies on, use a **large language model** to rewrite fake articles that exploit those features, and then build and evaluate two defenses — testing whether either one generalizes to news sources the model has never seen.

**Why it matters.** Automated misinformation detectors underpin news platforms, fact-checking tools, and election-integrity systems. Modern models achieve near-perfect accuracy in controlled evaluations — but a safety-critical model that works in the lab and fails in deployment is not a defense. It is a false sense of security.

---

## Background

This capstone extends prior work from CAP 5610 (Introduction to Machine Learning), where our team built **FactLens** — a dual-task explainable system detecting both fake news and political bias.

| Model | Task | Accuracy |
|-------|------|----------|
| DistilBERT (fine-tuned) | Fake vs. Real | 99.97% |
| Logistic Regression + TF-IDF | Fake vs. Real | 98.52% |
| DistilBERT (fine-tuned) | Left vs. Right bias | 86.6% |
| Logistic Regression + TF-IDF | Left vs. Right bias | 87.1% |

During explainability analysis, we discovered something uncomfortable: a **single token — `reuters` — accounted for roughly 86% of the model's confidence in the "Real" class.** The model had not learned to detect misinformation. It had learned to detect a wire service's formatting conventions.

That discovery is the foundation of this project.

---

## Research Questions

**RQ1** — Do feature attribution explanations from transformer-based fake news detectors systematically expose source-specific artifacts that can be exploited to generate adversarial examples?

**RQ2** — Does adversarial training on LLM-generated attacks reduce this vulnerability across news sources the model was never trained on?

**RQ3** — Can an attribution-regularized training architecture reduce this vulnerability across news sources the model was never trained on?

---

## Methodology

The project runs in four phases:

| Phase | Name | What happens |
|-------|------|--------------|
| **1** | **Map** | Retrain the DistilBERT baseline, apply Integrated Gradients to 50 sampled Real articles, extract the top 20 structural and stylistic artifacts — the *vulnerability blueprint*. |
| **2** | **Attack** | Design an LLM prompt from the blueprint. Use GPT-4o-mini to rewrite 500 fake articles that inject verified-source cues while preserving the false claim. Measure the baseline accuracy drop. |
| **3** | **Defend** | Train two defended models: (a) standard adversarial training on the 500 rewrites, (b) attribution-regularized loss penalizing high attribution scores on known artifact tokens. |
| **4** | **Test** | Evaluate all three models on in-distribution and out-of-distribution data. Statistical significance testing, ablation study, and a political-bias fairness audit. |

---

## Data

### Primary — FactLens Corpus (Bisaillon, Kaggle)

| Property | Value |
|----------|-------|
| Articles | 44,898 raw · 38,590 after cleaning |
| Split | 23,481 Fake · 21,417 Real |
| Sources | Real from Reuters; Fake from political blogs and PolitiFact-flagged outlets |
| Features | title, body text, subject, date |
| Role | Training corpus for the baseline; the vulnerability was discovered here |

### Secondary — NELA-GT-2022 (Gruppi, Horne & Adalı)

| Property | Value |
|----------|-------|
| Articles | 1,778,363 |
| Sources | 361 outlets (335 with veracity labels) |
| Labels | Source-level, from Media Bias/Fact Check — Reliable / Mixed / Unreliable |
| Coverage | January–December 2022, no collection gaps |
| Median length | 424 words |
| Role | Out-of-distribution evaluation — required to answer RQ2 and RQ3 |

> **Data access note.** The NELA-GT-2022 Harvard Dataverse listing was deaccessioned in 2026. We obtained the dataset directly from **Prof. Maurício Gruppi (Rensselaer Polytechnic Institute)**, one of its original creators. We gratefully acknowledge his assistance.

**Known data quality issues** (identified in our EDA and corrected before evaluation):

- Three sources have severe missing content: Bongino Report (91%), Sputnik (56%), TASS (31%) — excluded
- 118,268 duplicate titles, predominantly boilerplate (app promos, copyright notices) — filtered
- One source (`newsweek`) appeared twice in `labels.csv` with conflicting labels — resolved via deduplication

---

## Repository Structure

```
factlens-capstone/
├── notebooks/
│   ├── 00_setup.ipynb                  # Shared environment + seed
│   ├── 01_eda_nela.ipynb               # Exploratory data analysis
│   ├── 02_baseline_retrain.ipynb       # Phase 1 — baseline model
│   ├── 03_integrated_gradients.ipynb   # Phase 1 — vulnerability blueprint
│   ├── 04_adversarial_generation.ipynb # Phase 2 — LLM attack
│   ├── 05_defense_standard.ipynb       # Phase 3a — adversarial training
│   ├── 06_defense_regularized.ipynb    # Phase 3b — attribution regularization
│   └── 07_evaluation_ood.ipynb         # Phase 4 — cross-dataset evaluation
├── src/
│   ├── data_loader.py                  # Dataset loading and cleaning
│   ├── attribution.py                  # Integrated Gradients wrapper
│   ├── attack.py                       # LLM attack pipeline
│   ├── losses.py                       # Attribution-regularized loss
│   └── evaluate.py                     # Metrics and statistical tests
├── results/
│   ├── figures/                        # Charts and visualizations
│   ├── tables/                         # Results tables
│   └── logs/                           # Training logs
├── docs/
│   ├── proposal.pdf                    # Two-semester capstone plan
│   ├── eda_report.md                   # Written EDA findings
│   └── data_dictionary.md              # Field definitions
├── ENVIRONMENT.md
├── requirements.txt
└── README.md
```

> Datasets and model checkpoints are stored in Google Drive, not in this repository. See [ENVIRONMENT.md](ENVIRONMENT.md).

---

## Getting Started

### Prerequisites

- Google Colab (GPU runtime — T4 or better)
- Google Drive with access to the shared project folder
- OpenAI API key (Phase 2 only; estimated cost under $1)

### Setup

Every notebook begins with the standard environment block:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, random
import numpy as np
import torch

# Fixed seed — reproducibility across the team
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE = '/content/drive/MyDrive/Captsone Data Science AI/FactLens_Capstone'
DATA_DIR    = f'{BASE}/data'
MODELS_DIR  = f'{BASE}/models'
RESULTS_DIR = f'{BASE}/results'
```

### Dependencies

```bash
pip install -r requirements.txt
```

### Reproducibility

All experiments use a fixed random seed of **42** across `random`, `numpy`, and `torch`, with cuDNN determinism enabled. Library versions are pinned in `requirements.txt`. Any team member running the same notebook should obtain identical results.

---

## Team

**Group 9** · Florida International University

| Name | Role |
|------|------|
| Mauricio Velasquez | Offense — adversarial attack design and generation |
| Valentina Kloster | Defense — attribution regularization and evaluation |
| John Corbacho Soubal | Infrastructure — data pipeline and experiment tracking |

**Research Mentor:** Dr. Olaoluwa Adigun, Knight Foundation School of Computing and Information Sciences
**Capstone Instructor:** Dr. Ananda M. Mondal, IDC 6940 — Capstone in Data Science
**Endorsed by:** Dr. Fahad Saeed, Graduate Program Director, KFSCIS

---

## Timeline

**Fall 2026** — Phases 1–4a: baseline and blueprint (September), adversarial attack (October), both defenses and preliminary OOD evaluation (November)

**Spring 2027** — Full cross-dataset evaluation with statistical testing (January), VIS4ML dashboard and paper draft (February), venue submission and arXiv preprint (March), final capstone presentation and report (April)

---

## References

1. Ahmed, M. S., & Spezzano, F. (2026). A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors. *WSDM 2026*, 1058–1062. https://dl.acm.org/doi/10.1145/3773966.3779393

2. Chen, J., Wu, X., Rastogi, V., Liang, Y., & Jha, S. (2019). Robust Attribution Regularization. *NeurIPS 2019*. https://arxiv.org/abs/1905.09957

3. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML 2017*. https://arxiv.org/abs/1703.01365

4. Gruppi, M., Horne, B. D., & Adalı, S. (2022). NELA-GT-2022: A Large Multi-Labelled News Dataset for the Study of Misinformation in News Articles. arXiv:2203.05659. https://arxiv.org/abs/2203.05659

5. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT: a distilled version of BERT. arXiv:1910.01108. https://arxiv.org/abs/1910.01108

6. Wang, L., Nayyem, M. N., Al Rakin, A., Santosh, K. C., Zhang, C., & Zhou, Y. (2026). Explainability-Guided Defense: Attribution-Aware Model Refinement Against Adversarial Data Attacks. arXiv:2601.00968. https://arxiv.org/abs/2601.00968

7. Ma, S.-D., Liu, Y.-H., Lin, H.-Y., Chen, P.-Y., Huang, H.-Y., Hsu, S.-Y., & Chen, Y.-N. (2026). RADAR: Retrieval-Augmented Detector with Adversarial Refinement for Robust Fake News Detection. arXiv:2601.03981. https://arxiv.org/abs/2601.03981

8. Kishi, Y., Arima, Y., & Iyatomi, H. (2026). Fake News Detection Strategies under Dataset Bias: Using Large-scale Coarse-grained Labels. *EACL 2026 SRW*, 612–621. https://aclanthology.org/2026.eacl-srw.47.pdf

9. Kozik, R., Ficco, M., Pawlicka, A., Pawlicki, M., Palmieri, F., & Choraś, M. (2024). When explainability turns into a threat: using xAI to fool a fake news detection method. *Computers & Security*, 137, 103599.

---

## Acknowledgments

We thank **Dr. Maurício Gruppi** (RPI) for providing access to the NELA-GT datasets, and **Dr. Olaoluwa Adigun** for his mentorship and for defining the research questions that shape this work.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

Dataset licenses are held by their respective creators and are not redistributed in this repository.
