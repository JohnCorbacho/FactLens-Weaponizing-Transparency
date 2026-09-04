# Environment Setup

**Weaponizing Transparency (FactLens)** · Group 9 · IDC 6940

This document defines the shared working environment for the project. Every team member should follow these steps before running any notebook. The goal is that any of us can run any notebook and get **identical results**.

---

## 1. Platform

| Component | Choice | Notes |
|-----------|--------|-------|
| Compute | Google Colab | GPU runtime required (T4 minimum, A100 preferred for Phase 3) |
| Storage | Google Drive | Datasets, model checkpoints, results |
| Code | GitHub | Notebooks and source modules only — no data |
| Python | 3.12 | Colab default |

**Why this split:** datasets and model checkpoints are far too large for git. Code is versioned in GitHub; everything heavy lives in Drive.

---

## 2. Google Drive structure

All three team members must have access to the same shared folder. Create it once, share it with the team, and never change the paths — the notebooks depend on them.

```
Captsone Data Science AI/
└── FactLens_Capstone/
    ├── data/
    │   ├── bisaillon/                  # Primary dataset (Kaggle)
    │   └── nela-gt/                    # OOD dataset
    │       ├── nela-gt-2022.db
    │       ├── labels.csv
    │       └── tweet_table.csv
    ├── models/
    │   ├── baseline/                   # Phase 1 retrained DistilBERT
    │   ├── defended_standard/          # Phase 3a
    │   └── defended_regularized/       # Phase 3b
    └── results/
        ├── figures/
        ├── tables/
        └── logs/
```

**Action:** the folder owner shares `FactLens_Capstone` with edit access to both teammates.

---

## 3. Standard notebook header

**Every notebook starts with this block.** No exceptions — this is what makes our results reproducible across three separate machines.

```python
# ═══════════════════════════════════════════════
# FactLens Capstone — Standard Environment Block
# ═══════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
import random
import numpy as np
import torch

# ── Fixed seed for reproducibility ─────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Shared project paths ───────────────────────
BASE        = '/content/drive/MyDrive/Captsone Data Science AI/FactLens_Capstone'
DATA_DIR    = f'{BASE}/data'
MODELS_DIR  = f'{BASE}/models'
RESULTS_DIR = f'{BASE}/results'

NELA_DB     = f'{DATA_DIR}/nela-gt/nela-gt-2022.db'
NELA_LABELS = f'{DATA_DIR}/nela-gt/labels.csv'

# ── Verify ─────────────────────────────────────
print(f"Seed:        {SEED}")
print(f"Base path:   {os.path.exists(BASE)}")
print(f"GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

---

## 4. Dependencies

Install at the top of each Colab session, right after the header block:

```python
!pip install -q -r /content/factlens-capstone/requirements.txt
```

Or, if you have not cloned the repo:

```python
!pip install -q transformers==4.44.0 captum==0.7.0 datasets==2.20.0 accelerate==0.33.0 openai==1.40.0
```

**Versions are pinned deliberately.** Different library versions can produce different numerical results. If anyone needs to upgrade a package, propose it to the team first and update `requirements.txt` for everyone.

---

## 5. Reproducibility check

Before starting real work, all three of us run this and compare outputs. **The three results must match exactly.**

```python
test_random = [round(random.random(), 6) for _ in range(3)]
test_numpy  = np.round(np.random.rand(3), 6).tolist()
test_torch  = [round(x, 6) for x in torch.rand(3).tolist()]

print("random:", test_random)
print("numpy: ", test_numpy)
print("torch: ", test_torch)
```

Paste your output in the team chat. If they differ, the seed block did not run before some other random operation — re-run the notebook from the top.

---

## 6. GitHub workflow

### Cloning into Colab

```python
import os
REPO = '/content/factlens-capstone'
if not os.path.exists(REPO):
    !git clone https://github.com/<username>/factlens-capstone.git {REPO}
%cd {REPO}
```

For a **private repo**, use a fine-grained personal access token:

```python
from getpass import getpass
token = getpass('GitHub token: ')
!git clone https://{token}@github.com/<username>/factlens-capstone.git /content/factlens-capstone
```

Generate a token at: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens (grant Repo read/write).

### Saving notebooks back

Easiest route in Colab: **File → Save a copy in GitHub**. Choose the repo, branch, and path, then write a commit message.

Alternatively, from a cell:

```python
!git config --global user.email "you@fiu.edu"
!git config --global user.name  "Your Name"
!git add .
!git commit -m "Phase 1: baseline retrain + IG blueprint"
!git push
```

### Important

Colab sessions are ephemeral. Anything cloned into `/content/` disappears when the session ends. **Always push before disconnecting.** Data and models in Drive persist regardless.

---

## 7. What is never committed

Enforced by `.gitignore`:

```
# Data and models — live in Google Drive
data/
models/
*.db
*.pkl
*.pt
*.bin
*.csv

# Secrets
.env
*.key

# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# OS
.DS_Store
```

**Never commit the OpenAI API key.** Load it from a Colab secret or prompt for it:

```python
from getpass import getpass
OPENAI_API_KEY = getpass('OpenAI API key: ')
```

---

## 8. Team responsibilities

| Area | Owner |
|------|-------|
| Data pipeline, cleaning, experiment tracking | John Corbacho Soubal |
| Adversarial attack design and generation (Phase 2) | Mauricio Velasquez |
| Defense training and evaluation (Phases 3–4) | Valentina Kloster |
| Shared: environment consistency, code review, documentation | All |

---

## 9. Troubleshooting

**`os.path.exists(BASE)` returns False**
Drive is not mounted, or the folder name has a typo. Note the project folder is spelled `Captsone Data Science AI` — verify against your Drive.

**Out of memory when loading NELA-GT**
Do not load the full `content` column. Use `LENGTH(content) as content_length` for exploratory work, and load full text only for the subset you are actively modeling.

**Results differ between team members**
Confirm the seed block ran first and that library versions match `requirements.txt`. Run the reproducibility check in Section 5.

**Merge inflates row count after joining labels**
`labels.csv` contains a duplicate source (`newsweek`) with conflicting labels. Deduplicate before merging:

```python
labels_df = labels_df.drop_duplicates(subset='source', keep='first')
```

---

*Last updated: Fall 2026*
