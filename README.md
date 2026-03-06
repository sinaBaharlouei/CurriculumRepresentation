# CurriculumRepresentation

Implementation of **Representation Curriculum (RC)** methods for Learning-to-Rank on **MSLR-WEB10K** (LETOR format) and supporting scripts to reproduce **results + figures**. :contentReference[oaicite:1]{index=1}

---

## What’s in this repository

**Main training / sweep scripts**
- `GenerateForM.py`: runs an **M sweep** (curriculum depth) across **5 folds** and multiple repeats; outputs CSVs for overall + cold metrics and historical feature gains. :contentReference[oaicite:2]{index=2}
- `train_all.py`: trains the “all-features” baseline (full model). :contentReference[oaicite:3]{index=3}
- `content_only.py`: trains the content-only baseline. :contentReference[oaicite:4]{index=4}
- `CL.py`: curriculum-learning utilities / implementation helpers. :contentReference[oaicite:5]{index=5}
- `utilities.py`: shared utilities. :contentReference[oaicite:6]{index=6}

**Plotting**
- `AUC Plot.py`: generates plots across M (cold vs overall) for metrics such as NDCG@{1,3,5,10} and related curves. :contentReference[oaicite:7]{index=7}

**Included result tables (CSV)**
This repo already contains several summary CSVs (means/stds/per-fold) for baselines and/or sweeps. :contentReference[oaicite:8]{index=8}


## Setup

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # (mac/linux)
pip install -U pip
pip install numpy pandas scikit-learn lightgbm matplotlib
```
### 2) Dataset: MSLR-WEB10K (LETOR format)

Download **MSLR-WEB10K** and unzip it so the folder looks like:

MSLR-WEB10K/
- Fold1/
  - train.txt
  - vali.txt
  - test.txt
- Fold2/
  - train.txt
  - vali.txt
  - test.txt
- Fold3/
- Fold4/
- Fold5/

> The scripts expect 5 folds and the standard LETOR text format.


## Reproducing results (RC sweep over curriculum depth M)

Run the main sweep script (5 folds × repeats):

```bash
python GenerateForM.py
