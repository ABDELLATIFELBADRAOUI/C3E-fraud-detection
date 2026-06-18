# C3E — Chronological Cost-Calibrated Evaluation for Fraud Detection

Reference implementation for the paper:

> **The ROC–PR Divergence and Threshold Transferability: A Cost-Calibrated
> Evaluation Protocol for Fraud Detection**
> *International Journal of Data Science and Analytics* (under review).

This repository reproduces every table and figure in the paper. C3E is an
evaluation **protocol**, not a new classifier: it combines chronological
train/validation/test splitting, leakage-free preprocessing, optional
monotone recalibration, validation-based cost-optimal threshold selection,
and a single frozen-threshold evaluation on the held-out test set. It also
studies the ROC–PR divergence Δ = ROC-AUC − PR-AUC as a screening signal.

---

## Repository structure

```
C3E-fraud/
├── README.md
├── requirements.txt
├── C3E_master_notebook.ipynb      # main entry point; runs all experiments
├── c3e_framework.py               # core protocol: splitting, cost, thresholds
├── c3e_contribution3.py           # calibration (Temperature, Beta) + cost study
├── c3e_datasets_d5_d6.py          # loaders for Elliptic (D5) and GiveMeCredit (D6)
├── c3e_contribution4_nlp.py       # text-representation probe (negative result)
└── data/                          # NOT included — see "Datasets" below
```

---

## Installation

Python 3.10+ on CPU is sufficient (no GPU required).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Datasets

The six benchmarks are public and are **not** redistributed here. Download
each from its source and place the files under `data/` as indicated:

| ID | Dataset | Source | File expected in `data/` |
|----|---------|--------|--------------------------|
| D1 | creditcard | https://www.kaggle.com/mlg-ulb/creditcardfraud | `creditcard.csv` |
| D2 | IEEE-CIS | https://www.kaggle.com/c/ieee-fraud-detection | `train_transaction.csv`, `train_identity.csv` |
| D3 | BAF (NeurIPS 2022) | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 | `Base.csv` |
| D4 | PaySim | https://www.kaggle.com/ntnu-testimon/paysim1 | `PS_*.csv` |
| D5 | Elliptic Bitcoin | https://www.kaggle.com/ellipticco/elliptic-data-set | `elliptic_txs_*.csv` |
| D6 | Give Me Some Credit | https://www.kaggle.com/c/GiveMeSomeCredit | `cs-training.csv` |

---

## Reproducing the results

**Before running, edit one configuration cell.** Open
`C3E_master_notebook.ipynb` and go to the **configuration cell near the top**
(marked `# <-- EDIT`). Update two things to match your machine:

1. **`CODE_DIR`** — the folder containing the four `c3e_*.py` modules. If you
   keep the modules in the same folder as the notebook, set:

   ```python
   CODE_DIR = Path(".")
   ```

2. **`PATHS`** — the location of each downloaded dataset. Replace the example
   paths with the location where you saved the files:

   ```python
   PATHS = {
       "creditcard":        Path("data/creditcard.csv"),
       "ieee_cis":          Path("data/train_transaction.csv"),
       "baf":               Path("data/Base.csv"),
       "paysim":            Path("data/PaySim.csv"),
       "elliptic_features": Path("data/elliptic_txs_features.csv"),
       "elliptic_classes":  Path("data/elliptic_txs_classes.csv"),
       "giveme_train":      Path("data/cs-training.csv"),
   }
   ```

   (The notebook ships with absolute example paths such as
   `D:\fraud_data\creditcard.csv`; just point them to your own files.)

Then run all cells, or run headless:

```bash
jupyter nbconvert --to notebook --execute C3E_master_notebook.ipynb
```

Random seeds `{7, 21, 42, 84, 168}` are fixed; the six-dataset results are
averaged over these seeds, and single-dataset ablations use seed 42.

### Mapping: notebook → paper

| Paper item | Produced by |
|------------|-------------|
| Table (main six-dataset results) | core sweep over D1–D6, four models |
| Table (extended D1, six models) | `c3e_contribution3.run_contribution3` |
| Table (grid-resolution comparison) | grid-resolution cell (uniform / unique / quantile + Temperature) |
| Table (transaction-dependent costs) | linear/logarithmic cost regimes on D1 |
| Figures (Δ heatmap, scatter, τ* gap, ROC–PR, bootstrap CIs) | figure cells |
| Statistical comparison (Friedman, Wilcoxon) | statistical-validation cell |

---

## Method summary

- **Cost setting:** C_FN = 10, C_FP = 1 (r = 10), Bayes threshold τ_B ≈ 0.091.
- **Threshold grid:** 1001 uniform points in [0, 1].
- **Calibration:** none, Temperature Scaling, Beta Calibration (monotone).
- **Δ is computed on raw validation scores** (rank-invariant); only threshold
  *selection* uses calibrated scores when calibration is enabled.
- **Bootstrap:** 95% confidence intervals, 1000 resamples on the test split.

---

## Citation

```bibtex
@article{c3e2026,
  title   = {The ROC--PR Divergence and Threshold Transferability:
             A Cost-Calibrated Evaluation Protocol for Fraud Detection},
  author  = {Elbadraoui, Abdellatif and others},
  journal = {International Journal of Data Science and Analytics},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the MIT License (see `LICENSE`).
