# C3E — Chronological Cost-Calibrated Evaluation for Fraud Detection

Reference implementation and saved results for the paper:

> **The ROC–PR Divergence and Threshold Transferability: A Cost-Calibrated
> Evaluation Protocol for Fraud Detection**
> *International Journal of Data Science and Analytics* (revised version).
> Code release **`v2.0`** (the version cited in the paper's *Code availability*).

C3E is an evaluation **protocol**, not a new classifier: chronological
train/validation/test splitting, leakage-free preprocessing, optional monotone
recalibration (Temperature Scaling, Beta Calibration), cost-optimal threshold
selection on validation, and a single evaluation at the frozen threshold on the
held-out test split. The paper also studies the ROC–PR divergence
Δ = ROC-AUC − PR-AUC as a screening signal.

---

## What is in this release

```
C3E-fraud-detection/
├── README.md
├── LICENSE                          MIT
├── requirements.txt
├── c3e_framework.py                 core protocol: loaders, chronological split, preprocessing
│                                    (float32 cast, median imputer, scaler fitted on train only),
│                                    the four core trainers, uniform-grid threshold selection
├── c3e_contribution3.py             CatBoost / IF-Hybrid trainers, Temperature Scaling, Beta
│                                    Calibration (eps = 1e-7 clip), ECE (10 bins), cost regimes
├── c3e_datasets_d5_d6.py            loaders for Elliptic (D5) and Give Me Some Credit (D6)
├── c3e_contribution4_nlp.py         text-representation probe (negative result, Section 6.4)
├── C3E_master_notebook.ipynb        the full pipeline: trains every model on the six benchmarks
│                                    (five seeds), runs the D1 extended comparison, NLP probe,
│                                    Friedman/Wilcoxon tests, bootstrap CIs
├── reproduce_tables_8_10.ipynb      RETRAINS NOTHING: regenerates Tables 7, 8, 9, 10 and the
│                                    controls of Section 6.2 from the saved scores (see below)
├── make_figures.py                  regenerates Fig. 1, 2, 3, 4, 7, 8, 9, 10, 11 from results/
├── results/
│   ├── scores/
│   │   ├── scores_d1_seed42.npz              raw validation + test scores, D1, six models, seed 42,
│   │   │                                     with labels and transaction amounts (main pipeline)
│   │   ├── scores_d1_seed42_calibrated.npz   the same after Temperature Scaling and Beta Calibration
│   │   ├── scores_d1_seed42_val.csv          raw / TS / Beta validation scores, one column per model
│   │   └── scores_d1_seed42_test.csv         idem, test split
│   ├── c3e_all_seeds_raw.csv        every (dataset, model, seed) row of the six-dataset benchmark
│   ├── c3e_seed_aggregated.csv      five-seed mean ± std  -> Table 5
│   ├── friedman_summary.csv         Wilcoxon pairwise tests (reference XGB)  -> Table 6
│   ├── tab7_extended_d1.csv         D1 extended comparison from the saved scores  -> Table 7
│   ├── tab8_gridres.csv             threshold-grid audit  -> Table 8
│   ├── tab9_costregimes.csv         fixed vs logarithmic regime  -> Table 9
│   ├── fig11_cost_regimes_d1.csv    fixed / linear / logarithmic totals  -> Fig. 11
│   ├── tab10_pairs_labelled.csv     the 26 pairs with Δ, cost, cost ratio, costly/regular label
│   ├── tab10_screening.csv          flagged / missed / needless counts, κ ∈ {1.5, 2, 5}  -> Table 10
│   ├── tab_eps_sensitivity.csv      ε sweep 1e-6 … 1e-15 and the rank-map control (Section 6.2)
│   ├── tab_delta_after_calibration.csv   Δ on raw vs mapped scores (Section 6.2)
│   ├── tab_counterexample.csv       Remark 3.3: counter-example and the 2,000-draw simulation
│   ├── c3e_bootstrap_ci_all.csv     bootstrap 95 % CIs, 26 pairs, seed 42  -> Table 11
│   └── c3e_nlp_ieee_cis.csv, c3e_nlp_paysim.csv    NLP probe  -> Section 6.4
└── figures/                         Fig1.pdf … Fig11.pdf as included in the manuscript
                                     (Fig5 and Fig6 come from the master notebook)
```

The six datasets are public and are **not** redistributed here (see *Datasets*).

---

## Reproducing the paper without retraining

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter
jupyter nbconvert --to notebook --execute reproduce_tables_8_10.ipynb
python make_figures.py .
```

`reproduce_tables_8_10.ipynb` reads `results/scores/scores_d1_seed42.npz` and
`results/c3e_all_seeds_raw.csv`, and regenerates, in this order: the
Temperature-Scaling and Beta maps and the calibrated scores; Table 7; Table 8
(three grids × models, with the frozen threshold, the validation cost actually
minimised, the number of tied minimisers, the number of test transactions
flagged, the test cost, the oracle cost and the null-model cost 750 =
C_FN · N1_test); the ε sweep and the strictly increasing rank map; Δ on raw and
mapped scores; the Reviewer-2 counter-example and the 2,000-draw simulation of
Remark 3.3 (`np.random.default_rng(42)`; 6.0 % degraded, 5.8 % improved); the
26 pairs with their Δ, cost and costly/regular label, and the κ sweep of
Table 10; Table 9 and the three regimes of Fig. 11. Its last cell asserts every
number quoted in the paper against the regenerated values.

The two post-hoc maps are copied verbatim into the notebook from
`c3e_contribution3.py`, so the notebook needs neither LightGBM, XGBoost nor
CatBoost; when `c3e_contribution3` is importable the notebook checks that the
copies agree with the module.

---

## Reproducing from scratch (retraining)

Python 3.11 on CPU is sufficient. The paper's environment: Python 3.11.0,
NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, scikit-learn 1.5.0, XGBoost 2.0.3,
LightGBM 4.6.0, CatBoost 1.2.10.

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Open `C3E_master_notebook.ipynb`, edit the configuration cell (`CODE_DIR`,
`PATHS`), then run it top to bottom or headless:

```bash
jupyter nbconvert --to notebook --execute C3E_master_notebook.ipynb
```

### Seeds and determinism

* Six-dataset results: seeds `{7, 21, 42, 84, 168}`, reported as mean ± std.
  The D1 extended comparison uses seed 42.
* LR is deterministic. LightGBM's seed enters only through the subsample of at
  most 200,000 training rows (`bin_construct_sample_cnt`) used to build its
  histogram bins: on D1, D5 and D6 the training block is smaller than that
  count, so the LightGBM fit is seed-independent; on D2–D4 the bin boundaries,
  and hence the scores, vary with the seed. No bagging or feature subsampling
  is used. RF, XGBoost, CatBoost and IF-Hybrid vary with the seed.
* **Seed caveat.** The multi-seed loop of the master notebook sets the
  module-level seed of every `c3e_*` module and leaves it at the last value of
  the list (168). In the version that produced the first submission, the D1
  extended comparison (Section 4) was run after that loop, so its RF, XGBoost,
  CatBoost and IF-Hybrid rows were fitted at seed 168 while labelled "seed 42".
  Section 4 now resets the seed to 42 first; the paper's D1 tables are taken
  from the saved seed-42 scores of the main pipeline (`results/scores/`).
* The LightGBM fit on D1 depends on the preprocessing path: with the
  pipeline's `prepare_splits` (features cast to float32 before imputation and
  scaling) it yields 21 distinct validation scores and a test cost of 1,858;
  an earlier grid-resolution cell that re-implemented the preprocessing
  without the cast obtained 26 distinct values and a cost of 4,938. All
  reported D1 numbers use the pipeline path.

---

## Datasets

| ID | Dataset | Source | File expected in `data/` |
|----|---------|--------|--------------------------|
| D1 | creditcard | https://www.kaggle.com/mlg-ulb/creditcardfraud | `creditcard.csv` |
| D2 | IEEE-CIS | https://www.kaggle.com/c/ieee-fraud-detection | `train_transaction.csv`, `train_identity.csv` |
| D3 | BAF (NeurIPS 2022) | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 | `Base.csv` |
| D4 | PaySim | https://www.kaggle.com/ntnu-testimon/paysim1 | `PS_*.csv` |
| D5 | Elliptic Bitcoin | https://www.kaggle.com/ellipticco/elliptic-data-set | `elliptic_txs_*.csv` |
| D6 | Give Me Some Credit | https://www.kaggle.com/c/GiveMeSomeCredit | `cs-training.csv` |

---

## Method summary

- **Cost setting:** C_FN = 10, C_FP = 1 (r = 10), Bayes threshold τ_B = 1/11 ≈ 0.091.
- **Threshold grid:** 1001 uniform points in [0, 1]; ties broken by the smallest
  threshold (`numpy.argmin`). Table 8 also reports the unique-score and
  quantile grids.
- **Calibration:** none, Temperature Scaling, Beta Calibration. Both maps clip
  their input to [ε, 1 − ε], ε = 1e-7, before taking logarithms, so as
  implemented they are non-decreasing but not injective (paper, Section 6.2).
- **Δ is computed on raw validation scores**; only threshold selection uses the
  calibrated scores when calibration is enabled.
- **Null-model cost:** every realised cost is compared with C_FN · N1_test, the
  cost of flagging nothing (750 on D1); a frozen threshold that flags no
  instance is reported as such, not as an improvement.
- **Bootstrap:** 95 % confidence intervals, 1000 resamples of the test split.
- **ECE:** 10 equal-width bins on the validation split.

---

## Citation

```bibtex
@article{c3e2026,
  title   = {The ROC--PR Divergence and Threshold Transferability:
             A Cost-Calibrated Evaluation Protocol for Fraud Detection},
  author  = {Elbadraoui, Abdellatif and Mouhssine, Yassine and El Alaoui, Abdelkader
             and Ouatik El Alaoui, Said},
  journal = {International Journal of Data Science and Analytics},
  year    = {2026},
  note    = {Under review; code release v2.0}
}
```

## License

Released under the MIT License (see `LICENSE`).
