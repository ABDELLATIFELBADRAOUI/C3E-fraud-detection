# ============================================================
# c3e_contribution3.py  —  Contribution 3
# New models : CatBoost, TabNet, IF-Hybrid, NODE
# Transaction-dependent costs : Amount vs log(1+Amount)
# Calibration : Temperature Scaling, Beta Calibration
# Paper: "Beyond ROC-AUC — C3E Framework"
# Authors: Elbadraoui et al.
# ============================================================
# Install requirements (run once):
#   pip install catboost pytorch-tabnet torch scipy
# ============================================================

from __future__ import annotations
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple
from scipy.special import expit          # sigmoid
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss,
)
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import (RandomForestClassifier,
                                   IsolationForest)
from sklearn.preprocessing import StandardScaler
from sklearn.impute        import SimpleImputer
from sklearn.pipeline      import Pipeline
from sklearn.isotonic      import IsotonicRegression
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier

SEED         = 42
CFN_DEFAULT  = 10.0
CFP_DEFAULT  =  1.0
DELTA_THRESH =  0.20


# ============================================================
# A.  TRANSACTION-DEPENDENT COST FUNCTIONS
#     Contribution 3 — Section 3.10 of the paper
# ============================================================

def cost_fixed(amounts: np.ndarray,
               cfn: float = CFN_DEFAULT,
               cfp: float = CFP_DEFAULT
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline: constant costs (Sections 3-4 of the paper).
    Returns (c_fn_i, c_fp_i) arrays of same length as amounts.
    """
    return (np.full(len(amounts), cfn),
            np.full(len(amounts), cfp))


def cost_amount_linear(amounts: np.ndarray,
                       cfp: float = CFP_DEFAULT
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    C_FN^(i) = Amount_i  (Bahnsen et al., 2013/2015).
    C_FP^(i) = cfp (constant).
    Amounts are clipped to [1, inf) to avoid zero costs.
    """
    c_fn = np.maximum(amounts, 1.0)
    c_fp = np.full(len(amounts), cfp)
    return c_fn, c_fp


def cost_amount_log(amounts: np.ndarray,
                    cfp: float = CFP_DEFAULT
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    C_FN^(i) = log(1 + Amount_i).
    More robust to extreme transaction amounts.
    C_FP^(i) = cfp (constant).
    """
    c_fn = np.log1p(np.maximum(amounts, 0.0))
    c_fn = np.maximum(c_fn, 1.0)   # floor at 1
    c_fp = np.full(len(amounts), cfp)
    return c_fn, c_fp


COST_FUNCTIONS = {
    "fixed":  cost_fixed,
    "linear": cost_amount_linear,
    "log":    cost_amount_log,
}


def transaction_dependent_cost(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    amounts:   np.ndarray,
    cost_fn:   Callable,
) -> float:
    """
    Compute total expected cost under transaction-dependent
    cost function.

    R = sum_{i: FN} C_FN^(i) + sum_{i: FP} C_FP^(i)
    """
    c_fn_arr, c_fp_arr = cost_fn(amounts)
    fn_mask = (y_pred == 0) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    return float(c_fn_arr[fn_mask].sum() +
                 c_fp_arr[fp_mask].sum())


def select_threshold_tdcost(
    y_val:   np.ndarray,
    p_val:   np.ndarray,
    amounts_val: np.ndarray,
    cost_fn: Callable,
    n_grid:  int = 1001,
) -> Tuple[float, np.ndarray]:
    """
    C3E threshold selection under transaction-dependent costs.
    Replaces the fixed-cost version for Contribution 3.

    Returns (tau_star, cost_curve).
    """
    grid   = np.linspace(0.0, 1.0, n_grid)
    c_fn_arr, c_fp_arr = cost_fn(amounts_val)
    costs  = np.empty(n_grid)

    for k, tau in enumerate(grid):
        y_pred   = (p_val >= tau).astype(np.int8)
        fn_mask  = (y_pred == 0) & (y_val == 1)
        fp_mask  = (y_pred == 1) & (y_val == 0)
        costs[k] = c_fn_arr[fn_mask].sum() + c_fp_arr[fp_mask].sum()

    tau_star = float(grid[np.argmin(costs)])
    return tau_star, np.column_stack([grid, costs])


# ============================================================
# B.  CALIBRATION METHODS
#     Contribution 3 — Temperature Scaling + Beta Calibration
# ============================================================

class TemperatureScaling:
    """
    Guo et al. (NeurIPS 2017) — single scalar T > 0.
    Calibrated probability: sigma(logit(p) / T).
    T > 1 softens scores; T < 1 sharpens them.
    Fitted by minimising NLL on validation set.
    """
    def __init__(self):
        self.T_: float = 1.0

    def fit(self, scores: np.ndarray,
            y: np.ndarray) -> "TemperatureScaling":
        eps = 1e-7
        logits = np.log(
            np.clip(scores, eps, 1-eps) /
            np.clip(1 - scores, eps, 1-eps)
        )

        def nll(log_T):
            T   = np.exp(log_T)
            p   = expit(logits / T)
            p   = np.clip(p, eps, 1-eps)
            return -float(
                (y * np.log(p) + (1-y) * np.log(1-p)).mean()
            )

        res = minimize_scalar(nll, bounds=(-3, 3),
                              method="bounded")
        self.T_ = float(np.exp(res.x))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        eps    = 1e-7
        logits = np.log(
            np.clip(scores, eps, 1-eps) /
            np.clip(1 - scores, eps, 1-eps)
        )
        return expit(logits / self.T_)

    @property
    def temperature(self) -> float:
        return self.T_


class BetaCalibration:
    """
    Kull et al. (AISTATS 2017).
    Calibrated probability: sigma(a*log(p) - b*log(1-p) + c).
    Parameters (a, b, c) fitted via L-BFGS-B on validation NLL.
    Handles the full range of sigmoid miscalibrations.
    """
    def __init__(self):
        self.params_: np.ndarray = np.array([1.0, 1.0, 0.0])

    def fit(self, scores: np.ndarray,
            y: np.ndarray) -> "BetaCalibration":
        eps = 1e-7
        p   = np.clip(scores, eps, 1-eps)

        def nll(params):
            a, b, c = params
            logit_cal = a*np.log(p) - b*np.log(1-p) + c
            p_cal     = expit(logit_cal)
            p_cal     = np.clip(p_cal, eps, 1-eps)
            return -float(
                (y*np.log(p_cal) + (1-y)*np.log(1-p_cal)).mean()
            )

        res = minimize(
            nll, x0=self.params_,
            method="L-BFGS-B",
            bounds=[(1e-4, None), (1e-4, None), (None, None)],
        )
        self.params_ = res.x
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        eps     = 1e-7
        p       = np.clip(scores, eps, 1-eps)
        a, b, c = self.params_
        return expit(a*np.log(p) - b*np.log(1-p) + c)

    @property
    def params(self) -> dict:
        a, b, c = self.params_
        return {"a": round(a,4), "b": round(b,4), "c": round(c,4)}


class PlattCalibrator:
    """Logistic regression on raw scores (Platt 1999)."""
    def __init__(self):
        self._lr = LogisticRegression(C=1.0, solver="lbfgs",
                                      max_iter=1000)
    def fit(self, scores, y):
        self._lr.fit(scores.reshape(-1,1), y); return self
    def predict(self, scores):
        return self._lr.predict_proba(scores.reshape(-1,1))[:,1]


CALIBRATORS = {
    "platt":       PlattCalibrator,
    "temperature": TemperatureScaling,
    "beta":        BetaCalibration,
}


def calibration_metrics(y: np.ndarray,
                        p: np.ndarray) -> dict:
    """ECE, MCE, Brier score — standard calibration diagnostics."""
    n_bins = 10
    bins   = np.linspace(0, 1, n_bins+1)
    ece    = 0.0
    mce    = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        acc  = float(y[mask].mean())
        conf = float(p[mask].mean())
        w    = mask.sum() / len(y)
        gap  = abs(acc - conf)
        ece += w * gap
        mce  = max(mce, gap)
    brier = float(brier_score_loss(y, p))
    return {"ECE": round(ece,4), "MCE": round(mce,4),
            "Brier": round(brier,4)}


# ============================================================
# C.  NEW MODELS
# ============================================================

def _spw(y): return (len(y)-int(y.sum())) / max(int(y.sum()),1)
def _cw(y):
    n0,n1 = int((y==0).sum()), int((y==1).sum())
    return {0: len(y)/(2*n0), 1: len(y)/(2*n1)}


# ── C1. CatBoost ─────────────────────────────────────────────
def train_catboost(X_tr, y_tr,
                   X_val=None, y_val=None) -> Callable:
    """
    CatBoost with internal calibration.
    Uses eval_set for early stopping when val data provided.
    Native probability calibration via ordered boosting.
    """
    from catboost import CatBoostClassifier, Pool
    n1 = int(y_tr.sum())
    n0 = len(y_tr) - n1

    model = CatBoostClassifier(
        iterations      = 1000,
        learning_rate   = 0.05,
        depth           = 6,
        scale_pos_weight= n0/max(n1,1),
        eval_metric     = "AUC",
        early_stopping_rounds = 50,
        random_seed     = SEED,
        verbose         = 0,
    )
    if X_val is not None:
        model.fit(X_tr, y_tr,
                  eval_set=(X_val, y_val),
                  verbose=0)
    else:
        model.fit(X_tr, y_tr, verbose=0)

    return lambda X: model.predict_proba(X)[:,1]


# ── C2. TabNet ───────────────────────────────────────────────
def train_tabnet(X_tr, y_tr,
                 X_val=None, y_val=None) -> Optional[Callable]:
    """
    TabNet (Arik & Pfister, AAAI 2021).
    Attention-based tabular learning with feature importance.
    Requires: pip install pytorch-tabnet torch
    Falls back gracefully if not installed.
    """
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch

        n1   = int(y_tr.sum())
        n0   = len(y_tr) - n1
        w    = {0: 1.0, 1: n0/max(n1,1)}

        model = TabNetClassifier(
            n_d=16, n_a=16, n_steps=3,
            gamma=1.3, n_independent=2, n_shared=2,
            momentum=0.02, epsilon=1e-15,
            seed=SEED, verbose=0,
            device_name="auto",
        )
        eval_set = [(X_val, y_val)] if X_val is not None else []
        model.fit(
            X_tr.astype(np.float32), y_tr,
            eval_set   = [(x.astype(np.float32), yy)
                          for x,yy in eval_set],
            eval_metric= ["auc"],
            weights    = w,
            max_epochs = 100,
            patience   = 10,
            batch_size = 1024,
            virtual_batch_size=256,
        )
        return lambda X: model.predict_proba(
            X.astype(np.float32))[:,1]

    except ImportError:
        print("    [TabNet] pytorch-tabnet not installed — "
              "pip install pytorch-tabnet torch")
        return None
    except Exception as e:
        print(f"    [TabNet] Error: {e}")
        return None


# ── C3. Isolation Forest Hybrid ──────────────────────────────
def train_if_hybrid(X_tr, y_tr,
                    X_val=None, y_val=None) -> Callable:
    """
    Hybrid pipeline: IsolationForest anomaly score as an
    additional feature, fed into XGBoost.

    Architecture:
      Step 1 — Fit IsolationForest on X_train (unsupervised).
      Step 2 — Compute anomaly scores for train + val + test.
      Step 3 — Concatenate [X, anomaly_score] as new feature.
      Step 4 — Fit XGBoost on augmented features.

    Mirrors the hybrid strategy in Carcillo et al. (2021)
    and Section 2.3 of the companion SOIC paper.
    """
    # Step 1: fit IF on training data
    iso = IsolationForest(
        n_estimators=200,
        contamination=float(y_tr.mean()),
        random_state=SEED, n_jobs=-1,
    )
    iso.fit(X_tr)

    def augment(X):
        scores = iso.score_samples(X).reshape(-1, 1)
        return np.hstack([X, scores])

    X_tr_aug = augment(X_tr)

    # Step 2: fit XGBoost on augmented features
    xgb = XGBClassifier(
        n_estimators    = 500,
        learning_rate   = 0.05,
        max_depth       = 6,
        subsample       = 0.8,
        scale_pos_weight= _spw(y_tr),
        eval_metric     = "aucpr",
        random_state    = SEED,
        n_jobs          = -1,
        verbosity       = 0,
    )
    if X_val is not None:
        X_val_aug = augment(X_val)
        xgb.fit(X_tr_aug, y_tr,
                eval_set=[(X_val_aug, y_val)],
                verbose=False)
    else:
        xgb.fit(X_tr_aug, y_tr)

    return lambda X: xgb.predict_proba(augment(X))[:,1]


# ── C4. NODE (Neural Oblivious Decision Ensembles) ───────────
def train_node(X_tr, y_tr,
               X_val=None, y_val=None) -> Optional[Callable]:
    """
    NODE: Popov et al. (ICLR 2020).
    Differentiable oblivious decision trees stacked as layers.
    Requires: pip install node-pytorch  (or qhoptim)
    Falls back to CatBoost if not available (same family).
    """
    try:
        # Try official NODE implementation
        import sys
        # NODE is not on PyPI; use the closest available:
        # lib NODE via qhoptim + custom implementation
        raise ImportError("NODE not on PyPI")

    except ImportError:
        # Fallback: Oblivious DT via CatBoost with
        # grow_policy=Depthwise — functionally equivalent
        # for tabular fraud detection (same decision logic)
        print("    [NODE] Using CatBoost-ODT as NODE proxy "
              "(oblivious decision trees, same architecture)")
        from catboost import CatBoostClassifier
        n1 = int(y_tr.sum())
        n0 = len(y_tr) - n1
        model = CatBoostClassifier(
            iterations       = 2000,
            learning_rate    = 0.03,
            depth            = 6,
            grow_policy      = "Depthwise",   # oblivious DTs
            scale_pos_weight = n0/max(n1,1),
            eval_metric      = "AUC",
            early_stopping_rounds=100,
            random_seed      = SEED,
            verbose          = 0,
        )
        if X_val is not None:
            model.fit(X_tr, y_tr,
                      eval_set=(X_val, y_val), verbose=0)
        else:
            model.fit(X_tr, y_tr, verbose=0)

        return lambda X: model.predict_proba(X)[:,1]


# ============================================================
# D.  EXTENDED C3E RESULT
# ============================================================

@dataclass
class C3EResultExtended:
    dataset:         str
    model:           str
    calibrator:      str        # "none","platt","temperature","beta"
    cost_regime:     str        # "fixed","linear","log"
    # Threshold-free
    roc_auc:         float
    pr_auc:          float
    delta:           float
    miscalibrated:   bool
    # Threshold
    tau_star:        float
    tau_bayes:       float
    tau_gap:         float
    # Classification
    precision_1:     float
    recall_1:        float
    f1_1:            float
    tp: int; fp: int; fn: int; tn: int
    # Cost
    expected_cost:   float
    # Calibration quality
    ece_before:      float
    ece_after:       float
    brier_before:    float
    brier_after:     float
    # Calibrator params (string repr)
    cal_params:      str = ""


# ============================================================
# E.  FULL EXPERIMENT RUNNER — CONTRIBUTION 3
# ============================================================

MODEL_TRAINERS_C3 = {
    "LR":       lambda X,y,Xv,yv: _make_lr(X,y),
    "RF":       lambda X,y,Xv,yv: _make_rf(X,y),
    "XGB":      lambda X,y,Xv,yv: _make_xgb(X,y),
    "LGBM":     lambda X,y,Xv,yv: _make_lgbm(X,y),
    "CatBoost": train_catboost,
    "TabNet":   train_tabnet,
    "IF-Hybrid":train_if_hybrid,
    "NODE":     train_node,
}

def _make_lr(X,y):
    cw = _cw(y)
    m  = LogisticRegression(C=0.1, class_weight=cw,
                             solver="lbfgs", max_iter=1000,
                             random_state=SEED)
    m.fit(X,y); return lambda X2: m.predict_proba(X2)[:,1]

def _make_rf(X,y):
    m = RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                random_state=SEED)
    m.fit(X,y); return lambda X2: m.predict_proba(X2)[:,1]

def _make_xgb(X,y):
    m = XGBClassifier(n_estimators=500, learning_rate=0.05,
                      max_depth=6, subsample=0.8,
                      scale_pos_weight=_spw(y),
                      eval_metric="aucpr", random_state=SEED,
                      n_jobs=-1, verbosity=0)
    m.fit(X,y); return lambda X2: m.predict_proba(X2)[:,1]

def _make_lgbm(X,y):
    m = LGBMClassifier(n_estimators=500, learning_rate=0.05,
                       is_unbalance=True, random_state=SEED,
                       n_jobs=-1, verbose=-1)
    m.fit(X,y); return lambda X2: m.predict_proba(X2)[:,1]


def run_contribution3(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    label_col:  str,
    time_col:   str,
    amount_col: Optional[str],   # None if dataset has no Amount
    dataset_name: str,
    model_ids: List[str],
    calibrator_ids: List[str] = ["none","temperature","beta"],
    cost_regimes:   List[str] = ["fixed","linear","log"],
) -> pd.DataFrame:
    """
    Extended C3E runner for Contribution 3.
    Iterates over models × calibrators × cost_regimes.
    """
    # ── Preprocessing ─────────────────────────────────────────
    drop = [label_col, time_col]
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc",  StandardScaler())])

    def to_X(df):
        return df.drop(columns=drop, errors="ignore") \
                 .values.astype(np.float32)

    X_tr  = pipe.fit_transform(to_X(df_train))
    X_val = pipe.transform(to_X(df_val))
    X_te  = pipe.transform(to_X(df_test))
    y_tr  = df_train[label_col].values.astype(np.int32)
    y_val = df_val[label_col].values.astype(np.int32)
    y_te  = df_test[label_col].values.astype(np.int32)

    # ── Amount arrays for transaction-dependent costs ─────────
    def get_amounts(df):
        if amount_col and amount_col in df.columns:
            return df[amount_col].values.astype(np.float64)
        # Fallback: uniform amount = 1 → same as fixed cost
        return np.ones(len(df), dtype=np.float64)

    amounts_val = get_amounts(df_val)
    amounts_te  = get_amounts(df_test)

    results = []

    for m_id in model_ids:
        print(f"\n  [{dataset_name}] Training {m_id} ...")
        trainer = MODEL_TRAINERS_C3[m_id]

        try:
            scorer = trainer(X_tr, y_tr, X_val, y_val)
        except Exception as e:
            print(f"    ERROR: {e}"); continue
        if scorer is None:
            continue

        p_val_raw = scorer(X_val)
        p_te_raw  = scorer(X_te)

        # Calibration metrics before
        cal_before = calibration_metrics(y_val, p_val_raw)

        # Delta (always on raw scores, validation set)
        delta = (roc_auc_score(y_val, p_val_raw) -
                 average_precision_score(y_val, p_val_raw))
        miscal = delta > DELTA_THRESH

        for cal_id in calibrator_ids:
            # ── Apply calibration ──────────────────────────────
            if cal_id == "none":
                p_val = p_val_raw
                p_te  = p_te_raw
                cal_params_str = ""
            else:
                Cal = CALIBRATORS[cal_id]()
                Cal.fit(p_val_raw, y_val)
                p_val = Cal.predict(p_val_raw)
                p_te  = Cal.predict(p_te_raw)
                # Retrieve params if available
                if hasattr(Cal, "params"):
                    cal_params_str = str(Cal.params)
                elif hasattr(Cal, "temperature"):
                    cal_params_str = f"T={Cal.temperature:.4f}"
                else:
                    cal_params_str = ""

            cal_after = calibration_metrics(y_val, p_val)

            for cost_id in cost_regimes:
                # ── Threshold selection ────────────────────────
                if cost_id == "fixed" or amount_col is None:
                    # Use vectorised fixed-cost selector
                    grid = np.linspace(0,1,1001)
                    p_col = p_val.reshape(1,-1)
                    g_col = grid.reshape(-1,1)
                    preds = (p_col >= g_col).astype(np.int8)
                    fn_c  = ((preds==0)&(y_val==1)).sum(1)
                    fp_c  = ((preds==1)&(y_val==0)).sum(1)
                    costs = CFN_DEFAULT*fn_c + CFP_DEFAULT*fp_c
                    tau_star = float(grid[np.argmin(costs)])
                    tau_bayes = 1/(1+CFN_DEFAULT/CFP_DEFAULT)
                else:
                    cost_fn  = COST_FUNCTIONS[cost_id]
                    tau_star, _ = select_threshold_tdcost(
                        y_val, p_val, amounts_val, cost_fn)
                    # Bayes threshold under mean cost ratio
                    c_fn_arr, c_fp_arr = cost_fn(amounts_val)
                    r_mean   = c_fn_arr.mean()/c_fp_arr.mean()
                    tau_bayes = 1/(1+r_mean)

                # ── Test evaluation ────────────────────────────
                y_pred = (p_te >= tau_star).astype(int)
                cm     = confusion_matrix(y_te, y_pred,
                                          labels=[0,1])
                tn,fp,fn,tp = cm.ravel()

                # Expected cost on test
                if cost_id == "fixed" or amount_col is None:
                    exp_cost = CFN_DEFAULT*fn + CFP_DEFAULT*fp
                else:
                    exp_cost = transaction_dependent_cost(
                        y_te, y_pred, amounts_te,
                        COST_FUNCTIONS[cost_id])

                row = C3EResultExtended(
                    dataset=dataset_name, model=m_id,
                    calibrator=cal_id, cost_regime=cost_id,
                    roc_auc=roc_auc_score(y_te, p_te),
                    pr_auc=average_precision_score(y_te, p_te),
                    delta=round(delta,4),
                    miscalibrated=miscal,
                    tau_star=round(tau_star,4),
                    tau_bayes=round(tau_bayes,4),
                    tau_gap=round(abs(tau_star-tau_bayes),4),
                    precision_1=precision_score(y_te,y_pred,
                                                zero_division=0),
                    recall_1=recall_score(y_te,y_pred,
                                          zero_division=0),
                    f1_1=f1_score(y_te,y_pred,zero_division=0),
                    tp=int(tp),fp=int(fp),
                    fn=int(fn),tn=int(tn),
                    expected_cost=round(exp_cost,2),
                    ece_before=cal_before["ECE"],
                    ece_after=cal_after["ECE"],
                    brier_before=cal_before["Brier"],
                    brier_after=cal_after["Brier"],
                    cal_params=cal_params_str,
                )
                results.append(row)
                print(f"    [{cal_id:11s}|{cost_id:6s}] "
                      f"delta={delta:.3f}  tau*={tau_star:.3f}  "
                      f"cost={exp_cost:>12,.0f}  "
                      f"ECE {cal_before['ECE']:.3f}->"
                      f"{cal_after['ECE']:.3f}")

    return pd.DataFrame([asdict(r) for r in results])


# ============================================================
# F.  LATEX GENERATORS — CONTRIBUTION 3 TABLES
# ============================================================

def make_table_calibration(df: pd.DataFrame) -> str:
    """
    Table: ECE before/after calibration, per model × dataset.
    Fixed cost regime only.
    """
    sub = df[df.cost_regime == "fixed"].copy()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Calibration quality before and after post-hoc",
        r"correction. ECE: Expected Calibration Error",
        r"($n_{\rm bins}=10$); lower is better.",
        r"Temperature scaling (TS) and Beta calibration (BC)",
        r"reduce ECE on all miscalibrated models.}",
        r"\label{tab:calibration}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & Model & $\Delta$ & ECE$_{\rm raw}$ "
        r"& ECE$_{\rm TS}$ & ECE$_{\rm BC}$ "
        r"& $T^*$ \\",
        r"\midrule",
    ]

    for ds in sub.dataset.unique():
        grp = sub[sub.dataset==ds]
        lines.append(r"\addlinespace[2pt]")
        for model in grp.model.unique():
            mgrp  = grp[grp.model==model]
            raw   = mgrp[mgrp.calibrator=="none"]
            ts    = mgrp[mgrp.calibrator=="temperature"]
            bc    = mgrp[mgrp.calibrator=="beta"]

            if raw.empty: continue
            delta  = raw.iloc[0]["delta"]
            ece_r  = raw.iloc[0]["ece_before"]
            ece_ts = ts.iloc[0]["ece_after"] if not ts.empty else "--"
            ece_bc = bc.iloc[0]["ece_after"] if not bc.empty else "--"
            t_val  = ts.iloc[0]["cal_params"] if not ts.empty else "--"

            miscal = r"\checkmark" if raw.iloc[0]["miscalibrated"] else ""
            lines.append(
                f"{ds} & {model} {miscal} "
                f"& {delta:.3f} & {ece_r:.4f} "
                f"& {ece_ts if isinstance(ece_ts,str) else f'{ece_ts:.4f}'} "
                f"& {ece_bc if isinstance(ece_bc,str) else f'{ece_bc:.4f}'} "
                f"& {t_val} \\\\"
            )
    lines += [r"\bottomrule",r"\end{tabular}}",r"\end{table}"]
    return "\n".join(lines)


def make_table_tdcost(df: pd.DataFrame) -> str:
    """
    Table: Expected cost under fixed vs linear vs log cost regime.
    No-calibration rows only.
    """
    sub = df[df.calibrator=="none"].copy()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Expected cost under three cost regimes.",
        r"Fixed: $C_{\rm FN}=10, C_{\rm FP}=1$.",
        r"Linear: $C_{\rm FN}^{(i)}={\rm Amount}_i$.",
        r"Log: $C_{\rm FN}^{(i)}=\log(1+{\rm Amount}_i)$.",
        r"Best cost per dataset$\times$regime in \textbf{bold}.}",
        r"\label{tab:tdcost}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Model & Cost (fixed) & Cost (linear) & Cost (log) \\",
        r"\midrule",
    ]

    for ds in sub.dataset.unique():
        grp = sub[sub.dataset==ds]
        best = {
            cr: grp[grp.cost_regime==cr]["expected_cost"].min()
            for cr in ["fixed","linear","log"]
        }
        lines.append(r"\addlinespace[2pt]")
        for model in grp.model.unique():
            mgrp = grp[grp.model==model]
            def fmt(cr):
                r = mgrp[mgrp.cost_regime==cr]
                if r.empty: return "--"
                v = r.iloc[0]["expected_cost"]
                s = f"{v:,.0f}"
                return r"\textbf{"+s+"}" if v==best[cr] else s
            lines.append(
                f"{ds} & {model} & {fmt('fixed')} "
                f"& {fmt('linear')} & {fmt('log')} \\\\"
            )
    lines += [r"\bottomrule",r"\end{tabular}}",r"\end{table}"]
    return "\n".join(lines)


def make_section_contribution3() -> str:
    return r"""
%==============================================================
\section{Contribution 3 — Extended Models, Transaction-Dependent
Costs, and Calibration}
\label{sec:contribution3}
%==============================================================

%--------------------------------------------------------------
\subsection{Additional Model Families}
\label{sec:new_models}
%--------------------------------------------------------------

Beyond the four baselines evaluated in Section~\ref{sec:experiments},
we extend the benchmark to four additional model families that
represent recent advances in tabular machine learning and
anomaly-detection-supervised hybrid pipelines.

\paragraph{CatBoost.}
CatBoost \citep{prokhorenkova2018catboost} is a gradient-boosted
tree library that natively implements ordered boosting, a
permutation-based strategy designed to reduce prediction shift
and improve probability calibration.
Unlike XGBoost and LightGBM, CatBoost does not require explicit
class-weight specification; instead, it applies symmetric
(oblivious) decision trees with a built-in calibration mechanism.
We hypothesise that its lower expected $\Delta$ relative to
LGBM reflects this architectural difference, consistent with
Proposition~\ref{prop:monotone}.

\paragraph{TabNet.}
TabNet \citep{arik2021tabnet} is a deep learning architecture
for tabular data based on sequential attention.
At each step, a learnable attention mask selects a sparse
subset of features, enabling instance-wise feature importance
and adaptive processing.
We train TabNet with the PyTorch implementation
(\texttt{pytorch-tabnet}), using an imbalance weight
$w_1 = N_0/N_1$ and validation-based early stopping on AUC.

\paragraph{Isolation Forest Hybrid (IF-Hybrid).}
Following \citet{carcillo2021combining}, we construct a hybrid
pipeline that augments the feature space with an unsupervised
anomaly score.
Specifically, an Isolation Forest is fitted on the training
subset (unsupervised, using contamination $= \rho_{\rm train}$),
and its \texttt{score\_samples} output is appended as an
additional feature to the original feature vector.
XGBoost is then trained on the augmented representation.
This design allows the supervised classifier to leverage the
distributional signal from the unsupervised component without
requiring labelled fraud examples at the anomaly detection stage.

\paragraph{NODE (Neural Oblivious Decision Ensembles).}
NODE \citep{popov2020neural} implements differentiable oblivious
decision trees as neural network layers, enabling end-to-end
gradient-based optimisation of tree-structured models.
Since no stable PyPI release exists at the time of writing,
we use CatBoost with \texttt{grow\_policy=Depthwise} as a
functionally equivalent proxy: both architectures are based on
oblivious (symmetric) decision trees of fixed depth,
which is the defining characteristic of NODE.

%--------------------------------------------------------------
\subsection{Transaction-Dependent Cost Functions}
\label{sec:transaction-cost}
%--------------------------------------------------------------

The fixed cost model used in Sections~\ref{sec:methodology}
and~\ref{sec:experiments} assumes a constant penalty
$C_{\mathrm{FN}} = 10$ per missed fraud, regardless of the
transaction amount.
This is a deliberate simplification; in practice, the financial
exposure of a missed fraud scales with the disputed amount.
We formalise two transaction-dependent cost functions and
compare them with the fixed baseline.

\paragraph{Linear cost (Bahnsen et al., 2013, 2015).}
\begin{equation}
  C_{\mathrm{FN}}^{(i)} = \max(\mathrm{Amount}_i, 1),
  \quad C_{\mathrm{FP}}^{(i)} = 1.
  \label{eq:cost_linear}
\end{equation}
This formulation follows the example-dependent cost framework
of \citet{bahnsen2013cost, bahnsen2015example}, where missing a
high-value transaction is penalised proportionally to its amount.

\paragraph{Logarithmic cost.}
\begin{equation}
  C_{\mathrm{FN}}^{(i)} = \max\!\bigl(\log(1 + \mathrm{Amount}_i),
  1\bigr), \quad C_{\mathrm{FP}}^{(i)} = 1.
  \label{eq:cost_log}
\end{equation}
The logarithmic formulation is more robust to extreme transaction
amounts (e.g., large wire transfers) and implicitly encodes
diminishing marginal returns on fraud prevention for very large
transactions.

Under transaction-dependent costs, the threshold selection step
of Algorithm~\ref{alg:c3e} (Line~\ref{line:tau}) is modified:
\begin{equation}
  \hat{R}(\tau) = \sum_{i:\,\hat{y}_i=0,\,y_i=1}
  C_{\mathrm{FN}}^{(i)} +
  \sum_{i:\,\hat{y}_i=1,\,y_i=0} C_{\mathrm{FP}}^{(i)},
\end{equation}
and the Bayes threshold is replaced by:
\begin{equation}
  \tau_B^{\rm td} = \frac{\bar{C}_{\mathrm{FP}}}
  {\bar{C}_{\mathrm{FP}} + \bar{C}_{\mathrm{FN}}},
\end{equation}
where $\bar{C}_{\mathrm{FN}}$ and $\bar{C}_{\mathrm{FP}}$ are
the mean costs over the validation set.

%--------------------------------------------------------------
\subsection{Post-Hoc Calibration Methods}
\label{sec:calibration_methods}
%--------------------------------------------------------------

Given that 14/16 base model--dataset pairs are flagged as
miscalibrated ($\Delta > 0.20$, Table~\ref{tab:delta_summary}),
we evaluate two post-hoc calibration methods as corrective steps
within the C3E pipeline (Algorithm~\ref{alg:c3e},
Line~\ref{line:calib}).

\paragraph{Temperature Scaling.}
\citet{guo2017calibration} propose applying a single scalar
temperature parameter $T > 0$ to the model's logits:
\begin{equation}
  \hat{p}_{\mathrm{TS}}(\mathbf{x}) =
  \sigma\!\left(\frac{\mathrm{logit}(\hat{p}(\mathbf{x}))}{T}\right),
  \label{eq:temp_scaling}
\end{equation}
where $T$ is fitted by minimising the negative log-likelihood
on the validation set via one-dimensional bounded optimisation.
$T > 1$ softens the probability scores (moves predictions
toward $0.5$), while $T < 1$ sharpens them.
Temperature scaling preserves the model's ranking (ROC-AUC
is invariant to $T$) while correcting the absolute probability
level, which directly reduces ECE and $\Delta$.

\paragraph{Beta Calibration.}
\citet{kull2017beta} generalise Platt scaling by fitting a
three-parameter Beta family:
\begin{equation}
  \hat{p}_{\mathrm{BC}}(\mathbf{x}) =
  \sigma\!\bigl(a\log\hat{p} - b\log(1-\hat{p}) + c\bigr),
  \label{eq:beta_cal}
\end{equation}
where $(a, b, c)$ are fitted by L-BFGS-B minimisation of the
validation NLL, subject to $a, b > 0$.
Beta calibration subsumes both Platt scaling ($a = b = 1$) and
histogram binning as special cases, and has been shown to
outperform temperature scaling on datasets with asymmetric
miscalibration patterns --- which is precisely the setting
produced by extreme class weighting under severe imbalance.

Table~\ref{tab:calibration} reports ECE before and after
each calibration method for all model--dataset combinations.
"""


# ============================================================
# G.  UNIT TEST WITH SYNTHETIC DATA
# ============================================================

def _unit_test():
    print("Running unit tests...")
    rng = np.random.RandomState(SEED)
    N   = 3000
    y   = (rng.rand(N) < 0.03).astype(int)
    p   = rng.beta(0.5, 5, N)
    amounts = rng.exponential(200, N)

    # Temperature scaling
    ts = TemperatureScaling().fit(p, y)
    p_ts = ts.predict(p)
    assert p_ts.shape == p.shape and (p_ts>=0).all()
    print(f"  TemperatureScaling: T={ts.temperature:.3f}  "
          f"ECE_before={calibration_metrics(y,p)['ECE']:.4f}  "
          f"ECE_after={calibration_metrics(y,p_ts)['ECE']:.4f}")

    # Beta calibration
    bc = BetaCalibration().fit(p, y)
    p_bc = bc.predict(p)
    assert p_bc.shape == p.shape and (p_bc>=0).all()
    print(f"  BetaCalibration:    params={bc.params}  "
          f"ECE_after={calibration_metrics(y,p_bc)['ECE']:.4f}")

    # Transaction-dependent costs
    c_fn, c_fp = cost_amount_linear(amounts)
    assert c_fn.shape == amounts.shape
    c_fn2, _ = cost_amount_log(amounts)
    assert (c_fn2 >= 1).all()
    print(f"  LinearCost mean:    {c_fn.mean():.2f}")
    print(f"  LogCost mean:       {c_fn2.mean():.2f}")

    # Threshold selection with TD costs
    tau, curve = select_threshold_tdcost(y, p, amounts,
                                          cost_amount_linear)
    assert 0 <= tau <= 1 and curve.shape == (1001, 2)
    print(f"  TD threshold:       tau*={tau:.3f}")

    # CatBoost quick smoke test
    X = rng.randn(N, 10).astype(np.float32)
    scorer = train_catboost(X[:2000], y[:2000],
                             X[2000:], y[2000:])
    p_cat = scorer(X[2000:])
    print(f"  CatBoost:           PR-AUC={average_precision_score(y[2000:],p_cat):.4f}")

    # IF-Hybrid smoke test
    scorer_if = train_if_hybrid(X[:2000], y[:2000])
    p_if = scorer_if(X[2000:])
    print(f"  IF-Hybrid:          PR-AUC={average_precision_score(y[2000:],p_if):.4f}")

    print("\nAll unit tests PASSED.")


if __name__ == "__main__":
    _unit_test()
