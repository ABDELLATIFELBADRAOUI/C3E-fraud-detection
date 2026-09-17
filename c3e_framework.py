# ============================================================
# c3e_framework.py  —  Single-file, zero-import-error version
# Chronological Cost-Calibrated Evaluation (C3E)
# Paper: "Beyond ROC-AUC: A Calibration-Gap Diagnostic
#         Protocol for Cost-Sensitive Fraud Detection"
# Authors: Elbadraoui et al.
# Usage:
#   python c3e_framework.py                        (all datasets)
#   python c3e_framework.py --datasets creditcard  (one dataset)
# ============================================================

from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline
from sklearn.isotonic        import IsotonicRegression
from sklearn.metrics         import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# 0.  CONFIGURATION
# ============================================================

@dataclass
class DatasetSpec:
    name:        str
    path:        str
    time_col:    str
    label_col:   str
    drop_cols:   List[str] = field(default_factory=list)

DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "creditcard": DatasetSpec(
        name="creditcard",
        path=r"D:\fraud_data\creditcard.csv",
        time_col="Time",
        label_col="Class",
    ),
    "ieee_cis": DatasetSpec(
        name="ieee_cis",
        # path points to transaction table; identity table is auto-merged
        path=r"D:\fraud_data\IEEE_CIS\train_transaction.csv",
        time_col="TransactionDT",
        label_col="isFraud",
        drop_cols=["TransactionID"],
    ),
    "baf": DatasetSpec(
        name="baf",
        path=r"D:\fraud_data\BAF NeurIPS 2022\Base.csv",
        time_col="month",
        label_col="fraud_bool",
    ),
    "paysim": DatasetSpec(
        name="paysim",
        path=r"D:\fraud_data\PaySim.csv",
        time_col="step",
        label_col="isFraud",
        # NOTE: balance columns must NOT be used (see dataset doc)
        drop_cols=["nameOrig", "nameDest", "isFlaggedFraud",
                   "oldbalanceOrg", "newbalanceOrig",
                   "oldbalanceDest", "newbalanceDest"],
    ),
}

CFN_DEFAULT   = 10.0
CFP_DEFAULT   =  1.0
DELTA_THRESH  =  0.20   # miscalibration flag (Definition 2)
ALPHA, BETA, GAMMA = 0.60, 0.20, 0.20
SEED          = 42


# ============================================================
# 1.  DATA LOADING
# ============================================================

def load_dataset(dataset_id: str) -> Tuple[pd.DataFrame, DatasetSpec]:
    """
    Load a registered fraud detection dataset.
    Encodes categoricals, sorts chronologically.
    Raises FileNotFoundError with a helpful message if missing.
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_id}'. "
            f"Available: {list(DATASET_REGISTRY)}"
        )
    spec = DATASET_REGISTRY[dataset_id]
    p = Path(spec.path)
    if not p.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset file not found: {p.resolve()}\n"
            f"  Chemins attendus sur ta machine :\n"
            f"  creditcard -> D:\\fraud_data\\creditcard.csv\n"
            f"  ieee_cis   -> D:\\fraud_data\\IEEE_CIS\\train_transaction.csv\n"
            f"  baf        -> D:\\fraud_data\\BAF NeurIPS 2022\\Base.csv\n"
            f"  paysim     -> D:\\fraud_data\\PaySim.csv\n"
        )

    df = pd.read_csv(p, low_memory=False)

    # IEEE-CIS: merge identity table (left join on TransactionID)
    if dataset_id == "ieee_cis":
        identity_path = Path(spec.path).parent / "train_identity.csv"
        if identity_path.exists():
            df_id = pd.read_csv(identity_path, low_memory=False)
            df = df.merge(df_id, on="TransactionID", how="left")
            print(f"  [ieee_cis] identity merged: {df_id.shape[1]-1} extra features")
        else:
            print("  [ieee_cis] identity table not found — using transaction only")
    df = df.drop(columns=spec.drop_cols, errors="ignore")

    # Encode object columns (IEEE-CIS has many)
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.Categorical(df[col]).codes.astype(np.float32)
        df[col] = df[col].replace(-1, np.nan)

    # Mandatory chronological sort
    df = df.sort_values(spec.time_col).reset_index(drop=True)

    rho     = df[spec.label_col].mean()
    n_fraud = int(df[spec.label_col].sum())
    print(f"\n[load] {spec.name:12s}  N={len(df):>10,}  "
          f"N_fraud={n_fraud:>7,}  rho={rho:.4%}")
    return df, spec


# ============================================================
# 2.  CHRONOLOGICAL SPLIT  (Algorithm 1, Lines 1-2)
# ============================================================

def chronological_split(
    df: pd.DataFrame,
    label_col: str,
    alpha: float = ALPHA,
    beta:  float = BETA,
    gamma: float = GAMMA,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-free chronological 3-way split.
    df must already be sorted by timestamp.
    """
    assert abs(alpha + beta + gamma - 1.0) < 1e-9
    n       = len(df)
    i_train = int(n * alpha)
    i_val   = int(n * (alpha + beta))

    splits_  = {
        "train": df.iloc[:i_train],
        "val":   df.iloc[i_train:i_val],
        "test":  df.iloc[i_val:],
    }
    for name, s in splits_.items():
        print(f"  [{name:5s}]  N={len(s):>10,}  "
              f"rho={s[label_col].mean():.4%}")

    return (splits_["train"].copy(),
            splits_["val"].copy(),
            splits_["test"].copy())


# ============================================================
# 3.  PREPROCESSING  (Algorithm 1, Line 3)
# ============================================================

def prepare_splits(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    label_col: str,
    time_col:  str,
) -> dict:
    """
    Fit scaler/imputer on train ONLY; transform all three.
    Returns dict with X_train/val/test, y_train/val/test.
    """
    drop = [label_col, time_col]

    def to_X(df: pd.DataFrame) -> np.ndarray:
        return df.drop(columns=drop, errors="ignore") \
                 .values.astype(np.float32)

    X_tr  = to_X(df_train)
    X_val = to_X(df_val)
    X_te  = to_X(df_test)

    y_tr  = df_train[label_col].values.astype(np.int32)
    y_val = df_val[label_col].values.astype(np.int32)
    y_te  = df_test[label_col].values.astype(np.int32)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_tr  = pipe.fit_transform(X_tr)   # fit HERE only
    X_val = pipe.transform(X_val)       # transform only
    X_te  = pipe.transform(X_te)        # transform only

    return dict(
        X_train=X_tr,  y_train=y_tr,
        X_val=X_val,   y_val=y_val,
        X_test=X_te,   y_test=y_te,
        preprocessor=pipe,
    )


# ============================================================
# 4.  CALIBRATION  (Algorithm 1, Line 6)
# ============================================================

class PlattCalibrator:
    def __init__(self):
        self._lr = LogisticRegression(C=1.0, solver="lbfgs",
                                      max_iter=1000)
    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(scores.reshape(-1, 1), y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self._ir = IsotonicRegression(out_of_bounds="clip")
    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._ir.fit(scores, y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._ir.predict(scores)


# ============================================================
# 5.  MODELS  (Algorithm 1, Line 4)
# ============================================================

def _spw(y: np.ndarray) -> float:
    """scale_pos_weight = N_neg / N_pos"""
    n1 = int(y.sum())
    return (len(y) - n1) / max(n1, 1)

def _class_weight_dict(y: np.ndarray) -> dict:
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    return {0: len(y)/(2*n0), 1: len(y)/(2*n1)}


def train_lr(X_tr, y_tr) -> Callable:
    cw = _class_weight_dict(y_tr)
    m  = LogisticRegression(C=0.1, class_weight=cw,
                             solver="lbfgs", max_iter=1000,
                             random_state=SEED)
    m.fit(X_tr, y_tr)
    return lambda X: m.predict_proba(X)[:, 1]


def train_rf(X_tr, y_tr) -> Callable:
    m = RandomForestClassifier(
        n_estimators=500, n_jobs=-1,
        random_state=SEED,
    )
    m.fit(X_tr, y_tr)
    return lambda X: m.predict_proba(X)[:, 1]


def train_xgb(X_tr, y_tr) -> Callable:
    m = XGBClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        scale_pos_weight=_spw(y_tr),
        eval_metric="aucpr",
        random_state=SEED, n_jobs=-1,
        verbosity=0,
    )
    m.fit(X_tr, y_tr)
    return lambda X: m.predict_proba(X)[:, 1]


def train_lgbm(X_tr, y_tr) -> Callable:
    m = LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        is_unbalance=True,
        random_state=SEED, n_jobs=-1,
        verbose=-1,
    )
    m.fit(X_tr, y_tr)
    return lambda X: m.predict_proba(X)[:, 1]


def train_dnn(X_tr, y_tr, X_val, y_val,
              epochs=50, patience=5,
              batch_size=2048, lr=1e-3,
              dropout=0.3) -> Callable:
    """
    DNN from Elbadraoui et al. (SOIC 2026).
    Falls back gracefully if TensorFlow is not installed.
    """
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        cw  = _class_weight_dict(y_tr)
        d   = X_tr.shape[1]
        inp = tf.keras.Input(shape=(d,))
        x   = tf.keras.layers.Dense(128, activation="relu")(inp)
        x   = tf.keras.layers.Dropout(dropout)(x)
        x   = tf.keras.layers.Dense(64,  activation="relu")(x)
        x   = tf.keras.layers.Dropout(dropout)(x)
        x   = tf.keras.layers.Dense(32,  activation="relu")(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
        )

        best_pr, best_w, wait = -1.0, None, 0
        for epoch in range(1, epochs + 1):
            model.fit(X_tr, y_tr, epochs=1,
                      batch_size=batch_size,
                      class_weight=cw, verbose=0)
            p_v = model.predict(X_val, verbose=0).ravel()
            pr  = average_precision_score(y_val, p_v)
            if pr > best_pr:
                best_pr, best_w, wait = pr, model.get_weights(), 0
            else:
                wait += 1
            if wait >= patience:
                print(f"    [DNN] early stop @ epoch {epoch}  "
                      f"best val PR-AUC={best_pr:.4f}")
                break
        model.set_weights(best_w)
        return lambda X: model.predict(X, verbose=0).ravel()

    except ImportError:
        print("    [DNN] TensorFlow not found — skipping DNN.")
        return None


MODEL_TRAINERS: Dict[str, Callable] = {
    "LR":   train_lr,
    "RF":   train_rf,
    "XGB":  train_xgb,
    "LGBM": train_lgbm,
    "DNN":  train_dnn,
}


# ============================================================
# 6.  THRESHOLD SELECTION  (Algorithm 1, Lines 9-11)
# ============================================================

def select_threshold(
    y_val:  np.ndarray,
    p_val:  np.ndarray,
    cfn:    float = CFN_DEFAULT,
    cfp:    float = CFP_DEFAULT,
    n_grid: int   = 1001,
) -> Tuple[float, np.ndarray]:
    """
    Scan uniform grid; return (tau_star, cost_curve).
    cost_curve shape: (n_grid, 2) — columns [tau, cost].
    """
    grid  = np.linspace(0.0, 1.0, n_grid)
    costs = np.empty(n_grid, dtype=np.float64)

    # Vectorised computation — avoids Python loop overhead
    # for large validation sets (IEEE-CIS, PaySim)
    p_col = p_val.reshape(1, -1)          # (1, N_val)
    g_col = grid.reshape(-1, 1)           # (n_grid, 1)
    preds = (p_col >= g_col).astype(np.int8)  # (n_grid, N_val)
    y_row = y_val.reshape(1, -1)          # (1, N_val)

    fn_counts = ((preds == 0) & (y_row == 1)).sum(axis=1)
    fp_counts = ((preds == 1) & (y_row == 0)).sum(axis=1)
    costs     = cfn * fn_counts + cfp * fp_counts

    tau_star  = float(grid[np.argmin(costs)])
    return tau_star, np.column_stack([grid, costs])


# ============================================================
# 7.  METRICS  (Algorithm 1, Lines 7-8, 12)
# ============================================================

@dataclass
class C3EResult:
    dataset:       str
    model:         str
    # Threshold-free
    roc_auc:       float
    pr_auc:        float
    # Diagnostic
    delta:         float    # Δ = ROC-AUC − PR-AUC  (on val)
    miscalibrated: bool     # Δ > δ* = 0.20
    # Threshold analysis
    tau_star:      float    # empirical cost-optimal threshold
    tau_bayes:     float    # theoretical Bayes threshold 1/(1+r)
    tau_gap:       float    # |tau_star − tau_bayes|
    # Classification metrics at tau_star
    precision_1:   float
    recall_1:      float
    f1_1:          float
    tp:            int
    fp:            int
    fn:            int
    tn:            int
    expected_cost: float
    cfn:           float
    cfp:           float


def compute_metrics(
    dataset:  str,
    model:    str,
    y_val:    np.ndarray,
    p_val:    np.ndarray,
    y_test:   np.ndarray,
    p_test:   np.ndarray,
    tau_star: float,
    cfn: float = CFN_DEFAULT,
    cfp: float = CFP_DEFAULT,
) -> C3EResult:
    # Threshold-free on test
    roc_auc = roc_auc_score(y_test, p_test)
    pr_auc  = average_precision_score(y_test, p_test)

    # Delta on validation (before test evaluation — Algorithm 1 L7)
    delta         = roc_auc_score(y_val, p_val) \
                  - average_precision_score(y_val, p_val)
    miscalibrated = delta > DELTA_THRESH

    # Bayes threshold and gap
    r         = cfn / cfp
    tau_bayes = 1.0 / (1.0 + r)
    tau_gap   = abs(tau_star - tau_bayes)

    # Threshold-dependent on test
    y_pred = (p_test >= tau_star).astype(int)
    cm     = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    cost = cfn * fn + cfp * fp

    return C3EResult(
        dataset=dataset, model=model,
        roc_auc=roc_auc, pr_auc=pr_auc,
        delta=delta, miscalibrated=miscalibrated,
        tau_star=tau_star, tau_bayes=tau_bayes, tau_gap=tau_gap,
        precision_1 = precision_score(y_test, y_pred, zero_division=0),
        recall_1    = recall_score(y_test, y_pred, zero_division=0),
        f1_1        = f1_score(y_test, y_pred, zero_division=0),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        expected_cost=float(cost),
        cfn=cfn, cfp=cfp,
    )


# ============================================================
# 8.  FULL EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    dataset_ids: List[str],
    model_ids:   List[str],
    cfn:  float = CFN_DEFAULT,
    cfp:  float = CFP_DEFAULT,
    with_platt: bool = True,
    output_csv: str  = "results/c3e_results.csv",
) -> pd.DataFrame:
    """
    Full C3E loop over all (dataset × model) combinations.
    Saves results to CSV and returns the DataFrame.
    """
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    all_results: List[C3EResult] = []

    for ds_id in dataset_ids:
        # --- Load & split ---
        df, spec = load_dataset(ds_id)
        df_tr, df_val, df_te = chronological_split(
            df, spec.label_col
        )
        splits = prepare_splits(
            df_tr, df_val, df_te,
            spec.label_col, spec.time_col,
        )
        X_tr,  y_tr  = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"],   splits["y_val"]
        X_te,  y_te  = splits["X_test"],  splits["y_test"]

        for m_id in model_ids:
            print(f"\n  >> [{ds_id}] Training {m_id} ...")
            trainer = MODEL_TRAINERS[m_id]

            # Train
            if m_id == "DNN":
                scorer = trainer(X_tr, y_tr, X_val, y_val)
            else:
                scorer = trainer(X_tr, y_tr)

            if scorer is None:          # DNN skipped (no TF)
                continue

            p_val = scorer(X_val)
            p_te  = scorer(X_te)

            # --- DNN + Platt variant ---
            if m_id == "DNN" and with_platt:
                cal       = PlattCalibrator().fit(p_val, y_val)
                p_val_cal = cal.predict(p_val)
                p_te_cal  = cal.predict(p_te)
                tau_p, _  = select_threshold(y_val, p_val_cal, cfn, cfp)
                res_p     = compute_metrics(
                    ds_id, "DNN-Platt",
                    y_val, p_val_cal,
                    y_te,  p_te_cal,
                    tau_p, cfn, cfp,
                )
                all_results.append(res_p)
                _print_row(res_p)

            # --- Base model ---
            tau_star, _ = select_threshold(y_val, p_val, cfn, cfp)
            res = compute_metrics(
                ds_id, m_id,
                y_val, p_val,
                y_te,  p_te,
                tau_star, cfn, cfp,
            )
            all_results.append(res)
            _print_row(res)

    df_out = pd.DataFrame([asdict(r) for r in all_results])
    df_out.to_csv(output_csv, index=False)
    print(f"\n[done] Results saved to {output_csv}")
    return df_out


def _print_row(r: C3EResult) -> None:
    flag = " *** MISCALIBRATED" if r.miscalibrated else ""
    print(
        f"     Δ={r.delta:.3f}  tau*={r.tau_star:.3f}  "
        f"tau_Bayes={r.tau_bayes:.3f}  gap={r.tau_gap:.3f}  "
        f"cost={r.expected_cost:.0f}  "
        f"PR-AUC={r.pr_auc:.4f}{flag}"
    )


# ============================================================
# 9.  LATEX TABLE GENERATOR
# ============================================================

def results_to_latex(df: pd.DataFrame,
                     out_path: str = "results/table_main.tex") -> str:
    """
    Generate the main results table as LaTeX (booktabs style).
    One row per (dataset, model); columns match the paper.
    """
    cols_display = [
        "dataset", "model",
        "roc_auc", "pr_auc", "delta", "miscalibrated",
        "tau_star", "tau_gap",
        "precision_1", "recall_1", "f1_1",
        "expected_cost",
    ]
    col_headers = [
        "Dataset", "Model",
        "ROC-AUC", "PR-AUC", r"$\Delta$", "Miscal.",
        r"$\hat{\tau}^*$", r"$|\hat{\tau}^*-\tau_B|$",
        r"Prec$_1$", r"Rec$_1$", r"F$_1$",
        "Cost",
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main results under the C3E protocol "
        r"($C_{\mathrm{FN}}=10$, $C_{\mathrm{FP}}=1$, $r=10$). "
        r"Rows marked \checkmark\ in the Miscal.\ column have "
        r"$\Delta > 0.20$. Best cost per dataset in \textbf{bold}.}",
        r"\label{tab:main_results}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * (len(col_headers) - 2) + "}",
        r"\toprule",
        " & ".join(col_headers) + r" \\",
        r"\midrule",
    ]

    for ds_id, grp in df.groupby("dataset", sort=False):
        best_cost = grp["expected_cost"].min()
        lines.append(r"\addlinespace[2pt]")
        for _, row in grp.iterrows():
            miscal = r"\checkmark" if row["miscalibrated"] else ""
            cost_str = f"{row['expected_cost']:.0f}"
            if row["expected_cost"] == best_cost:
                cost_str = r"\textbf{" + cost_str + "}"
            cells = [
                row["dataset"], row["model"],
                f"{row['roc_auc']:.4f}",
                f"{row['pr_auc']:.4f}",
                f"{row['delta']:.3f}",
                miscal,
                f"{row['tau_star']:.3f}",
                f"{row['tau_gap']:.3f}",
                f"{row['precision_1']:.4f}",
                f"{row['recall_1']:.4f}",
                f"{row['f1_1']:.4f}",
                cost_str,
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(tex)
    print(f"[latex] Table saved to {out_path}")
    return tex


# ============================================================
# 10.  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C3E Framework — Chronological Cost-Calibrated Evaluation"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(DATASET_REGISTRY.keys()),
        choices=list(DATASET_REGISTRY.keys()),
        help="Datasets to run (default: all four)",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["LR", "RF", "XGB", "LGBM", "DNN"],
        choices=["LR", "RF", "XGB", "LGBM", "DNN"],
        help="Models to run (default: all five)",
    )
    parser.add_argument("--cfn", type=float, default=CFN_DEFAULT)
    parser.add_argument("--cfp", type=float, default=CFP_DEFAULT)
    parser.add_argument("--no_platt", action="store_true",
                        help="Disable Platt calibration for DNN")
    parser.add_argument("--output", default="results/c3e_results.csv")
    args = parser.parse_args()

    results_df = run_experiment(
        dataset_ids = args.datasets,
        model_ids   = args.models,
        cfn         = args.cfn,
        cfp         = args.cfp,
        with_platt  = not args.no_platt,
        output_csv  = args.output,
    )

    print("\n" + "="*70)
    print(results_df[[
        "dataset", "model", "roc_auc", "pr_auc",
        "delta", "miscalibrated",
        "tau_star", "tau_gap", "expected_cost",
    ]].to_string(index=False))

    results_to_latex(results_df)
