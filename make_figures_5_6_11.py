# -*- coding: utf-8 -*-
"""Redraw Fig. 5, 6 and 11 of the JDSA manuscript from the released result files
(no value typed in by hand). Same style as make_figures.py (release v2.1). Run it after make_figures.py.

Inputs (release v2.1, folder results/)
  c3e_seed_aggregated.csv            five-seed means (Table 5)           -> Fig 5
  c3e_bootstrap_ci_all.csv           seed-42 bootstrap CIs (Table 11)     -> Fig 6
  scores/scores_d1_seed42.npz        saved D1 scores + amounts (Table 9)  -> Fig 11
Changes with respect to v10
  Fig 5  : dataset names as in the text, lettering legible at print size
  Fig 6  : the four models of Table 11 only (the former Fig 6 carried CatBoost and
           IF-Hybrid values from an earlier run, 186 and 187 instead of 231 and 201)
  Fig 11 : y-axis label no longer truncated, legend no longer over the bars
Usage: python make_figures_5_6_11.py <release_root> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures"
RES = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
    "savefig.dpi": 300, "pdf.fonttype": 42, "figure.dpi": 110,
})
CFN, CFP, N_GRID = 10.0, 1.0, 1001
MODEL_ORDER = ["LR", "RF", "XGB", "LGBM", "CatBoost"]
SIX = ["LR", "RF", "XGB", "LGBM", "CatBoost", "IF-Hybrid"]
MODEL_COLOR = {"LR": "#0072B2", "RF": "#E69F00", "XGB": "#009E73", "LGBM": "#D55E00",
               "CatBoost": "#CC79A7", "IF-Hybrid": "#999999"}
DS_ORDER = ["creditcard", "ieee_cis", "baf", "paysim", "elliptic", "giveme"]
DS_LABEL = {"creditcard": "D1 creditcard", "ieee_cis": "D2 IEEE-CIS", "baf": "D3 BAF",
            "paysim": "D4 PaySim", "elliptic": "D5 Elliptic", "giveme": "D6 GiveMeSomeCredit"}
RHO = {"creditcard": 0.1727, "ieee_cis": 3.4990, "baf": 1.1029, "paysim": 0.1291,
       "elliptic": 9.7608, "giveme": 6.6840}          # in %, notebook [load] output (Table 1)
REG_COLOR = {"fixed": "#0072B2", "linear": "#D55E00", "log": "#009E73"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  [ok]", name)


def mean_of(cell):                       # "0.9820 ± 0.0000" -> 0.9820
    return float(str(cell).split("±")[0])


# =============================== Fig. 5 ======================================
A = pd.read_csv(RES / "c3e_seed_aggregated.csv")
A = A[A.model.isin(MODEL_ORDER)].copy()
for c in ("roc_auc", "pr_auc"):
    A[c] = A[c].map(mean_of)
assert len(A) == 26, len(A)
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), constrained_layout=True)
for ax, ds in zip(axes.ravel(), DS_ORDER):
    d = A[A.dataset == ds]
    xr = d.roc_auc.max() - d.roc_auc.min()
    yr = d.pr_auc.max() - d.pr_auc.min()
    xlo, xhi = d.roc_auc.min() - max(0.25 * xr, 0.01), d.roc_auc.max() + max(0.25 * xr, 0.01)
    rho = RHO[ds] / 100
    ylo = max(0.0, min(d.pr_auc.min(), rho) - max(0.2 * yr, 0.01))
    yhi = d.pr_auc.max() + max(0.2 * yr, 0.01)
    xs = np.linspace(xlo, xhi, 200)
    ax.fill_between(xs, ylo, np.clip(xs - 0.20, ylo, yhi), color="#D55E00", alpha=0.10, lw=0)
    ax.axhline(rho, ls=":", color="grey", lw=1)
    for _, r in d.iterrows():
        ax.scatter(r.roc_auc, r.pr_auc, s=38, color=MODEL_COLOR[r.model], edgecolor="black",
                   linewidth=0.4, zorder=3)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_title(DS_LABEL[ds]); ax.set_xlabel("ROC-AUC"); ax.set_ylabel("PR-AUC")
    ax.tick_params(labelsize=8)
handles = [Line2D([], [], marker="o", ls="", color=MODEL_COLOR[m], markeredgecolor="black",
                  markeredgewidth=0.4, label=m) for m in MODEL_ORDER]
handles += [Patch(color="#D55E00", alpha=0.10, label="ROC-AUC $-$ PR-AUC $> 0.20$"),
            Line2D([], [], ls=":", color="grey", label=r"prevalence $\rho$")]
fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.07),
           handletextpad=0.3, columnspacing=0.9)
save(fig, "Fig5")

# =============================== Fig. 6 ======================================
B = pd.read_csv(RES / "c3e_bootstrap_ci_all.csv")
B = B[B.dataset == "D1"].set_index("model").reindex(["LR", "RF", "XGB", "LGBM"])
assert B.notna().all().all()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)
x = np.arange(len(B))
a1.bar(x, B.pr_auc_mean, 0.6, color="#0072B2",
       yerr=[B.pr_auc_mean - B.pr_ci_lo, B.pr_ci_hi - B.pr_auc_mean], capsize=3,
       error_kw={"lw": 1})
a2.bar(x, B.cost_mean, 0.6, color="#D55E00",
       yerr=[B.cost_mean - B.cost_ci_lo, B.cost_ci_hi - B.cost_mean], capsize=3,
       error_kw={"lw": 1})
for i, m in enumerate(B.index):
    a1.text(i, B.pr_ci_hi[m] + 0.02, f"{B.pr_auc_mean[m]:.3f}", ha="center", va="bottom", fontsize=7.5)
    a2.text(i, B.cost_ci_hi[m] * 1.08, f"{B.cost_mean[m]:,.0f}", ha="center", va="bottom", fontsize=7.5)
a1.set_ylim(0, 1.0); a2.set_yscale("log"); a2.set_ylim(80, 5000)
for a, t, yl in ((a1, "(a) PR-AUC", "PR-AUC"), (a2, "(b) Expected test cost", "Cost (log scale)")):
    a.set_xticks(x); a.set_xticklabels(B.index); a.set_title(t); a.set_ylabel(yl)
save(fig, "Fig6")

# =============================== Fig. 11 =====================================
Z = np.load(RES / "scores" / "scores_d1_seed42.npz")
y_val, y_te = Z["y_val"].astype(int), Z["y_te"].astype(int)


def cost_at(y, p, tau, w_fn):
    yp = p >= tau
    return float(np.sum(w_fn[(y == 1) & (~yp)]) + CFP * np.sum((y == 0) & yp))


grid = np.linspace(0.0, 1.0, N_GRID)
W = {"fixed": (np.full(len(y_val), CFN), np.full(len(y_te), CFN)),
     "linear": (np.maximum(Z["amount_val"], 1.0), np.maximum(Z["amount_te"], 1.0)),
     "log": (np.log1p(Z["amount_val"]), np.log1p(Z["amount_te"]))}
rows = []
for m in SIX:
    pv, pt = Z[f"{m}_val"], Z[f"{m}_test"]
    r = {"model": m}
    for reg, (wv, wt) in W.items():
        tau = float(grid[int(np.argmin([cost_at(y_val, pv, t, wv) for t in grid]))])
        r[f"tau_{reg}"] = tau
        r[f"cost_{reg}"] = cost_at(y_te, pt, tau, wt)
    rows.append(r)
T11 = pd.DataFrame(rows).set_index("model")
ref = pd.read_csv(RES / "fig11_cost_regimes_d1.csv").set_index("model")
assert np.allclose(T11.round(2).values, ref[T11.columns].round(2).values), "Fig 11 values differ from the released CSV"
print(T11.round(2).to_string())
x = np.arange(len(T11)); w = 0.26
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for k, reg in enumerate(["fixed", "linear", "log"]):
    ax.bar(x + (k - 1) * w, T11[f"cost_{reg}"], w, label=reg, color=REG_COLOR[reg])
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(T11.index)
ax.set_ylabel("Test cost (log scale)")
ax.legend(title="Cost regime", ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "Fig11")
print("Done.")
