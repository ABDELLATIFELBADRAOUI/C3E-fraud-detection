# -*- coding: utf-8 -*-
"""Regenerate Fig. 1, 2, 3, 4, 7, 8, 9, 10, 11 of the JDSA manuscript from the
objects that produce the tables (no value typed in by hand).

Inputs
  results/tab10_pairs_labelled.csv   26 pairs, five-seed means (Table 5)  -> Fig 1, 3, 4, 7, 8
  results/tab7_extended_d1.csv       seed 42, saved scores (Table 7)       -> Fig 9, 10
  results/scores/scores_d1_seed42.npz raw scores + amounts (Table 9)       -> Fig 11
Outputs
  figures/FigN.pdf (+ .png)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
RES, OUT = ROOT / "results", ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
    "savefig.dpi": 300, "pdf.fonttype": 42, "figure.dpi": 110,
})

CFN, CFP, N_GRID, TAU_B, DELTA_STAR = 10.0, 1.0, 1001, 1 / 11, 0.20
MODEL_ORDER = ["LR", "RF", "XGB", "LGBM", "CatBoost"]
CORE = ["LR", "RF", "XGB", "LGBM"]
SIX = ["LR", "RF", "XGB", "LGBM", "CatBoost", "IF-Hybrid"]
# Okabe-Ito, validated (CVD-safe) for the five-model set
MODEL_COLOR = {"LR": "#0072B2", "RF": "#E69F00", "XGB": "#009E73", "LGBM": "#D55E00",
               "CatBoost": "#CC79A7", "IF-Hybrid": "#999999"}
DS_ORDER = ["creditcard", "ieee_cis", "baf", "paysim", "elliptic", "giveme"]
DS_LABEL = {"creditcard": "D1 creditcard", "ieee_cis": "D2 IEEE-CIS", "baf": "D3 BAF",
            "paysim": "D4 PaySim", "elliptic": "D5 Elliptic", "giveme": "D6 GiveMeSomeCredit"}
DS_COLOR = dict(zip(DS_ORDER, ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]))
DS_MARKER = dict(zip(DS_ORDER, ["o", "s", "^", "D", "v", "P"]))
RHO = {"creditcard": 0.173, "ieee_cis": 3.499, "baf": 1.103, "paysim": 0.129,
       "elliptic": 9.761, "giveme": 6.684}                       # Table 1, in %
CAL = [("none", "raw"), ("temperature", "TS"), ("beta", "Beta")]
CAL_COLOR = {"none": "#0072B2", "temperature": "#D55E00", "beta": "#009E73"}
REG_COLOR = {"fixed": "#0072B2", "linear": "#D55E00", "log": "#009E73"}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print("  [ok]", name)

# --------------------------------------------------------------------------- data
P = pd.read_csv(RES / "tab10_pairs_labelled.csv")
assert len(P) == 26, len(P)
P = P[P.model.isin(MODEL_ORDER)].copy()
T7 = pd.read_csv(RES / "tab7_extended_d1.csv").set_index("model").reindex(SIX)
Z = np.load(RES / "scores" / "scores_d1_seed42.npz")
y_val, y_te = Z["y_val"].astype(int), Z["y_te"].astype(int)
null_cost = float(CFN * y_te.sum())

# =============================== Fig. 1 ======================================
fig, ax = plt.subplots(figsize=(6.4, 3.3))
for i, m in enumerate(MODEL_ORDER):
    for _, r in P[P.model == m].iterrows():
        yy = i + (DS_ORDER.index(r.dataset) - 2.5) * 0.07        # small dodge: overlapping thresholds stay visible
        ax.scatter(r.tau_star, yy, s=46, color=DS_COLOR[r.dataset], marker=DS_MARKER[r.dataset],
                   edgecolor="black", linewidth=0.4, zorder=3)
ax.axvline(TAU_B, ls="--", color="grey", lw=1)
ax.text(TAU_B + 0.012, len(MODEL_ORDER) - 0.55, r"$\tau_B = %.3f$" % TAU_B, color="grey", fontsize=8)
ax.set_yticks(range(len(MODEL_ORDER))); ax.set_yticklabels(MODEL_ORDER)
ax.set_ylim(-0.6, len(MODEL_ORDER) - 0.4); ax.set_xlim(0, 1.03)
ax.set_xlabel(r"Seed-mean cost-optimal threshold $\hat{\tau}^*$")
ax.legend(handles=[Line2D([], [], marker=DS_MARKER[d], ls="", color=DS_COLOR[d],
                          markeredgecolor="black", markeredgewidth=0.4, label=DS_LABEL[d])
                   for d in DS_ORDER], ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
save(fig, "Fig1")

# =============================== Fig. 2 ======================================
# Compact layout: the drawing fills the figure (no in-figure title, the caption carries
# it), x and y share one scale (1 unit = 1/13.5 in), and the key sits beside the flag box.
C_TR, C_VAL, C_TE, C_ALL = "#2b6cb0", "#c77800", "#2f8f4e", "#6b6b6b"
W_IN, X_MAX, Y_MAX = 7.4, 100.0, 37.2
fig = plt.figure(figsize=(W_IN, W_IN * Y_MAX / X_MAX))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.grid(False)
ax.set_xlim(0, X_MAX); ax.set_ylim(0, Y_MAX)
ax.text(0.8, 36.1, "Chronological order  →", fontsize=7.5, color="grey", va="center")
BAR_Y0, BAR_H = 30.6, 4.2
for lab, x0, x1, col in [(r"$\mathcal{D}_{\rm tr}$  TRAIN 60%", 0, 60, C_TR),
                         (r"$\mathcal{D}_{\rm val}$  VAL 20%", 60, 80, C_VAL),
                         (r"$\mathcal{D}_{\rm te}$  TEST 20%", 80, 100, C_TE)]:
    ax.add_patch(Rectangle((x0 + 0.8, BAR_Y0), (x1 - x0) - 1.6, BAR_H, facecolor=col, edgecolor="none"))
    ax.text((x0 + x1) / 2, BAR_Y0 + BAR_H / 2, lab, ha="center", va="center", color="white",
            fontsize=8.2, fontweight="bold")
steps = [  # (number, box text, data read, colour, what it prevents)
    (1, "Sort by\ntimestamp",                          "all rows",                   C_ALL, "no future→past\nleakage"),
    (2, "Split\n60 / 20 / 20",                         "all rows",                   C_ALL, "time-ordered,\nnot random"),
    (3, "Fit\npreprocessing\non TRAIN",                r"$\mathcal{D}_{\rm tr}$ only", C_TR, "no preprocessing\nleakage"),
    (4, "Train $f$\non TRAIN",                         r"$\mathcal{D}_{\rm tr}$ only", C_TR, "test block never\nused in fitting"),
    (5, "Recalibrate\non VAL,\ncompute $\\Delta$",     r"$\mathcal{D}_{\rm val}$",     C_VAL, "calibrate before\nthresholding"),
    (6, "Select $\\hat{\\tau}^*$\non VAL,\nfreeze",     r"$\mathcal{D}_{\rm val}$",     C_VAL, "locks the\noperating point"),
    (7, "Evaluate at\nfrozen $\\hat{\\tau}^*$\non TEST", r"$\mathcal{D}_{\rm te}$, once", C_TE, "no test-set\noverfitting"),
]
xs = np.linspace(7.2, 92.8, 7); bw, bh, y0, PAD = 12.6, 9.6, 17.0, 0.25
for (n, title, data, col, prevents), x in zip(steps, xs):
    ax.add_patch(FancyBboxPatch((x - bw / 2, y0), bw, bh, boxstyle=f"round,pad={PAD}",
                                facecolor="white", edgecolor=col, linewidth=1.4))
    ax.add_patch(Circle((x - bw / 2 + 1.5, y0 + bh - 1.5), 1.3, facecolor=col, edgecolor="none"))
    ax.text(x - bw / 2 + 1.5, y0 + bh - 1.5, str(n), color="white", ha="center", va="center",
            fontsize=7, fontweight="bold")
    ax.text(x, y0 + bh / 2 - 0.5, title, ha="center", va="center", fontsize=7.0, linespacing=1.15)
    ax.text(x, y0 - 0.9, data, ha="center", va="top", fontsize=7.2, color=col)
    ax.text(x, y0 - 3.4, prevents, ha="center", va="top", fontsize=6.3, color="#a33",
            style="italic", linespacing=1.05)
for xa, xb in zip(xs[:-1], xs[1:]):
    ax.add_patch(FancyArrowPatch((xa + bw / 2 + PAD, y0 + bh / 2), (xb - bw / 2 - PAD, y0 + bh / 2),
                                 arrowstyle="-|>", mutation_scale=8, color="#444", lw=0.9,
                                 shrinkA=0, shrinkB=0))
# training-only zone around steps 3-4; its label sits inside the zone, under the notes
zx0 = (xs[1] + xs[2]) / 2          # zone borders run midway between neighbouring boxes
zx1 = (xs[3] + xs[4]) / 2
zb, zt = 8.4, y0 + bh + PAD + 0.8
ax.add_patch(Rectangle((zx0, zb), zx1 - zx0, zt - zb, fill=False, ls="--", edgecolor="#c0392b", lw=1.1))
ax.text((zx0 + zx1) / 2, zb + 0.35, r"training-only fitting zone ($\mathcal{D}_{\rm tr}$ only)",
        ha="center", va="bottom", fontsize=7, color="#c0392b")
for i, ((n, _, _, col, _), x) in enumerate(zip(steps, xs)):
    end = zt + 0.25 if i in (2, 3) else y0 + bh + PAD + 0.3   # steps 3-4: into the zone
    ax.add_patch(FancyArrowPatch((x, BAR_Y0 - 0.3), (x, end), arrowstyle="-|>", mutation_scale=7,
                                 color=col, lw=0.7, linestyle=(0, (2, 2)), shrinkA=0, shrinkB=0))
# bottom band: the key on the left, the flag box under steps 5-7
fx0, fx1, fy, fh = 58.6, 99.2, 0.9, 6.8
ax.add_patch(FancyBboxPatch((fx0, fy), fx1 - fx0, fh, boxstyle=f"round,pad={PAD}",
                            facecolor="#fdf2e9", edgecolor="#c77800", lw=1.1))
ax.text((fx0 + fx1) / 2, fy + fh / 2 + 1.2, "if $\\Delta > \\delta^\\dagger$  →  flag for threshold review",
        ha="center", va="center", fontsize=7.4)
ax.text((fx0 + fx1) / 2, fy + fh / 2 - 1.3, "advisory only — does not halt the pipeline",
        ha="center", va="center", fontsize=6.6, color="#8a5a00", style="italic")
ax.add_patch(FancyArrowPatch((xs[4], y0 - 6.2), (xs[4], fy + fh + PAD + 0.1), arrowstyle="-|>",
                             mutation_scale=8, color="#c77800", lw=0.9, shrinkA=0, shrinkB=0))
key = ax.text(0.8, fy + 0.6, "Key: $\\Delta$ = ROC-AUC $-$ PR-AUC is rank-invariant (Prop. 3.1). A strictly\n"
              "increasing recalibration cannot change $\\Delta$; it can change which grid\n"
              "thresholds are reachable (Prop. 3.3). $\\mathcal{D}_{\\rm te}$ is read once, at step 7.",
              fontsize=6.9, color="#333", va="bottom", ha="left", linespacing=1.25)
fig.canvas.draw()
_inv = ax.transData.inverted()
_kb = _inv.transform(key.get_window_extent())
assert _kb[1][0] < fx0 - PAD - 0.5 and _kb[1][1] < zb, ("key overlaps", _kb)
save(fig, "Fig2")

# =============================== Fig. 3 ======================================
def fmt_delta(v):
    """Three decimals; four when the value was stored to four decimals and ends in 5,
    where rounding to three decimals would be ambiguous (LR on Elliptic, 0.4025)."""
    r4 = round(v * 1e4)
    if abs(v * 1e4 - r4) < 1e-6 and r4 % 10 == 5:
        return f"{v:.4f}"
    return f"{v:.3f}"

M = P.pivot(index="dataset", columns="model", values="delta").reindex(index=DS_ORDER, columns=MODEL_ORDER)
fig, ax = plt.subplots(figsize=(6.4, 3.9)); ax.grid(False)
im = ax.imshow(M.to_numpy(dtype=float), cmap="viridis", vmin=0.0, vmax=0.9, aspect="auto")
ax.set_xticks(range(len(MODEL_ORDER))); ax.set_xticklabels(MODEL_ORDER)
ax.set_yticks(range(len(DS_ORDER))); ax.set_yticklabels([DS_LABEL[d] for d in DS_ORDER])
for i, d in enumerate(DS_ORDER):
    for j, m in enumerate(MODEL_ORDER):
        v = M.loc[d, m]
        if pd.isna(v):
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="white", edgecolor="none"))
            ax.text(j, i, "n/a", ha="center", va="center", color="grey", fontsize=7)
            continue
        ax.text(j, i, fmt_delta(v), ha="center", va="center", fontsize=8,
                color="white" if v < 0.55 else "black")
        if v > DELTA_STAR:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="red", lw=1.5))
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"$\Delta$ = ROC-AUC $-$ PR-AUC (validation, five-seed mean)")
n_flag = int((P.delta > DELTA_STAR).sum())
ax.set_title(rf"ROC–PR divergence per pair; red outline: flagged ($\Delta > {DELTA_STAR}$), {n_flag} of {len(P)}")
save(fig, "Fig3")

# =============================== Fig. 4 ======================================
fig, ax = plt.subplots(figsize=(6.3, 3.8))
for m in MODEL_ORDER:
    sub = P[P.model == m]
    ax.scatter(sub.delta, sub.cost, s=42, color=MODEL_COLOR[m], edgecolor="black",
               linewidth=0.4, label=m, zorder=3)
ax.set_yscale("log"); ax.axvline(DELTA_STAR, ls="--", color="grey", lw=1)
ax.text(DELTA_STAR + 0.01, ax.get_ylim()[1] * 0.55, r"$\delta^\dagger$", color="grey", fontsize=8)
ax.set_xlabel(r"$\Delta$ (validation, five-seed mean)")
ax.set_ylabel("Expected test cost\n(five-seed mean, log scale)")
ax.set_xlim(0, 0.95)
for _, r in P[P["case"] == "costly"].iterrows():
    ax.annotate(f"{DS_LABEL[r.dataset].split(' ', 1)[1]} / {r.model}", (r.delta, r.cost),
                xytext=(-7, 5), textcoords="offset points", ha="right", fontsize=7.5)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "Fig4")

# =============================== Fig. 7 ======================================
G = P.groupby("dataset").agg(mean_delta=("delta", "mean"), n=("delta", "size")).reindex(DS_ORDER)
G["rho"] = [RHO[d] for d in G.index]
fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.scatter(G.rho, G.mean_delta, s=58, color="#0072B2", edgecolor="black", linewidth=0.4, zorder=3)
OFF = {"paysim": (7, -12), "baf": (7, 5), "creditcard": (7, -12), "ieee_cis": (7, 4),
       "giveme": (7, 4), "elliptic": (-7, 5)}
for d, r in G.iterrows():
    dx, dy = OFF.get(d, (7, 4))
    ax.annotate(f"{DS_LABEL[d]} (n={int(r.n)})", (r.rho, r.mean_delta), xytext=(dx, dy),
                textcoords="offset points", ha="right" if dx < 0 else "left", fontsize=7.8)
ax.axhline(DELTA_STAR, ls="--", color="grey", lw=1)
ax.text(10.9, DELTA_STAR + 0.015, r"$\delta^\dagger$", color="grey", fontsize=8, ha="right")
ax.set_xlabel(r"Fraud prevalence $\rho$ (%)")
ax.set_ylabel(r"Mean $\Delta$ over the dataset's pairs (Table 5)")
ax.set_xlim(-0.4, 11.0); ax.set_ylim(0.0, 0.85)
save(fig, "Fig7")
print("     Fig7 per-dataset mean Delta:", G.mean_delta.round(3).to_dict())

# =============================== Fig. 8 ======================================
core = P[P.model.isin(CORE)].copy()
core["rank"] = core.groupby("dataset")["cost"].rank(method="average")
mr = core.groupby("model")["rank"].mean().reindex(CORE).sort_values()
fig, ax = plt.subplots(figsize=(5.2, 2.5)); ax.grid(axis="y", visible=False)
order = list(mr.index[::-1])
ax.barh(order, [mr[m] for m in order], color=[MODEL_COLOR[m] for m in order], height=0.6)
for i, m in enumerate(order):
    ax.text(mr[m] + 0.05, i, f"{mr[m]:.2f}", va="center", fontsize=8)
ax.set_xlim(0, 4.4); ax.set_xlabel("Mean cost rank across the six datasets (lower is better)")
save(fig, "Fig8")
print("     Fig8 mean cost ranks:", mr.round(2).to_dict())

# =============================== Fig. 9 ======================================
x = np.arange(len(T7)); w = 0.26
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.2), constrained_layout=True)
for k, (c, lab) in enumerate(CAL):
    a1.bar(x + (k - 1) * w, T7[f"ECE_{c}"], w, label=lab, color=CAL_COLOR[c])
    a2.bar(x + (k - 1) * w, T7[f"cost_{c}"], w, label=lab, color=CAL_COLOR[c])
    for i, m in enumerate(T7.index):
        if int(T7.loc[m, f"flag_{c}"]) == 0:
            a2.text(x[i] + (k - 1) * w, T7.loc[m, f"cost_{c}"] * 1.12, "flags 0", rotation=90,
                    ha="center", va="bottom", fontsize=6.5)
a2.set_yscale("log"); a2.axhline(null_cost, ls="--", color="grey", lw=1)
a2.text(x[0] - 0.45, null_cost * 1.07, f"null model = {null_cost:,.0f}", ha="left", fontsize=7.2, color="grey")
a2.set_ylim(100, max(T7[[f"cost_{c}" for c, _ in CAL]].max().max() * 2.2, null_cost * 2.2))
for k, (c, lab) in enumerate(CAL):                       # direct value labels
    for i, m in enumerate(T7.index):
        v = float(T7.loc[m, f"cost_{c}"])
        if int(T7.loc[m, f"flag_{c}"]) != 0:
            a2.text(x[i] + (k - 1) * w, v * 1.04, f"{v:,.0f}", rotation=90, ha="center",
                    va="bottom", fontsize=5.8, color="#333")
for a, t, yl in ((a1, "(a) Expected calibration error (validation, 10 bins)", "ECE"),
                 (a2, "(b) Expected test cost at the frozen threshold", "Cost (log scale)")):
    a.set_xticks(x); a.set_xticklabels(T7.index, rotation=20); a.set_title(t); a.set_ylabel(yl)
a1.legend(title="Calibrator", loc="upper left")
save(fig, "Fig9")

# =============================== Fig. 10 =====================================
fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(8.8, 2.9), constrained_layout=True)
for a, m, tag in ((b1, "XGB", "a"), (b2, "LGBM", "b")):
    vals = [float(T7.loc[m, f"cost_{c}"]) for c, _ in CAL]
    bars = a.bar([lab for _, lab in CAL], vals, color=[CAL_COLOR[c] for c, _ in CAL], width=0.6)
    for bar, v, (c, _) in zip(bars, vals, CAL):
        extra = "\n(flags 0)" if int(T7.loc[m, f"flag_{c}"]) == 0 else ""
        a.text(bar.get_x() + bar.get_width() / 2, v * 1.02, f"{v:,.0f}{extra}", ha="center",
               va="bottom", fontsize=7.3)
    a.set_ylim(0, max(vals) * 1.3); a.set_ylabel("Test cost"); a.set_title(f"({tag}) {m}")
b2.axhline(null_cost, ls="--", color="grey", lw=1)
b2.set_title(f"(b) LGBM  (dashed: null-model cost {null_cost:,.0f})")
for m in ("XGB", "LGBM"):
    b3.plot([lab for _, lab in CAL], [float(T7.loc[m, f"tau_{c}"]) for c, _ in CAL], marker="o",
            color=MODEL_COLOR[m], label=m)
b3.axhline(0.5, ls="--", color="grey", lw=1); b3.set_ylim(-0.03, 1.06)
b3.set_title(r"(c) frozen $\hat{\tau}^*$, XGB and LGBM, vs naive 0.5"); b3.set_ylabel(r"$\hat{\tau}^*$"); b3.legend()
save(fig, "Fig10")

# =============================== Fig. 11 =====================================
def cost_at(y, p, tau, w_fn):
    yp = p >= tau
    return float(np.sum(w_fn[(y == 1) & (~yp)]) + CFP * np.sum((y == 0) & yp))
grid = np.linspace(0.0, 1.0, N_GRID)
W = {"fixed":  (np.full(len(y_val), CFN), np.full(len(y_te), CFN)),
     "linear": (np.maximum(Z["amount_val"], 1.0), np.maximum(Z["amount_te"], 1.0)),
     "log":    (np.log1p(Z["amount_val"]), np.log1p(Z["amount_te"]))}
rows = []
for m in SIX:
    pv, pt = Z[f"{m}_val"], Z[f"{m}_test"]
    r = {"model": m}
    for reg, (wv, wt) in W.items():
        tau = float(grid[int(np.argmin([cost_at(y_val, pv, t, wv) for t in grid]))])
        r[f"tau_{reg}"] = tau; r[f"cost_{reg}"] = cost_at(y_te, pt, tau, wt)
    rows.append(r)
T11 = pd.DataFrame(rows).set_index("model")
T11.to_csv(RES / "fig11_cost_regimes_d1.csv")
print(T11.round(2).to_string())
assert all((T11.cost_linear > T11.cost_fixed) & (T11.cost_fixed > T11.cost_log)), "ordering linear>fixed>log fails"
x = np.arange(len(T11)); w = 0.26
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for k, reg in enumerate(["fixed", "linear", "log"]):
    ax.bar(x + (k - 1) * w, T11[f"cost_{reg}"], w, label=reg, color=REG_COLOR[reg])
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(T11.index, rotation=20)
ax.set_ylabel("Expected test cost at the frozen threshold (log scale)")
ax.legend(title="Cost regime", ncol=3, loc="upper left")
save(fig, "Fig11")
print("Done.")
