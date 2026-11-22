"""P01 step 18: manuscript figures from the analysis table written by step 12.
Usage: python 18_figures.py <results folder with analysis_table.csv> <output folder>
Colour encodes scanner in a fixed order and is always doubled by marker shape; the three hues were checked
for colour-vision-deficiency separation (all checks passed) before use.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m12", HERE / "12_models.py"); m12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m12)

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SCANNERS = [("Biograph mCT Flow 20", "#2a78d6", "o"), ("Discovery 690", "#e0762b", "s"), ("Biograph 64-4R TruePoint", "#1a9a77", "^")]
TRACERS = [("18F", "[18F]PSMA-1007", "#2a78d6", "o"), ("68Ga", "68Ga-PSMA-11", "#e0762b", "s")]
ORGANS = [("liver", "Liver"), ("spleen", "Spleen"), ("kidneys", "Kidneys"), ("aorta", "Aortic blood pool")]
CATS = ["negative", ">0-25", "25-189", "189-532", "532-1355", ">=1355"]
CAT_LAB = ["0", ">0-25", "25-189", "189-532", "532-1,355", "≥1,355"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.dpi": 300, "figure.dpi": 100})

a = pd.read_csv(RES / "analysis_table.csv")
win = a.uptake_series_min.between(45, 120) & a.qc_ok.astype(bool)
F, G = a[(a.tracer == "18F") & win].copy(), a[(a.tracer == "68Ga") & win].copy()


def save(fig, name):
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("wrote", name)


# Figure 2: organ uptake against tumour volume, [18F]PSMA-1007
fig, axs = plt.subplots(1, 4, figsize=(11.0, 3.1), sharex=True)
xt = [0, 10, 100, 1000]
for ax, (org, lab) in zip(axs, ORGANS):
    y = F["y_" + org]
    for name, col, mk in SCANNERS:
        s = F.scanner == name
        ax.scatter(F.ttv_ml[s] + 1, y[s], s=11, marker=mk, facecolor=col, edgecolor="white", linewidth=0.3, alpha=0.75, label=name, zorder=2)
    cx = [np.median(F.ttv_ml[F["cat"] == c] + 1) for c in CATS]
    med = [np.nanmedian(y[F["cat"] == c]) for c in CATS]
    q1 = [np.nanpercentile(y[F["cat"] == c], 25) for c in CATS]; q3 = [np.nanpercentile(y[F["cat"] == c], 75) for c in CATS]
    ax.errorbar(cx, med, yerr=[np.subtract(med, q1), np.subtract(q3, med)], color=INK, lw=1.2, marker="D", ms=4, mfc="white", mec=INK, capsize=2, zorder=3, label="Median and IQR by volume category")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(lab, fontsize=9, color=INK, loc="left")
    ax.set_xticks([v + 1 for v in xt]); ax.set_xticklabels([str(v) if v < 1000 else "1,000" for v in xt]); ax.minorticks_off()
    yt = [v for v in ([0.5, 1, 2, 4] if org == "aorta" else [2, 5, 10, 20, 40]) if y.min() * 0.8 <= v <= y.max() * 1.2]
    ax.set_yticks(yt); ax.set_yticklabels([str(v) for v in yt]); ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("PSMA-positive tumour volume (mL)")
axs[0].set_ylabel("SUVmean")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=1.4)
fig.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.27, wspace=0.22)
save(fig, "figure2_uptake_vs_tumour_volume_18F")

# Figure 3: within-patient change, [18F]PSMA-1007 (consecutive scan pairs)
fig, axs = plt.subplots(1, 4, figsize=(11.0, 3.0), sharex=True, sharey=True)
for ax, (org, lab) in zip(axs, ORGANS):
    d = m12.prep(F, "y_" + org); p = m12.pairs_of(d)
    for same, col, mk, nm in [(True, "#2a78d6", "o", "Both scans on the same scanner"), (False, "#e0762b", "s", "Different scanners")]:
        s = p.same_scanner == same
        ax.scatter(p.dx[s], (np.exp(p.dly[s]) - 1) * 100, s=13, marker=mk, facecolor=col, edgecolor="white", linewidth=0.3, alpha=0.8, label=nm, zorder=2)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1); ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_title(f"{lab} ({len(p)} pairs)", fontsize=9, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("Change in tumour volume (doublings)")
axs[0].set_ylabel("Change in SUVmean (%)"); axs[0].set_ylim(-80, 200)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5)
fig.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.27, wspace=0.10)
save(fig, "figure3_within_patient_change_18F")

# Figure 4: both tracers, median uptake by volume category relative to scans without lesions (descriptive)
fig, axs = plt.subplots(1, 4, figsize=(11.0, 3.0), sharex=True, sharey=True)
for ax, (org, lab) in zip(axs, ORGANS):
    for (tr, nm, col, mk), D in zip(TRACERS, [F, G]):
        ref = np.nanmedian(D.loc[D["cat"] == "negative", "y_" + org])
        n = [int((D["cat"] == c).sum()) for c in CATS]
        rel = [100 * np.nanmedian(D.loc[D["cat"] == c, "y_" + org]) / ref if k else np.nan for c, k in zip(CATS, n)]
        ax.plot(range(len(CATS)), rel, color=col, marker=mk, ms=5, lw=1.6, mec="white", mew=0.5, label=nm, zorder=3)
    ax.axhline(100, color=MUTED, lw=0.6, zorder=1); ax.axhline(70, color=MUTED, lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.set_title(lab, fontsize=9, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CAT_LAB, rotation=35, ha="right"); ax.set_xlabel("PSMA-positive tumour volume (mL)")
axs[0].set_ylabel("Median SUVmean\n(% of scans without lesions)"); axs[0].set_ylim(0, 160)
axs[0].annotate("70%", (0, 70), xytext=(2, 2), textcoords="offset points", fontsize=7, color=MUTED)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5)
fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.36, wspace=0.10)
save(fig, "figure4_both_tracers_by_volume_category")
