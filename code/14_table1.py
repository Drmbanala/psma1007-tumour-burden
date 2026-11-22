"""P01 step 14: cohort description (Table 1) and analysis-set counts.
Uses only the study table (headers and expert lesion masks); no organ uptake is read.
Usage: python 14_table1.py <study_table_v3_verified.csv> <out_dir>"""
import sys
import numpy as np, pandas as pd

st, out = sys.argv[1:3]
d = pd.read_csv(st)
CUTS = [-0.001, 0, 25, 189, 532, 1355, np.inf]
CATS = ["0 (no PSMA-positive lesion)", ">0 to 25", ">25 to 189", ">189 to 532", ">532 to 1355", ">1355"]
d["cat"] = pd.cut(d["ttv_ml"], CUTS, labels=CATS)
d["window"] = d["uptake_series_min"].between(45, 120)
d["bmi_calc"] = d["weight_kg"] / d["height_m"] ** 2

def miqr(s, nd=1):
    s = s.dropna()
    return "n/a" if s.empty else f"{s.median():.{nd}f} ({s.quantile(.25):.{nd}f} to {s.quantile(.75):.{nd}f})"

def col(x):
    r = {}
    r["Scans, n"] = len(x)
    r["Patients, n"] = x.PatientID.nunique()
    r["Patients with 2 or more scans, n"] = int((x.groupby("PatientID").size() >= 2).sum())
    r["Age, years"] = miqr(x.age_years, 0)
    r["Body weight, kg"] = miqr(x.weight_kg, 0)
    r["Height recorded, n (%)"] = f"{x.height_m.notna().sum()} ({100 * x.height_m.notna().mean():.0f})"
    r["Injected activity, MBq"] = miqr(x.dose_MBq, 0)
    r["Uptake time, min"] = miqr(x.uptake_series_min, 0)
    r["Uptake time 45 to 120 min, n (%)"] = f"{int(x.window.sum())} ({100 * x.window.mean():.0f})"
    for s in sorted(d.scanner.dropna().unique()):
        n = int((x.scanner == s).sum()); r[f"Scanner: {s}, n (%)"] = f"{n} ({100 * n / len(x):.0f})"
    r["Total tumour volume, mL"] = miqr(x.ttv_ml, 1)
    r["Total tumour volume, mL, scans with lesions"] = miqr(x.ttv_ml[x.ttv_ml > 0], 1)
    for c in CATS:
        n = int((x.cat == c).sum()); r[f"Tumour volume {c} mL, n (%)"] = f"{n} ({100 * n / len(x):.0f})"
    r["Total lesion PSMA, g"] = miqr(x.tl_psma_g[x.ttv_ml > 0], 0)
    r["Lesion SUVmean (scans with lesions)"] = miqr(x.lesion_suv_mean[x.ttv_ml > 0], 1)
    return r

F, G = d[d.tracer == "18F"], d[d.tracer == "68Ga"]
P = F[F.window]
cols = {"All scans": col(d), "[18F]PSMA-1007, all": col(F), "[18F]PSMA-1007, primary analysis set": col(P), "68Ga-PSMA-11, all": col(G)}
T = pd.DataFrame(cols)
T.to_csv(f"{out}/table1_cohort.csv")
both = set(F.PatientID) & set(G.PatientID)
excl = F[~F.window]
L = ["# P01 Table 1: cohort description", "",
     "Source: `01_derived_data/study_table_v3_verified.csv` (DICOM headers and expert lesion masks only). Values are median (IQR) unless stated.", "",
     T.to_markdown(), "",
     "## Counts for the flow diagram", "",
     f"- Dataset: {len(d)} PET/CT studies from {d.PatientID.nunique()} patients; tracer other than 18F or 68Ga: {int((~d.tracer.isin(['18F', '68Ga'])).sum())}.",
     f"- [18F]PSMA-1007: {len(F)} studies, {F.PatientID.nunique()} patients. 68Ga-PSMA-11: {len(G)} studies, {G.PatientID.nunique()} patients. Patients imaged with both tracers: {len(both)}.",
     f"- 18F studies outside the 45 to 120 min uptake window: {len(excl)} (below 45 min: {int((excl.uptake_series_min < 45).sum())}; above 120 min: {int((excl.uptake_series_min > 120).sum())}; uptake time missing: {int(excl.uptake_series_min.isna().sum())}).",
     f"- Primary analysis set before image QC: {len(P)} studies, {P.PatientID.nunique()} patients; patients contributing 2 or more scans: {int((P.groupby('PatientID').size() >= 2).sum())}; scans from those patients: {int(P.groupby('PatientID').PatientID.transform('size').ge(2).sum())}.",
     f"- In-field activity fraction outside 0.2 to 1.1 (QC rule N6): all scans {int((~d.fov_activity_fraction.between(0.2, 1.1)).sum())}; primary set {int((~P.fov_activity_fraction.between(0.2, 1.1)).sum())}.", ""]
open(f"{out}/table1_cohort.md", "w").write("\n".join(L))
print("\n".join(L))
