"""P01 post hoc sensitivity analysis of the kidney result (written 2026-11-15, after the final analysis).

Purpose: urinary obstruction is frequent in advanced prostate cancer and lowers renal tracer uptake.
The images were not read for hydronephrosis, so two objective checks that need no image reading are used:
  K1  symmetric kidneys: both kidneys valid and (lower / higher per-kidney SUVmean) >= 0.80
  K2  higher kidney: outcome = SUVmean of the kidney with the higher uptake
The criteria were fixed in this file before any of these models was run. Models are those of 12_models.py.

Usage: python 21_kidney_asymmetry_sensitivity.py <analysis_table.csv> <organ_uptake.csv> <12_models.py> <out dir>
"""
import sys, importlib.util
import numpy as np, pandas as pd
at, ou, mpath, out = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("m12", mpath); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a = pd.read_csv(at, low_memory=False); o = pd.read_csv(ou)
k = o[["StudyInstanceUID", "kidney_right_suvmean", "kidney_left_suvmean", "kidney_right_valid", "kidney_left_valid", "kidney_right_raw_ml", "kidney_left_raw_ml"]]
a = a.drop(columns=[c for c in k.columns if c in a.columns and c != "StudyInstanceUID"]).merge(k, on="StudyInstanceUID", how="left")
a = a[(a.tracer == "18F") & a.uptake_series_min.between(45, 120)].copy()
rv, lv = a.kidney_right_valid.astype(bool), a.kidney_left_valid.astype(bool)
r, l = a.kidney_right_suvmean.where(rv), a.kidney_left_suvmean.where(lv)
a["k_ratio"] = np.minimum(r, l) / np.maximum(r, l)
a["k_sym"] = rv & lv & (a.k_ratio >= 0.80)
a["y_kidney_high"] = np.fmax(r, l)
a["k_vol_max"] = np.fmax(a.kidney_right_raw_ml, a.kidney_left_raw_ml)
edges = [-1, 0, 25, 189, 532, 1355, np.inf]
a["cat"] = pd.cut(a.ttv_ml, edges, labels=m.CATS).astype(str)
rows = []
def run(name, d, ycol):
    res, dd = m.main_models(d, ycol, dict(analysis=name))
    for x in res: rows.append(x)
    dd = dd.copy(); cov = m.COV if dd.scanner.nunique() > 1 else "uptake_c + age_c + weight_c"
    rr = m.fit_mixed(dd, f"ly ~ C(cat, Treatment('negative')) + {cov}")
    nm = "C(cat, Treatment('negative'))[T.>=1355]"
    if nm in rr.params.index:
        b, se = rr.params[nm], rr.bse[nm]
        neg = dd.loc[dd.cat == "negative", ycol].median(); top = dd.loc[dd.cat == ">=1355", ycol]
        rows.append(dict(analysis=name, term=">=1355 mL vs negative", beta=b, se=se, pct_per_doubling=m.pct(b), ci_lo=m.pct(b - 1.96 * se), ci_hi=m.pct(b + 1.96 * se), p=float(rr.pvalues[nm]), n_scans=int(len(top)), n_patients=np.nan,
                         note=f"{int((top <= 0.7 * neg).sum())} of {len(top)} at or below 0.7 x negative median; negatives n={int((dd.cat == 'negative').sum())}"))
run("primary (as reported)", a, "y_kidneys")
run("K1 symmetric kidneys only", a[a.k_sym], "y_kidneys")
run("K2 kidney with higher uptake", a, "y_kidney_high")
pd.DataFrame(rows).to_csv(out + "/kidney_sensitivity_models.csv", index=False)
d = a[np.isfinite(a.y_kidneys)]
desc = d.groupby("cat").agg(n=("k_sym", "size"), asymmetric_or_single=("k_sym", lambda s: int((~s).sum())), median_ratio=("k_ratio", "median"), median_max_kidney_ct_ml=("k_vol_max", "median")).reindex(m.CATS)
desc.to_csv(out + "/kidney_asymmetry_by_category.csv")
pd.set_option("display.width", 250); print(pd.DataFrame(rows)[["analysis", "term", "pct_per_doubling", "ci_lo", "ci_hi", "p", "n_scans", "n_patients", "note"]].round(4).to_string()); print(desc.round(3))
