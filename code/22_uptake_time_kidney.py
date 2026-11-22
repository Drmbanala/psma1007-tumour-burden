"""P01 post hoc analysis of uptake time (injection to scan), suggested by a co-author (E.S.) on 2026-11-17
after the first complete draft. Models are those of 12_models.py (imported unchanged).
Usage: python 22_uptake_time_kidney.py <analysis_table.csv> <12_models.py> <out dir>
"""
import sys, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
at, mpath, out = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("m12", mpath); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a = pd.read_csv(at, low_memory=False); a = a[(a.tracer == "18F") & a.uptake_series_min.between(45, 120)].copy()
u = a.uptake_series_min; rows = []
rows.append(dict(item="uptake time, min", estimate=u.median(), lo=u.quantile(.25), hi=u.quantile(.75), note=f"median (IQR); range {u.min():.0f} to {u.max():.0f}; n={len(a)}"))
rows.append(dict(item="Spearman uptake time vs log2(PSMA-TV+1)", estimate=a[["uptake_series_min", "x"]].corr(method="spearman").iloc[0, 1]))
rest = m.COV.replace("uptake_c + ", "")
for org in ["kidneys", "liver", "spleen", "aorta"]:
    d = m.prep(a, "y_" + org)
    r = m.fit_mixed(d, f"ly ~ xx + {m.COV}"); b, se = r.params["uptake_c"], r.bse["uptake_c"]
    rows.append(dict(item=f"{org}: % change in SUVmean per 10 min later imaging", estimate=m.pct(10 * b), lo=m.pct(10 * (b - 1.96 * se)), hi=m.pct(10 * (b + 1.96 * se)), p=float(r.pvalues["uptake_c"])))
    r2 = m.fit_mixed(d, f"ly ~ xx*uptake_c + {rest}")
    rows.append(dict(item=f"{org}: interaction PSMA-TV x uptake time", estimate=r2.params["xx:uptake_c"], p=float(r2.pvalues["xx:uptake_c"])))
d = m.prep(a, "y_kidneys"); med = d.uptake_series_min.median()
for nm, s in [("at or below median uptake time", d[d.uptake_series_min <= med]), ("above median uptake time", d[d.uptake_series_min > med])]:
    o = m.main_models(s, "y_kidneys", {})[0][0]
    rows.append(dict(item=f"kidneys: % per doubling of PSMA-TV, studies {nm}", estimate=o["pct_per_doubling"], lo=o["ci_lo"], hi=o["ci_hi"], p=o["p"], note=f"{o['n_scans']} studies, {o['n_patients']} patients"))
pd.DataFrame(rows).to_csv(out + "/uptake_time_models.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())
