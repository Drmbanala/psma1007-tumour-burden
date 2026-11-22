"""Go/no-go part 2: tumour burden by radiotracer, and within-patient repeat-scan pairs.
Uses only tumour volume and acquisition metadata - no organ-uptake data are looked at before the decision.

Usage: python 03_gonogo_by_tracer.py <seg_tumour_burden.csv> <pet_header_fields.csv> <study_table_out.csv> <pairs_out.csv> <summary.md>
"""
import sys, time
import numpy as np, pandas as pd

seg_csv, hdr_csv, study_out, pairs_out, md_out = sys.argv[1:6]
seg = pd.read_csv(seg_csv)
hdr = pd.read_csv(hdr_csv)
d = seg.merge(hdr.drop(columns=["PatientID", "StudyDate"]), on="StudyInstanceUID", how="inner", suffixes=("", "_hdr"))
assert len(d) == len(seg) == len(hdr) == 597, (len(d), len(seg), len(hdr))
d["t"] = pd.to_datetime(d["StudyDate"].astype(str)) + pd.to_timedelta(d["uptake_this_slice_min"].fillna(0), unit="m")
d = d.sort_values(["PatientID", "t"])
d.to_csv(study_out, index=False)

L = ["# Go/no-go part 2: tumour burden by radiotracer and repeat scans\n",
     f"Run: {time.strftime('%Y-%m-%d %H:%M')} | inputs: seg_tumour_burden.csv + pet_header_fields.csv | code: 02_code/03_gonogo_by_tracer.py",
     "Only tumour volume and acquisition metadata are used here; no organ uptake has been measured or looked at.\n",
     "## Header check against the data descriptor\n",
     "| | This run | Descriptor |\n|---|---|---|"]
f, g = d[d.tracer == "18F"], d[d.tracer == "68Ga"]
L.append(f"| Studies 18F / 68Ga | {len(f)} / {len(g)} | 369 / 228 |")
L.append(f"| Injected MBq 18F | {f.injected_MBq.mean():.0f} +/- {f.injected_MBq.std():.0f} ({f.injected_MBq.min():.0f}-{f.injected_MBq.max():.0f}) | 246 +/- 27 (110-341) |")
L.append(f"| Injected MBq 68Ga | {g.injected_MBq.mean():.0f} +/- {g.injected_MBq.std():.0f} ({g.injected_MBq.min():.0f}-{g.injected_MBq.max():.0f}) | 214 +/- 45 (86-324) |")
L.append(f"| Uptake min 18F (one slice) | {f.uptake_this_slice_min.mean():.0f} +/- {f.uptake_this_slice_min.std():.0f} ({f.uptake_this_slice_min.min():.0f}-{f.uptake_this_slice_min.max():.0f}) | 74 +/- 22 (6-181) |")
L.append(f"| Uptake min 68Ga (one slice) | {g.uptake_this_slice_min.mean():.0f} +/- {g.uptake_this_slice_min.std():.0f} ({g.uptake_this_slice_min.min():.0f}-{g.uptake_this_slice_min.max():.0f}) | 74 +/- 19 (7-195) |\n")

L.append("## Tumour burden by tracer\n")
L.append("| Tracer | Studies (patients) | Positive studies | Median volume, mL (IQR) | >= 250 mL | >= 500 mL | >= 1,000 mL | >= 1,355 mL |\n|---|---|---|---|---|---|---|---|")
for tr, s in d.groupby("tracer"):
    c = lambda t: f"{int((s.ttv_ml >= t).sum())} ({s[s.ttv_ml >= t].PatientID.nunique()})"
    L.append(f"| {tr} | {len(s)} ({s.PatientID.nunique()}) | {int((s.ttv_ml > 0).sum())} | {s.ttv_ml.median():.1f} ({s.ttv_ml.quantile(.25):.1f}-{s.ttv_ml.quantile(.75):.1f}) | {c(250)} | {c(500)} | {c(1000)} | {c(1355)} |")
L.append("\nCounts are studies (patients).\n")
L.append("## Tracer by scanner\n")
L.append("| Scanner | Tracer | Studies | Median volume, mL | >= 500 mL | >= 1,355 mL |\n|---|---|---|---|---|---|")
for (sc, tr), s in d.groupby(["scanner", "tracer"]):
    L.append(f"| {sc} | {tr} | {len(s)} | {s.ttv_ml.median():.1f} | {int((s.ttv_ml >= 500).sum())} | {int((s.ttv_ml >= 1355).sum())} |")

pr = []
for pid, s in d.groupby("PatientID"):
    s = s.reset_index(drop=True)
    for a in range(len(s) - 1):
        x, y = s.loc[a], s.loc[a + 1]
        pr.append(dict(PatientID=pid, tracer0=x.tracer, tracer1=y.tracer, scanner0=x.scanner, scanner1=y.scanner,
                       days=(y.t - x.t).days, ttv0=x.ttv_ml, ttv1=y.ttv_ml,
                       uptake0=x.uptake_this_slice_min, uptake1=y.uptake_this_slice_min))
pr = pd.DataFrame(pr)
pr["same_tracer"] = pr.tracer0 == pr.tracer1
pr["same_scanner"] = pr.scanner0 == pr.scanner1
pr["big"] = pr[["ttv0", "ttv1"]].max(axis=1); pr["small"] = pr[["ttv0", "ttv1"]].min(axis=1)
pr["abs_change"] = (pr.ttv1 - pr.ttv0).abs()
pr["twofold"] = (pr.big >= 2 * pr.small) & (pr.big >= 10)
pr["abs_uptake_diff"] = (pr.uptake1 - pr.uptake0).abs()
pr.to_csv(pairs_out, index=False)
L.append("\n## Consecutive repeat-scan pairs\n")
L.append(f"- All pairs: {len(pr)} in {pr.PatientID.nunique()} patients; same tracer {int(pr.same_tracer.sum())}; tracer switched {int((~pr.same_tracer).sum())}\n")
L.append("| Pair type | Pairs (patients) | >= 2-fold change | Change >= 250 mL | Larger >= 500 mL | Larger >= 1,355 mL | Median interval, days | Median uptake-time difference, min |\n|---|---|---|---|---|---|---|---|")
def row(name, s):
    if len(s) == 0:
        return f"| {name} | 0 | | | | | | |"
    return (f"| {name} | {len(s)} ({s.PatientID.nunique()}) | {int(s.twofold.sum())} | {int((s.abs_change >= 250).sum())} | "
            f"{int((s.big >= 500).sum())} | {int((s.big >= 1355).sum())} | {s.days.median():.0f} | {s.abs_uptake_diff.median():.0f} |")
for tr in ["18F", "68Ga"]:
    st = pr[pr.same_tracer & (pr.tracer0 == tr)]
    L.append(row(f"Same tracer {tr}, any scanner", st))
    L.append(row(f"Same tracer {tr}, same scanner", st[st.same_scanner]))
L.append(row("Same tracer, both tracers, same scanner", pr[pr.same_tracer & pr.same_scanner]))
L.append(row("Tracer switched", pr[~pr.same_tracer]))
open(md_out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
