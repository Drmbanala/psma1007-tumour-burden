"""P01 step 12: pre-specified analysis (protocol v1.0, section 8). Written before any organ-uptake table existed.

Usage: python 12_models.py <study_table_v3_verified.csv> <organ_uptake.csv> <out_dir>
Writes analysis_table.csv (also the input for the Stata do-file), results_models.csv and results_summary.md.
"""
import sys, time, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
SEED = 20261114
PRIMARY_ORGANS = ["liver", "spleen", "kidneys"]
COV = "uptake_c + C(scanner) + age_c + weight_c"
CUTS = [-0.001, 0, 25, 189, 532, 1355, np.inf]
CATS = ["negative", ">0-25", "25-189", "189-532", "532-1355", ">=1355"]


def pct(b):
    return (np.exp(b) - 1) * 100


def fit_mixed(d, formula, reml=True):
    last = None
    for method in ("lbfgs", "bfgs", "powell", "cg"):
        try:
            r = smf.mixedlm(formula, d, groups=d["PatientID"]).fit(reml=reml, method=method)
            if np.all(np.isfinite(r.bse_fe)):
                return r
            last = r
        except Exception as e:
            last = e
    if isinstance(last, Exception):
        raise last
    return last


def coef(r, name, label, extra):
    b, se = r.params[name], r.bse[name]
    return dict(extra, term=label, beta=b, se=se, pct_per_doubling=pct(b), ci_lo=pct(b - 1.96 * se), ci_hi=pct(b + 1.96 * se),
                p=float(r.pvalues[name]), n_scans=int(r.nobs), n_patients=int(r.model.n_groups))


def prep(a, ycol, xcol="x"):
    d = a[np.isfinite(a[ycol]) & (a[ycol] > 0) & np.isfinite(a[xcol])].copy()
    d["ly"] = np.log(d[ycol])
    d["xm"] = d.groupby("PatientID")[xcol].transform("mean")
    d["xw"] = d[xcol] - d["xm"]
    d["xx"] = d[xcol]
    for c, src in (("uptake_c", "uptake_series_min"), ("age_c", "age_years"), ("weight_c", "weight_kg")):
        d[c] = d[src] - d[src].mean()
    return d


def main_models(a, ycol, tags, xcol="x"):
    d = prep(a, ycol, xcol)
    cov = COV if d["scanner"].nunique() > 1 else "uptake_c + age_c + weight_c"
    out = []
    if len(d) < 20 or d["PatientID"].nunique() < 10:
        return out, d
    r = fit_mixed(d, f"ly ~ xx + {cov}")
    out.append(coef(r, "xx", "overall (aim A)", tags))
    if (d["xw"].abs() > 1e-9).sum() >= 10:
        r2 = fit_mixed(d, f"ly ~ xw + xm + {cov}")
        out.append(coef(r2, "xw", "within-patient (aim B)", tags))
        out.append(coef(r2, "xm", "between-patient", tags))
    return out, d


def pairs_of(d):
    rows = []
    for pid, g in d.sort_values(["StudyDate", "uptake_series_min"]).groupby("PatientID"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            rows.append(dict(PatientID=pid, dx=g.loc[i + 1, "xx"] - g.loc[i, "xx"], dly=g.loc[i + 1, "ly"] - g.loc[i, "ly"],
                             days=(pd.to_datetime(g.loc[i + 1, "StudyDate"]) - pd.to_datetime(g.loc[i, "StudyDate"])).days,
                             same_scanner=g.loc[i + 1, "scanner"] == g.loc[i, "scanner"]))
    return pd.DataFrame(rows)


def spearman_cluster(p, n_boot=2000):
    if len(p) < 8:
        return None
    rho = sps.spearmanr(p["dx"], p["dly"]).statistic
    rng = np.random.default_rng(SEED)
    ids = p["PatientID"].unique(); grp = {k: v for k, v in p.groupby("PatientID")}
    bs = []
    for _ in range(n_boot):
        s = pd.concat([grp[k] for k in rng.choice(ids, len(ids))])
        if s["dx"].nunique() > 2:
            bs.append(sps.spearmanr(s["dx"], s["dly"]).statistic)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(rho=rho, lo=lo, hi=hi, n_pairs=len(p), n_patients=len(ids))


def rcs_term(x):
    t1, t2, t3 = np.percentile(x, [10, 50, 90])
    if not (t1 < t2 < t3):
        return None
    c = lambda v: np.clip(v, 0, None) ** 3
    return (c(x - t1) - c(x - t2) * (t3 - t1) / (t3 - t2) + c(x - t3) * (t2 - t1) / (t3 - t2)) / (t3 - t1) ** 2


def icc_agreement(x, y):
    """ICC(A,1): two-way, absolute agreement, single measurement."""
    m = np.column_stack([x, y]); n, k = m.shape
    gm = m.mean(); msr = k * ((m.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((m.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def fmt(r):
    return f"{r['pct_per_doubling']:+.2f}% ({r['ci_lo']:+.2f} to {r['ci_hi']:+.2f}), p = {r['p']:.3g}; {r['n_scans']} scans, {r['n_patients']} patients"


def main():
    st, ou, outdir = sys.argv[1:4]
    S = pd.read_csv(st); O = pd.read_csv(ou)
    a = S.merge(O.drop(columns=[c for c in ("PatientID",) if c in O]), on="StudyInstanceUID", how="left", suffixes=("", "_ou"))
    a["x"] = np.log2(a["ttv_ml"] + 1)
    # N7 (deviation D2): lesion voxels inside the blanked head region read SUV 0. They are imputed at the scan's mean
    # SUV of its measurable lesion voxels, i.e. SUV-based burden is divided by (1 - blanked share); not used above 50%.
    a["blank_share"] = a["lesion_frac_suv_zero"].fillna(0).clip(0, 1) if "lesion_frac_suv_zero" in a else 0.0
    ok50 = a["blank_share"] <= 0.5
    a["tl_psma_adj"] = (a["tl_psma_g"] / (1 - a["blank_share"])).where(ok50)
    a["tlf_adj"] = (a["tlf"] / (1 - a["blank_share"])).where(ok50)
    a["x_tlpsma"] = np.log2(a["tl_psma_adj"] + 1)
    a["x_tlf"] = np.log2(a["tlf_adj"] * 100 + 0.01)
    a["x_tlpsma_raw"] = np.log2(a["tl_psma_g"] + 1)
    a["x_tlf_raw"] = np.log2(a["tlf"] * 100 + 0.01)
    a["cat"] = pd.Categorical(pd.cut(a["ttv_ml"], CUTS, labels=CATS), categories=CATS)
    lbm = 9270 * a["weight_kg"] / (6680 + 216 * a["weight_kg"] / a["height_m"] ** 2)
    a["sul_factor"] = (lbm / a["weight_kg"]).where(a["height_m"].between(1.4, 2.1))
    err = a["error"].fillna("") if "error" in a else pd.Series("", index=a.index)
    a["qc_ok"] = (err == "") & a["fov_activity_fraction"].between(0.2, 1.1) & (a.get("pet_missing_slices", 0) == 0) & (a.get("pet_grid_misfit", 0) <= 0.1)
    for org in ("liver", "spleen", "aorta"):
        ok = a[org + "_valid"].fillna(False).astype(bool) & a["qc_ok"]
        a["y_" + org] = a[org + "_suvmean"].where(ok)
        a["ymed_" + org] = a[org + "_suvmedian"].where(ok)
        a["yne_" + org] = a[org + "_noerode_suvmean"].where(ok)
        a["y10_" + org] = a[org + "_erode10mm_suvmean"].where(ok & (a[org + "_erode10mm_ml"] >= 5))
    kok = (a["kidneys_n_valid"].fillna(0) >= 1) & a["qc_ok"]
    a["y_kidneys"] = a["kidneys_suvmean"].where(kok)
    kv = lambda suffix: pd.concat([a[f"kidney_{s}_{suffix}"].where(a[f"kidney_{s}_valid"].fillna(False).astype(bool)) for s in ("right", "left")], axis=1).mean(axis=1).where(kok)
    a["ymed_kidneys"], a["yne_kidneys"], a["y10_kidneys"] = kv("suvmedian"), kv("noerode_suvmean"), kv("erode10mm_suvmean")
    a["y_liver_sphere"] = a["liver_sphere_suvmean"].where(a["qc_ok"] & a["liver_sphere_fits"].fillna(False).astype(bool))
    for org in PRIMARY_ORGANS:
        a["ysul_" + org] = a["y_" + org] * a["sul_factor"]
    window = a["uptake_series_min"].between(45, 120)
    F = a[(a["tracer"] == "18F") & window].copy()
    G = a[(a["tracer"] == "68Ga") & window].copy()
    a.to_csv(f"{outdir}/analysis_table.csv", index=False)

    res, L = [], []
    L += ["# P01 results: tumour burden and normal-organ uptake", "",
          f"Run {time.strftime('%Y-%m-%d %H:%M')} | code 02_code/12_models.py | protocol v1.0 section 8.",
          "Effects are percentage change in organ SUVmean per doubling of (tumour volume + 1 mL), with 95% CI.", "",
          "## Analysis sets", "",
          f"- [18F]PSMA-1007, uptake 45-120 min: {len(F)} scans, {F.PatientID.nunique()} patients; passing image QC: {int(F.qc_ok.sum())}.",
          f"- 68Ga-PSMA-11, uptake 45-120 min: {len(G)} scans, {G.PatientID.nunique()} patients; passing image QC: {int(G.qc_ok.sum())}.",
          f"- Lesion volume inside the blanked head region (18F set, scans with lesions): more than 1% in {int((F.blank_share[F.ttv_ml > 0] > 0.01).sum())}, more than 5% in {int((F.blank_share[F.ttv_ml > 0] > 0.05).sum())}, more than 10% in {int((F.blank_share[F.ttv_ml > 0] > 0.10).sum())} of {int((F.ttv_ml > 0).sum())}. Tumour volume is unaffected; total lesion PSMA and TLF are imputed (rule N7).",
          "- Valid organ VOIs in the 18F set: " + ", ".join(f"{o} {int(np.isfinite(F['y_' + o]).sum())}" for o in PRIMARY_ORGANS + ["aorta"]) + ".",
          "- Median organ SUVmean in the 18F set: " + ", ".join(f"{o} {F['y_' + o].median():.2f}" for o in PRIMARY_ORGANS + ["aorta"]) + ".", ""]

    # ---- primary: 18F ----
    prim = {}
    for org in PRIMARY_ORGANS:
        rows, d = main_models(F, "y_" + org, dict(set="18F primary", organ=org, variant="primary"))
        res += rows; prim[org] = (rows, d)
    for aim in ("overall (aim A)", "within-patient (aim B)"):
        rs = [r for r in res if r["set"] == "18F primary" and r["term"] == aim]
        if rs:
            adj = multipletests([r["p"] for r in rs], method="holm")[1]
            for r, p in zip(rs, adj):
                r["p_holm"] = float(p)
    L += ["## Primary analysis, [18F]PSMA-1007", "", "| Organ | Overall, aim A | Holm p | Within-patient, aim B | Holm p | Between-patient |", "|---|---|---|---|---|---|"]
    for org in PRIMARY_ORGANS:
        g = {r["term"]: r for r in prim[org][0]}
        cell = lambda k: fmt(g[k]) if k in g else "not estimable"
        hp = lambda k: f"{g[k]['p_holm']:.3g}" if k in g and "p_holm" in g[k] else ""
        L.append(f"| {org} | {cell('overall (aim A)')} | {hp('overall (aim A)')} | {cell('within-patient (aim B)')} | {hp('within-patient (aim B)')} | {cell('between-patient')} |")
    L.append("")

    # ---- categories, 30% rule, pairs, mechanism, non-linearity ----
    L += ["## Tumour-volume categories (18F), percentage difference from negative scans", "", "| Organ | " + " | ".join(CATS[1:]) + " | Scans >=1,355 mL with >=30% lower uptake than the negative-scan median |", "|---|" + "---|" * 6]
    for org in PRIMARY_ORGANS:
        d = prim[org][1]
        if len(d) < 20:
            continue
        d = d.copy(); d["cat"] = d["cat"].cat.remove_unused_categories()
        cells = []
        try:
            r = fit_mixed(d, f"ly ~ C(cat, Treatment('negative')) + {COV if d.scanner.nunique() > 1 else 'uptake_c + age_c + weight_c'}")
            for c in CATS[1:]:
                nm = f"C(cat, Treatment('negative'))[T.{c}]"
                if nm in r.params:
                    b, se = r.params[nm], r.bse[nm]
                    cells.append(f"{pct(b):+.1f}% ({pct(b - 1.96 * se):+.1f} to {pct(b + 1.96 * se):+.1f}); n={int((d.cat == c).sum())}")
                    res.append(dict(set="18F primary", organ=org, variant="category", term=c, beta=b, se=se, pct_per_doubling=pct(b), ci_lo=pct(b - 1.96 * se), ci_hi=pct(b + 1.96 * se), p=float(r.pvalues[nm]), n_scans=int((d.cat == c).sum())))
                else:
                    cells.append("n=0")
        except Exception as e:
            cells = [f"model failed: {type(e).__name__}"] + [""] * 4
        neg = d.loc[d.cat == "negative", "y_" + org].median(); top = d.loc[d.cat == ">=1355", "y_" + org]
        prop = f"{int((top <= 0.7 * neg).sum())} of {len(top)}" if len(top) and np.isfinite(neg) else "n/a"
        L.append(f"| {org} | " + " | ".join(cells) + f" | {prop} |")
    L += ["", "## Within-patient repeat scans (18F), Spearman correlation of change in log2(volume + 1) with change in log(SUVmean)", "", "| Organ | All consecutive pairs | Same scanner | Interval <= 365 days |", "|---|---|---|---|"]
    for org in PRIMARY_ORGANS:
        p = pairs_of(prim[org][1]) if len(prim[org][1]) else pd.DataFrame()
        cells = []
        for sub in (p, p[p.same_scanner] if len(p) else p, p[p.days <= 365] if len(p) else p):
            s = spearman_cluster(sub) if len(sub) else None
            cells.append(f"rho {s['rho']:+.2f} ({s['lo']:+.2f} to {s['hi']:+.2f}); {s['n_pairs']} pairs, {s['n_patients']} patients" if s else "too few")
            if s:
                res.append(dict(set="18F primary", organ=org, variant="pairs", term=f"spearman_{len(res)}", beta=s["rho"], ci_lo=s["lo"], ci_hi=s["hi"], n_scans=s["n_pairs"], n_patients=s["n_patients"]))
        L.append(f"| {org} | " + " | ".join(cells) + " |")
    L += ["", "## Mechanistic check and non-linearity (18F)", "", "| Organ | Slope of log(SUVmean) on log(1 - TLF) (1 = proportional loss) | Non-linearity, likelihood-ratio p (restricted cubic spline, 3 knots) |", "|---|---|---|"]
    for org in PRIMARY_ORGANS:
        d = prim[org][1].copy()
        if len(d) < 20:
            continue
        cov = COV if d.scanner.nunique() > 1 else "uptake_c + age_c + weight_c"
        d = d[d["tlf_adj"].notna()].copy()
        d["l1mtlf"] = np.log(1 - d["tlf_adj"].clip(upper=0.95))
        try:
            r = fit_mixed(d, f"ly ~ l1mtlf + {cov}"); b, se = r.params["l1mtlf"], r.bse["l1mtlf"]
            mech = f"{b:.2f} ({b - 1.96 * se:.2f} to {b + 1.96 * se:.2f}), p = {r.pvalues['l1mtlf']:.3g}"
            res.append(dict(set="18F primary", organ=org, variant="mechanistic", term="log(1-TLF)", beta=b, se=se, p=float(r.pvalues["l1mtlf"]), n_scans=int(r.nobs)))
        except Exception as e:
            mech = f"failed: {type(e).__name__}"
        nl = "n/a"
        t = rcs_term(d["xx"].values)
        if t is not None:
            d["xs"] = t
            try:
                l0 = fit_mixed(d, f"ly ~ xx + {cov}", reml=False).llf; l1 = fit_mixed(d, f"ly ~ xx + xs + {cov}", reml=False).llf
                nl = f"{sps.chi2.sf(max(2 * (l1 - l0), 0), 1):.3g}"
            except Exception as e:
                nl = f"failed: {type(e).__name__}"
        L.append(f"| {org} | {mech} | {nl} |")

    # ---- sensitivity ----
    Fall = a[a["tracer"] == "18F"]
    variants = [("all uptake times", Fall, "y_{o}", "x"), ("positive scans only", F[F.ttv_ml > 0], "y_{o}", "x"), ("SUVmedian", F, "ymed_{o}", "x"),
                ("no erosion", F, "yne_{o}", "x"), ("10 mm erosion", F, "y10_{o}", "x"), ("SUL (height recorded)", F, "ysul_{o}", "x"),
                ("exposure: total lesion PSMA", F, "y_{o}", "x_tlpsma"), ("exposure: total lesion fraction", F, "y_{o}", "x_tlf"),
                ("exposure: total lesion PSMA, no head imputation", F, "y_{o}", "x_tlpsma_raw"), ("exposure: total lesion fraction, no head imputation", F, "y_{o}", "x_tlf_raw"),
                ("exposure: total lesion fraction, scans with at most 5% of lesion volume in the blanked head", F[F.blank_share <= 0.05], "y_{o}", "x_tlf")]
    for sc in sorted(F["scanner"].dropna().unique()):
        variants.append((f"scanner: {sc}", F[F.scanner == sc], "y_{o}", "x"))
    L += ["", "## Sensitivity analyses (18F)", "", "| Variant | Organ | Overall | Within-patient |", "|---|---|---|---|"]
    def add_variant(name, dset, ycol, xcol, org):
        try:
            rows, _ = main_models(dset, ycol, dict(set="18F sensitivity", organ=org, variant=name), xcol)
        except Exception as e:
            L.append(f"| {name} | {org} | failed: {type(e).__name__} | |"); return
        g = {r["term"]: r for r in rows}; res.extend(rows)
        L.append(f"| {name} | {org} | {fmt(g['overall (aim A)']) if 'overall (aim A)' in g else 'not estimable'} | {fmt(g['within-patient (aim B)']) if 'within-patient (aim B)' in g else 'not estimable'} |")
    for name, dset, ypat, xcol in variants:
        for org in PRIMARY_ORGANS:
            add_variant(name, dset, ypat.format(o=org), xcol, org)
    add_variant("3 cm sphere, right liver lobe", F, "y_liver_sphere", "x", "liver")
    for org in PRIMARY_ORGANS:                       # influence: drop |standardised residual| > 3 and refit
        d = prim[org][1]
        if len(d) < 30:
            continue
        cov = COV if d.scanner.nunique() > 1 else "uptake_c + age_c + weight_c"
        try:
            r = fit_mixed(d, f"ly ~ xx + {cov}"); z = r.resid / r.resid.std()
            keep = F.loc[d.index[np.abs(z) <= 3]]
            add_variant(f"without {int((np.abs(z) > 3).sum())} outlying scans", keep, "y_" + org, "x", org)
        except Exception:
            pass
    add_variant("exploratory: blood pool", F, "y_aorta", "x", "aorta")

    # ---- 68Ga descriptive ----
    L += ["", "## 68Ga-PSMA-11 scans (descriptive)", "", "| Organ | Overall | Within-patient |", "|---|---|---|"]
    for org in PRIMARY_ORGANS:
        try:
            rows, _ = main_models(G, "y_" + org, dict(set="68Ga descriptive", organ=org, variant="primary")); g = {r["term"]: r for r in rows}; res += rows
            L.append(f"| {org} | {fmt(g['overall (aim A)']) if 'overall (aim A)' in g else 'not estimable'} | {fmt(g['within-patient (aim B)']) if 'within-patient (aim B)' in g else 'not estimable'} |")
        except Exception as e:
            L.append(f"| {org} | failed: {type(e).__name__} | |")

    # ---- technical validation: fast vs full-resolution organ maps ----
    L += ["", "## Technical validation: 3 mm vs full-resolution organ maps", ""]
    if "fr_liver_suvmean" in a:
        L += ["| Organ | Scans | ICC (absolute agreement) | Mean difference, % (95% limits of agreement) |", "|---|---|---|---|"]
        for org in ("liver", "spleen", "kidneys"):
            v = a[[org + "_suvmean" if org != "kidneys" else "kidneys_suvmean", "fr_" + (org + "_suvmean" if org != "kidneys" else "kidneys_suvmean")]].dropna()
            v = v[(v > 0).all(axis=1)]
            if len(v) >= 5:
                x, y = v.iloc[:, 0].values, v.iloc[:, 1].values; dlog = np.log(x / y)
                L.append(f"| {org} | {len(v)} | {icc_agreement(x, y):.3f} | {pct(dlog.mean()):+.2f}% ({pct(dlog.mean() - 1.96 * dlog.std(ddof=1)):+.2f} to {pct(dlog.mean() + 1.96 * dlog.std(ddof=1)):+.2f}) |")
    else:
        L.append("Full-resolution organ maps not available in this table.")
    pd.DataFrame(res).to_csv(f"{outdir}/results_models.csv", index=False)
    open(f"{outdir}/results_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
