"""Whole-body tumour volume and lesion count from the expert DICOM SEG masks
of PSMA-PET-CT-Lesions (NCI Imaging Data Commons v25, CC BY 4.0).

Usage: python 01_tumour_burden_from_seg.py <raw_collection_dir> <idc_series_index.csv> <out.csv> <summary.md>

Volume = voxels in the union of all segments x (pixel spacing row x col x slice spacing).
Lesions = 3D connected components with 18-connectivity (as in the dataset descriptor).
Read-only on the raw data.
"""
import glob, os, sys, time
import numpy as np, pandas as pd, pydicom
from scipy import ndimage

raw_root, index_csv, out_csv, out_md = sys.argv[1:5]

def first(owner, name):
    seq = getattr(owner, name, None) if owner is not None else None
    return seq[0] if seq else None

rows = []
files = sorted(glob.glob(os.path.join(raw_root, "*", "*", "SEG_*", "*.dcm")))
print(f"{time.strftime('%H:%M:%S')} SEG files found: {len(files)}", flush=True)
struct18 = ndimage.generate_binary_structure(3, 2)
for i, fp in enumerate(files, 1):
    rec = {"SegSeriesInstanceUID": os.path.basename(os.path.dirname(fp))[4:]}
    try:
        ds = pydicom.dcmread(fp)
        rec.update(PatientID=str(ds.PatientID), StudyInstanceUID=str(ds.StudyInstanceUID),
                   SegSeriesDescription=str(ds.get("SeriesDescription", "")),
                   SegmentationType=str(ds.get("SegmentationType", "")),
                   n_segments=len(ds.SegmentSequence), n_frames=int(ds.get("NumberOfFrames", 1)),
                   rows=int(ds.Rows), cols=int(ds.Columns))
        ref = first(ds, "ReferencedSeriesSequence")
        rec["RefSeriesInstanceUID"] = str(ref.SeriesInstanceUID) if ref is not None else ""
        shared = first(ds, "SharedFunctionalGroupsSequence")
        frames = ds.PerFrameFunctionalGroupsSequence
        pm = first(shared, "PixelMeasuresSequence") or first(frames[0], "PixelMeasuresSequence")
        po = first(shared, "PlaneOrientationSequence") or first(frames[0], "PlaneOrientationSequence")
        iop = np.array([float(v) for v in po.ImageOrientationPatient])
        normal = np.cross(iop[:3], iop[3:]); normal /= np.linalg.norm(normal)
        sp = [float(v) for v in pm.PixelSpacing]
        thick = float(pm.SliceThickness) if "SliceThickness" in pm else np.nan
        sbs = float(pm.SpacingBetweenSlices) if "SpacingBetweenSlices" in pm else np.nan
        pos = np.array([float(np.dot(normal, [float(v) for v in first(f, "PlanePositionSequence").ImagePositionPatient])) for f in frames])
        upos = np.unique(np.round(pos, 2))
        dz_frames = float(np.min(np.diff(upos))) if len(upos) > 1 else np.nan
        # Slice spacing: the header SliceThickness is not the slice spacing on every scanner
        # (Biograph mCT Flow: SliceThickness 3.0 mm, slices every 2.0 mm), so the spacing measured
        # from the frame positions wins whenever it differs from the header by more than 0.05 mm.
        hdr = sbs if np.isfinite(sbs) else thick
        if np.isfinite(dz_frames) and (not np.isfinite(hdr) or abs(dz_frames - hdr) > 0.05):
            dz, dz_src = dz_frames, "frame positions"
        else:
            dz, dz_src = hdr, "header"
        arr = ds.pixel_array
        if arr.ndim == 2:
            arr = arr[None]
        if rec["SegmentationType"].upper() == "FRACTIONAL":
            arr = arr >= 0.5 * float(ds.get("MaximumFractionalValue", 255))
        else:
            arr = arr > 0
        k = np.round((pos - pos.min()) / dz).astype(int)
        misfit = float(np.max(np.abs((pos - pos.min()) / dz - k)))
        fg = np.where(arr.reshape(arr.shape[0], -1).any(axis=1))[0]
        n_vox, n_les, n_raw = 0, 0, int(arr.sum())
        if len(fg):
            sub = arr[fg]
            ys = np.where(sub.any(axis=(0, 2)))[0]; xs = np.where(sub.any(axis=(0, 1)))[0]
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            z0 = k[fg].min()
            vol = np.zeros((k[fg].max() - z0 + 1, y1 - y0, x1 - x0), bool)
            for fi in fg:
                vol[k[fi] - z0] |= arr[fi, y0:y1, x0:x1]
            n_vox = int(vol.sum())
            _, n_les = ndimage.label(vol, structure=struct18)
        vox_ml = sp[0] * sp[1] * dz / 1000.0
        rec.update(px_row_mm=sp[0], px_col_mm=sp[1], slice_thickness_mm=thick, spacing_between_slices_mm=sbs,
                   dz_from_frames_mm=dz_frames, dz_used_mm=dz, dz_source=dz_src, slice_grid_misfit=misfit,
                   n_voxels_raw=n_raw, n_voxels=n_vox, voxel_ml=vox_ml,
                   ttv_ml=n_vox * vox_ml, n_lesions=int(n_les), error="")
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rows.append(rec)
    if i % 50 == 0 or i == len(files):
        print(f"{time.strftime('%H:%M:%S')} {i}/{len(files)}", flush=True)

df = pd.DataFrame(rows)
idx = pd.read_csv(index_csv)
pt = idx[idx["Modality"] == "PT"][["StudyInstanceUID", "SeriesInstanceUID", "StudyDate", "ManufacturerModelName", "PatientAge"]]
pt = pt.rename(columns={"SeriesInstanceUID": "PetSeriesInstanceUID", "ManufacturerModelName": "PetScanner"})
df = df.merge(pt, on="StudyInstanceUID", how="left")
df["age_years"] = pd.to_numeric(df["PatientAge"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
df.to_csv(out_csv, index=False)

# ---------- summary ----------
L = []
ok = df[df["error"] == ""]
L.append("# Go/no-go part 1: tumour burden from the expert masks\n")
L.append(f"Run: {time.strftime('%Y-%m-%d %H:%M')} | source: IDC v25 psma_pet_ct_lesions SEG (CC BY 4.0) | code: 02_code/01_tumour_burden_from_seg.py\n")
L.append(f"- SEG files read: {len(df)}; errors: {int((df['error'] != '').sum())}")
L.append(f"- Patients: {ok['PatientID'].nunique()}; studies: {ok['StudyInstanceUID'].nunique()}")
pos = ok[ok["ttv_ml"] > 0]
det = ok["SegSeriesDescription"].str.contains("Tumor lesions detected", case=False) & ~ok["SegSeriesDescription"].str.contains("No tumor", case=False)
L.append(f"- Studies with tumour voxels: {len(pos)}; labelled 'lesions detected': {int(det.sum())}; disagreements: {int(((ok['ttv_ml'] > 0) != det).sum())}")
L.append(f"- Overlapping-segment voxels (raw minus union): {int((ok['n_voxels_raw'] - ok['n_voxels']).sum())}")
dzchk = ok[np.isfinite(ok["dz_from_frames_mm"])]
L.append(f"- Slice spacing taken from frame positions (header differed by > 0.05 mm): {int((ok['dz_source'] == 'frame positions').sum())} studies; max grid misfit after correction {ok['slice_grid_misfit'].max():.3f} slices\n")
q = lambda s: f"median {s.median():.2f} (IQR {s.quantile(.25):.2f}-{s.quantile(.75):.2f}; range {s.min():.2f}-{s.max():.2f})"
L.append("## Check against the published data descriptor (positive studies)\n")
L.append("| Metric | This run | Descriptor (doi:10.1038/s41597-026-07821-z) |\n|---|---|---|")
L.append(f"| Positive / negative studies | {len(pos)} / {len(ok) - len(pos)} | 539 / 58 |")
L.append(f"| Tumour volume, mL | {q(pos['ttv_ml'])} | median 44.08 (IQR 5.87-247.59; range 0.41-2885.14) |")
L.append(f"| Lesions per study | median {pos['n_lesions'].median():.0f} (IQR {pos['n_lesions'].quantile(.25):.0f}-{pos['n_lesions'].quantile(.75):.0f}; max {pos['n_lesions'].max():.0f}) | median 6 (IQR 1-41; max 318) |\n")
L.append("## High-burden studies\n")
L.append("| Tumour volume | Studies | Patients |\n|---|---|---|")
for t in [100, 250, 500, 1000, 1355, 2000]:
    s = ok[ok["ttv_ml"] >= t]
    L.append(f"| >= {t} mL | {len(s)} | {s['PatientID'].nunique()} |")
L.append("\n## By PET scanner\n")
L.append("| Scanner | Studies | Median volume, mL | >= 500 mL | >= 1,355 mL |\n|---|---|---|---|---|")
for sc, g in ok.groupby("PetScanner"):
    L.append(f"| {sc} | {len(g)} | {g['ttv_ml'].median():.1f} | {int((g['ttv_ml'] >= 500).sum())} | {int((g['ttv_ml'] >= 1355).sum())} |")
L.append("\n## Repeat scans (consecutive pairs within a patient; tracer not yet known)\n")
pr = []
for pid, g in ok.sort_values("StudyDate").groupby("PatientID"):
    g = g.reset_index(drop=True)
    for a in range(len(g) - 1):
        v0, v1 = g.loc[a, "ttv_ml"], g.loc[a + 1, "ttv_ml"]
        d0 = pd.to_datetime(g.loc[a, "StudyDate"]); d1 = pd.to_datetime(g.loc[a + 1, "StudyDate"])
        pr.append(dict(PatientID=pid, days=(d1 - d0).days, ttv0=v0, ttv1=v1, same_scanner=g.loc[a, "PetScanner"] == g.loc[a + 1, "PetScanner"]))
pr = pd.DataFrame(pr)
pr.to_csv(os.path.join(os.path.dirname(out_csv), "repeat_scan_pairs.csv"), index=False)
if len(pr):
    big = pr[["ttv0", "ttv1"]].max(axis=1); small = pr[["ttv0", "ttv1"]].min(axis=1)
    two_fold = (big >= 2 * small) & (big >= 10)
    L.append(f"- Patients with >= 2 studies: {pr['PatientID'].nunique()}; consecutive pairs: {len(pr)}; same scanner in both: {int(pr['same_scanner'].sum())}")
    L.append(f"- Interval between scans, days: median {pr['days'].median():.0f} (IQR {pr['days'].quantile(.25):.0f}-{pr['days'].quantile(.75):.0f})")
    L.append(f"- Pairs where volume changes >= 2-fold (larger value >= 10 mL): {int(two_fold.sum())}")
    L.append(f"- Pairs where the larger volume is >= 500 mL: {int((big >= 500).sum())}; >= 1,355 mL: {int((big >= 1355).sum())}")
    L.append(f"- Absolute change in volume, mL: median {(pr['ttv1'] - pr['ttv0']).abs().median():.1f}; pairs with change >= 250 mL: {int(((pr['ttv1'] - pr['ttv0']).abs() >= 250).sum())}")
L.append("\nPending: radiotracer per study (from one PET header per scan) to split every count above by 68Ga-PSMA-11 vs 18F-PSMA-1007 and keep same-tracer pairs only.")
open(out_md, "w").write("\n".join(L) + "\n")
print("\n".join(L))
