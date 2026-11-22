"""Radiotracer, injected activity, body size and timing from one PET DICOM header per study.

Usage: python 02_pet_headers.py <pet_header_samples_dir> <out.csv>
Read-only on the raw data. One row per PET series.
"""
import glob, os, sys
from datetime import datetime
import pandas as pd, pydicom

src, out_csv = sys.argv[1:3]

def dt(date, time):
    if not date or not time:
        return None
    t = str(time).split(".")[0].ljust(6, "0")[:6]
    try:
        return datetime.strptime(str(date)[:8] + t, "%Y%m%d%H%M%S")
    except ValueError:
        return None

rows = []
for fp in sorted(glob.glob(os.path.join(src, "*.dcm"))):
    ds = pydicom.dcmread(fp, stop_before_pixels=True)
    r = dict(PetSeriesInstanceUID=os.path.basename(fp)[:-4], PatientID=str(ds.get("PatientID", "")),
             StudyInstanceUID=str(ds.get("StudyInstanceUID", "")), manufacturer=str(ds.get("Manufacturer", "")),
             scanner=str(ds.get("ManufacturerModelName", "")), sex=str(ds.get("PatientSex", "")),
             age_years=pd.to_numeric(str(ds.get("PatientAge", "")).rstrip("Y"), errors="coerce"),
             weight_kg=pd.to_numeric(ds.get("PatientWeight", None), errors="coerce"),
             height_m=pd.to_numeric(ds.get("PatientSize", None), errors="coerce"),
             units=str(ds.get("Units", "")), decay_correction=str(ds.get("DecayCorrection", "")),
             recon=str(ds.get("ReconstructionMethod", "")), n_slices=ds.get("NumberOfSlices", None),
             frame_duration_s=pd.to_numeric(ds.get("ActualFrameDuration", None), errors="coerce") / 1000.0)
    rp = ds.get("RadiopharmaceuticalInformationSequence")
    rp = rp[0] if rp else None
    code = rp.RadionuclideCodeSequence[0] if rp is not None and rp.get("RadionuclideCodeSequence") else None
    r["radionuclide_code_meaning"] = str(code.CodeMeaning) if code is not None else ""
    r["radiopharmaceutical"] = str(rp.get("Radiopharmaceutical", "")) if rp is not None else ""
    r["half_life_s"] = float(rp.RadionuclideHalfLife) if rp is not None and rp.get("RadionuclideHalfLife") else None
    r["injected_MBq"] = float(rp.RadionuclideTotalDose) / 1e6 if rp is not None and rp.get("RadionuclideTotalDose") else None
    hl, cm = r["half_life_s"] or 0, r["radionuclide_code_meaning"].lower()
    by_hl = "68Ga" if 3800 <= hl <= 4300 else ("18F" if 6300 <= hl <= 6900 else "")
    by_code = "68Ga" if "gallium" in cm else ("18F" if "fluor" in cm else "")
    r["tracer"] = by_hl or by_code
    r["tracer_flag"] = "" if (by_hl and by_code and by_hl == by_code) else f"half-life says {by_hl or '?'}, code says {by_code or '?'}"
    inj = None
    if rp is not None:
        sdt = str(rp.get("RadiopharmaceuticalStartDateTime", "") or "")
        inj = dt(sdt[:8], sdt[8:]) if len(sdt) >= 12 else dt(ds.get("SeriesDate", ""), rp.get("RadiopharmaceuticalStartTime", ""))
    ser = dt(ds.get("SeriesDate", ""), ds.get("SeriesTime", ""))
    acq = dt(ds.get("AcquisitionDate", ""), ds.get("AcquisitionTime", ""))
    r["uptake_series_min"] = (ser - inj).total_seconds() / 60 if (ser and inj) else None
    r["uptake_this_slice_min"] = (acq - inj).total_seconds() / 60 if (acq and inj) else None
    r["StudyDate"] = str(ds.get("StudyDate", ""))
    rows.append(r)
df = pd.DataFrame(rows)
df["bmi"] = df["weight_kg"] / df["height_m"] ** 2
df.to_csv(out_csv, index=False)
print(f"rows {len(df)}; tracer counts {df['tracer'].value_counts(dropna=False).to_dict()}; tracer flags {int((df['tracer_flag'] != '').sum())}")
