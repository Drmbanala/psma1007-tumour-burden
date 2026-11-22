"""Header-only verification for every PET slice:
(1) is the decay-correction reference the series start?  DecayFactor should equal 2^(FrameReferenceTime / half-life),
    and the earliest AcquisitionTime should equal SeriesTime;
(2) does the expert SEG share the PET in-plane grid (origin, spacing, orientation)?
Resumable. Usage: python 05_verify_decay_reference_and_geometry.py <raw_root> <study_table.csv> <out_dir> <budget_s>"""
import glob, json, os, sys, time
from datetime import datetime
from multiprocessing import Pool
import numpy as np, pandas as pd, pydicom

raw_root, table, out_dir, budget = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
os.makedirs(out_dir, exist_ok=True); T0 = time.time()
TAGS = ["SeriesDate", "SeriesTime", "AcquisitionDate", "AcquisitionTime", "FrameReferenceTime", "DecayFactor", "ActualFrameDuration",
        "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "RadiopharmaceuticalInformationSequence", "FrameOfReferenceUID"]

def dt(d, t):
    return datetime.strptime(str(d)[:8] + str(t).split(".")[0].ljust(6, "0")[:6], "%Y%m%d%H%M%S")

def work(row):
    key = row["StudyInstanceUID"]; out = os.path.join(out_dir, key + ".json")
    if os.path.exists(out) or time.time() - T0 > budget:
        return 0
    base = os.path.join(raw_root, row["PatientID"], key); r = dict(StudyInstanceUID=key, error="")
    try:
        offs, dev, frt, ipp, dur = [], [], [], [], []
        for f in glob.glob(os.path.join(base, "PT_" + row["PetSeriesInstanceUID"], "*.dcm")):
            ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=TAGS)
            half = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            offs.append((dt(ds.AcquisitionDate, ds.AcquisitionTime) - dt(ds.SeriesDate, ds.SeriesTime)).total_seconds())
            t = float(ds.FrameReferenceTime) / 1000.0; frt.append(t); dur.append(float(ds.ActualFrameDuration) / 1000.0)
            dev.append(float(ds.DecayFactor) / 2.0 ** (t / half) - 1.0)
            ipp.append([float(v) for v in ds.ImagePositionPatient])
        iop = [float(v) for v in ds.ImageOrientationPatient]; sp = [float(v) for v in ds.PixelSpacing]
        ipp = np.array(ipp)
        r.update(acq_minus_series_min_s=min(offs), acq_minus_series_max_s=max(offs), frt_min_s=min(frt), frt_max_s=max(frt),
                 frame_duration_median_s=float(np.median(dur)), decayfactor_dev_max=float(np.max(np.abs(dev))),
                 frt_minus_acqoffset_median_s=float(np.median(np.array(frt) - np.array(offs))),
                 pet_ipp_xy_spread_mm=float(np.ptp(ipp[:, :2], axis=0).max()), pet_for=str(ds.get("FrameOfReferenceUID", "")))
        sg = pydicom.dcmread(glob.glob(os.path.join(base, "SEG_" + row["SegSeriesInstanceUID"], "*.dcm"))[0], stop_before_pixels=True)
        sh = sg.SharedFunctionalGroupsSequence[0]; fr = sg.PerFrameFunctionalGroupsSequence
        s_iop = [float(v) for v in sh.PlaneOrientationSequence[0].ImageOrientationPatient]
        s_sp = [float(v) for v in sh.PixelMeasuresSequence[0].PixelSpacing]
        s_ipp = np.array([[float(v) for v in f.PlanePositionSequence[0].ImagePositionPatient] for f in fr])
        r.update(seg_iop_maxdiff=float(np.max(np.abs(np.array(s_iop) - np.array(iop)))), seg_spacing_maxdiff_mm=float(np.max(np.abs(np.array(s_sp) - np.array(sp)))),
                 seg_origin_xy_maxdiff_mm=float(np.max(np.abs(s_ipp[:, :2] - ipp[0, :2]))), seg_for_same=bool(str(sg.get("FrameOfReferenceUID", "")) == r["pet_for"]),
                 seg_z_min=float(s_ipp[:, 2].min()), seg_z_max=float(s_ipp[:, 2].max()), pet_z_min=float(ipp[:, 2].min()), pet_z_max=float(ipp[:, 2].max()))
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    json.dump(r, open(out + ".tmp", "w")); os.replace(out + ".tmp", out); return 1

if __name__ == "__main__":
    rows = pd.read_csv(table)[["PatientID", "StudyInstanceUID", "PetSeriesInstanceUID", "SegSeriesInstanceUID"]].to_dict("records")
    with Pool(4) as p:
        done = sum(p.imap_unordered(work, rows, chunksize=2))
    print(f"processed this call: {done}; total finished: {len(glob.glob(os.path.join(out_dir, '*.json')))} of {len(rows)}; {time.time()-T0:.0f}s")
