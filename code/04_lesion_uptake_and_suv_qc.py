"""Per-study SUV calibration QC and lesion uptake (exposure metrics only; no normal-organ uptake).
Resumable: one small JSON per study. Usage: python 04_lesion_uptake_and_suv_qc.py <raw_root> <study_table.csv> <out_dir> <time_budget_s>"""
import glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from petlib import load_pet_suv, load_seg_on_pet

raw_root, table, out_dir, budget = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
os.makedirs(out_dir, exist_ok=True)
T0 = time.time()


def work(row):
    key = row["StudyInstanceUID"]
    out = os.path.join(out_dir, key + ".json")
    if os.path.exists(out) or time.time() - T0 > budget:
        return 0
    base = os.path.join(raw_root, row["PatientID"], key)
    r = dict(StudyInstanceUID=key, PatientID=row["PatientID"], error="")
    try:
        suv, g = load_pet_suv(os.path.join(base, "PT_" + row["PetSeriesInstanceUID"]))
        seg_file = glob.glob(os.path.join(base, "SEG_" + row["SegSeriesInstanceUID"], "*.dcm"))[0]
        m, worst = load_seg_on_pet(seg_file, g)
        n = int(m.sum())
        r.update({k: g[k] for k in ["n_slices", "dz", "dz_irregular", "px_row", "px_col", "voxel_ml", "rows", "cols", "weight_kg",
                                    "dose_MBq", "half_life_s", "decay_dt_s", "series_minus_inj_s", "ref_is_series_time", "frame_of_reference"]})
        r.update(seg_slice_misfit_mm=float(worst), lesion_voxels=n, ttv_ml_petgrid=n * g["voxel_ml"],
                 fov_activity_fraction=float(suv.sum(dtype=np.float64) * g["voxel_ml"] / (g["weight_kg"] * 1000.0)),
                 suv_max_volume=float(suv.max()))
        if n:
            v = suv[m]
            tlu = float(v.sum(dtype=np.float64) * g["voxel_ml"])
            r.update(lesion_suv_mean=float(v.mean()), lesion_suv_max=float(v.max()), tl_psma_g=tlu,
                     tlf=tlu / (g["weight_kg"] * 1000.0))
        else:
            r.update(lesion_suv_mean=None, lesion_suv_max=None, tl_psma_g=0.0, tlf=0.0)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    json.dump(r, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    return 1


if __name__ == "__main__":
    rows = pd.read_csv(table)[["PatientID", "StudyInstanceUID", "PetSeriesInstanceUID", "SegSeriesInstanceUID"]].to_dict("records")
    with Pool(4) as p:
        done = sum(p.imap_unordered(work, rows, chunksize=2))
    print(f"processed this call: {done}; total finished: {len(glob.glob(os.path.join(out_dir, '*.json')))} of {len(rows)}; {time.time()-T0:.0f}s")
