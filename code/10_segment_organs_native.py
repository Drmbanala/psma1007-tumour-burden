"""Runs natively on the Mac (needs the GPU/CPU and memory of the real machine).
For every study: CT DICOM -> temporary NIfTI -> TotalSegmentator 'total' (fast, 3 mm) -> one multilabel organ map.
Then a fixed random subset is re-segmented with the full-resolution model for technical validation.
Resumable: finished studies are skipped. Raw data are only read. No image leaves the computer; usage statistics are off."""
import json, os, random, shutil, sys, time, traceback
from pathlib import Path

CODE = Path(__file__).resolve().parent
PROJ = CODE.parent
ROOT = PROJ.parent.parent
RAW = ROOT / "01_raw_data" / "PSMA-PET-CT-Lesions_IDC" / "psma_pet_ct_lesions"
EXPECTED = ROOT / "01_raw_data" / "PSMA-PET-CT-Lesions_IDC" / "expected_pt_ct_series.csv"
TABLE = PROJ / "01_derived_data" / "study_table_v3_verified.csv"
OUT = PROJ / "01_derived_data" / "organ_seg"
SCRATCH = PROJ / "01_derived_data" / "_scratch"
TS_HOME = ROOT / "03_code_and_environments" / "totalseg_home"
STATUS = OUT / "_status.txt"
KEY = {"liver": 5, "spleen": 1, "kidney_right": 2, "kidney_left": 3}
SEED, N_FULLRES, FULLRES_CAP_S = 20261114, 40, 3 * 3600

os.environ["TOTALSEG_HOME_DIR"] = str(TS_HOME)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def status(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp"); tmp.write_text(msg + "\n"); os.replace(tmp, STATUS)
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def ct_to_nifti(ct_dir, out_file):
    import numpy as np, pydicom, SimpleITK as sitk
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(str(ct_dir))
    files = r.GetGDCMSeriesFileNames(str(ct_dir), ids[0])
    r.SetFileNames(files)
    img = r.Execute()
    sitk.WriteImage(img, str(out_file))
    zs = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "FrameOfReferenceUID", "ContrastBolusAgent"])
        iop = [float(v) for v in ds.ImageOrientationPatient]
        n = np.cross(iop[:3], iop[3:])
        zs.append(float(np.dot(n, [float(v) for v in ds.ImagePositionPatient])))
    dz = np.diff(np.array(zs))
    return dict(ct_size=list(img.GetSize()), ct_spacing=[round(v, 4) for v in img.GetSpacing()], ct_n_files=len(files), ct_n_series_in_folder=len(ids),
                ct_dz_min=float(np.min(np.abs(dz))) if len(dz) else None, ct_dz_max=float(np.max(np.abs(dz))) if len(dz) else None,
                ct_irregular_spacing=bool(len(dz) and (np.max(np.abs(dz)) - np.min(np.abs(dz))) > 0.05),
                ct_frame_of_reference=str(ds.get("FrameOfReferenceUID", "")), ct_contrast_tag=str(ds.get("ContrastBolusAgent", "")))


def label_volumes(nii_file):
    import nibabel as nib, numpy as np
    im = nib.load(str(nii_file)); a = np.asanyarray(im.dataobj).astype(np.int64)
    vox_ml = float(np.prod(im.header.get_zooms()[:3])) / 1000.0
    c = np.bincount(a.ravel(), minlength=118)
    return {int(i): round(float(c[i] * vox_ml), 2) for i in np.nonzero(c)[0] if i > 0}, a


def dice(a, b, lab):
    x, y = a == lab, b == lab
    s = int(x.sum()) + int(y.sum())
    return 1.0 if s == 0 else 2.0 * int((x & y).sum()) / s


def segment(ct_nii, out_file, device, fast, roi_subset=None):
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(input=Path(ct_nii), output=Path(out_file), ml=True, fast=fast, task="total", roi_subset=roi_subset,
                     device=device, quiet=True, nr_thr_resamp=1, nr_thr_saving=1)


def plausible(vol):
    liver, spleen = vol.get(5, 0), vol.get(1, 0)
    kid = max(vol.get(2, 0), vol.get(3, 0))
    return 600 <= liver <= 4500 and 30 <= spleen <= 2500 and 40 <= kid <= 700


def find_ct_dir(row):
    hits = sorted((RAW / row["PatientID"] / row["StudyInstanceUID"]).glob("CT_*"))
    return hits[0] if hits else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl                                          # only one job of this kind may run at a time
    global _LOCK
    _LOCK = open(OUT / "_outlining.lock", "w")
    try:
        fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another outlining job is already running; this one exits without doing anything.", flush=True)
        return 0
    import pandas as pd, torch
    OUT.mkdir(parents=True, exist_ok=True); SCRATCH.mkdir(parents=True, exist_ok=True); TS_HOME.mkdir(parents=True, exist_ok=True)
    cfg = TS_HOME / "config.json"
    if not cfg.exists():
        cfg.write_text(json.dumps({"totalseg_id": "totalseg_LOCALRUN", "send_usage_stats": False, "prediction_counter": 0}, indent=4))
    from totalsegmentator.map_to_binary import class_map
    (OUT / "class_map_total.json").write_text(json.dumps(class_map["total"], indent=1))
    assert all(class_map["total"][v] == k for k, v in KEY.items()), "unexpected TotalSegmentator label numbers"

    d = pd.read_csv(TABLE)
    exp = pd.read_csv(EXPECTED); exp = exp[exp["Modality"] == "CT"]
    n_expected = {p.split("/")[2]: int(n) for p, n in zip(exp["dest"], exp["instanceCount"])}
    primary = (d["tracer"] == "18F") & d["uptake_series_min"].between(45, 120)
    d = pd.concat([d[primary], d[~primary & (d["tracer"] == "18F")], d[d["tracer"] != "18F"]]).reset_index(drop=True)
    rows = d[["PatientID", "StudyInstanceUID", "tracer"]].to_dict("records")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    status(f"START torch {torch.__version__} first-choice device {device}; studies {len(rows)}")
    done = failed = skipped_incomplete = 0
    t_start = time.time(); times = []
    pilot_checked = (OUT / "_pilot_passed.json").exists()
    if pilot_checked:
        device = json.loads((OUT / "_pilot_passed.json").read_text())["device"]

    for i, row in enumerate(rows, 1):
        key = row["StudyInstanceUID"]
        odir = OUT / key; seg = odir / "total_fast.nii.gz"; meta_f = odir / "seg_meta.json"
        try:
            already = seg.exists() and meta_f.exists() and not json.loads(meta_f.read_text()).get("error")
        except Exception:
            already = False
        if already:
            done += 1
        else:
            ct_dir = find_ct_dir(row)
            n_have = len(list(ct_dir.glob("*.dcm"))) if ct_dir else 0
            if ct_dir is None or n_have < n_expected.get(key, 1):
                skipped_incomplete += 1
            else:
                odir.mkdir(parents=True, exist_ok=True)
                work = SCRATCH / f"w{i}"; work.mkdir(parents=True, exist_ok=True); ct_nii = work / "ct.nii"
                meta = dict(StudyInstanceUID=key, PatientID=row["PatientID"], error="")
                t0 = time.time()
                try:
                    meta.update(ct_to_nifti(ct_dir, ct_nii))
                    try:
                        segment(ct_nii, seg, device, fast=True)
                    except Exception as e:
                        if device == "cpu":
                            raise
                        print("   device", device, "failed:", type(e).__name__, str(e)[:200], "-> switching to cpu", flush=True)
                        device = "cpu"; segment(ct_nii, seg, device, fast=True)
                    vol, arr = label_volumes(seg)
                    meta.update(device=device, seconds=round(time.time() - t0, 1), volumes_ml=vol, plausible=plausible(vol))
                    if not pilot_checked and device == "mps":
                        ref = work / "cpu_check.nii.gz"; segment(ct_nii, ref, "cpu", fast=True)
                        _, arr_cpu = label_volumes(ref)
                        dc = {k: round(dice(arr, arr_cpu, v), 4) for k, v in KEY.items()}
                        meta["mps_vs_cpu_dice"] = dc
                        if min(dc.values()) < 0.95:
                            print("   GPU result differs from CPU", dc, "-> using cpu for everything", flush=True)
                            device = "cpu"; shutil.copyfile(ref, seg); vol, arr = label_volumes(seg)
                            meta.update(device="cpu", volumes_ml=vol, plausible=plausible(vol))
                    done += 1; times.append(time.time() - t0)
                except Exception as e:
                    meta["error"] = f"{type(e).__name__}: {str(e)[:300]}"; meta["traceback"] = traceback.format_exc()[-1500:]
                    failed += 1
                    if seg.exists():
                        seg.unlink()
                meta_f.write_text(json.dumps(meta, indent=1))
                shutil.rmtree(work, ignore_errors=True)
                if not pilot_checked and (done + failed) >= 3:
                    metas = [json.loads(p.read_text()) for p in OUT.glob("*/seg_meta.json")]
                    good = [m for m in metas if not m.get("error") and m.get("plausible")]
                    if len(good) < 2:
                        status(f"RESULT: pilot_failed - only {len(good)} of {len(metas)} first scans gave plausible organ volumes; stopped so nothing is wasted")
                        return 2
                    (OUT / "_pilot_passed.json").write_text(json.dumps({"device": device, "n_checked": len(metas), "n_plausible": len(good)}))
                    pilot_checked = True
                    status(f"PILOT PASSED on {device}: {len(good)} of {len(metas)} plausible; about {sum(times)/max(len(times),1):.0f} s per scan")
        if i % 10 == 0 or i == len(rows):
            per = sum(times[-20:]) / max(len(times[-20:]), 1)
            status(f"RUNNING fast segmentation {i}/{len(rows)} checked; done {done}; failed {failed}; waiting for CT {skipped_incomplete}; device {device}; about {per:.0f} s per scan; est. {(len(rows)-i)*per/3600:.1f} h left")

    # technical validation subset with the full-resolution model
    keys = sorted(r["StudyInstanceUID"] for r in rows)
    subset = random.Random(SEED).sample(keys, N_FULLRES)
    (OUT / "fullres_subset.json").write_text(json.dumps(subset, indent=1))
    pid = {r["StudyInstanceUID"]: r["PatientID"] for r in rows}
    t_full = time.time(); n_full = 0
    for j, key in enumerate(subset, 1):
        out_f = OUT / key / "organs_fullres.nii.gz"
        if out_f.exists():
            n_full += 1; continue
        if time.time() - t_full > FULLRES_CAP_S or not (OUT / key / "total_fast.nii.gz").exists():
            continue
        work = SCRATCH / f"f{j}"; work.mkdir(parents=True, exist_ok=True)
        try:
            ct_to_nifti(find_ct_dir({"PatientID": pid[key], "StudyInstanceUID": key}), work / "ct.nii")
            try:
                segment(work / "ct.nii", out_f, device, fast=False, roi_subset=list(KEY))
            except Exception:
                if device == "cpu":
                    raise
                segment(work / "ct.nii", out_f, "cpu", fast=False, roi_subset=list(KEY))
            n_full += 1
        except Exception as e:
            print("   full-resolution failed for", key[-12:], type(e).__name__, str(e)[:200], flush=True)
        shutil.rmtree(work, ignore_errors=True)
        status(f"RUNNING full-resolution validation subset {j}/{N_FULLRES}; done {n_full}")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    tag = "finished" if (failed == 0 and skipped_incomplete == 0) else "finished_with_gaps"
    status(f"RESULT: {tag} {time.strftime('%Y-%m-%d %H:%M')}; fast segmentations {done}/{len(rows)}; failed {failed}; CT missing {skipped_incomplete}; full-resolution subset {n_full}/{N_FULLRES}; device {device}; total {(time.time()-t_start)/3600:.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
