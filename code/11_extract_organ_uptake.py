"""P01 step 11: normal-organ uptake on the PET grid (protocol v1.0, section 6).

For every study that already has a TotalSegmentator organ map:
  1. PET DICOM -> body-weight SUV volume (petlib.load_pet_suv; decay reference = series start).
  2. Expert lesion SEG placed on the PET grid by slice position.
  3. Organ labels resampled to the PET grid (nearest neighbour).
  4. Organ VOI = label eroded by one PET voxel, minus every voxel within 10 mm of the lesion mask.
  5. SUVmean / SUVmedian per organ, validity rule, and the pre-specified sensitivity variants
     (no erosion, 10 mm erosion, 3 cm sphere in the right liver lobe).
The 40-scan technical-validation subset is measured a second time with the full-resolution organ map.

Runs natively on the Mac. Raw data are only read. Resumable: one JSON per study.
--follow keeps going while the organ-outlining job is still producing maps.
"""
import argparse, glob, json, os, random, sys, time, traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ORGANS = {"liver": [5], "spleen": [1], "kidney_right": [2], "kidney_left": [3], "aorta": [52]}
MIN_ML = {"liver": 200.0, "spleen": 30.0, "kidney_right": 30.0, "kidney_left": 30.0, "aorta": 5.0}
COLORS = {"liver": "tab:green", "spleen": "tab:purple", "kidney_right": "tab:blue", "kidney_left": "tab:cyan", "aorta": "tab:orange"}
LESION_MARGIN_MM, SPHERE_RADIUS_MM, ERODE10_MM = 10.0, 15.0, 10.0
SEED, N_OVERLAY = 20261114, 30
CFG = {}


def init(cfg):
    CFG.update(cfg)


def stats(v):
    if v.size == 0:
        return dict(n=0, mean=None, median=None, sd=None, p05=None, p95=None, max=None)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)), sd=float(v.std()),
                p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)), max=float(v.max()))


def build_pet_grid(pet_dir, force_dz=None):
    """SUV on a regular grid. Slices are placed by position; missing positions stay NaN and are reported."""
    from petlib import load_pet_suv
    stack, g = load_pet_suv(pet_dir)
    pos = g["pos"]
    dz = float(force_dz) if force_dz else float(g["dz"])
    kf = (pos - pos[0]) / dz
    k = np.round(kf).astype(int)
    nz = int(k.max()) + 1
    suv = np.full((nz,) + stack.shape[1:], np.nan, np.float32)
    suv[k] = stack
    info = dict(dz=dz, nz=nz, n_files=int(len(pos)), pet_missing_slices=int(nz - len(np.unique(k))),
                pet_duplicate_positions=int(len(k) - len(np.unique(k))), pet_grid_misfit=float(np.max(np.abs(kf - k))))
    return suv, g, info


def lesion_on_grid(seg_file, g, info, shape):
    import pydicom
    ds = pydicom.dcmread(seg_file)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None]
    arr = arr > 0
    m = np.zeros(shape, bool)
    worst, outside = 0.0, 0
    for fi, fr in enumerate(ds.PerFrameFunctionalGroupsSequence):
        if not arr[fi].any():
            continue
        p = float(np.dot(g["normal"], [float(v) for v in fr.PlanePositionSequence[0].ImagePositionPatient]))
        kf = (p - g["pos"][0]) / info["dz"]
        ki = int(round(kf))
        worst = max(worst, abs(kf - ki))
        if 0 <= ki < shape[0]:
            m[ki] |= arr[fi]
        else:
            outside += int(arr[fi].sum())
    return m, worst, outside


def labels_on_grid(label_file, g, info, shape):
    import SimpleITK as sitk
    ref = sitk.Image(int(shape[2]), int(shape[1]), int(shape[0]), sitk.sitkUInt8)
    ref.SetSpacing((float(g["px_col"]), float(g["px_row"]), float(info["dz"])))
    ref.SetOrigin([float(v) for v in g["ipp0"]])
    iop = np.array(g["iop"], float)
    ref.SetDirection(tuple(np.stack([iop[:3], iop[3:], np.array(g["normal"], float)], axis=1).ravel()))
    lab = sitk.ReadImage(str(label_file))
    out = sitk.Resample(lab, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out)


def crop_box(mask, sampling, shape):
    zz, yy, xx = np.nonzero(mask)
    box = []
    for idx, sp, n in ((zz, sampling[0], shape[0]), (yy, sampling[1], shape[1]), (xx, sampling[2], shape[2])):
        pad = int(np.ceil(SPHERE_RADIUS_MM / sp)) + 2
        box.append(slice(max(int(idx.min()) - pad, 0), min(int(idx.max()) + pad + 1, n)))
    return tuple(box)


def liver_sphere(raw, near, suv, sampling, x_sign):
    """3 cm sphere centred at the deepest point of the right half of the liver (PERCIST-style reference)."""
    from scipy import ndimage
    ok = raw & ~near
    if not ok.any():
        return dict(fits=False)
    din = ndimage.distance_transform_edt(ok, sampling=sampling)
    cx = np.nonzero(raw)[2].mean()
    cols = np.arange(raw.shape[2])
    right = cols < cx if x_sign > 0 else cols > cx       # patient right = lower LPS x
    cand = din * right[None, None, :]
    c = np.unravel_index(int(np.argmax(cand)), cand.shape)
    if cand[c] < SPHERE_RADIUS_MM:
        return dict(fits=False, deepest_mm=float(cand[c]))
    Z, Y, X = np.ogrid[:raw.shape[0], :raw.shape[1], :raw.shape[2]]
    d2 = ((Z - c[0]) * sampling[0]) ** 2 + ((Y - c[1]) * sampling[1]) ** 2 + ((X - c[2]) * sampling[2]) ** 2
    sph = (d2 <= SPHERE_RADIUS_MM ** 2) & np.isfinite(suv)
    v = suv[sph]
    return dict(fits=True, deepest_mm=float(cand[c]), n=int(v.size), mean=float(v.mean()), sd=float(v.std()))


def measure(labels, suv, near, vox_ml, sampling, x_sign, want_vois=False):
    from scipy import ndimage
    struct = ndimage.generate_binary_structure(3, 1)
    finite = np.isfinite(suv)
    out, vois = {}, {}
    for name, ids in ORGANS.items():
        raw = np.isin(labels, ids)
        r = dict(raw_ml=float(raw.sum() * vox_ml), voi_ml=0.0, valid=False)
        if raw.any():
            er = ndimage.binary_erosion(raw, struct)
            voi = er & ~near & finite
            n_er = max(int(er.sum()), 1)
            r["voi_ml"] = float(voi.sum() * vox_ml)
            r["frac_removed_near_lesion"] = float((er & near).sum() / n_er)
            r["frac_outside_pet"] = float((er & ~finite).sum() / n_er)
            r["suv"] = stats(suv[voi])
            v0 = raw & ~near & finite
            r["noerode"] = dict(ml=float(v0.sum() * vox_ml), mean=float(suv[v0].mean()) if v0.any() else None)
            v10 = (ndimage.distance_transform_edt(raw, sampling=sampling) >= ERODE10_MM) & ~near & finite
            r["erode10mm"] = dict(ml=float(v10.sum() * vox_ml), mean=float(suv[v10].mean()) if v10.any() else None)
            r["valid"] = bool(r["voi_ml"] >= MIN_ML[name])
            if name == "liver":
                r["sphere"] = liver_sphere(raw, near, suv, sampling, x_sign)
            if want_vois:
                vois[name] = (raw, voi)
        out[name] = r
    return out, vois


def overlay(png, suv, les, vois, sampling, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.2))
    ref = vois.get("liver") or next(iter(vois.values()))
    zc, yc, _ = [int(round(v.mean())) for v in np.nonzero(ref[0])]
    kid = [vois[k][0] for k in ("kidney_right", "kidney_left") if k in vois]
    yk = int(round(np.nonzero(np.logical_or.reduce(kid))[1].mean())) if kid else yc
    planes = [("coronal through liver", lambda a: a[:, yc, :][::-1], sampling[0] / sampling[2]),
              ("coronal through kidneys", lambda a: a[:, yk, :][::-1], sampling[0] / sampling[2]),
              ("axial through liver", lambda a: a[zc], sampling[1] / sampling[2])]
    vmax = 20.0
    for a, (name, cut, asp) in zip(ax, planes):
        a.imshow(np.nan_to_num(cut(suv)), cmap="gray_r", vmin=0, vmax=vmax, aspect=asp, interpolation="nearest")
        for org, (raw, voi) in vois.items():
            if cut(raw).any():
                a.contour(cut(raw), levels=[0.5], colors=COLORS[org], linewidths=0.5, linestyles="dotted")
            if cut(voi).any():
                a.contour(cut(voi), levels=[0.5], colors=COLORS[org], linewidths=1.0)
        if cut(les).any():
            a.contour(cut(les), levels=[0.5], colors="red", linewidths=0.8)
        a.set_title(name, fontsize=9); a.axis("off")
    fig.suptitle(title, fontsize=9)
    plt.tight_layout(); fig.savefig(png, dpi=int(os.environ.get("P01_OVERLAY_DPI", "60"))); plt.close(fig)


def fullres_ready(key):
    """The full-resolution map is used once it is stable: older than 90 s, or the outlining job has ended."""
    f = Path(CFG["segdir"]) / key / "organs_fullres.nii.gz"
    if not f.exists():
        return False
    return (time.time() - f.stat().st_mtime) > 90 or seg_status()[0].startswith("RESULT")


def needs_work(row):
    key = row["StudyInstanceUID"]
    meta = Path(CFG["segdir"]) / key / "seg_meta.json"
    if not (meta.exists() and (Path(CFG["segdir"]) / key / "total_fast.nii.gz").exists()):
        return False
    out = Path(CFG["out"]) / (key + ".json")
    if not out.exists():
        try:
            return not json.loads(meta.read_text()).get("error")
        except Exception:
            return False
    try:
        done = json.loads(out.read_text())
    except Exception:
        return True
    return done.get("fullres") is None and not done.get("error") and fullres_ready(key)


def process(row):
    key = row["StudyInstanceUID"]
    out = Path(CFG["out"]) / (key + ".json")
    r = dict(StudyInstanceUID=key, PatientID=row["PatientID"], error="", fullres=None)
    t0 = time.time()
    try:
        from scipy import ndimage
        base = Path(CFG["raw"]) / row["PatientID"] / key
        suv, g, info = build_pet_grid(str(base / ("PT_" + row["PetSeriesInstanceUID"])), CFG.get("force_dz"))
        r.update(info)
        sampling = (info["dz"], float(g["px_row"]), float(g["px_col"]))
        vox_ml = sampling[0] * sampling[1] * sampling[2] / 1000.0
        seg_file = glob.glob(str(base / ("SEG_" + row["SegSeriesInstanceUID"]) / "*.dcm"))[0]
        les, worst, outside = lesion_on_grid(seg_file, g, info, suv.shape)
        nles = int(les.sum())
        r.update(voxel_ml=vox_ml, seg_slice_misfit=float(worst), lesion_voxels_outside_grid=outside,
                 lesion_ml_grid=nles * vox_ml,
                 lesion_frac_suv_zero=float((suv[les] == 0).sum() / nles) if nles else 0.0,
                 lesion_frac_no_pet=float((~np.isfinite(suv[les])).sum() / nles) if nles else 0.0)
        segdir = Path(CFG["segdir"]) / key
        lab = labels_on_grid(segdir / "total_fast.nii.gz", g, info, suv.shape)
        lab_fr = labels_on_grid(segdir / "organs_fullres.nii.gz", g, info, suv.shape) if fullres_ready(key) else None
        ids = [i for v in ORGANS.values() for i in v]
        union = np.isin(lab, ids) | (np.isin(lab_fr, ids) if lab_fr is not None else False)
        if not union.any():
            raise RuntimeError("no target organ label inside the PET field of view")
        box = crop_box(union, sampling, suv.shape)
        suv_c, les_c = suv[box], les[box]
        near = (ndimage.distance_transform_edt(~les_c, sampling=sampling) <= LESION_MARGIN_MM) if les_c.any() else np.zeros(les_c.shape, bool)
        x_sign = 1.0 if float(g["iop"][0]) >= 0 else -1.0
        want = bool(row.get("overlay"))
        r["organs"], vois = measure(lab[box], suv_c, near, vox_ml, sampling, x_sign, want_vois=want)
        if lab_fr is not None:
            r["fullres"], _ = measure(lab_fr[box], suv_c, near, vox_ml, sampling, x_sign)
        if want and vois:
            o = r["organs"]
            f = lambda k: "n/a" if not o[k].get("suv") or o[k]["suv"]["mean"] is None else f"{o[k]['suv']['mean']:.1f}"
            title = (f"{row['PatientID'][-6:]} | {row.get('tracer', '')} | {row.get('scanner', '')} | tumour {r['lesion_ml_grid']:.0f} mL | "
                     f"SUVmean liver {f('liver')}, spleen {f('spleen')}, kidney R {f('kidney_right')} L {f('kidney_left')}, aorta {f('aorta')}\n"
                     "solid = measured VOI, dotted = organ outline, red = expert lesion mask")
            os.makedirs(CFG["qc"], exist_ok=True)
            overlay(os.path.join(CFG["qc"], key + ".png"), suv_c, les_c, vois, sampling, title)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        r["traceback"] = traceback.format_exc()[-1500:]
    r["seconds"] = round(time.time() - t0, 1)
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(r, fh)
    os.replace(tmp, out)
    return 0 if r["error"] else 1


def flatten(j):
    row = {k: v for k, v in j.items() if not isinstance(v, (dict, list)) and k != "traceback"}
    for tag, block in (("", j.get("organs")), ("fr_", j.get("fullres"))):
        if not block:
            continue
        for org, o in block.items():
            p = f"{tag}{org}_"
            row[p + "raw_ml"], row[p + "voi_ml"], row[p + "valid"] = o.get("raw_ml"), o.get("voi_ml"), o.get("valid")
            s = o.get("suv") or {}
            row[p + "suvmean"], row[p + "suvmedian"], row[p + "suvsd"], row[p + "n"] = s.get("mean"), s.get("median"), s.get("sd"), s.get("n")
            if tag == "":
                row[p + "frac_removed_near_lesion"], row[p + "frac_outside_pet"] = o.get("frac_removed_near_lesion"), o.get("frac_outside_pet")
                row[p + "noerode_suvmean"] = (o.get("noerode") or {}).get("mean")
                row[p + "erode10mm_suvmean"], row[p + "erode10mm_ml"] = (o.get("erode10mm") or {}).get("mean"), (o.get("erode10mm") or {}).get("ml")
                if org == "liver":
                    sp = o.get("sphere") or {}
                    row["liver_sphere_fits"], row["liver_sphere_suvmean"] = sp.get("fits"), sp.get("mean")
    return row


def compile_outputs(rows):
    import pandas as pd
    js = []
    for f in glob.glob(os.path.join(CFG["out"], "1.*.json")):
        try:
            js.append(flatten(json.load(open(f))))
        except Exception:
            pass
    if not js:
        return 0, 0
    d = pd.DataFrame(js)
    for side in ("", "fr_"):                              # kidneys = volume-weighted mean of the valid kidneys
        if side + "kidney_right_suvmean" not in d:
            continue
        num = den = 0.0
        for k in ("kidney_right", "kidney_left"):
            ok = d[side + k + "_valid"].fillna(False).astype(bool)
            num = num + (d[side + k + "_suvmean"].fillna(0) * d[side + k + "_n"].fillna(0)).where(ok, 0.0)
            den = den + d[side + k + "_n"].fillna(0).where(ok, 0.0)
        d[side + "kidneys_suvmean"] = (num / den).where(den > 0)
        d[side + "kidneys_n_valid"] = sum(d[side + k + "_valid"].fillna(False).astype(int) for k in ("kidney_right", "kidney_left"))
    d.to_csv(CFG["csv"], index=False)
    pngs = sorted(glob.glob(os.path.join(CFG["qc"], "1.*.png")))
    if pngs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for s in range(0, len(pngs), 6):
            fig, ax = plt.subplots(3, 2, figsize=(20, 12.5))
            for a in ax.ravel():
                a.axis("off")
            for a, p in zip(ax.ravel(), pngs[s:s + 6]):
                a.imshow(plt.imread(p))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
            fig.savefig(os.path.join(CFG["qc"], f"_sheet_{s // 6 + 1:02d}.png"), dpi=75); plt.close(fig)
    return len(d), int((d["error"].fillna("") != "").sum())


def seg_status():
    f = Path(CFG["segdir"]) / "_status.txt"
    try:
        return f.read_text().strip(), f.stat().st_mtime
    except Exception:
        return "", 0.0


def write_status(msg):
    os.makedirs(CFG["out"], exist_ok=True)
    tmp = os.path.join(CFG["out"], "_status.tmp")
    with open(tmp, "w") as fh:
        fh.write(msg + "\n")
    os.replace(tmp, os.path.join(CFG["out"], "_status.txt"))
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    proj = HERE.parent
    root = proj.parent.parent
    ap.add_argument("--raw", default=str(root / "01_raw_data" / "PSMA-PET-CT-Lesions_IDC" / "psma_pet_ct_lesions"))
    ap.add_argument("--proj", default=str(proj))
    ap.add_argument("--out", default=None)
    ap.add_argument("--qc", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--overlay-all", action="store_true")
    ap.add_argument("--overlay-uids", default=None, help="text file with StudyInstanceUIDs that should get an overlay image")
    ap.add_argument("--force-dz", type=float, default=None, help="testing only: slice spacing for a partial PET series")
    a = ap.parse_args()
    import pandas as pd
    from multiprocessing import get_context
    P = Path(a.proj)
    cfg = dict(raw=a.raw, segdir=str(P / "01_derived_data" / "organ_seg"),
               out=a.out or str(P / "01_derived_data" / "organ_uptake_json"),
               qc=a.qc or str(P / "03_results" / "qc_images" / "organ_overlays"),
               csv=a.csv or str(P / "01_derived_data" / "organ_uptake.csv"), force_dz=a.force_dz)
    init(cfg)
    os.makedirs(cfg["out"], exist_ok=True)
    import fcntl                                          # only one job of this kind may run at a time
    global _LOCK
    _LOCK = open(os.path.join(cfg["out"], "_measure.lock"), "w")
    try:
        fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another measurement job is already running; this one exits without doing anything.", flush=True)
        return 0
    d = pd.read_csv(P / "01_derived_data" / "study_table_v3_verified.csv")
    keys = sorted(d["StudyInstanceUID"])
    chosen = set(random.Random(SEED + 1).sample(keys, min(N_OVERLAY, len(keys))))
    if a.overlay_uids and os.path.exists(a.overlay_uids):
        chosen |= {s.strip() for s in open(a.overlay_uids) if s.strip()}
    rows = d[["PatientID", "StudyInstanceUID", "PetSeriesInstanceUID", "SegSeriesInstanceUID", "tracer", "scanner"]].to_dict("records")
    for r in rows:
        r["overlay"] = bool(a.overlay_all or r["StudyInstanceUID"] in chosen)
    if a.overlay_uids:                                     # redo the listed studies so that their images exist
        for k in chosen:
            f = Path(cfg["out"]) / (k + ".json")
            if f.exists() and not (Path(cfg["qc"]) / (k + ".png")).exists():
                f.unlink()
    write_status(f"START organ uptake; studies {len(rows)}; workers {a.workers}; follow {a.follow}")
    ok = bad = 0
    last_change = time.time(); last_seg = ""
    with get_context("spawn").Pool(a.workers, initializer=init, initargs=(cfg,)) as pool:
        while True:
            todo = [r for r in rows if needs_work(r)]
            for res in pool.imap_unordered(process, todo, chunksize=1):
                ok += res; bad += 1 - res
                if (ok + bad) % 25 == 0:
                    write_status(f"RUNNING organ uptake; measured this run {ok}; errors {bad}; in this batch {len(todo)}")
            seg, _ = seg_status()
            n_json = len(glob.glob(os.path.join(cfg["out"], "1.*.json")))
            if seg != last_seg:
                last_seg, last_change = seg, time.time()
            if not a.follow:
                break
            if seg.startswith("RESULT") and not any(needs_work(r) for r in rows):
                break
            if time.time() - last_change > 3600:
                print("The organ-outlining job has not reported progress for an hour; finishing with what is available.", flush=True)
                break
            if not todo:
                write_status(f"WAITING for organ outlining; measured so far {n_json} of {len(rows)}; outlining says: {seg[:110]}")
                time.sleep(60)
    n, nerr = compile_outputs(rows)
    seg, _ = seg_status()
    tag = "finished" if (n == len(rows) and nerr == 0) else "partial"
    write_status(f"RESULT: {tag} {time.strftime('%Y-%m-%d %H:%M')}; studies in table {n} of {len(rows)}; with errors {nerr}; outlining status: {seg[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
