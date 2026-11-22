"""Shared helpers: build a body-weight SUV volume and the expert lesion mask on the PET grid from raw DICOM.
Everything is rebuilt from the raw files on demand, so no intermediate images need to be stored."""
import glob, os
from datetime import datetime
import numpy as np, pydicom


def _dt(date, time):
    if not date or not time:
        return None
    t = str(time).split(".")[0].ljust(6, "0")[:6]
    try:
        return datetime.strptime(str(date)[:8] + t, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def load_pet_suv(pet_dir):
    """Return (suv[z,y,x] float32, geom dict). SUVbw = Bq/mL x weight[g] / (dose[Bq] x 2^(-dt/T_half)),
    dt = decay-correction reference time (series start; earliest acquisition time if earlier) minus injection time."""
    files = sorted(glob.glob(os.path.join(pet_dir, "*.dcm")))
    hs, pos, slabs = [], [], []
    normal = None
    for f in files:
        ds = pydicom.dcmread(f)
        if normal is None:
            iop = np.array([float(v) for v in ds.ImageOrientationPatient])
            normal = np.cross(iop[:3], iop[3:]); normal /= np.linalg.norm(normal)
            first = ds
        pos.append(float(np.dot(normal, [float(v) for v in ds.ImagePositionPatient])))
        slabs.append(ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept))
        hs.append((str(ds.get("AcquisitionDate", "")), str(ds.get("AcquisitionTime", "")), [float(v) for v in ds.ImagePositionPatient]))
    order = np.argsort(pos)
    pos = np.array(pos)[order]
    vol = np.stack([slabs[i] for i in order])
    dzs = np.diff(pos)
    dz = float(np.median(dzs)) if len(dzs) else float("nan")
    ds = first
    rp = ds.RadiopharmaceuticalInformationSequence[0]
    dose = float(rp.RadionuclideTotalDose); half = float(rp.RadionuclideHalfLife)
    sdt = str(rp.get("RadiopharmaceuticalStartDateTime", "") or "")
    inj = _dt(sdt[:8], sdt[8:]) if len(sdt) >= 12 else _dt(ds.SeriesDate, rp.RadiopharmaceuticalStartTime)
    ser = _dt(ds.SeriesDate, ds.SeriesTime)
    acq = [a for a in (_dt(d, t) for d, t, _ in hs) if a is not None]
    ref = ser
    if acq and min(acq) < ser:
        ref = min(acq)
    dt_s = (ref - inj).total_seconds()
    weight_g = float(ds.PatientWeight) * 1000.0
    scale = weight_g / (dose * 2.0 ** (-dt_s / half))
    assert str(ds.Units) == "BQML" and str(ds.DecayCorrection) == "START", (ds.Units, ds.DecayCorrection)
    sp = [float(v) for v in ds.PixelSpacing]
    geom = dict(n_slices=len(files), dz=dz, dz_irregular=bool(len(dzs) and np.max(np.abs(dzs - dz)) > 0.05),
                px_row=sp[0], px_col=sp[1], voxel_ml=sp[0] * sp[1] * dz / 1000.0, pos=pos, normal=normal,
                iop=[float(v) for v in ds.ImageOrientationPatient], ipp0=hs[order[0]][2],
                rows=int(ds.Rows), cols=int(ds.Columns), weight_kg=weight_g / 1000.0, dose_MBq=dose / 1e6,
                half_life_s=half, decay_dt_s=dt_s, series_minus_inj_s=(ser - inj).total_seconds(),
                ref_is_series_time=bool(ref == ser), frame_of_reference=str(ds.get("FrameOfReferenceUID", "")))
    return vol * np.float32(scale), geom


def load_seg_on_pet(seg_file, geom):
    """Expert lesion mask as bool[z,y,x] on the PET grid; frames matched to PET slices by position."""
    ds = pydicom.dcmread(seg_file)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None]
    arr = arr > 0
    mask = np.zeros((geom["n_slices"], geom["rows"], geom["cols"]), bool)
    assert arr.shape[1:] == mask.shape[1:], (arr.shape, mask.shape)
    worst = 0.0
    for fi, fr in enumerate(ds.PerFrameFunctionalGroupsSequence):
        if not arr[fi].any():
            continue
        p = float(np.dot(geom["normal"], [float(v) for v in fr.PlanePositionSequence[0].ImagePositionPatient]))
        k = int(np.argmin(np.abs(geom["pos"] - p)))
        worst = max(worst, abs(geom["pos"][k] - p))
        mask[k] |= arr[fi]
    return mask, worst
