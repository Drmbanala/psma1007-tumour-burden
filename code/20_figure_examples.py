"""P01 step 20: publication figure of organ volumes of interest (600 dpi).

For each listed study the PET (SUV), the organ labels of the 3 mm model (as used for measurement), the measured
volumes of interest (label eroded by one PET voxel, voxels within 10 mm of the lesion mask removed) and the
lesion mask are resampled to a common 1 mm grid, so that outlines are drawn without voxel steps.
No measurement is changed by this script; it only draws.
Usage: python 20_figure_examples.py [--uids FILE] [--raw DIR] [--out FILE_WITHOUT_EXTENSION]
"""
import argparse, glob, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; PROJ = HERE.parent
spec = importlib.util.spec_from_file_location("s11", HERE / "11_extract_organ_uptake.py"); s11 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s11)
ORG = {"liver": (5, "tab:green"), "spleen": (1, "tab:purple"), "kidney_right": (2, "tab:blue"), "kidney_left": (3, "tab:cyan"), "aorta": (52, "tab:orange")}
SP = 1.0                                                                   # display grid, mm


def pet_geometry(img, g, info):
    img.SetSpacing((float(g["px_col"]), float(g["px_row"]), float(info["dz"]))); img.SetOrigin([float(v) for v in g["ipp0"]])
    iop = np.array(g["iop"], float)
    img.SetDirection(tuple(np.stack([iop[:3], iop[3:], np.array(g["normal"], float)], axis=1).ravel()))
    return img


def study_layers(raw, pid, uid):
    base = Path(raw) / pid / uid
    pet_dir = sorted(glob.glob(str(base / "PT_*")))[0]; seg_file = sorted(glob.glob(str(base / "SEG_*" / "*.dcm")))[0]
    suv, g, info = s11.build_pet_grid(pet_dir); shape = suv.shape
    lesion = s11.lesion_on_grid(seg_file, g, info, shape)[0]
    label_file = PROJ / "01_derived_data" / "organ_seg" / uid / "total_fast.nii.gz"
    lab_pet = s11.labels_on_grid(label_file, g, info, shape)
    sampling = (float(info["dz"]), float(g["px_row"]), float(g["px_col"]))
    near = ndi.distance_transform_edt(~lesion, sampling=sampling) <= 10.0 if lesion.any() else np.zeros(shape, bool)
    lab_ct = sitk.ReadImage(str(label_file))
    # display grid: box around the organs, identity orientation, 1 mm
    arr = sitk.GetArrayFromImage(lab_ct); zz, yy, xx = np.nonzero(np.isin(arr, [v[0] for v in ORG.values()]))
    corners = np.array([lab_ct.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for x in (xx.min(), xx.max()) for y in (yy.min(), yy.max()) for z in (zz.min(), zz.max())])
    lo, hi = corners.min(0) - np.array([70, 60, 50.0]), corners.max(0) + np.array([70, 60, 50.0])
    ref = sitk.Image([int(np.ceil((h - l) / SP)) for l, h in zip(lo, hi)], sitk.sitkFloat32); ref.SetSpacing((SP,) * 3); ref.SetOrigin([float(v) for v in lo])
    def cidx(ids):                                             # centre of the given labels on the display grid (x, y, z)
        m = np.isin(arr, ids); z, y, x = np.nonzero(m)
        pt = lab_ct.TransformContinuousIndexToPhysicalPoint((float(x.mean()), float(y.mean()), float(z.mean())))
        return [int(round((pt[i] - lo[i]) / SP)) for i in range(3)]
    cl, ck = cidx([ORG["liver"][0]]), cidx([ORG["kidney_right"][0], ORG["kidney_left"][0]])
    planes = {"corL": ("cor", cl[1]), "corK": ("cor", ck[1]), "ax": ("ax", cl[2])}
    def show(img):                                             # resample one layer and keep only the three displayed planes
        v = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(img, sitk.sitkFloat32), ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        return {k: (v[:, i, :].copy() if pl == "cor" else v[i, :, :].copy()) for k, (pl, i) in planes.items()}
    def from_pet(a):
        return pet_geometry(sitk.GetImageFromArray(np.asarray(a, np.float32)), g, info)
    L = dict(pet=show(from_pet(np.nan_to_num(suv))), lesion=show(from_pet(lesion)), label={}, voi={})
    for name, (lid, _) in ORG.items():
        L["label"][name] = show(sitk.Cast(lab_ct == lid, sitk.sitkFloat32))
        voi = ndi.binary_erosion(lab_pet == lid, structure=ndi.generate_binary_structure(3, 1)) & ~near
        L["voi"][name] = show(from_pet(voi))
    return L


def draw(ax, L, key):
    ax.imshow(L["pet"][key], cmap="gray_r", vmin=0, vmax=20, origin="lower" if key.startswith("cor") else "upper", interpolation="bilinear")
    for name, (_, col) in ORG.items():
        if L["label"][name][key].max() > 0.5: ax.contour(L["label"][name][key], levels=[0.5], colors=col, linewidths=0.45, linestyles="dotted")
        if L["voi"][name][key].max() > 0.5: ax.contour(L["voi"][name][key], levels=[0.5], colors=col, linewidths=0.8)
    if L["lesion"][key].max() > 0.5: ax.contour(L["lesion"][key], levels=[0.5], colors="red", linewidths=0.7)
    ax.set_aspect("equal"); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", default=str(PROJ / "03_results" / "qc_images" / "figure_example_uids.txt"))
    ap.add_argument("--raw", default=str(HERE.parents[2] / "01_raw_data" / "PSMA-PET-CT-Lesions_IDC" / "psma_pet_ct_lesions"))
    ap.add_argument("--out", default=str(PROJ / "03_results" / "figures" / "Figure1_organ_volumes_of_interest_600dpi"))
    a = ap.parse_args()
    uids = [u.strip() for u in open(a.uids) if u.strip()]
    tab = pd.read_csv(PROJ / "01_derived_data" / "study_table_v3_verified.csv").set_index("StudyInstanceUID")
    layers = [study_layers(a.raw, tab.loc[u, "PatientID"], u) for u in uids]
    fig, axs = plt.subplots(len(layers), 3, figsize=(6.85, 2.75 * len(layers)), squeeze=False)
    for r, L in enumerate(layers):
        draw(axs[r, 0], L, "corL"); draw(axs[r, 1], L, "corK"); draw(axs[r, 2], L, "ax")
        axs[r, 0].text(0.02, 0.97, "ABCDEF"[r], transform=axs[r, 0].transAxes, fontsize=11, fontweight="bold", va="top", color="black")
        if r == 0:
            x0, y0 = 12, 12; axs[r, 0].plot([x0, x0 + 50 / SP], [y0, y0], color="black", lw=1.6); axs[r, 0].text(x0 + 25 / SP, y0 + 6, "5 cm", ha="center", va="bottom", fontsize=6.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02, hspace=0.03)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".png", dpi=600); fig.savefig(a.out + ".tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)
    print("wrote", a.out + ".png and .tif for", len(layers), "studies")


if __name__ == "__main__":
    main()
