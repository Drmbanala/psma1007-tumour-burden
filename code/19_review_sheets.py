"""P01 step 19: contact sheets of overlay images for a given list of studies, in list order.
Usage: python 19_review_sheets.py <overlay folder> <text file with StudyInstanceUIDs> <file prefix>"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

qc, lst, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
uids = [s.strip() for s in open(lst) if s.strip()]
pngs = [qc / f"{u}.png" for u in uids if (qc / f"{u}.png").exists()]
for s in range(0, len(pngs), 6):
    fig, ax = plt.subplots(3, 2, figsize=(20, 12.5))
    for a in ax.ravel():
        a.axis("off")
    for a, p in zip(ax.ravel(), pngs[s:s + 6]):
        a.imshow(plt.imread(p))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    fig.savefig(qc / f"{prefix}_{s // 6 + 1:02d}.png", dpi=75); plt.close(fig)
print(f"{prefix}: {len(pngs)} of {len(uids)} listed studies have an overlay image; {(len(pngs) + 5) // 6} sheets written")
