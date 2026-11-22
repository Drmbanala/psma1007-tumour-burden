# Tumour burden and normal-organ uptake of [18F]PSMA-1007

Analysis code, protocol and derived data tables for the study
"Tumour burden and normal-organ uptake of [18F]PSMA-1007: between-patient and within-patient analysis of an open, expert-annotated PET/CT dataset"
(Banala M, Nasir AA, Makanji R, Sfakianaki E).

## Data source
Imaging data are not redistributed here. They are available as the PSMA-PET-CT-Lesions collection from The Cancer Imaging Archive (https://doi.org/10.7937/r7ep-3x37) and the NCI Imaging Data Commons, under a CC BY 4.0 licence. Data descriptor: Jeblick K et al., Sci Data 2026;13:1023 (https://doi.org/10.1038/s41597-026-07821-z).

## Contents
- `protocol/` analysis protocol, fixed before organ uptake was measured. Decisions and additions made after the protocol are listed in Supplementary Table S1 of the article.
- `code/` scripts in the order in which they were run (01 to 22), the shared library `petlib.py`, the Stata replication (`13_models_stata.do`) and the pinned Python environment (`requirements_lock.txt`).
- `results/` derived tables: organ uptake per study (`organ_uptake.csv`), the analysis table, all model estimates (`results_models.csv`), the Stata replication output, and the post hoc kidney-asymmetry and uptake-time analyses.

## Reproducing the analysis
1. Download the collection (SEG, PET and CT series).
2. Run `01` to `05` (tumour volume, PET headers, SUV checks), `10` and `11` (organ segmentation with TotalSegmentator 2.18.0 and organ uptake), then `12_models.py`, `14_table1.py` and `18_figures.py`.
3. `21` and `22` reproduce the post hoc analyses; `13_models_stata.do` repeats the primary models in Stata.

The primary models can be re-run from `results/analysis_table.csv` without the images.

## Licence
Code: MIT licence. Derived tables: CC BY 4.0; they are derived from the collection named above, which must be cited when they are used.

## Use of a large language model
Claude (Anthropic) assisted with code development; the authors reviewed and verified the code and results.
