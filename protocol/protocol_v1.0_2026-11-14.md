# P01 analysis protocol, version 1.0

**Working title:** Tumour sink effect with [18F]PSMA-1007 PET/CT: between-patient and within-patient analysis of an open, expert-annotated dataset

**Frozen:** 2026-11-14, before any normal-organ uptake value was computed.
**Investigator:** Mallareddy Banala, MD (ORCID 0009-0000-0774-174X). Co-authors to be confirmed.
**Status of knowledge when frozen:** tumour volumes (expert masks), radiotracer, scanner, injected activity, body weight, uptake time and scan dates had been read for all 597 studies. No organ uptake and no lesion uptake had been computed.

## 1. Background and gap

Sequestration of PSMA ligands by a large tumour burden can lower uptake in normal organs (tumour sink effect). This matters for organ dosimetry and activity selection in PSMA radioligand therapy, and for any criterion that uses liver or salivary uptake as a reference.

- Shown for 68Ga-PSMA-11 at very high tumour volume (Gafita 2022, doi:10.2967/jnumed.121.261906) and for 68Ga-PSMA-617 (Zamanian 2025, doi:10.1186/s40644-025-00910-z).
- For [18F]PSMA-1007 the one large study (Tragardh 2025, n = 1,086, doi:10.1111/cpf.70016) had a median tumour volume of 3.8 mL and found almost no association.
- Published within-patient analyses used 68Ga tracers in about 30 patients each (doi:10.3390/cancers15092592; doi:10.23736/S1824-4785.25.03627-1).
- No within-patient or high-burden [18F]PSMA-1007 analysis was found (PubMed, 2026-11-14; Embase and Scopus still to be searched before submission).

## 2. Objectives

**Primary.** In [18F]PSMA-1007 PET/CT, test whether higher PSMA-positive tumour burden is associated with lower uptake in liver, spleen and kidneys:
- (A) across scans (between-patient association), and
- (B) within patients over time (repeat scans with the same tracer).

**Secondary.**
1. Shape and size of the association: categories of tumour volume, and the proportion of very-high-burden scans with a 30% or greater reduction (the threshold Gafita et al. called relevant).
2. Mechanistic check: whether organ uptake scales with the fraction of injected activity not sequestered in tumour (1 - TLF).
3. Descriptive repeat of all analyses in the 68Ga-PSMA-11 scans.
4. Exploratory organs: parotid and submandibular glands (field of view permitting) and blood pool (aorta).

## 3. Design and data

- Retrospective secondary analysis of open, de-identified data; reported to STROBE.
- Dataset: PSMA-PET-CT-Lesions, LMU Munich (TCIA doi:10.7937/r7ep-3x37; descriptor doi:10.1038/s41597-026-07821-z), obtained from NCI Imaging Data Commons v25, licence CC BY 4.0.
- 597 PET/CT studies in 378 men, 2014 to 2022 (dates shifted, intervals preserved): 369 [18F]PSMA-1007 and 228 68Ga-PSMA-11; three scanners.
- Expert whole-body lesion masks are part of the dataset.

**Change from the first plan.** A head-to-head 18F versus 68Ga comparison was planned. It was dropped as a primary aim after the go/no-go check (see `03_results/gonogo_decision.md`): only 6 68Ga scans reach 1,355 mL, and tracer is tied to scanner.

## 4. Analysis sets

- **Primary set:** [18F]PSMA-1007 studies with uptake time 45 to 120 min (343 of 369 by header) that pass image QC (section 7). The window was fixed from the metadata before outcomes were measured.
- **Within-patient set:** consecutive pairs of primary-set scans in the same patient.
- **Sensitivity sets:** all 18F scans regardless of uptake time; same-scanner pairs only; pairs no more than 365 days apart; positive scans only.
- **68Ga set:** 68Ga-PSMA-11 studies with the same uptake window.

## 5. Variables

**Exposure (tumour burden), from the expert masks**
- Primary: PSMA-positive tumour volume (PSMA-TV, mL), analysed as log2(PSMA-TV + 1).
- Secondary: total lesion fraction, TLF = (sum of lesion SUV x voxel volume) / body weight, i.e. the fraction of decay-corrected injected activity inside the lesion mask; total lesion PSMA (TV x SUVmean).

**Outcomes**
- Primary: SUVmean (body-weight SUV) of liver, spleen and kidneys (volume-weighted mean of both kidneys), log-transformed.
- Sensitivity: SUVmedian; SUL (Janmahasatian lean body mass) where height is recorded.
- Exploratory: parotid and submandibular glands, aortic blood pool.

**Covariates:** uptake time (min), scanner (3 levels), age, body weight. Injected activity and CT contrast are examined descriptively.

**Not available in this dataset:** therapy before or between scans, PSA, renal function, androgen-deprivation status. These are stated limitations.

## 6. Image processing

1. **PET to SUV.** DICOM PET (Bq/mL) to body-weight SUV with decay correction from injection time to series start, half-life from the header. Checked against the authors' published timing file and the descriptor's lesion statistics.
2. **Lesion mask.** Expert DICOM SEG placed on the PET grid by slice position.
3. **Organ masks.** TotalSegmentator 2.18.0, task `total`, fast (3 mm) model, on the co-acquired CT. Salivary glands: task `head_glands_cavities`, exploratory.
4. **Organ VOI on PET.** Mask resampled to the PET grid (nearest neighbour); eroded by one PET voxel; every voxel within 10 mm of the expert lesion mask removed.
5. **Validity rule.** VOI accepted if at least 200 mL (liver), 30 mL (spleen) or 30 mL (each kidney) remain; otherwise the organ value is missing for that scan.

## 7. Quality control

- SUV calibration: fraction of injected activity inside the PET field of view must be plausible (flag if below 0.2 or above 1.1).
- Geometry: PET and CT share a frame of reference; slice spacing taken from positions, not headers (a header error on one scanner was found and corrected during the go/no-go check).
- Segmentation: automated volume range checks for every organ; overlay images of a random 5% of scans and of every statistical outlier reviewed by the physician investigator.
- Technical validation: 40 randomly chosen scans (fixed seed 20261114) re-segmented with the full-resolution model; agreement of organ SUVmean by ICC and Bland-Altman limits.

## 8. Statistical analysis

**Main model (one per organ).** Linear mixed model with a random intercept per patient:

log(SUVmean) = b0 + bW x (x_ij - mean_i) + bB x mean_i + uptake time + scanner + age + weight + u_i + e_ij

where x_ij = log2(PSMA-TV + 1) for scan j of patient i and mean_i is that patient's mean. bW is the within-patient effect (aim B); bB is the between-patient effect. The overall cross-sectional effect (aim A) comes from the same model without the split. Effects are reported as percentage change in organ SUVmean per doubling of tumour volume, with 95% confidence intervals.

**Multiplicity.** Three organs per aim; Holm correction within each aim. Two-sided alpha 0.05.

**Categories.** Negative scans; more than 0 to 25; 25 to 189; 189 to 532; 532 to 1,355; 1,355 mL or more (Gafita cut-points). Pre-specified contrast: 1,355 mL or more versus negative scans.

**Within-patient pairs, descriptive.** Spearman correlation between change in log2(PSMA-TV + 1) and change in log(SUVmean); cluster-robust inference by patient.

**Mechanistic check.** Regress log(SUVmean) on log(1 - TLF); a slope near 1 means organ uptake falls in proportion to the activity sequestered.

**Non-linearity.** Restricted cubic spline (3 knots) for tumour burden.

**68Ga scans.** Same models, reported descriptively. Any tracer-by-burden interaction is exploratory only.

**Sensitivity analyses.** All uptake times; same-scanner pairs; pairs no more than 365 days apart; positive scans only; SUVmedian; no erosion and 10 mm erosion; 3 cm sphere in the right liver lobe; SUL; scanner-stratified models; exclusion of influential scans; full-resolution segmentation subset; TLF and total lesion PSMA as exposure.

**Missing data.** Complete cases per organ; counts reported.

**Precision available.** 116 same-tracer 18F pairs give 80% power for a correlation of 0.26 or stronger (two-sided alpha 0.05).

**Software.** Python 3.11 (versions in `03_code_and_environments/requirements_lock_nucrad.txt`); mixed models fitted in Stata/MP 19 and reproduced independently in Python (statsmodels).

## 9. Limitations accepted in advance

Single centre; no therapy, PSA or renal-function data; tracer, scanner and period are correlated; lesion masks from one reader with expert cross-check; radioligand therapy between scans lowers salivary uptake by itself, so salivary results are exploratory; kidney values on 68Ga scans include urinary activity.

## 10. Ethics, transparency, reporting

- Public, de-identified data released with ethics approval at the source institution. Whether the home institution requires a formal non-human-subjects determination is to be confirmed before submission.
- No attempt will be made to re-identify anyone.
- All code and derived tables will be released (GitHub and Zenodo) at submission.
- AI assistance in coding, analysis and drafting will be disclosed according to the target journal's policy; the investigator reviews and takes responsibility for all content.
- Any departure from this protocol is recorded with date and reason in `00_protocol/deviations_log.md`.
