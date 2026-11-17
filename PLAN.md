# Plan

Written 2026-11-17, before any pipeline code. The bars below are fixed; if one is missed the result is reported as a miss, not re-scoped. Changes to a bar after a run must be logged at the bottom with the reason.

Target role: medical-imaging software at a brain-computer-interface company (ingest scanner data, fuse fMRI with anatomy, place electrode targets in a GUI, hand images to a surgical robot, track the patient intraoperatively).

## Constraints

- Mac mini, 1.5 GB free disk at the start. Raw data is fetched per subject and can be deleted after processing; only derived outputs are kept.
- Python 3.13 with numpy, scipy, nibabel, pydicom. Validation-only extras (nilearn, SimpleITK) are allowed because they are the independent reference, never inside the pipeline.
- No cloud spend without Bilal's go-ahead and an estimate.

## Milestones and pre-registered bars

### M1 Ingest
DICOM series reader written on top of pydicom (pydicom only parses tags; the geometry is ours): groups files by SeriesInstanceUID, splits time points, sorts slices along the slice normal, builds the voxel-to-LPS matrix from ImageOrientationPatient, ImagePositionPatient and PixelSpacing, converts to RAS, applies RescaleSlope/Intercept, unpacks Siemens mosaics, and refuses (with a reason) gantry tilt, uneven slice spacing, missing slices and mixed orientations.

Bar B1: for each of the 8 dcm_qa_nih series, the volume we produce matches the dcm2niix reference NIfTI when both are sampled at the same world points: identical voxel values at every voxel and voxel-to-RAS matrices equal within 0.01 mm at the eight corners of the volume. Series we refuse count as misses.

Bar B1b: unit tests on synthetic series for each refusal case and for an oblique series (rotated 20 degrees about two axes), where the expected matrix is known by construction. All pass.

### M2 Motion correction
Rigid 6-parameter realignment of every fMRI volume to the first, least squares on intensity.

Bar B2: a single real fMRI volume moved by 30 known random rigid transforms (up to 3 degrees and 3 mm, typical head motion) is realigned with median error at most 0.2 mm and maximum at most 0.5 mm, measured at points on a 60 mm sphere around the brain centre.

### M3 Motor map
Own GLM: SPM-style double-gamma HRF, boxcar regressors for Finger, Foot and Lips convolved at the TR, cosine drift terms (128 s cut-off), the 6 motion parameters, OLS, t-statistic for Finger > mean(Foot, Lips).

Bar B3: nilearn's FirstLevelModel run on the same data with the same design and noise_model='ols' gives a t-map whose voxelwise Pearson correlation with ours is at least 0.999 inside the brain mask, and the peak voxel is the same.

### M4 Fusion
Rigid registration of the mean fMRI volume to the T1 by Mattes-style mutual information (joint histogram with Parzen smoothing), three resolution levels, Powell optimizer.

Bar B4a (known truth): a pseudo-EPI is made from the T1 (downsampled to 4 mm, contrast inverted, blurred, noise added) and moved by 30 random rigid transforms up to 10 degrees and 15 mm. Recovered transform error, measured at points on a 60 mm sphere: median at most 1.0 mm, 95th percentile at most 2.0 mm, none above 5 mm.

Bar B4b (agreement): on the real mean EPI to T1 for sub-01, both sessions, SimpleITK's Mattes MI rigid registration (its defaults with three levels) agrees with ours within 2.0 mm at those sphere points. Neither is ground truth; disagreement above 2 mm means at least one is wrong and gets investigated.

### M5 Planning GUI
Static HTML/JS (no framework) served by a stdlib Python server. Three orthogonal slices of the T1 resampled on the fly through the voxel-to-RAS matrices, fMRI t-map overlay with a threshold slider, crosshair readout in voxel, RAS and LPS, target and entry placement, a slice perpendicular to the trajectory, export of plan.json.

Bar B5: the JS world-to-voxel and voxel-to-world code agrees with the Python code on 1,000 random points within 1e-6 mm (run under node). A plan exported from the browser and re-read in Python reproduces the clicked target within 0.5 voxel. Checked in a real browser with screenshots.

### M6 Robot hand-off
Point-based rigid registration (SVD, with the reflection fix), fiducial registration error, and Fitzpatrick's predicted target registration error (TRE) for a given fiducial layout and localisation noise.

Bar B6: for 4 fiducial layouts (well spread, clustered, collinear-ish, planar) and noise 0.25 mm, Monte Carlo TRE at the planned target over 20,000 trials matches Fitzpatrick's prediction within 10% (RMS). The layout ranking must also match.

### M7 Test-retest of the target
For sub-01 (and every ds000114 subject if disk allows), plan the finger-peak target independently in each session, carry the retest target into the test T1 by T1-to-T1 registration, and report the distance.

No pass bar: this measures fMRI reproducibility more than this software. Stated expectation before running: under 10 mm for most subjects, based on published motor-fMRI peak reproducibility.

### M8 Cloud (needs Bilal)
DICOMweb STOW-RS upload, QIDO-RS search and WADO-RS retrieval against a Cloud Healthcare DICOM store, round trip byte-identical. Needs a GCP project, the API enabled and a login on the mini. Cost estimate before running.

### M9 Container
Dockerfile; `docker run` reproduces B1 and serves the viewer.

## Known limitations to state in the write-up
- No EPI distortion correction: the fMRI peak can sit a few mm off near air-tissue boundaries. ds000114 has no field map for this run, so it can't be done here.
- No slice timing correction (TR 2.5 s, block design: small effect on a 15 s block).
- Registration is rigid; the brain shifts after the skull is opened, which no pre-op plan accounts for.

## Change log
- 2026-11-17: plan written.
