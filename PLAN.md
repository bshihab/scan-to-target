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

### M8 Cloud (done 2026-11-17)
DICOMweb STOW-RS upload, QIDO-RS search and WADO-RS retrieval against a Cloud Healthcare DICOM store, round trip byte-identical. Needs a GCP project, the API enabled and a login on the mini. Cost estimate before running.

### M9 Container
Dockerfile; `docker run` reproduces B1 and serves the viewer.

## Known limitations to state in the write-up
- No EPI distortion correction: the fMRI peak can sit a few mm off near air-tissue boundaries. ds000114 has no field map for this run, so it can't be done here.
- No slice timing correction (TR 2.5 s, block design: small effect on a 15 s block).
- Registration is rigid; the brain shifts after the skull is opened, which no pre-op plan accounts for.

## Change log
- 2026-11-17: plan written.
- 2026-11-17, B2: run 1 (trilinear sampling, zero-filled simulation) failed: median 0.41 mm, max 1.82 mm. Started at the true answer the fit still drifted 0.2-0.5 mm, so the cost was biased, not the optimizer. Cause 1: trilinear interpolation pulls a least-squares fit towards whole-voxel offsets (0.1 voxel is 0.4 mm at 4 mm voxels); the final motion level now samples with cubic B-splines. Run 2 (cubic, same simulation) passed the median (0.15 mm) and failed the max (1.02 mm). Cause 2 is in the simulation: resampling one volume on its own grid fills tissue that moved out of the 30-slice slab with zeros, which a real scan never contains. The simulation now crops 12 mm off the top and bottom and resamples the moved image from the full volume (`--zero-fill` reproduces the old one). The bar itself is unchanged.
- 2026-11-17, added bar B1c before downloading its data: (1) every Siemens series in neurolabusc/dcm_qa `In/Orientation` (axial, coronal, sagittal, multiband axial; real oblique geometry) passes the B1 criteria against its dcm2niix reference; (2) the GE head CT in neurolabusc/dcm_qa_ct (variable gantry tilt and slice spacing) is refused with a gantry-tilt or spacing reason rather than loaded with a wrong matrix. Tilt correction is not attempted yet.
- 2026-11-17, M3 contrast, corrected the same day. First diagnosis (now withdrawn): the planned Finger > mean(Foot, Lips) map covered most of the brain in ses-retest (10,010 voxels at t > 3.1), and Foot or Lips against rest showed whole-brain signal drops, so I switched targeting to Finger > rest. Actual main cause: volume 0 of each run is 26-27% brighter (the magnetisation has not settled; the paper says each session began with dummy scans) and it was both an outlier in the GLM and the motion-correction reference. Leading volumes brighter than 105% of the run median are now dropped and the first kept volume is the reference. With that fix the ses-retest map shrinks to 134 voxels and the condition maps are no longer dominated by global drops, so the pre-registered contrast is restored as the primary map. Finger > rest is written alongside it.
- 2026-11-17, B4b run 1 FAILED: ours and SimpleITK disagreed by 14.5/19.0 mm (mean, test/retest). Overlays of the mean EPI's edges on the T1 (out/figs/b4b_test_overlays.png) show ours following the brain surface and SimpleITK's answer pulled onto the scalp; both moved the EPI 7-12 mm from the scanner alignment, which itself visibly misaligns. Ours was also unstable: refinements from three different starts landed up to 2.4 mm apart, because the T1 samples included scalp and neck that the EPI barely shows. Method change: EPI-to-T1 registration samples the T1 only inside the head mask eroded by 8 mm (a crude skull strip) and uses 200,000 points per level. Bars added before the re-run: B4b-2a, refinements from the scanner alignment, our run-1 answer and SimpleITK's run-1 answer agree within 1.0 mm (max over sphere points) in each session; B4b-2b, SimpleITK given the same fixed-image mask agrees with ours within 2.0 mm; B4a is re-run with the new method and must still pass its original bar.
- 2026-11-17, B4b run 2 FAILED: with the scalp-stripped mask, ses-test start spread 3.60 mm (bar 1.0) and SimpleITK (same mask) 13.73 mm max away; ses-retest spread 0.65 mm (pass) and SimpleITK 3.32 mm max (bar 2.0). Mutual information between a 4 mm EPI and a T1 is not well conditioned here. Method change for run 3: boundary-based registration (Greve and Fischl 2009, the method behind FreeSurfer bbregister and FSL epi_reg): white-matter boundary points with normals from the T1, cost from the EPI contrast across the boundary (grey matter brighter than white in BOLD EPI), started from the MI answer. Bars for run 3, written before it runs: (a) B4a known-truth test with the new method, original bar; (b) start spread <= 1.0 mm in each session, starting from the scanner alignment, the run-1 answer and the run-2 answer; (c) loop closure: EPI_retest->T1_retest->T1_test vs EPI_retest->EPI_test->T1_test, with T1-T1 and EPI-EPI registered within contrast, differ by <= 2.0 mm max over sphere points; (d) SimpleITK (masked) agreement is still reported, with the same 2.0 mm bar, and its run-1 overlay is kept as evidence of how it fails here.
- 2026-11-17, after an independent code review (11 findings, all addressed or disclosed; see RESULTS.md "Review"). B4a as run so far is partly circular for boundary-based registration: the pseudo-EPI is an inverted copy of the very T1 whose white-matter surface drives the cost, with no EPI distortion. Bar B4a-d added before running: the same 30 transforms, but the pseudo-EPI also gets (1) a smooth displacement along the EPI phase-encode axis, 4 mm peak over the orbitofrontal region and 2.5 mm over each temporal pole, (2) 70% signal dropout in the same regions, (3) a non-linear contrast (gamma 0.7 after inversion). Error is still measured against the known rigid transform. Bar: median <= 1.0 mm, 95th percentile <= 2.0 mm, none above 5 mm (the original B4a bar).
- 2026-11-17, GLM noise model: ordinary least squares replaced by AR(1) prewhitening (nilearn's algorithm: Yule-Walker on OLS residuals, coefficients truncated to 0.01 bins). B3 now runs with AR(1) in both tools; the OLS comparison is kept (`--ols`).
- 2026-11-17, M7 (no pass bar). With the AR(1) GLM, sub-01's peak-to-peak distance between sessions is 15.1 mm (it was 3.2 mm with OLS): the test-session peak moved from MNI (45, -12, 55) to (35, -18, 46). A single voxel's maximum is a fragile target. Added before running subjects 2-10: a second target rule reported alongside the peak, the t-weighted centre of mass of the connected cluster (t > 3.1, 6-connectivity) that contains the peak, in each session. Both rules are reported for every subject; neither is chosen after seeing the results.
- 2026-11-17, M8: run against Bilal's own Google Cloud project (gcloud configuration `bilal`, his account; the mini's default gcloud login belongs to someone else and is never used here). `gcloud_token()` now requires a configuration name and the script takes it as its first argument, so a token can't come from whichever configuration happens to be active. A first attempt got HTTP 404 because the dataset-creation commands had not run (a shell quoting mistake on my side, nothing reached the store); after creating the dataset and store the run passed. Store and dataset deleted afterwards.
- 2026-11-17, M7 result (no pass bar): the stated expectation failed. Peak rule median 77.0 mm (1/10 under 10 mm), cluster rule median 67.0 mm (2/10). 11 of 20 whole-brain peaks lie more than 25 mm from the hand area, 5 in the cerebellum or lower temporal/occipital lobe. A whole-brain maximum is not a usable cortical target here. Third target rule, written and committed BEFORE it is computed on any subject (all ten t-maps already exist, so this is a re-analysis, not a new scan): hand-area region = two spheres of radius 20 mm in approximate MNI space centred at (-37, -21, 58) and (37, -21, 58), intersected with the session's brain mask; each EPI voxel is placed in MNI through the session's EPI-to-T1 and T1-to-MNI matrices. Per session and hemisphere: (3a) the peak t voxel inside the sphere; (3b) the t-weighted centre of the connected t > 3.1 cluster that contains that peak, restricted to the sphere (none if the peak is below 3.1). Hemisphere for each subject: the one with the larger sum of the two sessions' 3a peak t-values (uses both sessions symmetrically; fixed before distances are seen). Distance as before: the retest point carried into the test T1 by the T1-to-T1 registration. Both hemispheres are reported; the chosen one is the headline. Expectation: median under 10 mm. Caveat stated now: the sphere caps any distance at about 40 mm plus registration differences, so this rule measures reproducibility inside the hand area, not whether the map finds the hand area unaided.

