# Results

Every number here is printed by a script in `validation/` and saved under `out/`. Failed runs are kept. Bars are defined in [PLAN.md](PLAN.md). The scripts reproduce the latest run of each bar from the current code; where an earlier run came from older code, the commit is named.

## B1 DICOM reader vs dcm2niix (pass, 8/8)

`python validation/b1_dicom_vs_dcm2niix.py`

| Series | Scanner | Layout | Shape | Corner error | Voxel values |
|---|---|---|---|---|---|
| GE mr_0004 to 0007 | GE fMRI, 3 mm slices with 2 mm gaps, 10 time points | one file per slice per time point, all with the same acquisition time | 64x64x15x10 | 5.9e-6 mm | identical |
| Siemens mr_0003 to 0006 | Siemens EPI, 4 phase-encoding directions | mosaic, 5 slices tiled 3x3 in each file | 72x72x5x2 | at most 3.2e-11 mm | identical |

The reference is dcm2niix v1.0.20261024's conversion shipped with the dataset. The comparison maps reference voxels to ours by an axis permutation with flips whose offset comes from the grid sizes, not from the matrices, so a position error can't hide. A mutation check confirms it: deleting the mosaic position correction moves the corners by 305 mm and fails the series. (An earlier version of the check took the offset from the matrices and reported 0 mm for that same broken reader; the value comparison still failed it. Fixed before any result was recorded.)

B1b, synthetic series with known geometry (`pytest tests/test_dicom.py`): 11/11 pass, covering an oblique stack rotated 20 degrees about two axes, a stack written top-down, a 3-time-point series in shuffled file order, per-slice rescale slopes, PHI kept out of metadata, and refusals for gantry tilt, uneven spacing, a missing slice, mixed orientations, two series in one folder, and two echoes sharing one series (added after the review; previously they would have loaded as two time points).

B1c, real oblique MRI and a tilted CT (`python validation/b1c_oblique_and_ct.py`): pass. All 21 Siemens series in dcm_qa's Orientation set (7 axial, 6 coronal, 6 sagittal, 2 multiband axial at 86x86x36; ascending, descending and interleaved slice orders) are voxel-identical to dcm2niix with corners within 6.8e-5 mm. Two of them are stored compressed (JPEG lossless and JPEG 2000) and failed on the first run because no decoder was installed; the pylibjpeg decoders are now dependencies. The GE head CT from dcm_qa_ct is refused: "slices are not stacked along the slice normal (in-plane offset 1.34 mm, 18.50 deg): gantry tilt or sheared stack". Loading it with a plain 4x4 matrix would have placed every slice up to 18.5 degrees off.

## B2 motion correction (pass on run 3)

`python validation/b2_motion_recovery.py`

| Run | Method | Simulation | Median error | Max error | Bar (0.2 / 0.5 mm) |
|---|---|---|---|---|---|
| 1 | trilinear | zero-filled | 0.41 mm | 1.82 mm | fail |
| 2 | cubic final level | zero-filled | 0.15 mm | 1.02 mm | fail (max) |
| 3 | cubic final level | cropped | 0.075 mm | 0.23 mm | pass |

Runs 1 and 2 ran on code from before commit f20ebbf; on current code they correspond to a trilinear final level and `--zero-fill`. 30 known motions up to 3 degrees and 3 mm, noise at the run's measured temporal SD (about 75 on a signal near 2,000), errors measured at 64 points on a 60 mm sphere. What changed between runs and why is in PLAN.md's change log. About 1.5 s per volume on the mini.

## B3 GLM vs nilearn (pass)

`python validation/b3_glm_vs_nilearn.py` (AR(1), the pipeline's model) and `--ols`

Same realigned data, mask, events, SPM HRF, 128 s cosine drift, six motion regressors, no smoothing; nilearn builds its own design matrix from the events. Contrast finger > mean(foot, lips).

| Session | Noise model | Pearson r (t-maps in mask) | Max abs t difference | Peak voxel | Peak t ours / nilearn |
|---|---|---|---|---|---|
| test | AR(1) | 0.99997 | 0.078 | identical | 6.03 / 5.99 |
| retest | AR(1) | 0.99997 | 0.061 | identical | 4.76 / 4.76 |
| test | OLS | 0.99997 | 0.052 | identical | 7.89 / 7.84 |
| retest | OLS | 0.99997 | 0.040 | identical | 4.79 / 4.75 |

The first version of the pipeline used OLS, which ignores the autocorrelation of fMRI noise and overstates t (7.9 against 6.0 at the test peak); the review flagged it and the pipeline now uses AR(1) prewhitening with nilearn's algorithm. Earlier the pipeline was also changed to drop non-steady-state volumes (volume 0 was 26-27% brighter than the rest of each run), and the contrast was briefly switched to finger vs rest on a diagnosis that turned out to be wrong, then restored (PLAN.md change log).

## B4a registration, known truth (pass)

`python validation/b4_registration.py a`

30 pseudo-EPIs made from the T1 (inverted contrast inside the head, 4 mm grid, 30-slice slab, 6 mm blur, 3% noise, 15% bias field), each moved by a random rigid transform up to 10 degrees and 15 mm, registered back by mutual information from the scanner alignment.

| Median | 95th percentile | Worst sphere point | Cases over 5 mm | Bar |
|---|---|---|---|---|
| 0.48 mm | 0.56 mm | 0.89 mm | 0 | pass (1.0 / 2.0 / 5.0 mm) |

Run 1 (commit 272978d) used mutual information alone; about 5 s per registration. Started at the true answer, the fit still drifts about 0.48 mm, so the floor is a small bias in the cost rather than capture range.

| Run | Method | Pseudo-fMRI | Median | 95th pct | Worst point | Bar |
|---|---|---|---|---|---|---|
| 1 | mutual information | inverted T1 | 0.48 mm | 0.56 mm | 0.89 mm | pass |
| 2 | template mask, MI, then boundary-based (the pipeline) | inverted T1 | 0.23 mm | 0.36 mm | 0.60 mm | pass |
| d | the pipeline | inverted T1 with simulated EPI distortion and dropout | 0.33 mm | 0.64 mm | 1.10 mm | pass |

Run 2 is partly circular, as the review pointed out: the pseudo-fMRI is an inverted copy of the same T1 whose white-matter surface drives the boundary cost, with no EPI distortion, so its 0.23 mm is optimistic. Bar B4a-d (written before it ran) adds a 4 mm displacement along the phase-encode axis over orbitofrontal cortex and 2.5 mm over each temporal pole, 70% signal dropout there, and a non-linear contrast; the error is still measured against the known rigid transform. It is still easier than real data (the real fMRI has a dark scalp and heavier inferior signal loss; `out/figs/pseudo_epi_distortion.png`).

## B4b registration on real data (run 1 fail, run 2 fail, run 3 partly pass)

`python validation/b4_registration.py b`

Run 1 ran at commit 272978d, run 2 on uncommitted code between 272978d and 14f0eae (MI with an 8 mm-eroded head mask), run 3 at 14f0eae. There is no ground truth for registering a real EPI to a real T1. Run 1's bar was agreement with SimpleITK; after it failed, bars that need no second tool were added before each later run (PLAN.md change log).

| Run | Method | Start spread (bar 1 mm) | Loop closure (bar 2 mm) | vs SimpleITK, max (bar 2 mm) |
|---|---|---|---|---|
| 1 | mutual information, whole head | not measured | not measured | 20.4 / 21.4 mm: fail |
| 2 | mutual information, scalp stripped by erosion | 3.60 / 0.65 mm: fail | not measured | 13.7 / 3.3 mm: fail |
| 3 | template brain mask, mutual information, then boundary-based | 0.001 / 0.001 mm: pass | 0.91 mm max, 0.68 mean: pass | 9.2 / 10.6 mm: fail |

(test / retest session.) Start spread: refinements from the scanner alignment, the run-1 answer and the run-2 answer. Loop closure: EPI_retest to T1_retest to T1_test against EPI_retest to EPI_test to T1_test, where the T1-T1 and EPI-EPI steps are within one contrast; the head moved 6.8 and 8.9 degrees between sessions, so the loop is not trivially short.

What this shows and what it doesn't. A start spread of 0.001 mm shows the optimizer is deterministic and lands in one basin from all three starts; it says nothing about accuracy by itself. Run 3 converges to one answer from very different starts, and the two sessions' answers are consistent with each other through independent within-contrast registrations to under a millimetre. It moved the EPI 5.6 mm (test) and 1.2 mm (retest) from the scanner's own alignment, and 5-7 mm from the run-1 and run-2 mutual-information answers. SimpleITK's mutual-information registration, configured as in PLAN.md, disagrees by about 10 mm; overlays from run 1 (`out/figs/b4b_test_overlays.png`) show its answer pulled onto the scalp, and boundary-based refinement started from its answer stops at a much worse cost (0.96 against 0.70). That is evidence about SimpleITK's setup here, not proof that ours is right; the B4b SimpleITK bar is reported as failed.

## B5 viewer geometry (pass, rebuilt after the review)

`python validation/b5_geometry_crosscheck.py` and `pytest tests/test_viewer.py`

The first B5 compared a JS matrix chain the viewer never ran with the same chain in numpy, and the server's "re-read" check multiplied the same matrix by the same point twice. Neither could fail, and both missed a real bug the review found: plans gave the target's voxel index in the cropped display volume while naming the original T1 file, 24 and 23 voxels (about 3 cm) away. Fixed; B5 was rebuilt so it can fail:

- `b5_geometry_crosscheck.py` recomputes every matrix the viewer uses from the source files (NIfTI headers via nibabel, the saved registration) and compares 1,000 points pushed through the viewer's own JS with the scene's matrices: display voxel to world, world to original-T1 voxel, world to t-map voxel, RAS to LPS. Largest difference 6e-14. Re-introducing the cropped-index bug makes it fail.
- `tests/viewer_phantom.mjs` renders a synthetic volume (flipped first axis, 7 degree tilt) with markers at the patient's left, front and top through the viewer's real rendering function, and checks screen positions against the radiological and neurological conventions, plus the edge letters, in axial, coronal and sagittal views. A single bright voxel is drawn centred at exactly the pixel position a click maps back to it (0.000 px). Swapping left and right fails 6 checks; going back to sampling pixel corners instead of centres (a half-pixel offset the review found) fails the click check.
- In Chrome: target at the map peak, entry snapped to the scalp, plan exported; the server stores the target's voxel index in the original T1 file computed from its own copy of that file's matrix.

## M7 test-retest of the target

`python validation/m7_test_retest.py --delete-raw sub-01 ... sub-10`

No pass bar. Expectation stated before running: under 10 mm for most subjects. Two target rules, both fixed before subjects 2-10 ran: the map's peak voxel, and the t-weighted centre of the cluster (t > 3.1) containing the peak. The retest target is carried into the test T1 by a T1-to-T1 registration inside the template brain mask.

M7_TABLE

MNI coordinates come from a 12-parameter affine and are good to several millimetres. sub-01's peak moved 15 mm between the OLS and AR(1) versions of the same GLM, which is why the second rule was added.

An earlier draft of this file argued from sub-01's right-hemisphere motor peak and left-cerebellum response that the subject tapped with the left hand. The review pointed out that a left-right flipped image would look exactly the same, so that inference is withdrawn. The dataset paper does not say which hand was used.

## Review

After the first full pass, an independent agent with no access to this conversation reviewed the code and docs and reported 11 findings. All are fixed or disclosed:

| # | Finding | What was done |
|---|---|---|
| 1 | Plan voxel index in the cropped display volume, labelled as the original T1 file (about 3 cm off) | Fixed; scene carries the original matrix; server recomputes the index; B5 and a server test now fail on the old behaviour |
| 2 | Overlay labelled "finger vs rest" while showing finger > mean(foot, lips) | Fixed; label now names the contrast and noise model |
| 3 | Plan and viewer checks could not fail | B5 rebuilt (above); server check renamed and described for what it is |
| 4 | Left-hand inference cannot exclude a flipped image | Withdrawn (M7 above); README limits say left and right come from the headers |
| 5 | B4 evidence did not cover the shipped method; B4a circular; reproducibility claim too broad | B4a run 2 and B4a-d added; commits named for older runs; start spread described as determinism |
| 6 | Docker image could not serve the viewer or find data | Viewer moved into the package as package data; CLI paths relative to the working directory; Dockerfile rewritten (still not built here: no disk space for a VM) |
| 7 | Two echoes in one series would load as two time points | Refused with the reason, or one echo chosen with `echo=` (new test) |
| 8 | Viewer drew each pixel at its corner, half a pixel from where clicks land | Fixed; covered by the phantom test |
| 9 | OLS overstates t; motion cache not tied to its input | AR(1) prewhitening (B3 against nilearn's AR(1)); cache keyed on the input hash and dropped volumes; threshold labelled uncorrected |
| 10 | Server: no Host or Origin check, robot's true pose served, negative Content-Length, same-second overwrite, free-text metadata kept | Host and Origin checks, whitelist of served files, length check, unique file names; SeriesDescription and input paths no longer stored (HTTP test covers each case) |
| 11 | README stale and overclaiming | Rewritten |
