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

`python validation/b4_registration.py a` and `... d`. Runs 3 and d2 are the current method (two starts, PLAN.md change log 2026-11-18/19); the table further down keeps every run.

30 pseudo-EPIs made from the T1 (inverted contrast inside the head, 4 mm grid, 30-slice slab, 6 mm blur, 3% noise, 15% bias field), each moved by a random rigid transform up to 10 degrees and 15 mm, registered back by the pipeline's own method (runs 3 and d2: boundary-based from the scanner alignment and from the mutual-information answer, the lower cost kept).

| Run | Median | 95th percentile | Worst sphere point | Cases over 5 mm | Bar |
|---|---|---|---|---|---|
| 3 (current) | 0.22 mm | 0.34 mm | 0.57 mm | 0 | pass (1.0 / 2.0 / 5.0 mm) |
| d2 (current, with distortion) | 0.41 mm | 0.60 mm | 0.98 mm | 0 | pass |

Run 1 (commit 272978d) used mutual information alone; about 5 s per registration (the current method takes about 20 s). Started at the true answer, the fit still drifts about 0.48 mm, so the floor is a small bias in the cost rather than capture range.

| Run | Method | Pseudo-fMRI | Median | 95th pct | Worst point | Bar |
|---|---|---|---|---|---|---|
| 1 | mutual information | inverted T1 | 0.48 mm | 0.56 mm | 0.89 mm | pass |
| 2 | template mask, MI, then boundary-based (the pipeline) | inverted T1 | 0.23 mm | 0.36 mm | 0.60 mm | pass |
| d | run-2 pipeline | inverted T1 with simulated EPI distortion and dropout | 0.33 mm | 0.64 mm | 1.10 mm | pass |
| 3 | boundary-based from two starts, scanner alignment and MI (the current pipeline) | inverted T1 | 0.22 mm | 0.34 mm | 0.57 mm | pass |
| d2 | the current pipeline | inverted T1 with simulated EPI distortion and dropout | 0.41 mm | 0.60 mm | 0.98 mm | pass |

Run 2 is partly circular, as the review pointed out: the pseudo-fMRI is an inverted copy of the same T1 whose white-matter surface drives the boundary cost, with no EPI distortion, so its 0.23 mm is optimistic. Bar B4a-d (written before it ran) adds a 4 mm displacement along the phase-encode axis over orbitofrontal cortex and 2.5 mm over each temporal pole, 70% signal dropout there, and a non-linear contrast; the error is still measured against the known rigid transform. It is still easier than real data (the real fMRI has a dark scalp and heavier inferior signal loss; `out/figs/pseudo_epi_distortion.png`).

## B4b registration on real data, sub-01 (run 1 fail, run 2 fail, runs 3 to 5 partly pass)

`python validation/b4_registration.py b`

Run 1 ran at commit 272978d, run 2 on uncommitted code between 272978d and 14f0eae (MI with an 8 mm-eroded head mask), run 3 at 14f0eae. There is no ground truth for registering a real EPI to a real T1. Run 1's bar was agreement with SimpleITK; after it failed, bars that need no second tool were added before each later run (PLAN.md change log).

| Run | Method | Start spread (bar 1 mm) | Loop closure (bar 2 mm) | vs SimpleITK, max (bar 2 mm) |
|---|---|---|---|---|
| 1 | mutual information, whole head | not measured | not measured | 20.4 / 21.4 mm: fail |
| 2 | mutual information, scalp stripped by erosion | 3.60 / 0.65 mm: fail | not measured | 13.7 / 3.3 mm: fail |
| 3 | template brain mask, mutual information, then boundary-based | 0.001 / 0.001 mm: pass | 0.91 mm max, 0.68 mean: pass | 9.2 / 10.6 mm: fail |
| 4 | boundary-based from two starts | 0.00 / 0.00 mm: pass | 1.23 mm max, 0.79 mean: pass | 9.0 / 10.4 mm: fail |
| 5 | run 4's method; loop measured in the right frame (current) | 0.00 / 0.00 mm: pass | 1.40 mm max, 0.91 mean: pass | 9.0 / 10.4 mm: fail |

(test / retest session.) Run 5 (2026-11-20) changes only the loop measurement: the second review found its 60 mm test sphere centred on the test T1's brain, while both composed transforms act on retest-EPI points; it is now centred in the retest frame. The registrations are bit-identical to run 4's. Run 4's output is kept as `out/b4b_run4.json` (its `run` field says 3, a label the script had not updated). Start spread: refinements from the scanner alignment, the run-1 answer and the run-2 answer. Loop closure: EPI_retest to T1_retest to T1_test against EPI_retest to EPI_test to T1_test, where the T1-T1 and EPI-EPI steps are within one contrast; the head moved 6.8 and 8.9 degrees between sessions, so the loop is not trivially short.

What this shows and what it doesn't. A start spread of 0.001 mm shows the optimizer is deterministic and lands in one basin from all three starts; it says nothing about accuracy by itself. Run 3 converges to one answer from very different starts, and the two sessions' answers are consistent with each other through independent within-contrast registrations to under a millimetre. It moved the EPI 5.6 mm (test) and 1.2 mm (retest) from the scanner's own alignment, and 5-7 mm from the run-1 and run-2 mutual-information answers. SimpleITK's mutual-information registration, configured as in PLAN.md, disagrees by about 10 mm; overlays from run 1 (`out/figs/b4b_test_overlays.png`) show its answer pulled onto the scalp, and boundary-based refinement started from its answer stops at a much worse cost (0.96 against 0.70). That is evidence about SimpleITK's setup here, not proof that ours is right; the B4b SimpleITK bar is reported as failed.

B4b was only ever run on sub-01, and sub-01 happened to be a subject where the run-3 method worked. M7 later showed it failing on other subjects; B4e below is the check across all of them.

## B4e registration across all 20 real sessions (fail: 9/10 loop closures)

Bar written before the run (PLAN.md change log 2026-11-18/19), judged on M7 run 3. (1) Every session passes the template-fit and EPI-to-T1 plausibility gates. (2) In every subject the cross-session loop closes within 2.0 mm (B4b(c)'s bar), with the EPI-to-EPI step started from identity and from the brain-mask centroid offset, never from the matrices under test.

What prompted it: in M7 runs 1 and 2, 4 of 20 sessions had EPI-to-T1 answers rotated 21-121 degrees from the scanner alignment and 3 more 13-18 degrees, with nothing flagging them. Run 3 (the B4b method) started boundary-based refinement only from the mutual-information answer, which locked onto the wrong structure in those sessions.

Result. (1) 20/20 sessions pass both gates: boundary cost 0.55-0.71, at most 7.5 degrees and 15.7 mm from the scanner alignment; one template fit (sub-07 test) needed its second start. (2) 9/10 subjects close the loop within 2.0 mm; sub-03 closes at 2.21 mm. By the bar as written, B4e fails. The loop values per subject are in the M7 table below (median 1.12 mm, range 0.55-2.21 mm).

Correction after the second review (2026-11-20): the loop's 60 mm test sphere was centred in the wrong session's frame (a test-T1 point, while both transforms act on retest-EPI points; the two centres are 6-34 mm apart). Recomputed with the right centre by `python validation/m7_test_retest.py --epi-loop-only` from the saved files: most values rose 0.05-0.3 mm (sub-03 2.12 to 2.21 mm, sub-09 0.81 to 1.09 mm); one fell (sub-04 0.63 to 0.55 mm). The verdict is unchanged, 9/10, fail. The earlier values are kept in `out/m7.json` as `epi_loop_before_centre_fix`. The gate thresholds were set after seeing 8 of these sessions, so (1) is weaker evidence than (2).

## B5 viewer geometry (pass, rebuilt after the review)

`python validation/b5_geometry_crosscheck.py` and `pytest tests/test_viewer.py`

The first B5 compared a JS matrix chain the viewer never ran with the same chain in numpy, and the server's "re-read" check multiplied the same matrix by the same point twice. Neither could fail, and both missed a real bug the review found: plans gave the target's voxel index in the cropped display volume while naming the original T1 file, 24 and 23 voxels (about 3 cm) away. Fixed; B5 was rebuilt so it can fail:

- `b5_geometry_crosscheck.py` recomputes every matrix the viewer uses from the source files (NIfTI headers via nibabel, the saved registration) and compares 1,000 points pushed through the viewer's own JS with the scene's matrices: display voxel to world, world to original-T1 voxel, world to t-map voxel, RAS to LPS. Largest difference 6e-14. Re-introducing the cropped-index bug makes it fail.
- `tests/viewer_phantom.mjs` renders a synthetic volume (flipped first axis, 7 degree tilt) with markers at the patient's left, front and top through the viewer's real rendering function, and checks screen positions against the radiological and neurological conventions, plus the edge letters, in axial, coronal and sagittal views. A single bright voxel is drawn centred at exactly the pixel position a click maps back to it (0.000 px). Swapping left and right fails 6 checks; going back to sampling pixel corners instead of centres (a half-pixel offset the review found) fails the click check.
- In Chrome: target at the map peak, entry snapped to the scalp, plan exported; the server stores the target's voxel index in the original T1 file computed from its own copy of that file's matrix.

## M7 test-retest of the target (expectation not met by the whole-brain rules; hand-area cluster rule 8/10 under 10 mm)

`python validation/m7_test_retest.py --delete-raw sub-01 ... sub-10`, then `python validation/m7_hand_roi.py`

No pass bar. Expectation stated before running: under 10 mm for most subjects. Three target rules, each fixed before it was computed: (1) the finger map's peak voxel in the whole brain; (2) the t-weighted centre of the cluster (t > 3.1) containing that peak; (3) the same two inside the hand area only: 20 mm spheres around MNI (-37, -21, 58) and (37, -21, 58), hemisphere chosen by the larger summed peak t across both sessions (3a peak, 3b cluster centre). The retest target is carried into the test T1 by a rigid T1-to-T1 registration inside the template brain mask.

This is run 3. Runs 1 and 2 had registration failures that nothing caught at the time (one T1-to-T1 registration 54 mm off, one template fit 100+ mm off, and EPI-to-T1 answers rotated up to 121 degrees in 4 sessions); every number in them is superseded, and both are kept in `out/m7_run1.json` and `out/m7_run2.json`. See B4e and the PLAN.md change log.

| Subject | Rule 1, mm | Rule 2, mm | Hand-area hemisphere | Rule 3a, mm | Rule 3b, mm | Whole-brain peak, MNI approx, test / retest | T1 loop, mm | EPI loop, mm |
|---|---|---|---|---|---|---|---|---|
| sub-01 | 15.1 | 7.8 | right | 15.1 | 7.4 | (35, -18, 47) / (48, -10, 57) | 0.4 | 1.60 |
| sub-02 | 87.9 | 76.7 | left | 19.3 | 2.1 | (-46, -37, 67) / (-22, 54, 40) | 0.5 | 0.84 |
| sub-03 | 101.8 | 101.2 | left | 21.8 | 19.9 | (-24, -90, -30) / (-50, -17, 61) | 0.3 | 2.21 |
| sub-04 | 59.3 | 55.4 | left | 31.6 | 32.0 | (-76, 8, -4) / (-61, -28, 47) | 0.3 | 0.55 |
| sub-05 | 14.9 | 7.2 | left | 11.8 | 2.3 | (-52, -33, 66) / (-54, -23, 54) | 0.5 | 1.14 |
| sub-06 | 13.0 | 4.8 | right | 13.0 | 3.7 | (36, -13, 71) / (36, -19, 56) | 1.0 | 1.16 |
| sub-07 | 89.7 | 83.4 | left | 3.4 | 1.5 | (21, -53, -20) / (-41, -16, 61) | 0.7 | 0.73 |
| sub-08 | 92.0 | 94.1 | left | 19.6 | 1.7 | (16, -52, -11) / (-35, -19, 72) | 0.9 | 1.70 |
| sub-09 | 77.0 | 83.1 | left | 11.5 | 3.6 | (27, 23, 62) / (-39, -19, 55) | 0.3 | 1.09 |
| sub-10 | 104.5 | 104.0 | right | 12.8 | 0.9 | (-22, -52, -23) / (23, -4, 82) | 0.8 | 1.10 |

| Rule | Median | Under 10 mm |
|---|---|---|
| 1, whole-brain peak | 82.5 mm | 0/10 |
| 2, whole-brain cluster centre | 79.9 mm | 3/10 |
| 3a, hand-area peak | 14.0 mm | 1/10 |
| 3b, hand-area cluster centre | 3.0 mm | 8/10 |
| baseline: whole hand-area region, no fMRI (added after the second review) | 0.45 mm | 10/10 |
| baseline: 3b centres of two different people, approx MNI (added after the second review) | 9.7 mm | 53% of 90 pairs |

What this says. Taking the strongest point of the finger map anywhere in the brain does not give a reproducible target: 9 of the 20 session peaks are more than 25 mm from the hand area, 5 of them below z = 0 (cerebellum or the lower temporal or occipital lobe), and the whole-brain peak changes hemisphere between sessions for 4 of 10 subjects. The cerebellum does take part in finger movement, so those are real responses, just not cortical targets. Restricted to the hand area, the single peak voxel still moves a median 14 mm, but the centre of the active cluster around it lands within 10 mm in 8 of 10 subjects (median 3.0 mm; sub-03 19.9 mm and sub-04 32.0 mm are the exceptions). Head motion was low throughout (mean framewise displacement 0.11-0.25 mm).

How much of 3b is the sphere. The second review pointed out that 3b has a floor built in: both sessions' clusters are cut by the same sphere, and the two template fits agree to under a millimetre. Two baselines, computed afterwards and not pre-registered (`baselines_post_hoc` in `out/m7_hand_roi.json`), put a size on it. Replacing each cluster with the whole region, which uses no fMRI at all, gives a median of 0.45 mm and 10/10 under 10 mm, so "under 10 mm" by itself is not evidence. Comparing cluster centres between different people (right-hemisphere centres mirrored, approximate MNI) gives a median of 9.7 mm, with 53% of pairs under 10 mm. The same person on two days is closer: median 3.1 mm in the same MNI comparison, and no random re-pairing of the ten retest sessions out of 20,000 had a median that low (p < 0.0001). So the cluster centre does carry something individual, but the 8/10 count overstates it: about half of all pairs of different people also land within 10 mm. The two failures, sub-03 and sub-04, are the only subjects with 5 voxels or fewer above threshold inside the region in both sessions.

Caveats. Rule 3 confines the answer to two 40 mm spheres, so it measures reproducibility inside the hand area, not whether the map finds the hand area unaided; a planner would need that anatomical prior. The dataset paper does not say which hand was tapped (3 of the 10 are left-handed), so the hemisphere cannot be checked. MNI coordinates come from a 12-parameter affine and are good to several millimetres. sub-01's peak moved 15 mm between the OLS and AR(1) versions of the same GLM, which is why rule 2 was added.

An earlier draft of this file argued from sub-01's right-hemisphere motor peak and left-cerebellum response that the subject tapped with the left hand. The review pointed out that a left-right flipped image would look exactly the same, so that inference is withdrawn.

## M8 Google Cloud DICOMweb round trip (pass)

`python validation/m8_gcp_roundtrip.py bilal testdicom-445700 us-central1 scan-to-target m8-roundtrip`, run 2026-11-17 against a Cloud Healthcare API DICOM store created for the test and deleted afterwards (with its dataset). The first argument names the gcloud configuration whose account signs the requests; there is no default.

Pass condition (PLAN.md M8): every uploaded instance is found by search, comes back byte-identical, and the retrieved series loads to the same voxels and matrix.

| Series | Files | Found by search | Byte-identical | Same voxels and matrix |
|---|---|---|---|---|
| 20180918GE/mr_0004 | 150 | 150 | 150 | yes |
| 20180918GE/mr_0005 | 150 | 150 | 150 | yes |
| 20180918GE/mr_0006 | 150 | 150 | 150 | yes |
| 20180918GE/mr_0007 | 150 | 150 | 150 | yes |
| 20180918Si/mr_0003 | 2 | 2 | 2 | yes |
| 20180918Si/mr_0004 | 2 | 2 | 2 | yes |
| 20180918Si/mr_0005 | 2 | 2 | 2 | yes |
| 20180918Si/mr_0006 | 2 | 2 | 2 | yes |

All 608 instances in 8 series passed. Upload (STOW-RS, 50 files per request) took 116 s in total and retrieval (WADO-RS, one request per instance, sequential) 247 s from the Mac mini; retrieval was not parallelised. The 2 studies were deleted through DICOMweb (HTTP 200 each) before the store itself was deleted.

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

### Second review (2026-11-20)

A second independent agent reviewed everything changed since the first review's fixes (commits 6a8e594 to a619bc0) and reported 7 findings plus design notes. It found no frame, matrix-direction or LPS/RAS errors, and every number it checked in the docs matched `out/` apart from item 7.

| # | Finding | What was done |
|---|---|---|
| 1 | Rule 3b's 8/10 is largely set by the hand-area sphere: with the cluster replaced by the whole region (no fMRI) the distance is 0.45 mm, 10/10 | Two baselines added and reported next to 3b (M7 above); the README and site no longer present 8/10 on its own |
| 2 | The probe view mirrored itself when the entry moved 2 mm forward or back on an oblique path (the convention flip keyed on the side axis's x component, which for an oblique path is the path's front-back tilt) | The convention now sets which way the view is seen (radiological from the target side, neurological from the entry side); new phantom checks on an oblique path fail on the old code |
| 3 | B4e(2) and B4b loop spheres were centred in the test frame while both transforms act on retest-EPI points | Fixed in both scripts; B4e(2) recomputed (sub-03 2.21 mm, still 9/10, fail); B4b run 5 (1.40 mm, pass) |
| 4 | The EPI-to-T1 cache ignored the mean EPI, and the template cache ignored the T1 | Both keys now include the input file's sha256 |
| 5 | `b4_registration.py b` overwrote the pipeline's cached EPI-to-T1 matrix without updating its key and gate record; its JSON said run 3 | It writes its own files (`out/b4b_epi_to_t1_*.npy`, bit-identical to the pipeline's today); label fixed |
| 6 | `scripts/build_static_demo.sh` deleted its destination with no guard (`... out` would have erased every result) | It refuses to replace a folder it did not make (marker file) |
| 7 | Stale wording: B4a's method, the site's "mutual information gets it close" and "under a millimetre", 121 vs 120 degrees, the count of validation scripts; the B4e(1) gate tally came from no validation script | Wording fixed; `m7_test_retest.py` now prints the gate tally (`out/b4e_gates.log`) |

Design notes it raised without calling them defects, kept here: the T1-to-T1 loop check is partly circular when the template-implied start wins (it still catches gross failures such as the 54 mm one); when the scanner start wins, the EPI-to-T1 "moved from scanner" gate passes almost automatically and only the cost threshold discriminates; `gcloud_token` names the configuration, but an environment variable such as `CLOUDSDK_CORE_ACCOUNT` would override its account (none is set on the machine used).
