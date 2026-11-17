# Results

Every number here is printed by a script in `validation/` and saved under `out/`. Failed runs are kept. Bars are defined in [PLAN.md](PLAN.md).

## B1 DICOM reader vs dcm2niix (pass, 8/8)

`python validation/b1_dicom_vs_dcm2niix.py`

| Series | Scanner | Layout | Shape | Corner error | Voxel values |
|---|---|---|---|---|---|
| GE mr_0004 to 0007 | GE fMRI, 3 mm slices with 2 mm gaps, 10 time points | one file per slice per time point, all with the same acquisition time | 64x64x15x10 | 5.9e-6 mm | identical |
| Siemens mr_0003 to 0006 | Siemens EPI, 4 phase-encoding directions | mosaic, 5 slices tiled 3x3 in each file | 72x72x5x2 | at most 3.2e-11 mm | identical |

The reference is dcm2niix v1.0.20261024's conversion shipped with the dataset. The comparison maps reference voxels to ours by an axis permutation with flips whose offset comes from the grid sizes, not from the matrices, so a position error can't hide. A mutation check confirms it: deleting the mosaic position correction moves the corners by 305 mm and fails the series. (An earlier version of the check took the offset from the matrices and reported 0 mm for that same broken reader; the value comparison still failed it. Fixed before any result was recorded.)

B1b, synthetic series with known geometry (`pytest tests/test_dicom.py`): 10/10 pass, covering an oblique stack rotated 20 degrees about two axes, a stack written top-down, a 3-time-point series in shuffled file order, per-slice rescale slopes, PHI kept out of metadata, and refusals for gantry tilt, uneven spacing, a missing slice, mixed orientations and two series in one folder.

B1c, real oblique MRI and a tilted CT (`python validation/b1c_oblique_and_ct.py`): pass. All 21 Siemens series in dcm_qa's Orientation set (7 axial, 6 coronal, 6 sagittal, 2 multiband axial at 86x86x36; ascending, descending and interleaved slice orders) are voxel-identical to dcm2niix with corners within 6.8e-5 mm. Two of them are stored compressed (JPEG lossless and JPEG 2000) and failed on the first run because no decoder was installed; the pylibjpeg decoders are now dependencies. The GE head CT from dcm_qa_ct is refused: "slices are not stacked along the slice normal (in-plane offset 1.34 mm, 18.50 deg): gantry tilt or sheared stack". Loading it with a plain 4x4 matrix would have placed every slice up to 18.5 degrees off.

## B2 motion correction (pass on run 3)

`python validation/b2_motion_recovery.py`

| Run | Method | Simulation | Median error | Max error | Bar (0.2 / 0.5 mm) |
|---|---|---|---|---|---|
| 1 | trilinear | zero-filled | 0.41 mm | 1.82 mm | fail |
| 2 | cubic final level | zero-filled | 0.15 mm | 1.02 mm | fail (max) |
| 3 | cubic final level | cropped | 0.075 mm | 0.23 mm | pass |

30 known motions up to 3 degrees and 3 mm, noise at the run's measured temporal SD (about 75 on a signal near 2,000), errors measured at 64 points on a 60 mm sphere. What changed between runs and why is in PLAN.md's change log. About 1.5 s per volume on the mini.

## B3 GLM vs nilearn (pass)

`python validation/b3_glm_vs_nilearn.py`

Same realigned data, mask, events, SPM HRF, 128 s cosine drift, six motion regressors, OLS, no smoothing; nilearn builds its own design matrix from the events.

| Session | Pearson r (t-maps in mask) | Max abs t difference | Peak voxel | Peak t ours / nilearn |
|---|---|---|---|---|
| test | 0.99997 | under 0.06 | identical | 7.89 / 7.84 |
| retest | 0.99997 | under 0.06 | identical | 4.79 / 4.75 |

Before this run, the pipeline was changed twice (see PLAN.md change log): leading non-steady-state volumes are dropped (volume 0 was 26-27% brighter than the rest of each run), and the contrast was briefly switched to finger vs rest on a diagnosis that turned out to be wrong, then restored to the pre-registered finger > mean(foot, lips). B3 passed both before and after these changes.

## B4a registration, known truth (pass)

`python validation/b4_registration.py a`

30 pseudo-EPIs made from the T1 (inverted contrast inside the head, 4 mm grid, 30-slice slab, 6 mm blur, 3% noise, 15% bias field), each moved by a random rigid transform up to 10 degrees and 15 mm, registered back by mutual information from the scanner alignment.

| Median | 95th percentile | Worst sphere point | Cases over 5 mm | Bar |
|---|---|---|---|---|
| 0.48 mm | 0.56 mm | 0.89 mm | 0 | pass (1.0 / 2.0 / 5.0 mm) |

Run 1 used mutual information alone; about 5 s per registration. Started at the true answer, the fit still drifts about 0.48 mm, so the floor is a small bias in the cost rather than capture range. Run 2 (the run-3 method from B4b below: template brain mask, mutual information, then boundary-based refinement) is listed when it finishes.

## B4b registration on real data (run 1 fail, run 2 fail, run 3 partly pass)

`python validation/b4_registration.py b`

There is no ground truth for registering a real EPI to a real T1. Run 1's bar was agreement with SimpleITK; after it failed, bars that need no second tool were added before each later run (PLAN.md change log).

| Run | Method | Start spread (bar 1 mm) | Loop closure (bar 2 mm) | vs SimpleITK, max (bar 2 mm) |
|---|---|---|---|---|
| 1 | mutual information, whole head | not measured | not measured | 20.4 / 21.4 mm: fail |
| 2 | mutual information, scalp stripped by erosion | 3.60 / 0.65 mm: fail | not measured | 13.7 / 3.3 mm: fail |
| 3 | template brain mask, mutual information, then boundary-based | 0.001 / 0.001 mm: pass | 0.91 mm max, 0.68 mean: pass | 9.2 / 10.6 mm: fail |

(test / retest session.) Start spread: refinements from the scanner alignment, the run-1 answer and the run-2 answer. Loop closure: EPI_retest to T1_retest to T1_test against EPI_retest to EPI_test to T1_test, where the T1-T1 and EPI-EPI steps are within one contrast; the head moved 6.8 and 8.9 degrees between sessions, so the loop is not trivially short.

What this shows and what it doesn't. Run 3 converges to one answer from very different starts, and the two sessions' answers are consistent with each other through independent within-contrast registrations to under a millimetre. It moved the EPI 5.6 mm (test) and 1.2 mm (retest) from the scanner's own alignment, and 5-7 mm from the run-1 and run-2 mutual-information answers. SimpleITK's mutual-information registration, configured as in PLAN.md, disagrees by about 10 mm; overlays from run 1 (`out/figs/b4b_test_overlays.png`) show its answer pulled onto the scalp, and boundary-based refinement started from its answer stops at a much worse cost (0.96 against 0.70). That is evidence about SimpleITK's setup here, not proof that ours is right; the B4b SimpleITK bar is reported as failed.

## B5 viewer geometry (pass)

`python validation/b5_geometry_crosscheck.py`, plus a manual run in Chrome.

The viewer's JS (`viewer/geometry.js`) and the Python code agree on 1,000 random points to 6e-14 mm (voxel to world, world to voxel, RAS to LPS, and the chain T1 world to EPI voxel through the real registration) and on trilinear sampling to 6e-8. In Chrome: target placed at the map peak, entry on the scalp, plan exported; the server re-read the target at 6.9e-4 voxel (bar 0.5 voxel) and converted it to the simulated robot frame (6 fiducials, FRE 0.32 mm, predicted TRE 0.20 mm, simulated error in that draw 0.13 mm). The browser test found that an entry could be placed in the air; the viewer now warns when the entry is off the scalp and can slide it onto the scalp along the planned line.

## B6 fiducial registration error (pass)

`python validation/b6_fiducial_tre.py`

Robot-space fiducial noise sigma 0.25 mm per axis, 20,000 trials per layout, target 40 mm left, 20 mm back, 55 mm up from head centre.

| Layout | Fiducials | Predicted RMS TRE | Simulated RMS TRE | Difference | Simulated 95th pct |
|---|---|---|---|---|---|
| Spread over the head | 6 | 0.227 mm | 0.227 mm | -0.1% | 0.37 mm |
| Ring in one plane | 5 | 0.287 mm | 0.286 mm | -0.2% | 0.47 mm |
| Clustered on the forehead | 4 | 2.574 mm | 2.582 mm | +0.3% | 4.53 mm |
| Nearly in a line | 4 | 4.429 mm | 4.457 mm | +0.6% | 8.78 mm |

The ranking matches. With the same tracker noise, four fiducials bunched on the forehead put the hand-area target about 11 times further off than six spread around the skull.

## M7 test-retest of the target

`python validation/m7_test_retest.py sub-01 ...`

| Subject | Distance between sessions (T1 frame) | Test peak MNI (approx.) | Retest peak MNI (approx.) |
|---|---|---|---|
| sub-01 | 3.2 mm | 45, -12, 55 (t 8.7) | 48, -10, 57 (t 5.0) |

Other subjects are added as they finish. The dataset paper does not say which hand was used; sub-01's map peaks on the right precentral gyrus and the finger-vs-rest map also lights the left cerebellum, which together point to left-hand tapping. MNI coordinates come from a 12-parameter affine alignment and are good to several millimetres.
