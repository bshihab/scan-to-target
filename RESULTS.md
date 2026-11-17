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

## B4a registration, known truth (pass)

`python validation/b4_registration.py a`

30 pseudo-EPIs made from the T1 (inverted contrast inside the head, 4 mm grid, 30-slice slab, 6 mm blur, 3% noise, 15% bias field), each moved by a random rigid transform up to 10 degrees and 15 mm, registered back by mutual information from the scanner alignment.

| Median | 95th percentile | Worst sphere point | Cases over 5 mm | Bar |
|---|---|---|---|---|
| 0.48 mm | 0.56 mm | 0.89 mm | 0 | pass (1.0 / 2.0 / 5.0 mm) |

About 5 s per registration. The error does not grow with the size of the motion, which points to a fixed floor (interpolation or the pseudo-EPI construction) rather than capture range.

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
