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

## B2 motion correction (pass on run 3)

`python validation/b2_motion_recovery.py`

| Run | Method | Simulation | Median error | Max error | Bar (0.2 / 0.5 mm) |
|---|---|---|---|---|---|
| 1 | trilinear | zero-filled | 0.41 mm | 1.82 mm | fail |
| 2 | cubic final level | zero-filled | 0.15 mm | 1.02 mm | fail (max) |
| 3 | cubic final level | cropped | 0.075 mm | 0.23 mm | pass |

30 known motions up to 3 degrees and 3 mm, noise at the run's measured temporal SD (about 75 on a signal near 2,000), errors measured at 64 points on a 60 mm sphere. What changed between runs and why is in PLAN.md's change log. About 1.5 s per volume on the mini.
