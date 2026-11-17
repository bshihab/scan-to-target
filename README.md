# Scan to Target

A pipeline that takes brain scans as they come off an MRI scanner and ends with an electrode target written in the coordinates a surgical robot would use. It reads DICOM and NIfTI files, maps where the hand area of motor cortex is from a finger-tapping fMRI scan, lines that map up with the anatomical scan, lets a person place a target and an insertion path in a browser, and converts the plan into a robot's frame through fiducial registration.

Everything runs on public research data. This is a portfolio project and has never been used on a patient. It is not a medical device.

Status: in progress. The table below is updated as each piece is built and measured; numbers only appear after the run that produced them. Measured results live in [RESULTS.md](RESULTS.md), the plan and the pass/fail bars (written before any code ran) in [PLAN.md](PLAN.md).

| Step | What it does | Status |
|---|---|---|
| 1. Ingest | DICOM series to a 3D/4D volume with its voxel-to-patient matrix; NIfTI loading with the sform/qform rules | planned |
| 2. Motion correction | Rigid realignment of each fMRI volume to a reference | planned |
| 3. Motor map | General linear model on the finger/foot/lips task, t-map for finger movement | planned |
| 4. Fusion | Rigid registration of the fMRI to the T1 anatomy by mutual information | planned |
| 5. Planning GUI | Browser viewer: three orthogonal slices, fMRI overlay, target and entry placement, a view along the trajectory | planned |
| 6. Robot hand-off | Fiducial registration from image space to a robot frame, with predicted and simulated target error | planned |
| 7. Cloud storage | Upload and query through the DICOMweb API of Google Cloud Healthcare | planned, needs a GCP project |
| 8. Container | Dockerfile that runs the pipeline and serves the viewer | planned |

## The pipeline

```
 scanner files            image space (T1, mm)                     robot space (mm)
 -------------            --------------------                     ----------------
 DICOM series ──┐
                ├─ ingest ─► T1 volume ◄────────────┐
 NIfTI files ───┘           fMRI 4D ─ realign ─ GLM ─► t-map ─ register ─┘
                                                           │
                                         planning GUI: target + entry point
                                                           │
                                         plan.json (RAS and LPS mm, input hashes)
                                                           │
                         fiducials touched by tracked probe ─► point registration ─► target in robot frame
```

## Coordinate frames

Most bugs in this kind of software are frame bugs, so every point in a plan carries the name of its frame.

| Frame | Axes | Where it comes from |
|---|---|---|
| Voxel | (i, j, k) array indices | The array in memory |
| DICOM patient (LPS) | +x to patient Left, +y to Posterior, +z to Superior, in mm | ImagePositionPatient, ImageOrientationPatient, PixelSpacing and slice positions |
| Scanner RAS | +x Right, +y Anterior, +z Superior, in mm | NIfTI convention; LPS with x and y negated |
| T1 space | Scanner RAS of the anatomical scan | All planning happens here |
| Robot | Whatever the robot's base defines | Fiducial registration from T1 space |

A volume is stored as an array plus one 4x4 matrix from voxel indices to scanner RAS. Moving between images means composing these matrices with the rigid transforms found by registration; images are only resampled for display.

## Data

| Source | What | Size | License |
|---|---|---|---|
| [OpenNeuro ds000114](https://openneuro.org/datasets/ds000114) (Gorgolewski et al., 2013) | T1 and a finger/foot/lips block-design fMRI run, each subject scanned in two sessions on different days. The dataset was collected to test how reliable fMRI is for neurosurgical planning. | about 33 MB per session | CC0 |
| [dcm_qa_nih](https://github.com/neurolabusc/dcm_qa_nih) | Real GE and Siemens fMRI DICOM series with reference NIfTI conversions from dcm2niix | 23 MB | BSD-2-Clause |

Neither contains identifiable patient data. Data is downloaded by `scripts/fetch_data.sh` and is not committed.

## Validation

Each step has a test whose pass bar was written down before the code that it measures existed (see [PLAN.md](PLAN.md)). Where real data has no ground truth, which is the case for registering one real scan to another, the test says so and measures agreement with an independent tool or recovery of a known simulated transform instead.

## What this does not do

- No skull stripping, distortion correction of the EPI, or slice timing correction. Each one is listed in PLAN.md with the effect it would have on target accuracy.
- No real tracking hardware. Step 6 simulates a tracked probe with Gaussian localisation noise.
- No clinical judgment. The GUI suggests the peak of the finger-movement map; a person decides where the target goes.

## Running it

Coming with step 1.

## How this was made

[Bilal: one or two sentences in your own words on your part.] Code and docs were written with Claude Opus 5.5 in Claude Code. Every number in RESULTS.md comes from a script in `validation/` that anyone can re-run.
