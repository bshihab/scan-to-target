# Scan to Target

A pipeline from MRI scanner files to an electrode target written in a robot's coordinate frame. It reads DICOM and NIfTI, finds the hand area of motor cortex from a finger-tapping fMRI run, registers the fMRI to the anatomical scan, lets a person place a target and an insertion path in a browser, and carries the plan into a robot frame through fiducial registration.

It runs on public research data. It is a portfolio project, has never touched a patient, and is not a medical device.

Measured results, including failed runs, are in [RESULTS.md](RESULTS.md). The plan, the pass bars and every change to them are in [PLAN.md](PLAN.md).

| Step | What it does | Status |
|---|---|---|
| 1. Ingest | DICOM series to a 3D or 4D volume with its voxel-to-patient matrix (stacks, time series, Siemens mosaics, compressed transfer syntaxes); refuses gantry tilt, uneven spacing, missing slices, mixed orientations and mixed echoes with the reason | done: voxel-identical to dcm2niix on 29 real series |
| 2. Motion correction | Rigid realignment of each fMRI volume, after dropping non-steady-state volumes | done |
| 3. Motor map | GLM on the finger/foot/lips blocks with an AR(1) noise model; t-map for finger > mean(foot, lips) | done: matches nilearn at r = 0.99997 |
| 4. Fusion | MNI152 template alignment for a brain mask, then fMRI to T1 by boundary-based registration from two starts; plausibility gates refuse a bad fit | done, with two failed bars (see below) |
| 5. Planning GUI | Browser viewer: three orthogonal slices, the fMRI map, target and entry placement, a slice along the path and one across it at a chosen depth | done, tested in Chrome |
| 6. Robot hand-off | Point-based registration from image to a simulated robot frame, with Fitzpatrick's predicted target error | done (simulated tracker) |
| 7. Cloud storage | Upload, search and retrieve through DICOMweb (Google Cloud Healthcare or any DICOMweb server) | done: 608 instances uploaded to a Google Cloud Healthcare DICOM store, found, retrieved byte-identical and reloaded to the same voxels |
| 8. Container | Dockerfile for the pipeline and the server | written, not yet built |

## The pipeline

```
 scanner files                 image space: T1 scanner RAS (mm)                         robot space (mm)
 -------------                 --------------------------------                         ----------------
 DICOM series ──┐
                ├─ ingest ─► T1 ──── MNI152 template (affine) ─► brain mask, approximate MNI coordinates
 NIfTI files ───┘           fMRI ─ drop non-steady volumes ─ realign ─ GLM, AR(1) ─► t-map
                                                     │
      mean fMRI ─ boundary-based, from scanner alignment and from mutual information ─ gate ─► fMRI-to-T1 matrix
                                                     │
                                       planning GUI: target and entry on the T1
                                                     │
                                       plan.json: RAS and LPS mm, T1-file voxel, input hashes
                                                     │
                          fiducials measured in the robot frame ─► point registration ─► target and path in robot mm
```

## Coordinate frames

Every point in a plan carries the name of its frame.

| Frame | Axes | Where it comes from |
|---|---|---|
| Voxel | (i, j, k) array indices | The array in memory. Plans give indices into the original T1 file, never into the cropped copy the viewer draws |
| DICOM patient (LPS) | +x patient left, +y posterior, +z superior, mm | ImagePositionPatient, ImageOrientationPatient, PixelSpacing, slice positions |
| Scanner RAS | +x right, +y anterior, +z superior, mm | The NIfTI convention: LPS with x and y negated |
| T1 space | Scanner RAS of the anatomical scan | All planning happens here |
| MNI (approximate) | MNI152 2009 template space | A 12-parameter affine fit, good to several millimetres; shown as a sanity check, never used for the plan |
| Robot | Defined by the robot base | Fiducial registration from T1 space |

A volume is an array plus one 4x4 matrix from voxel indices to scanner RAS. Moving between images means composing these matrices with the transforms found by registration. The viewer draws the fMRI map by carrying each pixel's world point through the registration into the map's own voxel grid, so nothing is resampled ahead of time.

## Data

| Source | What | Size | License |
|---|---|---|---|
| [OpenNeuro ds000114](https://openneuro.org/datasets/ds000114) (Gorgolewski et al., 2013) | T1 and a finger/foot/lips block-design fMRI run for 10 healthy volunteers over 50, each scanned in two sessions on different days. Collected to test how reliable fMRI is for neurosurgical planning | about 33 MB per session | CC0 |
| [dcm_qa_nih](https://github.com/neurolabusc/dcm_qa_nih), [dcm_qa](https://github.com/neurolabusc/dcm_qa) (Orientation set), [dcm_qa_ct](https://github.com/neurolabusc/dcm_qa_ct) (GE head CT) | Real GE and Siemens MRI DICOM series with dcm2niix reference conversions, and a CT with a variable gantry tilt | about 110 MB | BSD-2-Clause |

None of it contains identifiable patient data. `scripts/fetch_data.sh` downloads it; nothing is committed.

## Validation in one table

Each bar was written in PLAN.md before the run it judges. When a run failed and the method changed, the failed run stays in RESULTS.md and any new bar was written before the next run. Scripts in `validation/` reproduce the latest run of each bar from the current code; earlier, failed runs came from earlier commits named in RESULTS.md.

| Bar | Question | Result |
|---|---|---|
| B1, B1c | Does the DICOM reader match dcm2niix on real series, and refuse a tilted CT? | 29/29 voxel-identical, corners within 7e-5 mm; CT refused for its 18.5 degree tilt |
| B2 | Does motion correction recover known head motion? | median 0.075 mm, max 0.23 mm (third run; two failed runs logged) |
| B3 | Does the GLM match nilearn with the same model? | r = 0.99997, same peak voxel (AR(1) and OLS) |
| B4a | Is a known fMRI-to-T1 transform recovered? | median 0.41 mm, worst point 0.98 mm, with simulated EPI distortion and dropout (0.22 mm without) |
| B4b | On one subject's real data, is the registration stable and self-consistent? | same answer from every start; cross-session loop closes to 1.40 mm; SimpleITK disagrees by about 10 mm (bar failed) |
| B4e | Does registration hold on all 20 real sessions? | all 20 pass the plausibility gates; loop closes within 2 mm in 9 of 10 subjects, the tenth at 2.21 mm (bar failed). Added after earlier runs had silent failures on other subjects |
| B5 | Does the viewer draw and export the right coordinates? | matches values recomputed from the source files to 1e-13; left-right phantom passes in both display conventions |
| B6 | Does the predicted target error match simulation? | within 0.6% for four fiducial layouts |
| M8 | Does a Google Cloud DICOMweb round trip preserve the files? | 608/608 instances byte-identical, same voxels and matrix after retrieval |
| M7 | Does the planned target land in the same place on two days? | no pass bar. Whole-brain peak: median 82 mm apart. Centre of the active cluster inside the hand area: median 3.0 mm, under 10 mm in 8 of 10 subjects, but the region itself sets a floor: cluster centres of two different people are a median 9.7 mm apart, half of them under 10 mm |

## Limits

- The fMRI is not corrected for EPI distortion (this run has no field map) or slice timing. Near the sinuses and temporal poles the map can sit a few millimetres off.
- The t threshold in the viewer (3.1 by default) is uncorrected for the number of voxels tested.
- Registration is rigid. Brain shift after the skull is opened is not modelled; no pre-operative plan accounts for it.
- Left and right come from the scan headers. The code's own handling is tested (DICOM references, an orientation phantom in the viewer), but nothing in this dataset can confirm the headers themselves; a real system would check a lateral marker on the scan.
- The robot and its tracker are simulated: a random rigid pose plus Gaussian noise on six scalp fiducials.
- The server checks a plan against the scene it was built from (the T1's sha256 recorded at build time, the voxel index recomputed from the original T1 matrix, RAS and LPS consistency). It does not re-hash the scan on disk.
- The GUI suggests the map's peak; a person decides where the target goes. The whole-brain peak is not a reproducible target (M7): it often falls in the cerebellum or outside motor cortex. The cluster centre inside the hand area was closer between one person's two days (median 3 mm) than between different people (median 10 mm), but that relies on an anatomical prior, and the region alone makes about half of all pairs land within 10 mm.
- The pipeline refuses to build a planning scene when the template fit or the fMRI-to-T1 registration fails a plausibility check (brain volume and position; rotation and shift from the scanner's own alignment, boundary cost). The thresholds were set from this dataset.

## Running it

Python 3.11 or newer, about 200 MB of data and 600 MB of packages (half of that is the validation-only references: nilearn, SimpleITK).

```
python -m venv .venv && .venv/bin/pip install -e ".[validation]"
scripts/fetch_data.sh                      # public data into data/ (sub-01 by default; pass more subjects)

.venv/bin/s2t ingest data/dicom/dcm_qa_nih/In/20180918Si/mr_0003 -o epi.nii.gz   # any DICOM series folder
.venv/bin/s2t subject sub-01 test          # realign, GLM, template, registration, viewer scene (about 6 min)
.venv/bin/s2t serve out/scenes/sub-01_ses-test
```

Open http://127.0.0.1:8770. The server listens on localhost only and refuses requests whose Host or Origin is not localhost; from another machine use `ssh -L 8770:localhost:8770 <host>`. Click to move the cursor, press T at the target and E at the entry, use "Snap entry to scalp" if the entry is off the scalp, check the two trajectory views, and export.

```
.venv/bin/python -m pytest                              # unit tests, including the viewer phantom (needs node)
.venv/bin/python validation/b1_dicom_vs_dcm2niix.py      # each script prints its bar and writes out/<bar>.json
```

## Layout

```
s2t/            dicom, volume, transforms, motion, glm, fmri, register (MI, SSD, boundary-based),
                template, fiducials, robot, scene, serve, dicomweb, cli; viewer/ (HTML, JS)
validation/     one script per bar
tests/          unit tests, viewer orientation phantom
scripts/        data download, static copy of the planner for a website
```

## How this was made

[Bilal: one or two sentences in your own words on your part.] Code and docs were written with Claude Opus 5.5 in Claude Code, and two independent review agents' findings (11, then 7) are listed with their fixes in RESULTS.md. Every number in RESULTS.md comes from a script in `validation/`.
