"""fMRI run -> motion-corrected mean image + finger-movement t-map."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import glm
from .motion import framewise_displacement, matrix_to_params, realign
from .register import foreground_threshold
from .volume import Volume, resample_to

# Pre-registered contrast: finger movement against the mean of the other two
# movements. Finger against rest is written as a second map.
FINGER_CONTRAST = {"Finger": 1.0, "Foot": -0.5, "Lips": -0.5}
FINGER_VS_REST = {"Finger": 1.0}

NONSTEADY_THRESHOLD = 1.05  # leading volumes brighter than 105% of the run median are dropped


def brain_mask(mean_epi: Volume) -> np.ndarray:
    from scipy import ndimage
    m = mean_epi.data > foreground_threshold(mean_epi.data)
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m)


def nonsteady_volumes(bold: Volume) -> int:
    """Number of leading volumes acquired before the magnetisation settled.
    They are much brighter than the rest (26-27% for volume 0 on ds000114)."""
    m = brain_mask(bold.mean())
    g = bold.data[m].mean(axis=0)
    med = np.median(g)
    k = 0
    while k < len(g) // 4 and g[k] > NONSTEADY_THRESHOLD * med:
        k += 1
    return k


def run(bold_path, events_path, out_dir, tr: float | None = None, smooth_fwhm_mm: float = 5.0,
        log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bold_full = Volume.from_nifti(bold_path)
    if tr is None:
        tr = float(bold_full.meta["zooms"][3])
    drop = nonsteady_volumes(bold_full)
    bold = Volume(bold_full.data[..., drop:], bold_full.affine.copy(), dict(bold_full.meta))
    n = bold.data.shape[3]
    if drop:
        log(f"dropped {drop} non-steady-state volume(s)")

    mats_file = out / "motion_matrices.npy"
    if mats_file.exists():
        mats = list(np.load(mats_file))
        ref = bold.frame(0)
        realigned = Volume(np.stack([resample_to(bold.frame(t), ref, mats[t]).data for t in range(n)], 3),
                           ref.affine.copy(), dict(bold.meta))
        log(f"motion: reused {mats_file.name}")
    else:
        log(f"motion: realigning {n} volumes")
        realigned, mats, _ = realign(bold, progress=lambda i, k: log(f"  {i}/{k}") if i % 46 == 0 else None)
        np.save(mats_file, np.array(mats))
    params = np.array([matrix_to_params(M) for M in mats])
    fd = framewise_displacement(mats)

    mean_epi = realigned.mean()
    mask = brain_mask(mean_epi)
    events = glm.read_events(events_path)
    for e in events:  # time zero is now the first kept volume
        e["onset"] -= drop * tr
    design = glm.design_matrix(n, tr, events, confounds=params,
                               confound_names=["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"])
    c = glm.contrast_vector(design, FINGER_CONTRAST)
    tmap, stats = glm.fit_t(realigned, design, c, mask, smooth_fwhm_mm=smooth_fwhm_mm)
    t_rest, _ = glm.fit_t(realigned, design, glm.contrast_vector(design, FINGER_VS_REST), mask,
                          smooth_fwhm_mm=smooth_fwhm_mm)
    t_rest.to_nifti(out / "tmap_finger_vs_rest.nii.gz")

    mean_epi.to_nifti(out / "mean_epi.nii.gz")
    tmap.to_nifti(out / "tmap_finger.nii.gz")
    Volume(mask.astype(np.float32), mean_epi.affine).to_nifti(out / "brain_mask.nii.gz")
    np.savetxt(out / "motion_params.tsv", params, delimiter="\t", fmt="%.5f",
               header="rot_x_deg\trot_y_deg\trot_z_deg\ttrans_x_mm\ttrans_y_mm\ttrans_z_mm", comments="")
    np.save(out / "design.npy", design.X)
    peak = np.unravel_index(int(np.argmax(tmap.data)), tmap.shape3)
    summary = {
        "bold": str(bold_path), "bold_sha256": bold.meta["sha256"], "tr": tr, "n_scans": n,
        "dropped_volumes": drop, "n_vox_t_gt_3_1": int((tmap.data[mask] > 3.1).sum()),
        "n_vox_t_gt_3_1_finger_vs_rest": int((t_rest.data[mask] > 3.1).sum()),
        "smooth_fwhm_mm": smooth_fwhm_mm, "contrast": FINGER_CONTRAST, "dof": stats["dof"],
        "max_abs_translation_mm": float(np.abs(params[:, 3:]).max()),
        "max_abs_rotation_deg": float(np.abs(params[:, :3]).max()),
        "mean_fd_mm": float(fd.mean()), "max_fd_mm": float(fd.max()),
        "peak_t": float(tmap.data.max()), "peak_voxel": [int(x) for x in peak],
        "peak_ras_mm": [round(float(x), 2) for x in tmap.voxel_to_world(np.array(peak, float))],
        "design_columns": design.names,
    }
    (out / "fmri_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
