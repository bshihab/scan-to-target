"""Package a subject/session for the planning viewer.

A scene folder holds:
  scene.json        geometry, input hashes, peak, head centre
  t1.u8.gz          T1 cropped to the head, windowed to 0-255, C order
  overlay.f32.gz    finger-movement t-map on its own EPI grid, C order
  robot_demo.json   simulated fiducials for the robot hand-off (see robot.py)
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import transforms as T
from .register import foreground_threshold
from .volume import Volume, sha256


def head_mask(t1: Volume) -> np.ndarray:
    m = t1.data > foreground_threshold(t1.data)
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == 1 + int(np.argmax(sizes))
    return m


def scalp_points(t1: Volume, mask: np.ndarray, directions) -> np.ndarray:
    """Where rays from the head centre leave the head mask: plausible places
    for skull-mounted fiducials."""
    c = t1.voxel_to_world(np.argwhere(mask).mean(0))
    inv = np.linalg.inv(t1.affine)
    pts = []
    for d in directions:
        d = np.asarray(d, float) / np.linalg.norm(d)
        last = None
        for r in np.arange(0, 160, 0.5):
            ijk = np.rint(T.apply(inv, c + r * d)).astype(int)
            if np.any(ijk < 0) or np.any(ijk >= np.array(mask.shape)):
                break
            if mask[tuple(ijk)]:
                last = c + r * d
        if last is not None:
            pts.append(last)
    return np.array(pts)


FIDUCIAL_DIRECTIONS = [(0, 1, 0.35), (1, 0.2, 0.45), (-1, 0.2, 0.45), (0, -1, 0.45), (0.25, -0.1, 1), (-0.6, 0.6, 0.6)]


def build(t1_path, mean_epi_path, tmap_path, epi_to_t1: np.ndarray, out_dir, title: str, t1_to_mni=None,
          overlay_label: str = "t-map",
          extra_inputs: dict | None = None, fiducial_sigma_mm: float = 0.25, seed: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t1 = Volume.from_nifti(t1_path)
    tmap = Volume.from_nifti(tmap_path)
    mask = head_mask(t1)

    # crop to the head with a 4-voxel margin; shift the matrix's origin to match
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - 4, 0)
    hi = np.minimum(idx.max(0) + 5, np.array(t1.shape3))
    crop = t1.data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    A = t1.affine.copy()
    A[:3, 3] = t1.voxel_to_world(lo.astype(float))
    w0, w1 = np.percentile(crop[mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]], [0.5, 99.5])
    u8 = np.clip((crop - w0) / (w1 - w0) * 255, 0, 255).astype(np.uint8)
    with gzip.open(out / "t1.u8.gz", "wb", compresslevel=6) as f:
        f.write(np.ascontiguousarray(u8).tobytes(order="C"))
    with gzip.open(out / "overlay.f32.gz", "wb", compresslevel=6) as f:
        f.write(np.ascontiguousarray(tmap.data.astype(np.float32)).tobytes(order="C"))

    peak_ijk = np.array(np.unravel_index(int(np.argmax(tmap.data)), tmap.shape3), float)
    peak_ras = T.apply(epi_to_t1, tmap.voxel_to_world(peak_ijk))
    head_c = t1.voxel_to_world(idx.mean(0))

    scene = {
        "schema": "scan-to-target/scene/1",
        "title": title,
        # "affine" maps voxels of the CROPPED display volume in t1.u8.gz; voxel
        # indices in the original NIfTI use "affine_original" (crop offset below)
        "t1": {"file": "t1.u8.gz", "dtype": "uint8", "shape": [int(x) for x in u8.shape], "affine": A.tolist(),
               "affine_original": t1.affine.tolist(), "shape_original": [int(x) for x in t1.shape3],
               "crop_offset_voxels": [int(x) for x in lo], "window": [float(w0), float(w1)], "order": "C"},
        "overlay": {"file": "overlay.f32.gz", "dtype": "float32", "shape": [int(x) for x in tmap.shape3],
                    "label": overlay_label, "short_label": "t",
                    "affine_epi": tmap.affine.tolist(), "epi_to_t1": epi_to_t1.tolist(),
                    # T1 world -> EPI world -> EPI voxel
                    "world_to_voxel": (np.linalg.inv(tmap.affine) @ np.linalg.inv(epi_to_t1)).tolist()},
        "peak_ras": [float(x) for x in peak_ras],
        "peak_t": float(tmap.data.max()),
        "t1_to_mni": None if t1_to_mni is None else np.asarray(t1_to_mni).tolist(),
        "head_center_ras": [float(x) for x in head_c],
        "inputs": {"t1_sha256": sha256(t1_path), "tmap_sha256": sha256(tmap_path),
                   "mean_epi_sha256": sha256(mean_epi_path), "epi_to_t1": epi_to_t1.tolist(), **(extra_inputs or {})},
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1))

    # Simulated robot: fiducials on the scalp, a hidden robot pose, noisy measurements.
    fids = scalp_points(t1, mask, FIDUCIAL_DIRECTIONS)
    rng = np.random.default_rng(seed)
    M_true = T.random_rigid(rng, 180.0, 1.0)
    M_true[:3, 3] = [400.0, -150.0, 900.0]
    measured = T.apply(M_true, fids) + rng.normal(0, fiducial_sigma_mm, fids.shape)
    (out / "robot_demo.json").write_text(json.dumps({
        "note": "Simulated. Fiducials are scalp points of this T1; robot-space positions are a hidden rigid pose "
                f"plus Gaussian noise (sigma {fiducial_sigma_mm} mm per axis), standing in for a tracked probe.",
        "sigma_mm": fiducial_sigma_mm, "fiducials_image_ras": fids.tolist(),
        "fiducials_robot_measured": measured.tolist(), "true_image_to_robot": M_true.tolist()}, indent=1))
    return scene
