"""Rigid registration.

register_rigid(fixed, moving) returns the 4x4 matrix M that maps a point in the
moving image's world frame to the matching point in the fixed image's world
frame. To display moving on fixed's grid: resample_to(moving, fixed, M).

Method
- Cost is computed on a fixed set of sample points taken from the fixed image's
  foreground on a regular grid at the current level's spacing. Each point is
  carried into the moving image through inv(M) and the moving image's own
  voxel matrix, then interpolated (trilinear).
- Metric "mi": mutual information from a 32x32 joint histogram with linear
  (two-bin) soft assignment, so the cost changes smoothly with the parameters.
  Used across contrasts (fMRI to T1).
- Metric "ssd": mean squared intensity difference after scaling moving to
  fixed's mean. Used within one contrast (motion correction).
- Parameters are three rotations about the fixed image's foreground centre and
  three translations. Rotations are expressed to the optimizer as arc length in
  mm at a 60 mm radius, so one unit means about 1 mm of motion for every
  parameter.
- Powell's method at each level of a coarse-to-fine schedule; each level starts
  from the previous result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, optimize

from . import transforms as T
from .volume import Volume

RADIUS_MM = 60.0


@dataclass
class Level:
    spacing_mm: float      # sample grid spacing in the fixed image
    smooth_mm: float       # Gaussian FWHM applied to both images
    max_points: int = 40000
    order: int = 1         # interpolation of the moving image: 1 trilinear, 3 cubic B-spline


DEFAULT_MI_LEVELS = (Level(8, 8), Level(4, 4), Level(2, 2))
DEFAULT_SSD_LEVELS = (Level(8, 6), Level(4, 0, order=3))


@dataclass
class Result:
    matrix: np.ndarray
    cost: float
    n_evals: int
    levels: list = field(default_factory=list)


def _smooth(vol: Volume, fwhm_mm: float) -> np.ndarray:
    arr = vol.data if vol.data.ndim == 3 else vol.data.mean(axis=3)
    arr = np.asarray(arr, dtype=np.float32)
    if fwhm_mm <= 0:
        return arr
    sigma_vox = fwhm_mm / 2.3548 / vol.spacing[:3]
    return ndimage.gaussian_filter(arr, sigma_vox, mode="nearest")


def foreground_threshold(arr: np.ndarray) -> float:
    """Simple head/brain threshold: a fraction of the robust maximum, as in
    many fMRI packages (values above 1/8 of the 98th percentile)."""
    return float(np.percentile(arr, 98)) / 8.0


def _fixed_samples(fixed: Volume, arr: np.ndarray, spacing_mm: float, max_points: int,
                   rng: np.random.Generator, mask: np.ndarray | None):
    step = np.maximum(spacing_mm / fixed.spacing[:3], 1.0)
    axes = [np.arange(0, n - 0.5, s) for n, s in zip(fixed.shape3, step)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    vals = ndimage.map_coordinates(arr, grid.T, order=1)
    keep = vals > foreground_threshold(arr)
    if mask is not None:
        keep &= ndimage.map_coordinates(mask.astype(np.float32), grid.T, order=0) > 0.5
    grid, vals = grid[keep], vals[keep]
    if len(grid) > max_points:
        idx = rng.choice(len(grid), max_points, replace=False)
        grid, vals = grid[idx], vals[idx]
    return fixed.voxel_to_world(grid), vals


def _mi(a: np.ndarray, b: np.ndarray, a_rng, b_rng, bins: int = 32) -> float:
    """Mutual information (nats) with linear soft binning."""
    fa = np.clip((a - a_rng[0]) / (a_rng[1] - a_rng[0]), 0, 1) * (bins - 1)
    fb = np.clip((b - b_rng[0]) / (b_rng[1] - b_rng[0]), 0, 1) * (bins - 1)
    ia = np.minimum(fa.astype(int), bins - 2)
    ib = np.minimum(fb.astype(int), bins - 2)
    wa = fa - ia
    wb = fb - ib
    h = np.zeros(bins * bins)
    for da, wa_ in ((0, 1 - wa), (1, wa)):
        for db, wb_ in ((0, 1 - wb), (1, wb)):
            h += np.bincount((ia + da) * bins + (ib + db), weights=wa_ * wb_, minlength=bins * bins)
    p = h.reshape(bins, bins)
    p /= p.sum()
    pa = p.sum(1)
    pb = p.sum(0)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum())


def _params_to_matrix(x, center):
    rot = np.asarray(x[:3]) / RADIUS_MM
    return T.rigid(np.concatenate([rot, x[3:]]), center)


def register_rigid(fixed: Volume, moving: Volume, metric: str = "mi", init: np.ndarray | None = None,
                   levels=None, fixed_mask: np.ndarray | None = None, seed: int = 0,
                   bins: int = 32) -> Result:
    if levels is None:
        levels = DEFAULT_MI_LEVELS if metric == "mi" else DEFAULT_SSD_LEVELS
    rng = np.random.default_rng(seed)
    M = np.eye(4) if init is None else np.asarray(init, dtype=float).copy()

    fixed_arr0 = fixed.data if fixed.data.ndim == 3 else fixed.data.mean(axis=3)
    fg = np.argwhere(fixed_arr0 > foreground_threshold(fixed_arr0))
    center = fixed.voxel_to_world(fg.mean(axis=0)) if len(fg) else fixed.center_world()

    mov_inv_aff = np.linalg.inv(moving.affine)
    total_evals = 0
    info = []
    cost = np.nan
    for lv in levels:
        f_arr = _smooth(fixed, lv.smooth_mm)
        m_arr = _smooth(moving, lv.smooth_mm)
        # cubic B-spline sampling avoids the pull towards whole-voxel offsets
        # that trilinear interpolation gives a least-squares cost
        m_coef = ndimage.spline_filter(m_arr, order=lv.order, mode="nearest") if lv.order > 1 else m_arr
        pts, fvals = _fixed_samples(fixed, f_arr, lv.spacing_mm, lv.max_points, rng, fixed_mask)
        f_rng = np.percentile(fvals, [0.5, 99.5])
        m_rng = np.percentile(m_arr[m_arr > foreground_threshold(m_arr)], [0.5, 99.5]) if metric == "mi" else None
        m_scale = None
        # The optimizer works on a correction applied on top of the current M.
        base = M.copy()

        def moving_values(x):
            Mx = _params_to_matrix(x, center) @ base
            vox = T.apply(mov_inv_aff @ np.linalg.inv(Mx), pts)
            inside = np.all((vox >= 0) & (vox <= np.array(moving.shape3) - 1), axis=1)
            vals = ndimage.map_coordinates(m_coef, vox.T, order=lv.order, mode="nearest", prefilter=False)
            return vals, inside

        if metric == "ssd":
            v0, in0 = moving_values(np.zeros(6))
            m_scale = fvals[in0].mean() / max(v0[in0].mean(), 1e-6)

        def cost_fn(x):
            vals, inside = moving_values(x)
            frac = inside.mean()
            if frac < 0.2:
                return 1e6
            if metric == "mi":
                return -_mi(fvals[inside], vals[inside], f_rng, m_rng, bins)
            d = fvals[inside] - m_scale * vals[inside]
            return float(np.mean(d * d)) / (1e-12 + np.var(fvals))

        res = optimize.minimize(cost_fn, np.zeros(6), method="Powell",
                                options={"xtol": 1e-3, "ftol": 1e-7, "maxfev": 4000,
                                         "direc": np.eye(6) * max(lv.spacing_mm / 2, 1.0)})
        M = _params_to_matrix(res.x, center) @ base
        total_evals += res.nfev
        cost = float(res.fun)
        info.append({"spacing_mm": lv.spacing_mm, "smooth_mm": lv.smooth_mm, "n_points": int(len(pts)),
                     "evals": int(res.nfev), "cost": cost})
    return Result(M, cost, total_evals, info)
