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
    """6 parameters: rigid. 12 parameters: rigid plus log-scales (x[6:9]) and
    shears (x[9:12]), all expressed so one unit moves a point ~1 mm at 60 mm
    from the centre. Matrix: x -> R S H (x - c) + c + t."""
    rot = np.asarray(x[:3]) / RADIUS_MM
    M = T.rigid(np.concatenate([rot, x[3:6]]), center)
    if len(x) == 6:
        return M
    S = np.diag(np.exp(np.asarray(x[6:9]) / RADIUS_MM))
    sh = np.asarray(x[9:12]) / RADIUS_MM
    H = np.array([[1, sh[0], sh[1]], [0, 1, sh[2]], [0, 0, 1]])
    L = M[:3, :3] @ S @ H
    c = np.asarray(center, float)
    A = np.eye(4)
    A[:3, :3] = L
    A[:3, 3] = c - L @ c + np.asarray(x[3:6])
    return A


def register_rigid(fixed: Volume, moving: Volume, metric: str = "mi", init: np.ndarray | None = None,
                   levels=None, fixed_mask: np.ndarray | None = None, seed: int = 0,
                   bins: int = 32, dof: int = 6) -> Result:
    if levels is None:
        levels = DEFAULT_MI_LEVELS if metric == "mi" else DEFAULT_SSD_LEVELS
    rng = np.random.default_rng(seed)
    M = np.eye(4) if init is None else np.asarray(init, dtype=float).copy()

    fixed_arr0 = fixed.data if fixed.data.ndim == 3 else fixed.data.mean(axis=3)
    fg = np.argwhere(fixed_mask if fixed_mask is not None else fixed_arr0 > foreground_threshold(fixed_arr0))
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
            v0, in0 = moving_values(np.zeros(dof))
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

        res = optimize.minimize(cost_fn, np.zeros(dof), method="Powell",
                                options={"xtol": 1e-3, "ftol": 1e-7, "maxfev": 4000 * dof // 6,
                                         "direc": np.eye(dof) * max(lv.spacing_mm / 2, 1.0)})
        M = _params_to_matrix(res.x, center) @ base
        total_evals += res.nfev
        cost = float(res.fun)
        info.append({"spacing_mm": lv.spacing_mm, "smooth_mm": lv.smooth_mm, "n_points": int(len(pts)),
                     "evals": int(res.nfev), "cost": cost})
    return Result(M, cost, total_evals, info)


# ---------------------------------------------------------------------------
# fMRI to anatomy

EPI_TO_T1_LEVELS = (Level(8, 8, max_points=200000), Level(4, 4, max_points=200000), Level(2, 2, max_points=200000))
SCALP_STRIP_MM = 8.0


def t1_registration_mask(t1: Volume, strip_mm: float = SCALP_STRIP_MM) -> np.ndarray:
    """Head mask with the outer `strip_mm` removed: a crude skull strip. The EPI
    shows little of the scalp, fat and neck that dominate a T1, and letting them
    into the cost made the answer depend on where the optimizer started."""
    m = t1.data > foreground_threshold(t1.data)
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == 1 + int(np.argmax(sizes))
    depth = ndimage.distance_transform_edt(m, sampling=t1.spacing[:3])
    return depth > strip_mm


def register_epi_to_t1(t1: Volume, mean_epi: Volume, init: np.ndarray | None = None) -> Result:
    """Matrix mapping EPI world to T1 world (mutual information)."""
    return register_rigid(t1, mean_epi, metric="mi", init=init, levels=EPI_TO_T1_LEVELS,
                          fixed_mask=t1_registration_mask(t1))


# ---------------------------------------------------------------------------
# Boundary-based registration (Greve and Fischl, NeuroImage 2009)

def _kmeans_1d(v: np.ndarray, k: int = 3, iters: int = 50) -> np.ndarray:
    c = np.percentile(v, np.linspace(15, 85, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(v[:, None] - c[None]), axis=1)
        new = np.array([v[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c


def white_matter_boundary(t1: Volume, region: np.ndarray, n_points: int = 20000, seed: int = 0):
    """White-matter surface points and outward normals (T1 world mm).

    `region` is a brain-only mask on the T1 grid. Inside it the T1 is divided
    by a 15 mm local mean (removes the slow brightness drift), the intensities
    are split into three classes, and the brightest class is white matter."""
    sig = 15.0 / 2.3548 / t1.spacing[:3]
    num = ndimage.gaussian_filter(np.where(region, t1.data, 0).astype(np.float32), sig)
    den = ndimage.gaussian_filter(region.astype(np.float32), sig)
    norm = np.where(region, t1.data / np.maximum(num / np.maximum(den, 1e-3), 1e-3), 0)
    centers = np.sort(_kmeans_1d(norm[region]))
    thr = 0.5 * (centers[1] + centers[2])
    wm = region & (norm > thr)
    wm = ndimage.binary_opening(wm, iterations=1)
    lab, n = ndimage.label(wm)
    if n > 1:
        sizes = ndimage.sum(wm, lab, range(1, n + 1))
        wm = lab == 1 + int(np.argmax(sizes))
    inner = ndimage.binary_erosion(region, iterations=3)
    edge = wm & ~ndimage.binary_erosion(wm) & inner
    idx = np.argwhere(edge)
    rng = np.random.default_rng(seed)
    if len(idx) > n_points:
        idx = idx[rng.choice(len(idx), n_points, replace=False)]
    smooth = ndimage.gaussian_filter(wm.astype(np.float32), 1.0 / t1.spacing[:3])
    g = np.stack([ndimage.map_coordinates(np.gradient(smooth, axis=a), idx.T.astype(float), order=1) for a in range(3)], 1)
    # voxel-space gradient -> world-space gradient: inverse transpose of the linear part
    gw = g @ np.linalg.inv(t1.affine[:3, :3])
    nrm = -gw / np.maximum(np.linalg.norm(gw, axis=1, keepdims=True), 1e-9)  # out of white matter
    ok = np.linalg.norm(gw, axis=1) > 1e-6
    info = {"class_centers": centers.tolist(), "n_boundary_voxels": int(edge.sum()), "n_points": int(ok.sum())}
    return t1.voxel_to_world(idx[ok].astype(float)), nrm[ok], info


def register_bbr(t1: Volume, mean_epi: Volume, init: np.ndarray, t1_brain: np.ndarray,
                 dist_mm: float = 2.0, slope: float = 0.5, seed: int = 0) -> Result:
    """Refine an EPI-to-T1 matrix by boundary-based registration.

    For each white-matter boundary point p with outward normal n, the EPI is
    sampled at p - d n (white matter) and p + d n (grey matter). BOLD EPI shows
    grey matter brighter than white, so the percent contrast
    Q = 100 (g - w) / ((g + w) / 2) should be positive all along the surface.
    Cost: mean of 1 - tanh(slope * Q); lower is better."""
    # t1_brain: brain mask on the T1 grid (template-based, see template.py),
    # fixed before optimisation so every start sees the same surface
    region = t1_brain
    pts, nrm, info = white_matter_boundary(t1, region, seed=seed)
    w_pts, g_pts = pts - dist_mm * nrm, pts + dist_mm * nrm
    center = pts.mean(0)
    inv_epi = np.linalg.inv(mean_epi.affine)
    out_info = []
    M = init.copy()
    cost = np.nan
    evals = 0
    for fwhm in (4.0, 0.0):
        arr = _smooth(mean_epi, fwhm)
        coef = ndimage.spline_filter(arr, order=3, mode="nearest")
        base = M.copy()

        def sample(P, Minv):
            v = T.apply(inv_epi @ Minv, P)
            return ndimage.map_coordinates(coef, v.T, order=3, mode="nearest", prefilter=False)

        def cost_fn(x):
            Mx = _params_to_matrix(x, center) @ base
            Minv = np.linalg.inv(Mx)
            w = sample(w_pts, Minv)
            g = sample(g_pts, Minv)
            q = 100.0 * (g - w) / np.maximum(0.5 * (g + w), 1e-6)
            return float(np.mean(1.0 - np.tanh(slope * q)))

        res = optimize.minimize(cost_fn, np.zeros(6), method="Powell",
                                options={"xtol": 1e-3, "ftol": 1e-7, "maxfev": 4000, "direc": np.eye(6) * 1.0})
        M = _params_to_matrix(res.x, center) @ base
        cost = float(res.fun)
        evals += res.nfev
        out_info.append({"epi_fwhm_mm": fwhm, "cost": cost, "evals": int(res.nfev)})
    out_info.insert(0, info)
    return Result(M, cost, evals, out_info)


def register_epi_to_t1_bbr(t1: Volume, mean_epi: Volume, t1_brain: np.ndarray,
                           init: np.ndarray | None = None) -> Result:
    """The pipeline's fMRI-to-anatomy step (run 3 in PLAN.md)."""
    # mutual information for the coarse alignment, then boundary-based refinement
    if init is None:
        init = register_epi_to_t1(t1, mean_epi).matrix
    return register_bbr(t1, mean_epi, init, t1_brain)
