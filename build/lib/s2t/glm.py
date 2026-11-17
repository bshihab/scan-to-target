"""First-level general linear model for a block-design fMRI run.

Model per voxel: y = X b + e, fitted by ordinary least squares.
X columns: each condition's boxcar convolved with the SPM canonical HRF,
cosine drift terms below the high-pass cut-off, optional nuisance regressors
(motion), and a constant.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.stats import gamma

from .volume import Volume


def spm_hrf(dt: float, length: float = 32.0) -> np.ndarray:
    """SPM's canonical haemodynamic response sampled every dt seconds:
    gamma(6) peak minus gamma(16)/6 undershoot, normalised to unit sum."""
    t = np.arange(0, length, dt)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()


def read_events(path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [{"onset": float(r["onset"]), "duration": float(r["duration"]),
             "trial_type": r["trial_type"], "weight": float(r.get("weight", 1) or 1)} for r in rows]


def cosine_drift(n: int, tr: float, cutoff_s: float = 128.0) -> np.ndarray:
    """DCT-II basis up to the high-pass cut-off (constant term excluded),
    as in SPM and nilearn's 'cosine' drift model."""
    k_max = int(np.floor(2 * n * tr / cutoff_s))
    t = np.arange(n)
    cols = [np.sqrt(2.0 / n) * np.cos(np.pi * (2 * t + 1) * k / (2 * n)) for k in range(1, k_max + 1)]
    return np.stack(cols, 1) if cols else np.zeros((n, 0))


@dataclass
class Design:
    X: np.ndarray
    names: list


def design_matrix(n_scans: int, tr: float, events: list[dict], confounds: np.ndarray | None = None,
                  confound_names=None, cutoff_s: float = 128.0, oversampling: int = 50) -> Design:
    dt = tr / oversampling
    n_hi = n_scans * oversampling
    t_hi = np.arange(n_hi) * dt
    hrf = spm_hrf(dt)
    conditions = sorted({e["trial_type"] for e in events})
    cols, names = [], []
    for cond in conditions:
        box = np.zeros(n_hi)
        for e in events:
            if e["trial_type"] == cond:
                box[(t_hi >= e["onset"]) & (t_hi < e["onset"] + e["duration"])] += e["weight"]
        conv = np.convolve(box, hrf)[:n_hi]
        cols.append(conv[::oversampling])  # value at each scan's start time
        names.append(cond)
    drift = cosine_drift(n_scans, tr, cutoff_s)
    for k in range(drift.shape[1]):
        cols.append(drift[:, k])
        names.append(f"drift_{k + 1}")
    if confounds is not None:
        for k in range(confounds.shape[1]):
            cols.append(confounds[:, k])
            names.append(confound_names[k] if confound_names else f"confound_{k}")
    cols.append(np.ones(n_scans))
    names.append("constant")
    return Design(np.stack(cols, 1), names)


def contrast_vector(design: Design, spec: dict) -> np.ndarray:
    c = np.zeros(len(design.names))
    for name, w in spec.items():
        c[design.names.index(name)] = w
    return c


def _t_ols(X, Y, contrast):
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ X.T @ Y
    resid = Y - X @ B
    dof = X.shape[0] - np.linalg.matrix_rank(X)
    sigma2 = (resid ** 2).sum(0) / dof
    var_c = float(contrast @ XtX_inv @ contrast)
    return (contrast @ B) / np.sqrt(sigma2 * var_c + 1e-30), resid, dof


def ar1_coefficients(resid: np.ndarray, bins: int = 100) -> np.ndarray:
    """Lag-1 autocorrelation of each voxel's OLS residuals (Yule-Walker),
    truncated to 1/bins steps, as nilearn does so that voxels with the same
    coefficient can share one whitened fit."""
    n = resid.shape[0]
    y = resid - resid.mean()
    r0 = (y * y).sum(0) / (n * n)
    r1 = (y[1:] * y[:-1]).sum(0) / ((n - 1) * n)
    rho = r1 / np.maximum(r0, 1e-30)
    return (rho * bins).astype(int) / bins


def ar1_whiten(A: np.ndarray, rho: float) -> np.ndarray:
    """Row t becomes A[t] - rho A[t-1]; the first row is kept."""
    W = np.array(A, dtype=np.float64, copy=True)
    W[1:] = W[1:] - rho * np.asarray(A, np.float64)[:-1]
    return W


def fit_t(bold: Volume, design: Design, contrast: np.ndarray, mask: np.ndarray,
          smooth_fwhm_mm: float = 0.0, noise_model: str = "ar1") -> tuple[Volume, dict]:
    """Fit inside `mask`; returns the t-map as a Volume and fit stats.

    noise_model "ols": ordinary least squares. "ar1" (default, as in nilearn
    and SPM): fMRI noise is autocorrelated, which makes OLS t-values too large;
    the residuals' lag-1 autocorrelation is estimated per voxel and data and
    design are prewhitened before refitting."""
    data = bold.data
    if smooth_fwhm_mm > 0:
        sig = smooth_fwhm_mm / 2.3548 / bold.spacing[:3]
        data = np.stack([ndimage.gaussian_filter(data[..., t], sig) for t in range(data.shape[3])], 3)
    Y = data[mask].T.astype(np.float64)          # (n_scans, n_voxels)
    X = design.X
    t, resid, dof = _t_ols(X, Y, contrast)
    stats = {"dof": int(dof), "noise_model": noise_model}
    if noise_model == "ar1":
        rho = ar1_coefficients(resid)
        t = np.empty(Y.shape[1])
        for r in np.unique(rho):
            idx = rho == r
            t[idx], _, _ = _t_ols(ar1_whiten(X, r), ar1_whiten(Y[:, idx], r), contrast)
        stats.update({"ar1_median": float(np.median(rho)), "ar1_p95": float(np.percentile(rho, 95))})
    elif noise_model != "ols":
        raise ValueError(noise_model)
    out = np.zeros(bold.shape3, np.float32)
    out[mask] = t
    return Volume(out, bold.affine.copy(), {"kind": "t-map", "dof": int(dof), "noise_model": noise_model}), stats
