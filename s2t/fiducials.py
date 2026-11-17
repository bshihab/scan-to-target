"""Point-based rigid registration from image space to a robot (or tracker) frame.

In surgery the image-to-robot transform comes from fiducials: markers whose
positions are known in the image and are touched or tracked in the room. The
fit, its residual and the error it implies at the target are computed here.

Terms (Fitzpatrick, West and Maurer, IEEE TMI 1998):
- FLE, fiducial localisation error: noise in measuring each marker.
- FRE, fiducial registration error: RMS residual of the fit at the markers.
  Small FRE does not mean a small error at the target.
- TRE, target registration error: the error at the point that matters.
"""
from __future__ import annotations

import numpy as np


def fit_rigid(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform M (4x4) with dst ~ R src + t.
    SVD solution of Arun et al. (1987) with Umeyama's reflection correction:
    without it, noisy near-planar layouts can return a mirror image."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need at least 3 matching points")
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, S, Vt = np.linalg.svd(H)
    if S[1] < 1e-9 * max(S[0], 1e-12):
        raise ValueError("fiducials are collinear: rotation about their line is undetermined")
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = cd - R @ cs
    return M


def fre(M: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    pred = src @ M[:3, :3].T + M[:3, 3]
    return float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))


def predicted_tre_rms(fiducials: np.ndarray, target: np.ndarray, fle_rms: float) -> float:
    """Fitzpatrick's expected RMS TRE at `target`:
        <TRE^2> = <FLE^2>/N * (1 + 1/3 * sum_k d_k^2 / f_k^2)
    d_k: distance of the target from principal axis k of the fiducial
    configuration; f_k: RMS distance of the fiducials from that axis.
    `fle_rms` is the RMS 3D localisation error (sqrt(3)*sigma for isotropic
    per-axis noise sigma)."""
    P = np.asarray(fiducials, float)
    c = P.mean(0)
    X = P - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    r = np.asarray(target, float) - c
    total = 0.0
    for k in range(3):
        axis = Vt[k]
        d2 = np.sum(r ** 2) - (r @ axis) ** 2              # target to axis k
        f2 = np.mean(np.sum(X ** 2, 1) - (X @ axis) ** 2)  # fiducials to axis k
        total += d2 / f2
    n = len(P)
    return float(np.sqrt(fle_rms ** 2 / n * (1 + total / 3.0)))


def simulate_tre(fiducials: np.ndarray, target: np.ndarray, sigma: float, M_true: np.ndarray,
                 trials: int, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo: robot-space fiducial measurements = true positions plus
    isotropic Gaussian noise (sigma per axis); refit; return the target error
    of every trial."""
    P = np.asarray(fiducials, float)
    true_dst = P @ M_true[:3, :3].T + M_true[:3, 3]
    true_t = M_true[:3, :3] @ target + M_true[:3, 3]
    out = np.empty(trials)
    for i in range(trials):
        noisy = true_dst + rng.normal(0, sigma, true_dst.shape)
        M = fit_rigid(P, noisy)
        out[i] = np.linalg.norm(M[:3, :3] @ target + M[:3, 3] - true_t)
    return out
