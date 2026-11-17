"""Rigid transforms and frame conversions.

Conventions used everywhere in this package:
- Points are rows: an (N, 3) array of x, y, z in mm.
- A transform is a 4x4 homogeneous matrix M acting on column vectors, so
  world = M @ [x, y, z, 1]^T. `apply` handles the row/column bookkeeping.
- Rigid parameters are (rx, ry, rz, tx, ty, tz): rotations in radians about the
  x, y, z axes applied in that order (R = Rz @ Ry @ Rx), about a given centre,
  followed by a translation in mm.
"""
from __future__ import annotations

import numpy as np

# DICOM patient coordinates are LPS (+x left, +y posterior, +z superior).
# NIfTI scanner coordinates are RAS. The two differ by negating x and y.
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
RAS_TO_LPS = LPS_TO_RAS  # its own inverse


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 matrix to (N, 3) points (or a single (3,) point)."""
    pts = np.asarray(pts, dtype=float)
    single = pts.ndim == 1
    p = np.atleast_2d(pts)
    out = p @ M[:3, :3].T + M[:3, 3]
    return out[0] if single else out


def rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rigid(params, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """4x4 matrix for x -> R (x - c) + c + t."""
    rx, ry, rz, tx, ty, tz = params
    R = rotation_xyz(rx, ry, rz)
    c = np.asarray(center, dtype=float)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c + np.array([tx, ty, tz])
    return M


def rotation_from_axis_angle(axis, angle: float) -> np.ndarray:
    """Rodrigues' formula."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def random_rigid(rng: np.random.Generator, max_deg: float, max_mm: float,
                 center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A rigid transform with a uniformly random rotation axis, angle uniform in
    [0, max_deg], and a translation with uniformly random direction and length
    uniform in [0, max_mm]. Rotation is about `center`."""
    axis = rng.normal(size=3)
    angle = np.deg2rad(rng.uniform(0, max_deg))
    R = rotation_from_axis_angle(axis, angle)
    d = rng.normal(size=3)
    t = d / np.linalg.norm(d) * rng.uniform(0, max_mm)
    c = np.asarray(center, dtype=float)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c + t
    return M


def is_rigid(M: np.ndarray, tol: float = 1e-6) -> bool:
    R = M[:3, :3]
    return (np.allclose(R.T @ R, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1) < tol
            and np.allclose(M[3], [0, 0, 0, 1]))


def rotation_angle_deg(M: np.ndarray) -> float:
    R = M[:3, :3]
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def sphere_points(center, radius: float = 60.0, n: int = 64) -> np.ndarray:
    """n points spread evenly over a sphere (Fibonacci lattice). Used to turn a
    transform error into a distance in mm at brain-sized radii."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    p = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    return np.asarray(center, dtype=float) + radius * p


def transform_error_mm(A: np.ndarray, B: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance between where A and B send each point."""
    return np.linalg.norm(apply(A, pts) - apply(B, pts), axis=1)
