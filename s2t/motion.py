"""Head-motion correction for a 4D fMRI run."""
from __future__ import annotations

import numpy as np

from . import transforms as T
from .register import register_rigid
from .volume import Volume, resample_to


def realign(bold: Volume, ref_index: int = 0, progress=None):
    """Rigidly register every volume to volume `ref_index` (least squares).

    Returns (realigned 4D Volume on the reference grid, list of 4x4 matrices,
    (T, 6) motion parameters). Matrix t maps volume t's world frame onto the
    reference's. Motion parameters are rotations in degrees about x, y, z and
    translations in mm, extracted from the matrices, for use as GLM nuisance
    regressors."""
    ref = bold.frame(ref_index)
    mats, frames = [], []
    prev = np.eye(4)
    n = bold.data.shape[3]
    for t in range(n):
        if t == ref_index:
            M = np.eye(4)
        else:
            # start from the previous volume's answer: motion is continuous
            M = register_rigid(ref, bold.frame(t), metric="ssd", init=prev).matrix
        prev = M
        mats.append(M)
        frames.append(resample_to(bold.frame(t), ref, M, order=1).data)
        if progress:
            progress(t + 1, n)
    data = np.stack(frames, axis=3)
    params = np.array([matrix_to_params(M) for M in mats])
    return Volume(data, ref.affine.copy(), dict(bold.meta, realigned_to=ref_index)), mats, params


def matrix_to_params(M: np.ndarray) -> np.ndarray:
    """(rx, ry, rz) in degrees for R = Rz Ry Rx, and translation in mm."""
    R = M[:3, :3]
    ry = np.arcsin(-np.clip(R[2, 0], -1, 1))
    rx = np.arctan2(R[2, 1], R[2, 2])
    rz = np.arctan2(R[1, 0], R[0, 0])
    return np.concatenate([np.degrees([rx, ry, rz]), M[:3, 3]])


def framewise_displacement(mats, radius: float = 50.0) -> np.ndarray:
    """Power et al. FD: sum of absolute changes in the 6 parameters, rotations
    turned into mm of arc on a 50 mm sphere."""
    p = np.array([matrix_to_params(M) for M in mats])
    d = np.abs(np.diff(p, axis=0))
    d[:, :3] = np.deg2rad(d[:, :3]) * radius
    return np.concatenate([[0.0], d.sum(axis=1)])
