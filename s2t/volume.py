"""A volume is an array plus the 4x4 matrix that maps voxel indices to scanner RAS mm."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from . import transforms as T


@dataclass
class Volume:
    data: np.ndarray            # (I, J, K) or (I, J, K, T)
    affine: np.ndarray          # 4x4, voxel (i, j, k) -> scanner RAS mm
    meta: dict = field(default_factory=dict)

    # ---- geometry -------------------------------------------------------
    @property
    def shape3(self):
        return self.data.shape[:3]

    @property
    def spacing(self) -> np.ndarray:
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    def voxel_to_world(self, ijk) -> np.ndarray:
        return T.apply(self.affine, ijk)

    def world_to_voxel(self, xyz) -> np.ndarray:
        return T.apply(np.linalg.inv(self.affine), xyz)

    def center_world(self) -> np.ndarray:
        return self.voxel_to_world((np.array(self.shape3) - 1) / 2.0)

    def corners_world(self) -> np.ndarray:
        I, J, K = (np.array(self.shape3) - 1)
        c = np.array([[i, j, k] for i in (0, I) for j in (0, J) for k in (0, K)], float)
        return self.voxel_to_world(c)

    # ---- sampling -------------------------------------------------------
    def sample(self, xyz, order: int = 1, cval: float = 0.0, frame: int | None = None) -> np.ndarray:
        """Sample the volume at world points (N, 3). Points outside get `cval`."""
        arr = self.data if self.data.ndim == 3 else self.data[..., 0 if frame is None else frame]
        ijk = self.world_to_voxel(np.atleast_2d(xyz))
        return ndimage.map_coordinates(arr, ijk.T, order=order, mode="constant", cval=cval)

    def mean(self) -> "Volume":
        if self.data.ndim == 3:
            return self
        return Volume(self.data.mean(axis=3).astype(np.float32), self.affine.copy(), dict(self.meta))

    def frame(self, t: int) -> "Volume":
        return Volume(np.asarray(self.data[..., t], dtype=np.float32), self.affine.copy(), dict(self.meta))

    # ---- io -------------------------------------------------------------
    @classmethod
    def from_nifti(cls, path) -> "Volume":
        """Load a NIfTI file. The matrix follows the NIfTI-1 rule as nibabel
        implements it: the sform if sform_code > 0, otherwise the qform if
        qform_code > 0, otherwise a scaling-only fallback (flagged in meta)."""
        img = nib.load(str(path))
        h = img.header
        sc, qc = int(h["sform_code"]), int(h["qform_code"])
        source = "sform" if sc > 0 else ("qform" if qc > 0 else "fallback")
        data = np.asarray(img.dataobj, dtype=np.float32)  # applies scl_slope/inter
        meta = {"source": str(path), "affine_from": source, "sha256": sha256(path),
                "zooms": [float(z) for z in h.get_zooms()]}
        return cls(data, img.affine.astype(float), meta)

    def to_nifti(self, path) -> None:
        img = nib.Nifti1Image(np.asarray(self.data, dtype=np.float32), self.affine)
        img.set_sform(self.affine, code=1)
        img.set_qform(self.affine, code=1)
        nib.save(img, str(path))


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resample_to(src: Volume, ref: Volume, src_to_ref: np.ndarray | None = None,
                order: int = 1, cval: float = 0.0) -> Volume:
    """Resample `src` onto `ref`'s grid. `src_to_ref` maps src world points to ref
    world points (identity if both are already in the same world frame)."""
    M = np.eye(4) if src_to_ref is None else src_to_ref
    # ref voxel -> ref world -> src world -> src voxel, as one matrix
    vox = np.linalg.inv(src.affine) @ np.linalg.inv(M) @ ref.affine
    arr = src.data if src.data.ndim == 3 else src.data.mean(axis=3)
    out = ndimage.affine_transform(arr, vox[:3, :3], vox[:3, 3], output_shape=ref.shape3,
                                   order=order, mode="constant", cval=cval)
    return Volume(out.astype(np.float32), ref.affine.copy(), {"resampled_from": src.meta.get("source")})
