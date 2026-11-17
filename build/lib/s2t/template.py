"""MNI152 template (ICBM 2009a, bundled with nilearn) aligned to a subject's T1.

Used for two things: a brain mask for the T1 (template-based skull strip) and
approximate MNI coordinates for a planned target. The alignment is affine
(12 parameters), so MNI coordinates are good to several millimetres, not to
the gyrus; they are a sanity check, never the plan.
"""
from __future__ import annotations

import numpy as np

from .register import Level, register_rigid
from .volume import Volume, resample_to


def load_template():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nilearn import datasets
        t = datasets.load_mni152_template(resolution=1)
        m = datasets.load_mni152_brain_mask(resolution=1)
        wm = datasets.load_mni152_wm_template(resolution=1)
    tv = Volume(np.asarray(t.get_fdata(), np.float32), t.affine.astype(float), {"source": "MNI152 2009a (nilearn)"})
    return tv, np.asarray(m.get_fdata()) > 0.5, np.asarray(wm.get_fdata(), np.float32)


def align_template(t1: Volume, init: np.ndarray | None = None):
    """Returns (M, info): M maps T1 world -> MNI world (12 parameters).
    The template is the fixed image and only its brain voxels are sampled, so
    the subject's scalp and neck never enter the cost."""
    tv, brain, _ = load_template()
    # rigid first, then affine from the rigid answer
    lv_r = (Level(8, 8, max_points=60000), Level(4, 4, max_points=60000))
    lv_a = (Level(4, 4, max_points=100000), Level(2, 2, max_points=100000))
    if init is None:
        # start by matching centres of mass: T1 head vs template brain
        from .register import foreground_threshold
        c_t1 = t1.voxel_to_world(np.argwhere(t1.data > foreground_threshold(t1.data)).mean(0))
        c_mni = tv.voxel_to_world(np.argwhere(brain).mean(0))
        init = np.eye(4)
        init[:3, 3] = c_mni - c_t1
    r1 = register_rigid(tv, t1, metric="mi", init=init, levels=lv_r, fixed_mask=brain)
    r2 = register_rigid(tv, t1, metric="mi", init=r1.matrix, levels=lv_a, fixed_mask=brain, dof=12)
    return r2.matrix, {"rigid_cost": r1.cost, "affine_cost": r2.cost}


def brain_mask_in_t1(t1: Volume, t1_to_mni: np.ndarray, erode_mm: float = 0.0) -> np.ndarray:
    tv, brain, _ = load_template()
    mv = Volume(brain.astype(np.float32), tv.affine)
    # resample_to(src, ref, src_to_ref): src world (MNI) -> ref world (T1) is inv(t1_to_mni)
    m = resample_to(mv, t1, np.linalg.inv(t1_to_mni), order=1).data > 0.5
    if erode_mm > 0:
        from scipy import ndimage
        m = ndimage.distance_transform_edt(m, sampling=t1.spacing[:3]) > erode_mm
    return m
