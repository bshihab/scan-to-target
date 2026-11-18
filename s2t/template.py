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


# Heights (mm, along the T1's superior axis) tried for the brain centre above the
# head's centre of mass. T1 fields of view include different amounts of neck,
# which drags the centre of mass down; one start at 0 left sub-09's retest fit
# stuck 100+ mm away (PLAN.md change log), so several are tried at the coarse
# level and the best one is refined.
START_HEIGHTS_MM = (0.0, 15.0, 30.0, 45.0, -15.0)
METHOD = "multistart-heights-v2"  # cache key: fits made before this change are redone


def align_template(t1: Volume, init: np.ndarray | None = None):
    """Returns (M, info): M maps T1 world -> MNI world (12 parameters).
    The template is the fixed image and only its brain voxels are sampled, so
    the subject's scalp and neck never enter the cost."""
    tv, brain, _ = load_template()
    lv_coarse = (Level(8, 8, max_points=60000),)
    lv_r = (Level(8, 8, max_points=60000), Level(4, 4, max_points=60000))
    lv_a = (Level(4, 4, max_points=100000), Level(2, 2, max_points=100000))
    starts = []
    if init is not None:
        starts.append(np.asarray(init, float))
    else:
        from .register import foreground_threshold
        c_t1 = t1.voxel_to_world(np.argwhere(t1.data > foreground_threshold(t1.data)).mean(0))
        c_mni = tv.voxel_to_world(np.argwhere(brain).mean(0))
        for h in START_HEIGHTS_MM:
            s0 = np.eye(4)
            s0[:3, 3] = c_mni - (c_t1 + np.array([0.0, 0.0, h]))
            starts.append(s0)
    coarse = [register_rigid(tv, t1, metric="mi", init=s0, levels=lv_coarse, fixed_mask=brain) for s0 in starts]
    best = int(np.argmin([c.cost for c in coarse]))
    r1 = register_rigid(tv, t1, metric="mi", init=coarse[best].matrix, levels=lv_r, fixed_mask=brain)
    r2 = register_rigid(tv, t1, metric="mi", init=r1.matrix, levels=lv_a, fixed_mask=brain, dof=12)
    info = {"start_costs": [c.cost for c in coarse], "best_start": best,
            "rigid_cost": r1.cost, "affine_cost": r2.cost}
    info.update(template_qc(t1, r2.matrix))
    return r2.matrix, info


def template_qc(t1: Volume, t1_to_mni: np.ndarray) -> dict:
    """Plausibility of a template fit: the template brain, carried into the T1,
    must sit inside the head (T1 foreground) and have a brain-like volume.
    Thresholds: at least 80% inside, 900-2000 mL. Set after seeing five fits
    (good fits 0.89-0.93 and 1090-1653 mL; the failed sub-09 retest fit 0.36 and
    576 mL): dark CSF inside the brain sits below the foreground threshold."""
    from .register import foreground_threshold
    m = brain_mask_in_t1(t1, t1_to_mni)
    vol_ml = float(m.sum() * np.prod(t1.spacing[:3]) / 1000.0)
    inside = float((t1.data[m] > foreground_threshold(t1.data)).mean()) if m.any() else 0.0
    ok = inside >= 0.80 and 900.0 <= vol_ml <= 2000.0
    return {"qc_brain_inside_head": inside, "qc_brain_volume_ml": vol_ml, "qc_ok": bool(ok)}


def brain_mask_in_t1(t1: Volume, t1_to_mni: np.ndarray, erode_mm: float = 0.0) -> np.ndarray:
    tv, brain, _ = load_template()
    mv = Volume(brain.astype(np.float32), tv.affine)
    # resample_to(src, ref, src_to_ref): src world (MNI) -> ref world (T1) is inv(t1_to_mni)
    m = resample_to(mv, t1, np.linalg.inv(t1_to_mni), order=1).data > 0.5
    if erode_mm > 0:
        from scipy import ndimage
        m = ndimage.distance_transform_edt(m, sampling=t1.spacing[:3]) > erode_mm
    return m
