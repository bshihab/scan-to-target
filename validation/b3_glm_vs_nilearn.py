"""Bar B3: our GLM vs nilearn's FirstLevelModel on the same data.

Same realigned data (from the cached motion matrices), same brain mask, no
smoothing, same events, SPM HRF, cosine drift with a 128 s cut-off, the six
motion parameters as confounds, OLS noise model. nilearn builds its own design
matrix from the events; we only hand it the confounds.
Bar: voxelwise Pearson r >= 0.999 inside the mask and the same peak voxel.

Usage: python validation/b3_glm_vs_nilearn.py [ses ...]
"""
import json
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2t import fmri, glm  # noqa: E402
from s2t.volume import Volume, resample_to  # noqa: E402

DATA = ROOT / "data/ds000114"


def realigned(bold, mats, drop):
    ref = bold.frame(drop)
    n = bold.data.shape[3]
    return Volume(np.stack([resample_to(bold.frame(t), ref, mats[t - drop]).data for t in range(drop, n)], 3),
                  ref.affine.copy(), dict(bold.meta))


def run(ses, sub="sub-01"):
    from nilearn.glm.first_level import FirstLevelModel
    out = ROOT / f"out/{sub}/ses-{ses}/fmri"
    summ = json.loads((out / "fmri_summary.json").read_text())
    drop = int(summ.get("dropped_volumes", 0))
    bold = Volume.from_nifti(DATA / f"{sub}/ses-{ses}/func/{sub}_ses-{ses}_task-fingerfootlips_bold.nii.gz")
    mats = list(np.load(out / "motion_matrices.npy"))
    rv = realigned(bold, mats, drop)
    n, tr = rv.data.shape[3], summ["tr"]
    params = np.loadtxt(out / "motion_params.tsv", skiprows=1)
    mask = nib.load(str(out / "brain_mask.nii.gz")).get_fdata() > 0.5

    events = glm.read_events(DATA / "task-fingerfootlips_events.tsv")
    for e in events:
        e["onset"] -= drop * tr
    design = glm.design_matrix(n, tr, events, confounds=params,
                               confound_names=["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"])
    c = glm.contrast_vector(design, fmri.FINGER_CONTRAST)
    ours, _ = glm.fit_t(rv, design, c, mask, smooth_fwhm_mm=0)

    img = nib.Nifti1Image(rv.data, rv.affine)
    mimg = nib.Nifti1Image(mask.astype(np.uint8), rv.affine)
    ev = pd.DataFrame([{"onset": e["onset"], "duration": e["duration"], "trial_type": e["trial_type"]} for e in events])
    conf = pd.DataFrame(params, columns=["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flm = FirstLevelModel(t_r=tr, slice_time_ref=0.0, hrf_model="spm", drift_model="cosine",
                              high_pass=1.0 / 128, noise_model="ols", mask_img=mimg, smoothing_fwhm=None,
                              signal_scaling=False, minimize_memory=False)
        flm.fit(img, events=ev, confounds=conf)
        theirs = flm.compute_contrast("Finger - 0.5*Foot - 0.5*Lips", stat_type="t", output_type="stat").get_fdata()

    a, b = ours.data[mask], theirs[mask]
    r = float(np.corrcoef(a, b)[0, 1])
    pa = np.unravel_index(int(np.argmax(np.where(mask, ours.data, -np.inf))), mask.shape)
    pb = np.unravel_index(int(np.argmax(np.where(mask, theirs, -np.inf))), mask.shape)
    res = {"session": ses, "n_scans": n, "dropped_volumes": drop, "pearson_r": r,
           "max_abs_t_diff": float(np.abs(a - b).max()), "peak_ours": [int(x) for x in pa],
           "peak_nilearn": [int(x) for x in pb], "peak_t_ours": float(ours.data[pa]), "peak_t_nilearn": float(theirs[pb]),
           "n_vox_t_gt_3_1_ours": int((a > 3.1).sum()), "n_vox_t_gt_3_1_nilearn": int((b > 3.1).sum()),
           "passed": bool(r >= 0.999 and pa == pb)}
    return res


def main():
    sessions = sys.argv[1:] or ["test", "retest"]
    rows = [run(s) for s in sessions]
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out/b3.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
