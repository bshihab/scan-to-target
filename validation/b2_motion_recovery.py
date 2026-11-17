"""Bar B2: recovery of known head motion.

Volume 0 of sub-01 ses-test is moved by 30 random rigid transforms (up to 3 deg
and 3 mm), resampled with cubic splines, given Gaussian noise at the run's
measured temporal noise level, and realigned back to the original.

Both images are cropped 3 slices (12 mm) from the top and bottom of the full
volume, and the moved image is resampled from the full volume. Without the crop
the simulation fills whatever moves out of the field of view with zeros, which
never happens in a real scan (the scanner images whatever tissue is in the slab)
and biases the fit at the edges. Run with --zero-fill for the original version.
Error is the distance between the recovered and true transforms at 64 points on
a 60 mm sphere around the brain centre.
Bar: median <= 0.2 mm and max <= 0.5 mm.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2t import transforms as T  # noqa: E402
from s2t.register import foreground_threshold, register_rigid  # noqa: E402
from s2t.volume import Volume, resample_to  # noqa: E402

BOLD = ROOT / "data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz"


def main(n=30, seed=7, zero_fill=False):
    bold = Volume.from_nifti(BOLD)
    full = bold.frame(0)
    if zero_fill:
        ref = full
    else:
        A = full.affine.copy()
        A[:3, 3] = full.voxel_to_world([0, 0, 3])
        ref = Volume(full.data[:, :, 3:-3].copy(), A, dict(full.meta))
    fg = ref.data > foreground_threshold(ref.data)
    # temporal noise: median over foreground voxels of the std of the detrended series
    ts = bold.data[:, :, 3:-3][fg] if not zero_fill else bold.data[fg]
    ts = ts - np.linspace(0, 1, ts.shape[1])[None] * (ts[:, -1:] - ts[:, :1]) - ts.mean(1, keepdims=True)
    noise_sd = float(np.median(ts.std(axis=1)))
    center = ref.voxel_to_world(np.argwhere(fg).mean(0))
    pts = T.sphere_points(center, 60.0, 64)
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for k in range(n):
        Tk = T.random_rigid(rng, 3.0, 3.0, center)
        moved = resample_to(full, ref, Tk, order=3)  # point x in ref -> Tk x in moved
        moved.data = moved.data + rng.normal(0, noise_sd, moved.data.shape).astype(np.float32)
        M = register_rigid(ref, moved, metric="ssd").matrix  # moved world -> ref world
        err = T.transform_error_mm(M, np.linalg.inv(Tk), pts)
        rows.append({"true_deg": T.rotation_angle_deg(Tk), "true_mm": float(np.linalg.norm(Tk[:3, 3])),
                     "err_mean_mm": float(err.mean()), "err_max_mm": float(err.max())})
    errs = np.array([r["err_mean_mm"] for r in rows])
    worst = np.array([r["err_max_mm"] for r in rows])
    summary = {"simulation": "zero-fill" if zero_fill else "crop", "n": n, "noise_sd": noise_sd, "median_mm": float(np.median(errs)),
               "max_mm": float(worst.max()), "seconds_per_registration": (time.time() - t0) / n,
               "passed": bool(np.median(errs) <= 0.2 and worst.max() <= 0.5)}
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out/b2.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(zero_fill="--zero-fill" in sys.argv))
