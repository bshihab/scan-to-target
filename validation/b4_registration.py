"""Bar B4: fMRI-to-T1 rigid registration.

B4a, known truth. A pseudo-EPI is made from sub-01's T1: inside the head the
contrast is inverted (EPI shows fluid bright and white matter dark, the T1 the
opposite), it is resampled onto the real EPI's 4 mm grid and 30-slice slab
through a known random rigid transform (up to 10 deg and 15 mm), blurred to
6 mm FWHM, and given 3% Gaussian noise plus a smooth multiplicative bias field.
Registration starts from the scanner alignment (identity). Error is measured
at 64 points on a 60 mm sphere around the brain centre.
Bar: median <= 1.0 mm, 95th percentile <= 2.0 mm, none above 5 mm.

B4b, agreement on real data. The real motion-corrected mean EPI of each
session is registered to that session's T1 by ours and by SimpleITK (Mattes
MI, 3 levels, gradient descent with physical-shift scaling). Neither is ground
truth. Bar: the two agree within 2.0 mm at every sphere point.

Usage: python validation/b4_registration.py [a|b]   (default both; writes out/b4a.json, out/b4b.json)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2t import transforms as T  # noqa: E402
from s2t.register import foreground_threshold, register_rigid  # noqa: E402
from s2t.volume import Volume, resample_to  # noqa: E402

D = ROOT / "data/ds000114/sub-01"
F = np.diag([-1.0, -1.0, 1.0, 1.0])  # RAS <-> LPS


def t1_path(ses):
    return D / f"ses-{ses}/anat/sub-01_ses-{ses}_T1w.nii.gz"


def bold_path(ses):
    return D / f"ses-{ses}/func/sub-01_ses-{ses}_task-fingerfootlips_bold.nii.gz"


def head_mask(t1: Volume) -> np.ndarray:
    m = t1.data > foreground_threshold(t1.data)
    m = ndimage.binary_closing(m, iterations=2)
    return ndimage.binary_fill_holes(m)


def brain_center(t1: Volume) -> np.ndarray:
    return t1.voxel_to_world(np.argwhere(head_mask(t1)).mean(0))


def pseudo_epi(t1: Volume, grid: Volume, Tk: np.ndarray, rng) -> Volume:
    hm = head_mask(t1)
    hi = np.percentile(t1.data[hm], 99)
    inv = np.where(hm, np.clip(1.15 * hi - t1.data, 0, None), 0).astype(np.float32)
    src = Volume(inv, t1.affine)
    out = resample_to(src, grid, Tk, order=1)  # point x in T1 -> Tk x in pseudo
    sig = 6.0 / 2.3548 / grid.spacing[:3]
    arr = ndimage.gaussian_filter(out.data, sig)
    ijk = np.indices(arr.shape).astype(np.float32)
    bias = 1 + 0.15 * np.sin(ijk[0] / arr.shape[0] * np.pi) * np.cos(ijk[2] / arr.shape[2] * np.pi / 2)
    arr = arr * bias
    arr = arr + rng.normal(0, 0.03 * arr.max(), arr.shape)
    return Volume(arr.astype(np.float32), grid.affine.copy())


def part_a(n=30, seed=11):
    t1 = Volume.from_nifti(t1_path("test"))
    grid = Volume.from_nifti(bold_path("test")).frame(1)
    c = brain_center(t1)
    pts = T.sphere_points(c, 60.0, 64)
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for k in range(n):
        Tk = T.random_rigid(rng, 10.0, 15.0, c)
        pe = pseudo_epi(t1, grid, Tk, rng)
        M = register_rigid(t1, pe, metric="mi").matrix
        err = T.transform_error_mm(M, np.linalg.inv(Tk), pts)
        rows.append({"true_deg": T.rotation_angle_deg(Tk), "true_mm": float(np.linalg.norm(Tk[:3, 3])),
                     "err_mean_mm": float(err.mean()), "err_max_mm": float(err.max())})
        print(f"  a {k + 1}/{n}: {rows[-1]['true_deg']:.1f} deg {rows[-1]['true_mm']:.1f} mm -> "
              f"{rows[-1]['err_mean_mm']:.2f} mm (max {rows[-1]['err_max_mm']:.2f})", flush=True)
    e = np.array([r["err_mean_mm"] for r in rows])
    emax = np.array([r["err_max_mm"] for r in rows])
    s = {"n": n, "median_mm": float(np.median(e)), "p95_mm": float(np.percentile(e, 95)),
         "worst_point_mm": float(emax.max()), "n_over_5mm": int((emax > 5).sum()),
         "seconds_per_registration": (time.time() - t0) / n}
    s["passed"] = bool(s["median_mm"] <= 1.0 and s["p95_mm"] <= 2.0 and s["n_over_5mm"] == 0)
    return {"summary": s, "rows": rows}


def sitk_register(fixed_path, moving_path):
    import SimpleITK as sitk
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=300,
                                                 relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # identity start (both images are in the same scanner frame), rotation about the fixed centre
    tx = sitk.Euler3DTransform()
    tx.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in fixed.GetSize()]))
    reg.SetInitialTransform(tx, inPlace=False)
    out = reg.Execute(fixed, moving)
    final = sitk.CompositeTransform(out).GetBackTransform() if out.GetName() == "CompositeTransform" else out
    e = sitk.Euler3DTransform(final) if final.GetName() != "Euler3DTransform" else final
    A = np.eye(4)
    A[:3, :3] = np.array(e.GetMatrix()).reshape(3, 3)
    ctr = np.array(e.GetCenter())
    A[:3, 3] = np.array(e.GetTranslation()) + ctr - A[:3, :3] @ ctr
    # A maps fixed LPS -> moving LPS. Ours maps moving RAS -> fixed RAS.
    return np.linalg.inv(F @ A @ F), reg.GetMetricValue()


def part_b():
    rows = []
    for ses in ("test", "retest"):
        t1 = Volume.from_nifti(t1_path(ses))
        mean_path = ROOT / f"out/sub-01/ses-{ses}/fmri/mean_epi.nii.gz"
        mean = Volume.from_nifti(mean_path)
        t0 = time.time()
        ours = register_rigid(t1, mean, metric="mi").matrix
        t_ours = time.time() - t0
        t0 = time.time()
        theirs, metric = sitk_register(t1_path(ses), mean_path)
        t_sitk = time.time() - t0
        pts = T.sphere_points(brain_center(t1), 60.0, 64)
        d = T.transform_error_mm(ours, theirs, pts)
        moved_by = T.transform_error_mm(ours, np.eye(4), pts)
        rows.append({"session": ses, "agreement_max_mm": float(d.max()), "agreement_mean_mm": float(d.mean()),
                     "ours_vs_scanner_alignment_mean_mm": float(moved_by.mean()),
                     "ours_rotation_deg": T.rotation_angle_deg(ours), "seconds_ours": t_ours, "seconds_sitk": t_sitk,
                     "ours_matrix": ours.tolist(), "sitk_matrix": theirs.tolist(), "passed": bool(d.max() <= 2.0)})
        np.save(ROOT / f"out/sub-01/ses-{ses}/fmri/epi_to_t1.npy", ours)
        print(f"  b {ses}: agreement max {d.max():.2f} mm, mean {d.mean():.2f} mm; "
              f"registration moved the EPI {moved_by.mean():.2f} mm from the scanner alignment", flush=True)
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    ok = True
    (ROOT / "out").mkdir(exist_ok=True)
    if "a" in which:
        a = part_a()
        (ROOT / "out/b4a.json").write_text(json.dumps(a, indent=2))
        print("B4a", json.dumps(a["summary"]))
        ok &= a["summary"]["passed"]
    if "b" in which:
        b = part_b()
        (ROOT / "out/b4b.json").write_text(json.dumps(b, indent=2))
        print("B4b passed:", b["passed"])
        ok &= b["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
