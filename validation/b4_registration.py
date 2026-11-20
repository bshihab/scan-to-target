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

B4a-d adds susceptibility-like distortion and dropout to the pseudo-EPI (PLAN.md).

Usage: python validation/b4_registration.py [a|b|d]   (default a and b; writes out/b4a.json, out/b4b.json, out/b4a_distorted.json)
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
from s2t import template  # noqa: E402
from s2t.register import (foreground_threshold, register_bbr, register_epi_to_t1,  # noqa: E402
                          register_epi_to_t1_bbr, register_rigid, t1_registration_mask)
from s2t.register import Level  # noqa: E402
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


def distorted_pseudo_epi(t1: Volume, grid: Volume, Tk: np.ndarray, rng, mni: np.ndarray) -> Volume:
    """Pseudo-EPI with susceptibility-like distortion and dropout (bar B4a-d).
    Regions are placed with the T1-to-MNI matrix: orbitofrontal cortex near
    MNI (0, 35, -20), temporal poles near (+-40, 15, -35)."""
    hm = head_mask(t1)
    hi = np.percentile(t1.data[hm], 99)
    inv = np.where(hm, np.clip(1.15 * hi - t1.data, 0, None), 0).astype(np.float32)
    inv = (inv / max(inv.max(), 1e-6)) ** 0.7 * 1000.0
    to_t1 = np.linalg.inv(mni)
    centres = [(T.apply(to_t1, [0, 35, -20]), 4.0), (T.apply(to_t1, [40, 15, -35]), 2.5), (T.apply(to_t1, [-40, 15, -35]), 2.5)]
    ijk = np.indices(grid.shape3).reshape(3, -1).T.astype(float)
    world = grid.voxel_to_world(ijk)                      # pseudo-EPI world
    t1_world = T.apply(np.linalg.inv(Tk), world)          # the same anatomy in T1 world
    pe = grid.affine[:3, 1] / np.linalg.norm(grid.affine[:3, 1])  # phase-encode axis = EPI j axis
    shift = np.zeros(len(world))
    drop = np.ones(len(world))
    for c, amp in centres:
        g = np.exp(-np.sum((t1_world - c) ** 2, 1) / (2 * 18.0 ** 2))
        shift += amp * g
        drop *= 1 - 0.7 * g
    src = t1_world + shift[:, None] * pe                  # each EPI voxel shows tissue displaced along PE
    vox = T.apply(np.linalg.inv(t1.affine), src)
    vals = ndimage.map_coordinates(inv, vox.T, order=1, cval=0.0) * drop
    arr = vals.reshape(grid.shape3)
    sig = 6.0 / 2.3548 / grid.spacing[:3]
    arr = ndimage.gaussian_filter(arr, sig)
    arr = arr + rng.normal(0, 0.03 * arr.max(), arr.shape)
    return Volume(arr.astype(np.float32), grid.affine.copy())


def part_a(n=30, seed=11, distorted=False):
    t1 = Volume.from_nifti(t1_path("test"))
    grid = Volume.from_nifti(bold_path("test")).frame(1)
    c = brain_center(t1)
    pts = T.sphere_points(c, 60.0, 64)
    brain = template.brain_mask_in_t1(t1, t1_to_mni("test", t1))
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for k in range(n):
        Tk = T.random_rigid(rng, 10.0, 15.0, c)
        pe = distorted_pseudo_epi(t1, grid, Tk, rng, t1_to_mni("test", t1)) if distorted else pseudo_epi(t1, grid, Tk, rng)
        M = register_epi_to_t1_bbr(t1, pe, brain).matrix
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


def t1_to_mni(ses, t1=None):
    f = ROOT / f"out/sub-01/ses-{ses}/t1_to_mni.npy"
    if not f.exists():
        np.save(f, template.align_template(t1 if t1 is not None else Volume.from_nifti(t1_path(ses)))[0])
    return np.load(f)


def sitk_register(fixed_path, moving_path, mask=None):
    """mask: boolean array on the fixed image's grid (numpy [i, j, k] order)."""
    import SimpleITK as sitk
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    if mask is not None:
        mimg = sitk.GetImageFromArray(np.transpose(mask.astype(np.uint8), (2, 1, 0)))  # SimpleITK arrays are [k, j, i]
        mimg.CopyInformation(fixed)
        reg.SetMetricFixedMask(mimg)
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
    """Run 3 (boundary-based registration; bars in PLAN.md written before it
    ran). Run-1 and run-2 answers are read from out/b4b_run1.json and _run2."""
    import nibabel as nib
    prev = {k: {r["session"]: r for r in json.loads((ROOT / f"out/b4b_{k}.json").read_text())["rows"]}
            for k in ("run1", "run2")}
    rows, A, t1s, means = [], {}, {}, {}
    for ses in ("test", "retest"):
        t1 = Volume.from_nifti(t1_path(ses))
        mean_path = ROOT / f"out/sub-01/ses-{ses}/fmri/mean_epi.nii.gz"
        mean = Volume.from_nifti(mean_path)
        t1s[ses], means[ses] = t1, mean
        brain = template.brain_mask_in_t1(t1, t1_to_mni(ses, t1))
        pts = T.sphere_points(brain_center(t1), 60.0, 64)
        t0 = time.time()
        ours = register_epi_to_t1_bbr(t1, mean, brain).matrix
        t_ours = time.time() - t0
        starts = {"scanner": np.eye(4), "run1": np.array(prev["run1"][ses]["ours_matrix"]),
                  "run2": np.array(prev["run2"][ses]["ours_matrix"])}
        refined = {k: register_bbr(t1, mean, v, brain).matrix for k, v in starts.items()}
        allm = list(refined.values()) + [ours]
        spread = max(T.transform_error_mm(a, b, pts).max() for a in allm for b in allm)
        theirs, _ = sitk_register(t1_path(ses), mean_path, mask=t1_registration_mask(t1))
        d = T.transform_error_mm(ours, theirs, pts)
        A[ses] = ours
        rows.append({"session": ses, "start_spread_max_mm": float(spread), "sitk_agreement_max_mm": float(d.max()),
                     "sitk_agreement_mean_mm": float(d.mean()),
                     "ours_vs_scanner_alignment_mean_mm": float(T.transform_error_mm(ours, np.eye(4), pts).mean()),
                     "ours_vs_run1_mean_mm": float(T.transform_error_mm(ours, starts["run1"], pts).mean()),
                     "ours_vs_run2_mean_mm": float(T.transform_error_mm(ours, starts["run2"], pts).mean()),
                     "seconds_ours": t_ours, "ours_matrix": ours.tolist(), "sitk_matrix": theirs.tolist(),
                     "passed_b_spread": bool(spread <= 1.0), "passed_d_sitk": bool(d.max() <= 2.0)})
        # kept apart from the pipeline's cached matrix (fmri/epi_to_t1.npy), whose
        # epi_to_t1.json key and gate record describe the pipeline's own run
        np.save(ROOT / f"out/b4b_epi_to_t1_{ses}.npy", ours)
        print(f"  b {ses}: start spread {spread:.2f} mm; vs SimpleITK max {d.max():.2f} mm; EPI moved "
              f"{rows[-1]['ours_vs_scanner_alignment_mean_mm']:.2f} mm from the scanner alignment", flush=True)

    # (c) loop closure across sessions
    brain_test = template.brain_mask_in_t1(t1s["test"], t1_to_mni("test"))
    C = register_rigid(t1s["test"], t1s["retest"], metric="mi", fixed_mask=brain_test,
                       levels=(Level(8, 8, max_points=100000), Level(4, 4, max_points=100000),
                               Level(2, 2, max_points=100000, order=3))).matrix  # T1_retest -> T1_test
    epi_mask = nib.load(str(ROOT / "out/sub-01/ses-test/fmri/brain_mask.nii.gz")).get_fdata() > 0.5
    D = register_rigid(means["test"], means["retest"], metric="ssd", fixed_mask=epi_mask).matrix  # EPI_retest -> EPI_test
    # both sides act on retest-EPI points, so the sphere sits in the retest frame
    # (it was centred on the test T1's brain before the second review; run 5 fixes it)
    pts = T.sphere_points(brain_center(t1s["retest"]), 60.0, 64)
    loop = T.transform_error_mm(A["test"] @ D, C @ A["retest"], pts)
    loop_info = {"max_mm": float(loop.max()), "mean_mm": float(loop.mean()), "passed_c_loop": bool(loop.max() <= 2.0),
                 "t1_retest_to_test_rotation_deg": T.rotation_angle_deg(C),
                 "epi_retest_to_test_rotation_deg": T.rotation_angle_deg(D)}
    print(f"  b loop closure: max {loop.max():.2f} mm, mean {loop.mean():.2f} mm", flush=True)
    passed = all(r["passed_b_spread"] for r in rows) and loop_info["passed_c_loop"]
    return {"run": 5, "rows": rows, "loop_closure": loop_info,
            "passed_b_and_c": bool(passed), "passed_d_sitk": all(r["passed_d_sitk"] for r in rows),
            "passed": bool(passed and all(r["passed_d_sitk"] for r in rows))}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    ok = True
    (ROOT / "out").mkdir(exist_ok=True)
    if which == "d":
        a = part_a(distorted=True)
        (ROOT / "out/b4a_distorted.json").write_text(json.dumps(a, indent=2))
        print("B4a-d", json.dumps(a["summary"]))
        return 0 if a["summary"]["passed"] else 1
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
