"""M7: does the planned target land in the same place on two different days?

For each subject: run the pipeline on ses-test and ses-retest independently,
take the peak of the finger map as the automatic target in each session,
carry the retest target into the test T1 (T1-to-T1 rigid registration inside
the template brain mask) and measure the distance. MNI coordinates (affine,
approximate) are reported so the location can be judged against the hand area.

No pass bar (PLAN.md): this measures fMRI reproducibility more than software.
Stated expectation before running: under 10 mm for most subjects.

Usage: python validation/m7_test_retest.py [--delete-raw] sub-01 sub-02 ...
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2t import template  # noqa: E402
from s2t import transforms as T  # noqa: E402
from s2t.cli import run_subject, subject_paths  # noqa: E402
from s2t.register import Level, register_rigid  # noqa: E402
from s2t.volume import Volume  # noqa: E402

DATA = ROOT / "data/ds000114"
OUT = ROOT / "out"


def peak_world(tmap_path, epi_to_t1):
    t = Volume.from_nifti(tmap_path)
    ijk = np.array(np.unravel_index(int(np.argmax(t.data)), t.shape3), float)
    return T.apply(epi_to_t1, t.voxel_to_world(ijk)), float(t.data.max())


def cluster_centroid_world(tmap_path, epi_to_t1, thr=3.1):
    """t-weighted centre of mass of the connected supra-threshold cluster that
    contains the peak (second target rule, PLAN.md)."""
    from scipy import ndimage
    t = Volume.from_nifti(tmap_path)
    lab, _ = ndimage.label(t.data > thr)
    pk = np.unravel_index(int(np.argmax(t.data)), t.shape3)
    if lab[pk] == 0:
        return None, 0
    m = lab == lab[pk]
    idx = np.argwhere(m).astype(float)
    w = t.data[m]
    c = (idx * w[:, None]).sum(0) / w.sum()
    return T.apply(epi_to_t1, t.voxel_to_world(c)), int(m.sum())


LOOP_MAX_MM = 5.0


def rigid_from_templates(Nt, Nr):
    """Rigid part of inv(Nt) @ Nr (retest T1 -> MNI -> test T1), fixed at the
    template's brain centre. Used to start the T1-to-T1 registration."""
    A = np.linalg.inv(Nt) @ Nr
    U, _, Vt = np.linalg.svd(A[:3, :3])
    R = U @ Vt
    c = np.array([0.0, -18.0, 18.0])
    p, q = T.apply(np.linalg.inv(Nr), c), T.apply(np.linalg.inv(Nt), c)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, q - R @ p
    return M


def t1_retest_to_test(t1_test, t1_retest, Nt, Nr):
    """Rigid retest-T1 -> test-T1 inside the template brain mask. Two starts
    (the template-implied answer and identity), lower cost kept. Loop check:
    Nt @ C must agree with Nr within LOOP_MAX_MM over a 60 mm sphere of points,
    since both map retest-T1 points into MNI (PLAN.md change log: one
    identity-started run of sub-05 converged 54 mm away and nothing flagged it)."""
    brain = template.brain_mask_in_t1(t1_test, Nt)
    levels = (Level(8, 8, max_points=100000), Level(4, 4, max_points=100000), Level(2, 2, max_points=100000, order=3))
    runs = {name: register_rigid(t1_test, t1_retest, metric="mi", fixed_mask=brain, init=init, levels=levels)
            for name, init in (("templates", rigid_from_templates(Nt, Nr)), ("identity", np.eye(4)))}
    best = min(runs, key=lambda k: runs[k].cost)
    C = runs[best].matrix
    P = T.sphere_points(np.array([0.0, -18.0, 18.0]), 60.0, 64)
    loop = float(np.linalg.norm(T.apply(Nt @ C, T.apply(np.linalg.inv(Nr), P)) - P, axis=1).max())
    return C, {"costs": {k: v.cost for k, v in runs.items()}, "start": best,
               "loop_max_mm": loop, "loop_ok": loop <= LOOP_MAX_MM}


def process(sub, log=print):
    if not subject_paths(DATA, sub, "test")[0].exists() or not subject_paths(DATA, sub, "retest")[0].exists():
        subprocess.run([str(ROOT / "scripts/fetch_data.sh"), sub], check=True)
    res = {"subject": sub}
    for ses in ("test", "retest"):
        run_subject(sub, ses, DATA, OUT, log=lambda m: log(f"  {sub} {ses}: {m}"))
        f = OUT / sub / f"ses-{ses}"
        A = np.load(f / "fmri/epi_to_t1.npy")
        p, tpk = peak_world(f / "fmri/tmap_finger.nii.gz", A)
        cc, csize = cluster_centroid_world(f / "fmri/tmap_finger.nii.gz", A)
        summ = json.loads((f / "fmri/fmri_summary.json").read_text())
        N = np.load(f / "t1_to_mni.npy")
        res[ses] = {"peak_t1_ras": p.tolist(), "peak_t": tpk, "peak_mni_approx": T.apply(N, p).tolist(),
                    "cluster_centroid_t1_ras": None if cc is None else cc.tolist(), "cluster_voxels": csize,
                    "cluster_centroid_mni_approx": None if cc is None else T.apply(N, cc).tolist(),
                    "mean_fd_mm": summ["mean_fd_mm"], "max_fd_mm": summ["max_fd_mm"], "n_vox_t_gt_3_1": summ["n_vox_t_gt_3_1"]}
    t1_test = Volume.from_nifti(subject_paths(DATA, sub, "test")[0])
    t1_retest = Volume.from_nifti(subject_paths(DATA, sub, "retest")[0])
    Nt = np.load(OUT / sub / "ses-test/t1_to_mni.npy")
    Nr = np.load(OUT / sub / "ses-retest/t1_to_mni.npy")
    C, res["t1_registration"] = t1_retest_to_test(t1_test, t1_retest, Nt, Nr)
    np.save(OUT / sub / "t1_retest_to_test.npy", C)
    res["t1_retest_to_test"] = C.tolist()
    retest_in_test = T.apply(C, np.array(res["retest"]["peak_t1_ras"]))
    res["distance_mm"] = float(np.linalg.norm(retest_in_test - np.array(res["test"]["peak_t1_ras"])))
    if res["test"]["cluster_centroid_t1_ras"] and res["retest"]["cluster_centroid_t1_ras"]:
        cr = T.apply(C, np.array(res["retest"]["cluster_centroid_t1_ras"]))
        res["centroid_distance_mm"] = float(np.linalg.norm(cr - np.array(res["test"]["cluster_centroid_t1_ras"])))
    res["distance_mni_mm"] = float(np.linalg.norm(np.array(res["retest"]["peak_mni_approx"]) - np.array(res["test"]["peak_mni_approx"])))
    res["same_hemisphere"] = bool(np.sign(res["test"]["peak_mni_approx"][0]) == np.sign(res["retest"]["peak_mni_approx"][0]))
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    delete_raw = "--delete-raw" in sys.argv
    out_file = OUT / "m7.json"
    rows = {r["subject"]: r for r in json.loads(out_file.read_text())["rows"]} if out_file.exists() else {}
    for sub in args:
        try:
            rows[sub] = process(sub)
            r = rows[sub]
            print(f"{sub}: T1 loop {r['t1_registration']['loop_max_mm']:.1f} mm ({r['t1_registration']['start']} start); peak distance {r['distance_mm']:.1f} mm (MNI {r['distance_mni_mm']:.1f}), "
                  f"cluster-centroid distance {r.get('centroid_distance_mm', float('nan')):.1f} mm; "
                  f"test MNI {np.round(r['test']['peak_mni_approx']).astype(int).tolist()} t={r['test']['peak_t']:.1f}; "
                  f"retest MNI {np.round(r['retest']['peak_mni_approx']).astype(int).tolist()} t={r['retest']['peak_t']:.1f}", flush=True)
        except Exception as e:
            rows[sub] = {"subject": sub, "error": f"{type(e).__name__}: {e}"}
            print(f"{sub}: ERROR {e}", flush=True)
        if delete_raw and sub != "sub-01":
            shutil.rmtree(DATA / sub, ignore_errors=True)
        out_file.write_text(json.dumps({"rows": sorted(rows.values(), key=lambda r: r["subject"])}, indent=2))
    ok = [r for r in rows.values() if "distance_mm" in r and r.get("t1_registration", {}).get("loop_ok")]
    bad = [r["subject"] for r in rows.values() if "distance_mm" in r and not r.get("t1_registration", {}).get("loop_ok")]
    if bad:
        print("excluded, T1-to-T1 loop check failed:", ", ".join(bad))
    if ok:
        d = np.array([r["distance_mm"] for r in ok])
        print(f"M7 peak rule: n={len(ok)} median {np.median(d):.1f} mm, under 10 mm: {(d < 10).sum()}/{len(d)}")
        dc = np.array([r["centroid_distance_mm"] for r in ok if "centroid_distance_mm" in r])
        if len(dc):
            print(f"M7 cluster-centroid rule: n={len(dc)} median {np.median(dc):.1f} mm, under 10 mm: {(dc < 10).sum()}/{len(dc)}")


if __name__ == "__main__":
    main()
