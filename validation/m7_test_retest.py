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


def process(sub, log=print):
    if not subject_paths(DATA, sub, "test")[0].exists() or not subject_paths(DATA, sub, "retest")[0].exists():
        subprocess.run([str(ROOT / "scripts/fetch_data.sh"), sub], check=True)
    res = {"subject": sub}
    for ses in ("test", "retest"):
        run_subject(sub, ses, DATA, OUT, log=lambda m: log(f"  {sub} {ses}: {m}"))
        f = OUT / sub / f"ses-{ses}"
        A = np.load(f / "fmri/epi_to_t1.npy")
        p, tpk = peak_world(f / "fmri/tmap_finger.nii.gz", A)
        summ = json.loads((f / "fmri/fmri_summary.json").read_text())
        res[ses] = {"peak_t1_ras": p.tolist(), "peak_t": tpk, "peak_mni_approx": T.apply(np.load(f / "t1_to_mni.npy"), p).tolist(),
                    "mean_fd_mm": summ["mean_fd_mm"], "max_fd_mm": summ["max_fd_mm"], "n_vox_t_gt_3_1": summ["n_vox_t_gt_3_1"]}
    t1_test = Volume.from_nifti(subject_paths(DATA, sub, "test")[0])
    t1_retest = Volume.from_nifti(subject_paths(DATA, sub, "retest")[0])
    brain = template.brain_mask_in_t1(t1_test, np.load(OUT / sub / "ses-test/t1_to_mni.npy"))
    C = register_rigid(t1_test, t1_retest, metric="mi", fixed_mask=brain,
                       levels=(Level(8, 8, max_points=100000), Level(4, 4, max_points=100000),
                               Level(2, 2, max_points=100000, order=3))).matrix
    retest_in_test = T.apply(C, np.array(res["retest"]["peak_t1_ras"]))
    res["distance_mm"] = float(np.linalg.norm(retest_in_test - np.array(res["test"]["peak_t1_ras"])))
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
            print(f"{sub}: distance {r['distance_mm']:.1f} mm (MNI {r['distance_mni_mm']:.1f}); "
                  f"test MNI {np.round(r['test']['peak_mni_approx']).astype(int).tolist()} t={r['test']['peak_t']:.1f}; "
                  f"retest MNI {np.round(r['retest']['peak_mni_approx']).astype(int).tolist()} t={r['retest']['peak_t']:.1f}", flush=True)
        except Exception as e:
            rows[sub] = {"subject": sub, "error": f"{type(e).__name__}: {e}"}
            print(f"{sub}: ERROR {e}", flush=True)
        if delete_raw and sub != "sub-01":
            shutil.rmtree(DATA / sub, ignore_errors=True)
        out_file.write_text(json.dumps({"rows": sorted(rows.values(), key=lambda r: r["subject"])}, indent=2))
    ok = [r for r in rows.values() if "distance_mm" in r]
    if ok:
        d = np.array([r["distance_mm"] for r in ok])
        print(f"M7: n={len(ok)} median {np.median(d):.1f} mm, under 10 mm: {(d < 10).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
