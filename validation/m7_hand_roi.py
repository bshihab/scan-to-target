"""M7 third target rule (pre-registered in PLAN.md, 2026-11-17, before this ran):
the target is looked for only inside the hand area of primary motor cortex.

Hand area: spheres of radius 20 mm in approximate MNI space at (-37, -21, 58)
and (37, -21, 58), intersected with the session's brain mask. Each EPI voxel is
placed in MNI through epi_to_t1 then t1_to_mni.
  3a: peak t inside the sphere.
  3b: t-weighted centre of the connected t > 3.1 cluster (inside the sphere)
      that contains the 3a peak; none if that peak is below 3.1.
Hemisphere per subject: larger sum of the two sessions' 3a peak t.
Distance: retest point carried into the test T1 by t1_retest_to_test.

Baselines (added after the second code review, 2026-11-20; not pre-registered):
  roi_only: the same distance with each cluster replaced by the whole hand-area
      region, i.e. no fMRI information. This is the floor the sphere imposes.
  cross_subject: 3b cluster centres compared between different people in
      approximate MNI (right-hemisphere centres mirrored to the left), against
      the same distance within a person; p from random re-pairings of retest
      sessions to other subjects (derangements).

Re-analysis of the t-maps written by m7_test_retest.py; no raw data needed.
Usage: python validation/m7_hand_roi.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2t import transforms as T  # noqa: E402
from s2t.volume import Volume  # noqa: E402

OUT = ROOT / "out"
CENTRES = {"left": np.array([-37.0, -21.0, 58.0]), "right": np.array([37.0, -21.0, 58.0])}
RADIUS = 20.0
THR = 3.1


def session(sub, ses):
    f = OUT / sub / f"ses-{ses}"
    t = Volume.from_nifti(f / "fmri/tmap_finger.nii.gz")
    brain = Volume.from_nifti(f / "fmri/brain_mask.nii.gz").data > 0
    A = np.load(f / "fmri/epi_to_t1.npy")
    N = np.load(f / "t1_to_mni.npy")
    ijk = np.indices(t.shape3).reshape(3, -1).T.astype(float)
    t1 = T.apply(A, t.voxel_to_world(ijk))
    mni = T.apply(N, t1)
    out = {}
    for hemi, c in CENTRES.items():
        roi = (np.linalg.norm(mni - c, axis=1) <= RADIUS).reshape(t.shape3) & brain
        if not roi.any():
            # the sphere falls outside this session's EPI coverage or brain mask
            out[hemi] = {"roi_voxels": 0, "peak_t": None, "peak_t1_ras": None, "centroid_t1_ras": None}
            continue
        vals = np.where(roi, t.data, -np.inf)
        k = int(np.argmax(vals))
        pk = np.unravel_index(k, t.shape3)
        roi_c = T.apply(A, t.voxel_to_world(np.argwhere(roi).mean(0).astype(float)))
        r = {"roi_voxels": int(roi.sum()), "peak_t": float(t.data[pk]), "roi_centroid_t1_ras": roi_c.tolist(),
             "peak_t1_ras": t1[k].tolist(), "peak_mni_approx": mni[k].tolist(), "centroid_t1_ras": None}
        if t.data[pk] > THR:
            lab, _ = ndimage.label((t.data > THR) & roi)
            m = lab == lab[pk]
            w = t.data[m]
            cv = (np.argwhere(m).astype(float) * w[:, None]).sum(0) / w.sum()
            cw = T.apply(A, t.voxel_to_world(cv))
            r.update(centroid_t1_ras=cw.tolist(), centroid_mni_approx=T.apply(N, cw).tolist(), cluster_voxels=int(m.sum()))
        out[hemi] = r
    return out


def main():
    m7 = {r["subject"]: r for r in json.loads((OUT / "m7.json").read_text())["rows"]}
    subs = sorted(p.name for p in OUT.glob("sub-*") if (p / "t1_retest_to_test.npy").exists())
    rows, excluded = [], []
    for sub in subs:
        if not m7.get(sub, {}).get("t1_registration", {}).get("loop_ok"):
            excluded.append(sub)
            continue
        s = {ses: session(sub, ses) for ses in ("test", "retest")}
        C = np.load(OUT / sub / "t1_retest_to_test.npy")
        row = {"subject": sub, "sessions": s, "by_hemisphere": {}}
        for hemi in CENTRES:
            a, b = s["test"][hemi], s["retest"][hemi]
            h = {"peak_t_test": a["peak_t"], "peak_t_retest": b["peak_t"]}
            if a["peak_t1_ras"] and b["peak_t1_ras"]:
                h["peak_distance_mm"] = float(np.linalg.norm(T.apply(C, np.array(b["peak_t1_ras"])) - np.array(a["peak_t1_ras"])))
            if a.get("roi_centroid_t1_ras") and b.get("roi_centroid_t1_ras"):
                h["roi_only_distance_mm"] = float(np.linalg.norm(T.apply(C, np.array(b["roi_centroid_t1_ras"])) - np.array(a["roi_centroid_t1_ras"])))
            if a["centroid_t1_ras"] and b["centroid_t1_ras"]:
                h["centroid_distance_mm"] = float(np.linalg.norm(T.apply(C, np.array(b["centroid_t1_ras"])) - np.array(a["centroid_t1_ras"])))
            row["by_hemisphere"][hemi] = h
        row["hemisphere"] = max(CENTRES, key=lambda hm: (s["test"][hm]["peak_t"] or 0) + (s["retest"][hm]["peak_t"] or 0))
        rows.append(row)
        h = row["by_hemisphere"][row["hemisphere"]]
        print(f"{sub}: {row['hemisphere']:5s} peak {h.get('peak_distance_mm', float('nan')):5.1f} mm (t {h['peak_t_test'] or 0:.1f} / {h['peak_t_retest'] or 0:.1f}), "
              f"centroid {h.get('centroid_distance_mm', float('nan')):5.1f} mm", flush=True)
    base = baselines(rows)
    (OUT / "m7_hand_roi.json").write_text(json.dumps({"centres_mni": {k: v.tolist() for k, v in CENTRES.items()},
                                                      "radius_mm": RADIUS, "excluded_loop_check": excluded, "rows": rows,
                                                      "baselines_post_hoc": base}, indent=2))
    if excluded:
        print("excluded, T1-to-T1 loop check failed:", ", ".join(excluded))
    d = np.array([r["by_hemisphere"][r["hemisphere"]]["peak_distance_mm"] for r in rows
                  if "peak_distance_mm" in r["by_hemisphere"][r["hemisphere"]]])
    print(f"Rule 3a (peak in hand area): n={len(d)} median {np.median(d):.1f} mm, under 10 mm {(d < 10).sum()}/{len(d)}")
    dc = np.array([r["by_hemisphere"][r["hemisphere"]]["centroid_distance_mm"] for r in rows
                   if "centroid_distance_mm" in r["by_hemisphere"][r["hemisphere"]]])
    print(f"Rule 3b (cluster centre in hand area): n={len(dc)} of {len(rows)} median "
          f"{np.median(dc) if len(dc) else float('nan'):.1f} mm, under 10 mm {(dc < 10).sum()}/{len(dc)}")

    print(f"Baseline, whole hand-area region instead of the cluster (no fMRI): median {base['roi_only']['median_mm']:.2f} mm, "
          f"under 10 mm {base['roi_only']['under_10']}/{base['roi_only']['n']}")
    cs = base["cross_subject"]
    print(f"Baseline, 3b centres in approximate MNI: same person median {cs['within_median_mm']:.1f} mm "
          f"({cs['within_under_10']}/{cs['within_n']} under 10); different people median {cs['across_median_mm']:.1f} mm "
          f"({100 * cs['across_under_10_fraction']:.0f}% of {cs['across_n']} pairs under 10); "
          f"re-pairing p = {cs['permutation_p_median']:.4f} ({cs['permutations']} derangements)")


def baselines(rows, n_perm=20000, seed=0):
    roi = np.array([r["by_hemisphere"][r["hemisphere"]]["roi_only_distance_mm"] for r in rows
                    if "roi_only_distance_mm" in r["by_hemisphere"][r["hemisphere"]]])
    cen = {}
    for r in rows:
        hm = r["hemisphere"]
        c = [r["sessions"][s][hm].get("centroid_mni_approx") for s in ("test", "retest")]
        if all(c):
            cen[r["subject"]] = [np.array(v) * np.array([-1.0 if hm == "right" else 1.0, 1, 1]) for v in c]
    subs = sorted(cen)
    D = np.array([[np.linalg.norm(cen[a][0] - cen[b][1]) for b in subs] for a in subs])
    n = len(subs)
    within = np.diag(D)
    across = D[~np.eye(n, dtype=bool)]
    rng = np.random.default_rng(seed)
    obs = np.median(within)
    hits, done = 0, 0
    while done < n_perm:
        perm = rng.permutation(n)
        if (perm == np.arange(n)).any():
            continue
        done += 1
        hits += np.median(D[np.arange(n), perm]) <= obs
    return {"roi_only": {"n": len(roi), "median_mm": float(np.median(roi)), "under_10": int((roi < 10).sum())},
            "cross_subject": {"subjects": subs, "within_n": n, "within_median_mm": float(obs),
                              "within_under_10": int((within < 10).sum()), "across_n": int(across.size),
                              "across_median_mm": float(np.median(across)),
                              "across_under_10_fraction": float((across < 10).mean()),
                              "permutations": n_perm, "permutation_p_median": float((hits + 1) / (n_perm + 1))}}


if __name__ == "__main__":
    main()
