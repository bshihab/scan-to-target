"""Bar B6: simulated target error vs Fitzpatrick's prediction.

Four fiducial layouts on a head-sized scene (mm, head centre at the origin),
target at a hand-knob-like point 40 mm left, 20 mm back, 55 mm up. Robot-space
measurements get isotropic Gaussian noise, sigma 0.25 mm per axis. 20,000
trials per layout. The true image-to-robot transform is a large arbitrary
rigid motion (the robot base is nowhere near the image origin).
Bar: Monte Carlo RMS TRE within 10% of the prediction for every layout, and
the same ranking of layouts.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2t import fiducials as fid  # noqa: E402
from s2t import transforms as T  # noqa: E402

TARGET = np.array([-40.0, -20.0, 55.0])
SIGMA = 0.25
TRIALS = 20000


def layouts(rng):
    sph = lambda th, ph, r=90: [r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)]
    spread = np.array([sph(np.pi / 2, 0), sph(np.pi / 2, np.pi / 2), sph(np.pi / 2, np.pi),
                       sph(np.pi / 2, 3 * np.pi / 2), sph(0.2, 0.3), sph(np.pi / 3, np.pi / 4)])
    clustered = np.array([[0, 90, 20], [15, 88, 22], [-12, 89, 30], [3, 86, 38]], float)  # 20-30 mm patch on the forehead
    line = np.linspace(-60, 60, 4)
    collinearish = np.stack([line, 85 + rng.normal(0, 2, 4), 10 + rng.normal(0, 2, 4)], 1)
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    planar = np.stack([80 * np.cos(ang), 80 * np.sin(ang), np.full(5, -10.0)], 1)
    return {"well_spread": spread, "clustered": clustered, "collinear_ish": collinearish, "planar_ring": planar}


def main():
    rng = np.random.default_rng(3)
    M_true = T.random_rigid(rng, 180.0, 1.0)
    M_true[:3, 3] = [512.0, -230.0, 840.0]
    rows = []
    for name, P in layouts(rng).items():
        pred = fid.predicted_tre_rms(P, TARGET, np.sqrt(3) * SIGMA)
        sim = fid.simulate_tre(P, TARGET, SIGMA, M_true, TRIALS, rng)
        rms = float(np.sqrt(np.mean(sim ** 2)))
        rows.append({"layout": name, "n_fiducials": len(P), "predicted_rms_mm": pred, "simulated_rms_mm": rms,
                     "simulated_p95_mm": float(np.percentile(sim, 95)), "rel_diff": rms / pred - 1,
                     "within_10pct": bool(abs(rms / pred - 1) <= 0.10)})
    rank_pred = [r["layout"] for r in sorted(rows, key=lambda r: r["predicted_rms_mm"])]
    rank_sim = [r["layout"] for r in sorted(rows, key=lambda r: r["simulated_rms_mm"])]
    out = {"sigma_mm": SIGMA, "trials": TRIALS, "target": TARGET.tolist(), "rows": rows,
           "rank_predicted": rank_pred, "rank_simulated": rank_sim,
           "passed": bool(all(r["within_10pct"] for r in rows) and rank_pred == rank_sim)}
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out/b6.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['layout']:14s} N={r['n_fiducials']} predicted {r['predicted_rms_mm']:.3f} mm  simulated {r['simulated_rms_mm']:.3f} mm "
              f"({100 * r['rel_diff']:+.1f}%)  p95 {r['simulated_p95_mm']:.3f} mm")
    print("ranking predicted:", rank_pred)
    print("ranking simulated:", rank_sim)
    print("B6 passed:", out["passed"])
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
