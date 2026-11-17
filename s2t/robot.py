"""Carry a plan from T1 space into a robot frame."""
from __future__ import annotations

import numpy as np

from . import fiducials as fid
from . import transforms as T


def to_robot(plan: dict, fiducials_image: np.ndarray, fiducials_robot: np.ndarray, sigma_mm: float,
             true_image_to_robot: np.ndarray | None = None) -> dict:
    P = np.asarray(fiducials_image, float)
    Q = np.asarray(fiducials_robot, float)
    M = fid.fit_rigid(P, Q)
    target = np.array(plan["target"]["ras_mm"])
    out = {
        "n_fiducials": int(len(P)),
        "image_to_robot": M.tolist(),
        "fre_mm": fid.fre(M, P, Q),
        "predicted_tre_rms_mm": fid.predicted_tre_rms(P, target, np.sqrt(3) * sigma_mm),
        "target_robot_mm": T.apply(M, target).tolist(),
        "entry_robot_mm": T.apply(M, np.array(plan["entry"]["ras_mm"])).tolist() if plan.get("entry") else None,
    }
    if plan.get("entry"):
        d = T.apply(M, target) - T.apply(M, np.array(plan["entry"]["ras_mm"]))
        out["direction_robot"] = (d / np.linalg.norm(d)).tolist()
    if true_image_to_robot is not None:
        true_t = T.apply(np.asarray(true_image_to_robot), target)
        out["simulated_true_tre_mm"] = float(np.linalg.norm(true_t - out["target_robot_mm"]))
        out["note"] = (f"Simulated tracker. Error at the target in this draw: {out['simulated_true_tre_mm']:.2f} mm; "
                       f"FRE says little about it (Fitzpatrick 1998).")
    else:
        out["note"] = "Predicted TRE assumes isotropic localisation noise of the stated sigma."
    return out
