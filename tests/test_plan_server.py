import json

import numpy as np
import pytest

from s2t import serve
from s2t import transforms as T


def make_scene(tmp_path, robot=True):
    A = np.array([[-1.0, 0, 0, 90], [0, 1.2, 0, -100], [0, 0, 1, -60], [0, 0, 0, 1]])
    scene = {"t1": {"affine": A.tolist(), "shape": [180, 200, 160]}, "inputs": {"t1_sha256": "abc"}}
    (tmp_path / "scene.json").write_text(json.dumps(scene))
    if robot:
        rng = np.random.default_rng(0)
        fids = rng.normal(0, 70, (6, 3))
        M = T.random_rigid(rng, 90, 300)
        (tmp_path / "robot_demo.json").write_text(json.dumps({
            "sigma_mm": 0.0, "fiducials_image_ras": fids.tolist(),
            "fiducials_robot_measured": T.apply(M, fids).tolist(), "true_image_to_robot": M.tolist()}))
    return A


def plan_for(A, ras, sha="abc"):
    ras = np.array(ras, float)
    return {"schema": "scan-to-target/plan/1", "inputs": {"t1_sha256": sha},
            "target": {"ras_mm": ras.tolist(), "lps_mm": T.apply(T.RAS_TO_LPS, ras).tolist(),
                       "t1_voxel": T.apply(np.linalg.inv(A), ras).tolist()},
            "entry": {"ras_mm": (ras + [0, 0, 60]).tolist()}}


def test_plan_round_trip_and_robot(tmp_path):
    A = make_scene(tmp_path)
    res = serve.save_plan(tmp_path, plan_for(A, [-38, -20, 55]))
    assert res["check"]["target_voxel_error"] < 1e-9
    assert res["check"]["lps_consistent"]
    # noise-free fiducials: the robot target is exact
    assert res["robot"]["simulated_true_tre_mm"] < 1e-9
    saved = json.loads((tmp_path / res["saved_as"]).read_text())
    assert saved["server_check"]["inside_t1"]


def test_refuses_plan_from_another_scan(tmp_path):
    A = make_scene(tmp_path, robot=False)
    with pytest.raises(ValueError, match="different T1"):
        serve.save_plan(tmp_path, plan_for(A, [0, 0, 0], sha="other"))


def test_refuses_target_outside_volume(tmp_path):
    A = make_scene(tmp_path, robot=False)
    with pytest.raises(ValueError, match="outside"):
        serve.save_plan(tmp_path, plan_for(A, [0, 0, 400]))


def test_detects_lps_sign_error(tmp_path):
    A = make_scene(tmp_path, robot=False)
    p = plan_for(A, [-38, -20, 55])
    p["target"]["lps_mm"] = p["target"]["ras_mm"]  # the classic bug: RAS written as LPS
    res = serve.save_plan(tmp_path, p)
    assert res["check"]["lps_consistent"] is False
