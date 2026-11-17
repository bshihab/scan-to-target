import json

import numpy as np
import pytest

from s2t import serve
from s2t import transforms as T


def make_scene(tmp_path, robot=True):
    A = np.array([[-1.0, 0, 0, 90], [0, 1.2, 0, -100], [0, 0, 1, -60], [0, 0, 0, 1]])
    scene = {"t1": {"affine": A.tolist(), "shape": [180, 200, 160], "affine_original": A.tolist(),
                    "shape_original": [180, 200, 160]}, "inputs": {"t1_sha256": "abc"}}
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
    assert res["check"]["client_voxel_disagreement"] < 1e-9
    assert res["check"]["lps_consistent"]
    # noise-free fiducials: the robot target is exact
    assert res["robot"]["simulated_true_tre_mm"] < 1e-9
    saved = json.loads((tmp_path / res["saved_as"]).read_text())
    assert saved["server_check"]["inside_t1"]


def test_refuses_plan_from_another_scan(tmp_path):
    A = make_scene(tmp_path, robot=False)
    with pytest.raises(ValueError, match="different T1 than this scene"):
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


def test_server_voxel_is_in_the_original_file_not_the_crop(tmp_path):
    """The display volume is cropped; the plan's voxel index must refer to the
    original NIfTI. A client reporting a cropped index is flagged."""
    A = make_scene(tmp_path, robot=False)
    scene = json.loads((tmp_path / "scene.json").read_text())
    crop = A.copy()
    crop[:3, 3] = T.apply(A, [24, 0, 23])  # display volume starts 24, 0, 23 voxels in
    scene["t1"]["affine"] = crop.tolist()
    (tmp_path / "scene.json").write_text(json.dumps(scene))
    p = plan_for(A, [-38, -20, 55])
    p["target"]["t1_voxel"] = T.apply(np.linalg.inv(crop), [-38, -20, 55]).tolist()  # the old bug
    res = serve.save_plan(tmp_path, p)
    assert np.allclose(res["check"]["t1_file_voxel"], T.apply(np.linalg.inv(A), [-38, -20, 55]), atol=1e-3)
    assert res["check"]["client_voxel_disagreement"] > 20


def test_http_hardening(tmp_path):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    A = make_scene(tmp_path)
    (tmp_path / "t1.u8.gz").write_bytes(b"x")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    port = srv.server_port
    serve.Handler.scene_dir = tmp_path
    serve.Handler.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def req(method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        return r.status, r.read()

    try:
        assert req("GET", "/scene/scene.json")[0] == 200
        assert req("GET", "/")[0] == 200
        # DNS rebinding: the browser sends the attacker's host name
        assert req("GET", "/scene/t1.u8.gz", headers={"Host": "evil.example:80"})[0] == 403
        # files that are not served: the robot's true pose, saved plans, anything else
        for p in ("/scene/robot_demo.json", "/scene/plans/x.json", "/scene/../scene.json", "/../pyproject.toml"):
            assert req("GET", p)[0] == 404, p
        good = json.dumps(plan_for(A, [-38, -20, 55]))
        assert req("POST", "/api/plan", good, {"Content-Type": "application/json"})[0] == 200
        assert req("POST", "/api/plan", good, {"Content-Type": "application/json", "Origin": "http://evil.example"})[0] == 403
        assert req("POST", "/api/plan", good, {"Content-Type": "text/plain"})[0] == 415
        assert req("POST", "/api/plan", b"", {"Content-Type": "application/json", "Content-Length": "-1"})[0] == 400
    finally:
        srv.shutdown()
