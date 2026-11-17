"""Local planning server: serves the viewer and one scene, and saves plans.

Binds to 127.0.0.1 by default. Imaging data stays on this machine; reach it
from another computer through an SSH port-forward, not by opening the port.

Web hardening, because a page on any other site can still send requests to
localhost:
- Host header must be localhost/127.0.0.1 on this port (blocks DNS rebinding).
- POST needs Content-Type application/json and, when the browser sends an
  Origin, the same origin (blocks cross-site writes).
- Only the viewer's files and the three scene files the viewer needs are
  served. robot_demo.json, which holds the simulated robot's true pose, and
  saved plans are not.
"""
from __future__ import annotations

import json
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import robot
from . import transforms as T

VIEWER = Path(__file__).resolve().parent / "viewer"
SCENE_FILES = {"scene.json", "t1.u8.gz", "overlay.f32.gz"}
VIEWER_FILES = {"index.html", "app.js", "geometry.js", "style.css"}
MAX_BODY = 1_000_000


class Handler(SimpleHTTPRequestHandler):
    scene_dir: Path = None
    allowed_hosts: set = set()

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "") in self.allowed_hosts

    def _fail(self, code, msg):
        body = msg.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def translate_path(self, path):
        p = path.split("?", 1)[0].split("#", 1)[0]
        if p.startswith("/scene/"):
            name = p[len("/scene/"):]
            return str(self.scene_dir / name) if name in SCENE_FILES else str(self.scene_dir / "__nope__")
        name = p.lstrip("/") or "index.html"
        return str(VIEWER / name) if name in VIEWER_FILES else str(VIEWER / "__nope__")

    def do_GET(self):
        if not self._host_ok():
            return self._fail(HTTPStatus.FORBIDDEN, "bad Host header")
        return super().do_GET()

    def do_HEAD(self):
        if not self._host_ok():
            return self._fail(HTTPStatus.FORBIDDEN, "bad Host header")
        return super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if not self._host_ok():
            return self._fail(HTTPStatus.FORBIDDEN, "bad Host header")
        origin = self.headers.get("Origin")
        if origin is not None and origin.replace("http://", "", 1) not in self.allowed_hosts:
            return self._fail(HTTPStatus.FORBIDDEN, "cross-origin request refused")
        if self.path != "/api/plan":
            return self._fail(HTTPStatus.NOT_FOUND, "not found")
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return self._fail(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "plan must be application/json")
        try:
            n = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            n = -1
        if not 0 < n <= MAX_BODY:
            return self._fail(HTTPStatus.BAD_REQUEST, "missing or oversized body")
        try:
            plan = json.loads(self.rfile.read(n))
            body = json.dumps(save_plan(self.scene_dir, plan)).encode()
        except Exception as e:  # report the reason to the page
            return self._fail(HTTPStatus.BAD_REQUEST, str(e))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def save_plan(scene_dir: Path, plan: dict) -> dict:
    """Validate a plan against the scene and save it.

    What the checks do and don't establish: the plan must name the same input
    T1 (by the sha256 recorded when the scene was built) as this scene; the
    server recomputes the target's voxel index in the ORIGINAL T1 file from its
    own copy of that file's matrix and stores that, and flags a plan whose LPS
    coordinates are not the RAS ones with x and y negated. It does not re-hash
    the T1 on disk."""
    scene = json.loads((Path(scene_dir) / "scene.json").read_text())
    if plan.get("schema") != "scan-to-target/plan/1" or not plan.get("target"):
        raise ValueError("not a plan with a target")
    if plan.get("inputs", {}).get("t1_sha256") != scene["inputs"]["t1_sha256"]:
        raise ValueError("plan names a different T1 than this scene was built from")
    A_orig = np.array(scene["t1"]["affine_original"])
    shape_orig = np.array(scene["t1"]["shape_original"])
    ras = np.array(plan["target"]["ras_mm"], float)
    vox = T.apply(np.linalg.inv(A_orig), ras)
    check = {
        "t1_file_voxel": [round(float(x), 3) for x in vox],
        "client_voxel_disagreement": (float(np.abs(vox - np.array(plan["target"]["t1_voxel"])).max())
                                      if plan["target"].get("t1_voxel") is not None else None),
        "lps_consistent": bool(np.allclose(T.apply(T.RAS_TO_LPS, ras), plan["target"]["lps_mm"], atol=1e-3)),
        "inside_t1": bool(np.all(vox >= -0.5) and np.all(vox <= shape_orig - 0.5)),
    }
    if not check["inside_t1"]:
        raise ValueError("target is outside the T1 volume")
    plan["server_check"] = check
    rd = Path(scene_dir) / "robot_demo.json"
    rb = None
    if rd.exists():
        demo = json.loads(rd.read_text())
        rb = robot.to_robot(plan, demo["fiducials_image_ras"], demo["fiducials_robot_measured"], demo["sigma_mm"],
                            np.array(demo["true_image_to_robot"]))
        plan["robot"] = rb
    plans = Path(scene_dir) / "plans"
    plans.mkdir(exist_ok=True)
    name = time.strftime("plan-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6] + ".json"
    (plans / name).write_text(json.dumps(plan, indent=2))
    return {"saved_as": f"plans/{name}", "check": check, "robot": rb}


def serve(scene_dir, host: str = "127.0.0.1", port: int = 8770):
    Handler.scene_dir = Path(scene_dir).resolve()
    Handler.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"planner: http://127.0.0.1:{port}/   scene: {Handler.scene_dir}")
    httpd.serve_forever()
