"""Local planning server: serves the viewer and one scene, and saves plans.

Binds to 127.0.0.1 by default. Imaging data stays on this machine; reach it
from another computer through an SSH port-forward, not by opening the port.
"""
from __future__ import annotations

import json
import time
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import robot
from . import transforms as T

VIEWER = Path(__file__).resolve().parents[1] / "viewer"


class Handler(SimpleHTTPRequestHandler):
    scene_dir: Path = None

    def translate_path(self, path):
        p = path.split("?", 1)[0].split("#", 1)[0]
        if p.startswith("/scene/"):
            target = (self.scene_dir / p[len("/scene/"):]).resolve()
            root = self.scene_dir.resolve()
        else:
            target = (VIEWER / (p.lstrip("/") or "index.html")).resolve()
            root = VIEWER.resolve()
        if root not in target.parents and target != root:
            return str(root / "__nope__")
        return str(target)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if self.path != "/api/plan":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 1_000_000:
                raise ValueError("plan too large")
            plan = json.loads(self.rfile.read(n))
            result = save_plan(self.scene_dir, plan)
            body = json.dumps(result).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # report the reason to the page
            body = str(e).encode()
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def save_plan(scene_dir: Path, plan: dict) -> dict:
    scene = json.loads((scene_dir / "scene.json").read_text())
    if plan.get("schema") != "scan-to-target/plan/1" or not plan.get("target"):
        raise ValueError("not a plan with a target")
    if plan.get("inputs", {}).get("t1_sha256") != scene["inputs"]["t1_sha256"]:
        raise ValueError("plan was made on a different T1 than this scene")
    A = np.array(scene["t1"]["affine"])
    ras = np.array(plan["target"]["ras_mm"], float)
    vox = T.apply(np.linalg.inv(A), ras)
    check = {
        "target_voxel_error": float(np.abs(vox - np.array(plan["target"]["t1_voxel"])).max()),
        "lps_consistent": bool(np.allclose(T.apply(T.RAS_TO_LPS, ras), plan["target"]["lps_mm"], atol=1e-3)),
        "inside_t1": bool(np.all(vox >= -0.5) and np.all(vox <= np.array(scene["t1"]["shape"]) - 0.5)),
    }
    if not check["inside_t1"]:
        raise ValueError("target is outside the T1 volume")
    plan["server_check"] = check
    rd = scene_dir / "robot_demo.json"
    rb = None
    if rd.exists():
        demo = json.loads(rd.read_text())
        rb = robot.to_robot(plan, demo["fiducials_image_ras"], demo["fiducials_robot_measured"], demo["sigma_mm"],
                            np.array(demo["true_image_to_robot"]))
        plan["robot"] = rb
    plans = scene_dir / "plans"
    plans.mkdir(exist_ok=True)
    name = time.strftime("plan-%Y%m%d-%H%M%S.json")
    (plans / name).write_text(json.dumps(plan, indent=2))
    return {"saved_as": f"plans/{name}", "check": check, "robot": rb}


def serve(scene_dir, host: str = "127.0.0.1", port: int = 8770):
    Handler.scene_dir = Path(scene_dir).resolve()
    httpd = ThreadingHTTPServer((host, port), partial(Handler))
    print(f"planner: http://{host}:{port}/   scene: {Handler.scene_dir}")
    httpd.serve_forever()
