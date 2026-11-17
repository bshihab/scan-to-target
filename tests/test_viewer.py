import subprocess
from pathlib import Path


def test_viewer_orientation_phantom():
    r = subprocess.run(["node", str(Path(__file__).with_name("viewer_phantom.mjs"))], capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr
