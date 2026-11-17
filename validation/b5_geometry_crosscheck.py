"""Bar B5 (part 1): the viewer's JS geometry vs the Python geometry.

1,000 random points and the real T1 and EPI matrices of sub-01 (plus the
EPI-to-T1 registration if it exists) go through voxel->world, world->voxel,
RAS->LPS and trilinear sampling in both languages.
Bar: every coordinate agrees within 1e-6 mm (or 1e-6 voxel) and sampled
values agree to float32 precision.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2t import transforms as T  # noqa: E402
from s2t.volume import Volume  # noqa: E402

JS = r"""
import { inv4, apply, matmul, rasToLps, trilinear } from "%s";
import { readFileSync } from "fs";
const c = JSON.parse(readFileSync(process.argv[2], "utf8"));
const vol = new Float32Array(c.vol);
const out = { world: [], vox: [], lps: [], chain: [], samp: [] };
const inv = inv4(c.A);
const chain = matmul(inv4(c.A_epi), inv4(c.M));
for (const p of c.ijk) out.world.push(apply(c.A, p));
for (const w of c.xyz) out.vox.push(apply(inv, w));
for (const w of c.xyz) out.lps.push(rasToLps(w));
for (const w of c.xyz) out.chain.push(apply(chain, w));
for (const q of c.q) out.samp.push(trilinear(vol, c.shape, q[0], q[1], q[2]));
console.log(JSON.stringify(out));
"""


def main():
    t1 = Volume.from_nifti(ROOT / "data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")
    epi = Volume.from_nifti(ROOT / "data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz")
    mfile = ROOT / "out/sub-01/ses-test/fmri/epi_to_t1.npy"
    M = np.load(mfile) if mfile.exists() else T.random_rigid(np.random.default_rng(5), 5, 5)
    rng = np.random.default_rng(0)
    ijk = rng.uniform(0, 1, (1000, 3)) * (np.array(t1.shape3) - 1)
    xyz = rng.uniform(-120, 120, (1000, 3))
    small = rng.normal(size=(9, 8, 7)).astype(np.float32)
    q = rng.uniform(0, 1, (1000, 3)) * (np.array(small.shape) - 1)
    case = {"A": t1.affine.tolist(), "A_epi": epi.affine.tolist(), "M": M.tolist(), "ijk": ijk.tolist(),
            "xyz": xyz.tolist(), "shape": list(small.shape), "vol": small.ravel(order="C").tolist(), "q": q.tolist()}
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "case.json"
        cf.write_text(json.dumps(case))
        js = Path(td) / "check.mjs"
        js.write_text(JS % (ROOT / "viewer/geometry.js").as_uri())
        res = json.loads(subprocess.run(["node", str(js), str(cf)], capture_output=True, text=True, check=True).stdout)

    py_world = T.apply(t1.affine, ijk)
    py_vox = T.apply(np.linalg.inv(t1.affine), xyz)
    py_lps = T.apply(T.RAS_TO_LPS, xyz)
    py_chain = T.apply(np.linalg.inv(epi.affine) @ np.linalg.inv(M), xyz)
    py_samp = ndimage.map_coordinates(small, q.T, order=1)
    err = {
        "voxel_to_world_mm": float(np.abs(np.array(res["world"]) - py_world).max()),
        "world_to_voxel_vox": float(np.abs(np.array(res["vox"]) - py_vox).max()),
        "ras_to_lps_mm": float(np.abs(np.array(res["lps"]) - py_lps).max()),
        "t1_world_to_epi_voxel_vox": float(np.abs(np.array(res["chain"]) - py_chain).max()),
        "trilinear_abs": float(np.abs(np.array(res["samp"]) - py_samp).max()),
    }
    ok = all(v <= 1e-6 for k, v in err.items() if k != "trilinear_abs") and err["trilinear_abs"] <= 1e-5
    out = {"errors": err, "used_real_registration": mfile.exists(), "passed": bool(ok)}
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out/b5_geometry.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
