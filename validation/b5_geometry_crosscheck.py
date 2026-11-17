"""Bar B5 (part 1): the viewer's geometry against values recomputed from the source files.

The viewer only ever uses the matrices stored in scene.json. This script
recomputes each of them from the original inputs (the T1 and t-map NIfTI
headers read by nibabel, and the saved registration) and pushes 1,000 random
points through the viewer's own JS (viewer/geometry.js, under node) with the
SCENE's matrices, and through numpy with the independently recomputed ones:
  - T1 world -> overlay voxel (the map lookup for every pixel),
  - T1 world -> voxel of the original T1 file (the index written into plans),
  - cropped display voxel -> world (what is drawn),
  - RAS -> LPS, and trilinear sampling.
Bar: coordinates agree within 1e-6 mm or voxel, samples to float32 precision.
Display orientation and click/draw consistency are tested separately by
tests/viewer_phantom.mjs.
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
import { inv4, apply, rasToLps, trilinear } from "%s";
import { readFileSync } from "fs";
const c = JSON.parse(readFileSync(process.argv[2], "utf8"));
const sc = c.scene;
const vol = new Float32Array(c.vol);
const out = { world: [], vox: [], lps: [], chain: [], samp: [] };
const invOrig = inv4(sc.t1.affine_original);
for (const p of c.ijk) out.world.push(apply(sc.t1.affine, p));            // cropped display voxel -> world
for (const w of c.xyz) out.vox.push(apply(invOrig, w));                    // world -> original T1 file voxel
for (const w of c.xyz) out.lps.push(rasToLps(w));
for (const w of c.xyz) out.chain.push(apply(sc.overlay.world_to_voxel, w)); // world -> t-map voxel
for (const q of c.q) out.samp.push(trilinear(vol, c.shape, q[0], q[1], q[2]));
console.log(JSON.stringify(out));
"""


def main():
    import nibabel as nib
    scene_dir = ROOT / "out/scenes/sub-01_ses-test"
    scene = json.loads((scene_dir / "scene.json").read_text())
    # independent recomputation from the source files
    t1_img = nib.load(str(ROOT / "data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"))
    A_orig = t1_img.affine
    A_crop = A_orig.copy()
    A_crop[:3, 3] = A_orig[:3, :3] @ np.array(scene["t1"]["crop_offset_voxels"], float) + A_orig[:3, 3]
    A_epi = nib.load(str(ROOT / "out/sub-01/ses-test/fmri/tmap_finger.nii.gz")).affine
    M = np.load(ROOT / "out/sub-01/ses-test/fmri/epi_to_t1.npy")
    rng = np.random.default_rng(0)
    ijk = rng.uniform(0, 1, (1000, 3)) * (np.array(scene["t1"]["shape"]) - 1)
    xyz = rng.uniform(-120, 120, (1000, 3))
    small = rng.normal(size=(9, 8, 7)).astype(np.float32)
    q = rng.uniform(0, 1, (1000, 3)) * (np.array(small.shape) - 1)
    case = {"scene": scene, "ijk": ijk.tolist(), "xyz": xyz.tolist(), "shape": list(small.shape),
            "vol": small.ravel(order="C").tolist(), "q": q.tolist()}
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "case.json"
        cf.write_text(json.dumps(case))
        js = Path(td) / "check.mjs"
        js.write_text(JS % (ROOT / "s2t/viewer/geometry.js").as_uri())
        res = json.loads(subprocess.run(["node", str(js), str(cf)], capture_output=True, text=True, check=True).stdout)

    py_world = T.apply(A_crop, ijk)
    py_vox = T.apply(np.linalg.inv(A_orig), xyz)
    py_lps = xyz * np.array([-1, -1, 1])
    py_chain = T.apply(np.linalg.inv(A_epi) @ np.linalg.inv(M), xyz)
    py_samp = ndimage.map_coordinates(small, q.T, order=1)
    err = {
        "display_voxel_to_world_mm": float(np.abs(np.array(res["world"]) - py_world).max()),
        "world_to_t1_file_voxel_vox": float(np.abs(np.array(res["vox"]) - py_vox).max()),
        "ras_to_lps_mm": float(np.abs(np.array(res["lps"]) - py_lps).max()),
        "world_to_overlay_voxel_vox": float(np.abs(np.array(res["chain"]) - py_chain).max()),
        "trilinear_abs": float(np.abs(np.array(res["samp"]) - py_samp).max()),
    }
    ok = all(v <= 1e-6 for k, v in err.items() if k != "trilinear_abs") and err["trilinear_abs"] <= 1e-5
    out = {"errors": err, "scene": str(scene_dir.relative_to(ROOT)), "passed": bool(ok)}
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out/b5_geometry.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
