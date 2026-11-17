"""Bar B1: our DICOM reader vs dcm2niix's reference NIfTIs on dcm_qa_nih.

For each series: load with s2t.dicom, load the dcm2niix reference, and compare
in world space. The map from reference voxels to our voxels must be an exact
axis permutation/flip; under it, every voxel value must be identical and the
two voxel-to-RAS matrices must agree within 0.01 mm at the 8 grid corners.

Usage: python validation/b1_dicom_vs_dcm2niix.py  (writes out/b1.json)
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2t import dicom  # noqa: E402
from s2t import transforms as T  # noqa: E402

QA = ROOT / "data/dicom/dcm_qa_nih"
CORNER_TOL_MM = 0.01


def ref_for(series_dir: Path, series_number: int) -> Path:
    vendor = series_dir.parent.name
    prefix = "DCM2NIIX_regression_test_" if vendor.endswith("GE") else "EPI_PE="
    hits = sorted((QA / "Ref").glob(f"{prefix}*{series_number}.nii"))
    hits = [h for h in hits if h.stem.split("_")[-1] == str(series_number)]
    if len(hits) != 1:
        raise RuntimeError(f"no unique reference for {series_dir} (series {series_number}): {hits}")
    return hits[0]


def compare(series_dir: Path) -> dict:
    row = {"series": f"{series_dir.parent.name}/{series_dir.name}"}
    try:
        vol = dicom.load_series(series_dir)
    except dicom.DicomGeometryError as e:
        row.update(status="refused", reason=str(e), passed=False)
        return row
    ref_path = ref_for(series_dir, int(vol.meta["SeriesNumber"]))
    ref = nib.load(str(ref_path))
    rdata = ref.get_fdata(dtype=np.float32)
    row["reference"] = ref_path.name
    row["shape_ours"] = list(vol.data.shape)
    row["shape_ref"] = list(rdata.shape)
    row["mosaic"] = vol.meta["mosaic"]

    # ref voxel -> world -> our voxel. The rotation part must be an axis
    # permutation with flips. The translation is NOT taken from M: it is fixed
    # by the grids (a flipped axis starts at the far end), so a position error
    # in either matrix shows up as a corner error instead of being absorbed.
    M = np.linalg.inv(vol.affine) @ ref.affine
    P = np.round(M[:3, :3])
    perm_ok = (np.allclose(np.abs(P).sum(0), 1) and np.allclose(np.abs(P).sum(1), 1)
               and np.allclose(M[:3, :3], P, atol=1e-4))
    Mi = np.eye(4)
    Mi[:3, :3] = P
    for b in range(3):
        a = int(np.argmax(np.abs(P[b])))
        Mi[b, 3] = 0 if P[b, a] > 0 else vol.shape3[b] - 1
    row["max_index_mismatch_voxels"] = float(np.abs(M - Mi).max())

    # corner agreement in mm: where each matrix puts the reference grid corners
    I, J, K = np.array(rdata.shape[:3]) - 1
    corners = np.array([[i, j, k] for i in (0, I) for j in (0, J) for k in (0, K)], float)
    ours_world = T.apply(vol.affine @ Mi, corners)
    ref_world = T.apply(ref.affine, corners)
    row["corner_max_err_mm"] = float(np.linalg.norm(ours_world - ref_world, axis=1).max())

    values_equal = False
    if perm_ok and rdata.shape[3:] == vol.data.shape[3:]:
        idx = np.indices(rdata.shape[:3]).reshape(3, -1).T
        ours_idx = np.rint(T.apply(Mi, idx)).astype(int)
        inside = np.all((ours_idx >= 0) & (ours_idx < np.array(vol.shape3)), axis=1)
        if inside.all() and np.prod(vol.shape3) == np.prod(rdata.shape[:3]):
            ours_vals = vol.data[ours_idx[:, 0], ours_idx[:, 1], ours_idx[:, 2]]
            ref_vals = rdata.reshape(-1, *rdata.shape[3:])
            diff = np.abs(ours_vals - ref_vals)
            row["max_value_diff"] = float(diff.max())
            values_equal = bool(diff.max() == 0)
    row["values_identical"] = values_equal
    row["passed"] = bool(values_equal and row["corner_max_err_mm"] <= CORNER_TOL_MM)
    row["status"] = "compared"
    return row


def main():
    rows = [compare(d) for d in sorted((QA / "In").glob("*/mr_*"))]
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    (out / "b1.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        if r["status"] == "refused":
            print(f"{r['series']:22s} REFUSED  {r['reason']}")
        else:
            print(f"{r['series']:22s} {'PASS' if r['passed'] else 'FAIL'}  ref={r['reference']:34s} "
                  f"shape={r['shape_ours']} corners={r['corner_max_err_mm']:.2e} mm  "
                  f"values_identical={r['values_identical']} mosaic={r['mosaic']}")
    n = sum(r["passed"] for r in rows)
    print(f"B1: {n}/{len(rows)} series pass")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
