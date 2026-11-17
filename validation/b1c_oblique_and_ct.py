"""Bar B1c: real oblique MRI and a gantry-tilted CT.

(1) Every Siemens series in neurolabusc/dcm_qa In/Orientation is compared with
    its dcm2niix reference exactly as in B1.
(2) The GE head CT in neurolabusc/dcm_qa_ct must be refused with a gantry-tilt
    or slice-spacing reason.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
import b1_dicom_vs_dcm2niix as b1  # noqa: E402
from s2t import dicom  # noqa: E402

QA = ROOT / "data/dicom/dcm_qa"
CT = ROOT / "data/dicom/dcm_qa_ct"


def ref_by_number(series_dir: Path, num: int) -> Path:
    hits = [h for h in sorted((QA / "Ref").glob(f"*_{num}.nii")) if h.stem.split("_")[-1] == str(num)
            and h.name.split("_")[0] in ("ax", "cor", "sag", "fMRI")]
    if len(hits) != 1:
        raise RuntimeError(f"no unique reference for series {num}: {hits}")
    return hits[0]


def main():
    b1.QA = QA
    b1.ref_for = ref_by_number
    rows = []
    for d in sorted((QA / "In/Orientation").glob("*/*")):
        if d.is_dir():
            try:
                rows.append(b1.compare(d))
            except Exception as e:  # anything other than a clean refusal is a failure, with the reason
                rows.append({"series": f"{d.parent.name}/{d.name}", "status": "error", "reason": f"{type(e).__name__}: {e}", "passed": False})
    try:
        vol = dicom.load_series(CT / "In/GE")
        ct = {"status": "loaded (should have been refused)", "shape": list(vol.data.shape), "passed": False}
    except dicom.DicomGeometryError as e:
        msg = str(e)
        ct = {"status": "refused", "reason": msg, "passed": ("tilt" in msg or "spacing" in msg or "stacked" in msg)}
    out = {"orientation": rows, "ct_ge": ct,
           "passed": bool(all(r["passed"] for r in rows) and ct["passed"])}
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out/b1c.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        if r["status"] == "compared":
            print(f"{r['series']:22s} {'PASS' if r['passed'] else 'FAIL'}  ref={r['reference']:24s} shape={r['shape_ours']} "
                  f"corners={r['corner_max_err_mm']:.1e} mm values_identical={r['values_identical']} mosaic={r['mosaic']}")
        else:
            print(f"{r['series']:22s} {r['status'].upper()}  {r.get('reason')}")
    print("GE CT:", ct)
    print(f"B1c: {sum(r['passed'] for r in rows)}/{len(rows)} MRI series pass; CT {'refused correctly' if ct['passed'] else 'FAILED'}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
