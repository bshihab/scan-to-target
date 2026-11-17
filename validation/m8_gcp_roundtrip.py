"""M8: DICOMweb round trip through a Google Cloud Healthcare DICOM store.

Uploads the public dcm_qa_nih test series (no patient data), searches them,
retrieves every instance in its original transfer syntax, and checks:
  - every retrieved file is byte-identical to the uploaded one;
  - the retrieved series loads to the same voxels and matrix (B1 still holds).
Then deletes the studies it created.

Usage (after `gcloud auth login` and creating the dataset and store):
  python validation/m8_gcp_roundtrip.py PROJECT LOCATION DATASET STORE
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2t import dicom  # noqa: E402
from s2t.dicomweb import DicomWeb, gcloud_token, gcp_base, tag  # noqa: E402


def main():
    project, location, dataset, store = sys.argv[1:5]
    client = DicomWeb(gcp_base(project, location, dataset, store), token=gcloud_token())
    rows, studies = [], set()
    for sdir in sorted((ROOT / "data/dicom/dcm_qa_nih/In").glob("*/mr_*")):
        files = sorted(p for p in sdir.iterdir() if dicom.is_dicom_file(p))
        orig = {str(pydicom.dcmread(f, stop_before_pixels=True).SOPInstanceUID): f.read_bytes() for f in files}
        uid = str(pydicom.dcmread(files[0], stop_before_pixels=True).SeriesInstanceUID)
        t0 = time.time()
        for i in range(0, len(files), 50):
            client.store(files[i:i + 50])
        t_up = time.time() - t0
        hits = client.search_instances(SeriesInstanceUID=uid)
        with tempfile.TemporaryDirectory() as td:
            same = 0
            t0 = time.time()
            for h in hits:
                st, se, sop = tag(h, "0020000D"), tag(h, "0020000E"), tag(h, "00080018")
                studies.add(st)
                raw = client.retrieve_instance(st, se, sop)
                same += raw == orig.get(sop)
                (Path(td) / f"{sop}.dcm").write_bytes(raw)
            t_down = time.time() - t0
            a, b = dicom.load_series(sdir), dicom.load_series(td)
            geom = bool(np.array_equal(a.data, b.data) and np.allclose(a.affine, b.affine))
        rows.append({"series": f"{sdir.parent.name}/{sdir.name}", "files": len(files), "found": len(hits),
                     "byte_identical": same, "geometry_identical": geom,
                     "upload_s": round(t_up, 2), "download_s": round(t_down, 2)})
        print(rows[-1], flush=True)
    deleted = {s: client.delete_study(s) for s in studies}
    ok = all(r["found"] == r["files"] == r["byte_identical"] and r["geometry_identical"] for r in rows)
    out = {"store": f"{project}/{location}/{dataset}/{store}", "rows": rows, "deleted": deleted, "passed": ok}
    (ROOT / "out/m8.json").write_text(json.dumps(out, indent=2))
    print("M8 passed:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
