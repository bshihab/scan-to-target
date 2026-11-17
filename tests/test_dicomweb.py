"""DICOMweb client against an in-memory fake server: upload a real series,
search it, retrieve every instance, check bytes and geometry survive."""
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pydicom
import pytest

from s2t import dicom
from s2t.dicomweb import DicomWeb, _multipart, parse_multipart, tag

SERIES = Path(__file__).resolve().parents[1] / "data/dicom/dcm_qa_nih/In/20180918Si/mr_0003"


class Fake(BaseHTTPRequestHandler):
    store: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/dicom+json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        assert self.headers["Authorization"] == "Bearer tok"
        body = self.rfile.read(int(self.headers["Content-Length"]))
        for part in parse_multipart(body, self.headers["Content-Type"]):
            ds = pydicom.dcmread(io.BytesIO(part), stop_before_pixels=True)
            self.store[str(ds.SOPInstanceUID)] = (ds, part)
        self._send(200, b"{}")

    def do_GET(self):
        if self.path.startswith("/instances"):
            from urllib.parse import parse_qs, urlparse
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            out = [{"0020000D": {"vr": "UI", "Value": [str(ds.StudyInstanceUID)]},
                    "0020000E": {"vr": "UI", "Value": [str(ds.SeriesInstanceUID)]},
                    "00080018": {"vr": "UI", "Value": [uid]}}
                   for uid, (ds, _) in self.store.items()
                   if str(ds.SeriesInstanceUID) == q.get("SeriesInstanceUID", str(ds.SeriesInstanceUID))]
            return self._send(200 if out else 204, json.dumps(out).encode() if out else b"")
        uid = self.path.rstrip("/").split("/")[-1]
        _, raw = self.store[uid]
        b = "fakeboundary"
        self._send(200, _multipart([raw], b), f'multipart/related; type="application/dicom"; boundary={b}')


@pytest.mark.skipif(not SERIES.exists(), reason="run scripts/fetch_data.sh")
def test_round_trip(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = DicomWeb(f"http://127.0.0.1:{srv.server_port}", token=lambda: "tok")
        files = sorted(p for p in SERIES.iterdir() if dicom.is_dicom_file(p))
        client.store(files)
        uid = str(pydicom.dcmread(files[0], stop_before_pixels=True).SeriesInstanceUID)
        hits = client.search_instances(SeriesInstanceUID=uid)
        assert len(hits) == len(files)
        out = tmp_path / "back"
        out.mkdir()
        originals = {pydicom.dcmread(f, stop_before_pixels=True).SOPInstanceUID: f.read_bytes() for f in files}
        for h in hits:
            raw = client.retrieve_instance(tag(h, "0020000D"), tag(h, "0020000E"), tag(h, "00080018"))
            assert raw == originals[tag(h, "00080018")]
            (out / f"{tag(h, '00080018')}.dcm").write_bytes(raw)
        a, b = dicom.load_series(SERIES), dicom.load_series(out)
        assert np.array_equal(a.data, b.data) and np.allclose(a.affine, b.affine)
    finally:
        srv.shutdown()


def test_multipart_round_trip():
    parts = [b"\x00\x01--not-a-boundary\r\n\r\n", b"second"]
    body = _multipart(parts, "abc123")
    assert parse_multipart(body, 'multipart/related; type="application/dicom"; boundary=abc123') == parts
