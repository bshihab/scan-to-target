"""Minimal DICOMweb client (PS3.18): STOW-RS upload, QIDO-RS search, WADO-RS retrieve.

Works against any DICOMweb server. For Google Cloud Healthcare the base URL is
  https://healthcare.googleapis.com/v1/projects/P/locations/L/datasets/D/dicomStores/S/dicomWeb
and the token comes from `gcloud auth print-access-token` (gcloud_token()).
Standard library only.
"""
from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def gcloud_token() -> str:
    return subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True).stdout.strip()


def gcp_base(project: str, location: str, dataset: str, store: str) -> str:
    return (f"https://healthcare.googleapis.com/v1/projects/{project}/locations/{location}"
            f"/datasets/{dataset}/dicomStores/{store}/dicomWeb")


def _multipart(parts: list[bytes], boundary: str) -> bytes:
    out = []
    for p in parts:
        out.append(f"--{boundary}\r\nContent-Type: application/dicom\r\n\r\n".encode())
        out.append(p)
        out.append(b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out)


def parse_multipart(body: bytes, content_type: str) -> list[bytes]:
    """Split a multipart/related body into its part payloads."""
    params = dict(p.strip().split("=", 1) for p in content_type.split(";")[1:] if "=" in p)
    boundary = params.get("boundary", "").strip('"')
    if not boundary:
        raise ValueError("multipart response without a boundary")
    delim = b"--" + boundary.encode()
    parts = []
    for chunk in body.split(delim)[1:]:
        if chunk.startswith(b"--"):
            break
        head, _, payload = chunk.partition(b"\r\n\r\n")
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        parts.append(payload)
    return parts


class DicomWeb:
    def __init__(self, base_url: str, token=None, timeout: float = 60.0):
        self.base = base_url.rstrip("/")
        self.token = token  # str, callable returning str, or None
        self.timeout = timeout

    def _req(self, method, path, body=None, headers=None):
        h = dict(headers or {})
        tok = self.token() if callable(self.token) else self.token
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        req = urllib.request.Request(self.base + path, data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def store(self, files) -> dict:
        """STOW-RS: upload DICOM files in one multipart request."""
        boundary = uuid.uuid4().hex
        body = _multipart([Path(f).read_bytes() for f in files], boundary)
        status, _, resp = self._req("POST", "/studies", body, {
            "Content-Type": f'multipart/related; type="application/dicom"; boundary={boundary}',
            "Accept": "application/dicom+json"})
        if status not in (200, 202):
            raise RuntimeError(f"STOW-RS failed: HTTP {status}: {resp[:300]!r}")
        return json.loads(resp) if resp else {}

    def search_instances(self, **filters) -> list[dict]:
        """QIDO-RS: e.g. search_instances(SeriesInstanceUID=uid)."""
        q = "?" + urllib.parse.urlencode(filters) if filters else ""
        status, _, resp = self._req("GET", "/instances" + q, headers={"Accept": "application/dicom+json"})
        if status == 204:
            return []
        if status != 200:
            raise RuntimeError(f"QIDO-RS failed: HTTP {status}: {resp[:300]!r}")
        return json.loads(resp)

    def retrieve_instance(self, study: str, series: str, instance: str) -> bytes:
        """WADO-RS: the stored instance, in its original transfer syntax."""
        status, headers, resp = self._req(
            "GET", f"/studies/{study}/series/{series}/instances/{instance}",
            headers={"Accept": 'multipart/related; type="application/dicom"; transfer-syntax=*'})
        if status != 200:
            raise RuntimeError(f"WADO-RS failed: HTTP {status}: {resp[:300]!r}")
        ctype = {k.lower(): v for k, v in headers.items()}.get("content-type", "")
        parts = parse_multipart(resp, ctype)
        if len(parts) != 1:
            raise RuntimeError(f"expected one instance, got {len(parts)} parts")
        return parts[0]

    def delete_study(self, study: str) -> int:
        status, _, _ = self._req("DELETE", f"/studies/{study}")
        return status


def tag(ds_json: dict, keyword_hex: str):
    v = ds_json.get(keyword_hex, {}).get("Value")
    return v[0] if v else None
