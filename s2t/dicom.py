"""DICOM series -> Volume.

pydicom parses the tags and decodes pixels. Everything geometric is done here:
grouping files into a series, splitting time points, ordering slices, building
the voxel-to-patient matrix, unpacking Siemens mosaics, and refusing series
whose geometry can't be represented by one 4x4 matrix.

DICOM geometry in one paragraph. ImageOrientationPatient (IOP) holds two unit
vectors in patient LPS: r, the direction of increasing column index (along a
row), and c, the direction of increasing row index (down a column).
PixelSpacing is [spacing between rows, spacing between columns], so moving one
column moves PixelSpacing[1] mm along r and one row moves PixelSpacing[0] mm
along c. ImagePositionPatient (IPP) is the LPS centre of the first transmitted
pixel. The slice normal is n = r x c. The array here is indexed [i, j, k] =
[column, row, slice], so the voxel-to-LPS matrix has columns r*dc, c*dr, the
step between slices, and IPP of the first slice.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from . import transforms as T
from .volume import Volume

# Only these header fields are copied into Volume.meta. Patient name, ID,
# birth date, accession number and so on never leave this module.
# SeriesDescription is free text typed at the scanner and sometimes holds names
# or record numbers, so it is left out too.
SAFE_META = ("Modality", "Manufacturer", "SeriesNumber",
             "MagneticFieldStrength", "RepetitionTime", "EchoTime", "SliceThickness",
             "SpacingBetweenSlices", "SOPClassUID")

POSITION_TOL_MM = 0.05     # IPPs must lie on one line through the stack within this
SPACING_TOL_MM = 0.05      # neighbouring slice gaps must agree within this
ORIENT_TOL = 1e-4          # IOP components must agree within this across files
TILT_TOL_DEG = 0.05        # max angle between the stack direction and r x c


class DicomGeometryError(ValueError):
    """The series can't be represented as a regular grid with one matrix."""


@dataclass
class _File:
    path: Path
    ds: Dataset  # header only


def is_dicom_file(path: Path) -> bool:
    """True if the file has the 128-byte preamble followed by 'DICM'."""
    try:
        with open(path, "rb") as f:
            head = f.read(132)
        return len(head) == 132 and head[128:] == b"DICM"
    except OSError:
        return False


def find_series(folder) -> dict[str, list[Path]]:
    """Walk a folder and group DICOM files by SeriesInstanceUID. Non-DICOM files
    (READMEs, .DS_Store, thumbnails) are skipped."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and is_dicom_file(p):
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            groups[str(ds.SeriesInstanceUID)].append(p)
    return dict(groups)


def _kind(ds: Dataset) -> tuple:
    """What a file is, apart from where and when: echo number and image type.
    Two echoes, or magnitude and phase, can share one SeriesInstanceUID; they
    must not be mistaken for time points."""
    echo = ds.get("EchoNumbers")
    # Only the first three ImageType values are defined across vendors (value 3
    # tells magnitude M from phase P on Siemens MR, AXIAL from LOCALIZER on CT);
    # later values are vendor-specific (GE CT adds "ADD" to some slices).
    it = tuple(str(x).upper() for x in (ds.get("ImageType") or [])[:3])
    return (int(echo) if echo not in (None, "") else None, it)


def load_series(source, series_uid: str | None = None, echo: int | None = None,
                image_type: str | None = None) -> Volume:
    """Load one series from a folder or a list of files.

    Returns a Volume whose matrix maps voxels to scanner RAS (the NIfTI
    convention). The LPS matrix is kept in meta['affine_lps'].
    Raises DicomGeometryError with the reason when the series can't be loaded
    as a regular grid.
    """
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        groups = find_series(source)
        if not groups:
            raise DicomGeometryError(f"no DICOM files in {source}")
        if series_uid is None:
            if len(groups) > 1:
                raise DicomGeometryError(f"{len(groups)} series in folder; pass series_uid (one of {list(groups)})")
            series_uid = next(iter(groups))
        paths = groups[series_uid]
    else:
        paths = [Path(p) for p in source]

    files = [_File(p, pydicom.dcmread(p, stop_before_pixels=True)) for p in paths]
    uids = {str(f.ds.SeriesInstanceUID) for f in files}
    if len(uids) != 1:
        raise DicomGeometryError(f"files belong to {len(uids)} different series")
    if echo is not None:
        files = [f for f in files if _kind(f.ds)[0] == echo]
    if image_type is not None:
        files = [f for f in files if image_type.upper() in _kind(f.ds)[1]]
    if not files:
        raise DicomGeometryError("no files match the requested echo / image type")
    kinds = sorted({_kind(f.ds) for f in files}, key=str)
    if len(kinds) > 1:
        desc = "; ".join(f"echo {e}, {'/'.join(t) or 'no ImageType'}" for e, t in kinds)
        raise DicomGeometryError(f"series mixes {len(kinds)} kinds of image ({desc}): choose one with echo= or image_type=")

    for f in files:
        for tag in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "Rows", "Columns"):
            if tag not in f.ds:
                raise DicomGeometryError(f"{f.path.name}: missing {tag}")

    _check_same(files, "Rows")
    _check_same(files, "Columns")
    iop = np.array([[float(x) for x in f.ds.ImageOrientationPatient] for f in files])
    if np.abs(iop - iop[0]).max() > ORIENT_TOL:
        raise DicomGeometryError("mixed slice orientations in one series")
    ps = np.array([[float(x) for x in f.ds.PixelSpacing] for f in files])
    if np.abs(ps - ps[0]).max() > 1e-4:
        raise DicomGeometryError("pixel spacing changes within the series")

    r = iop[0, :3]
    c = iop[0, 3:]
    if abs(np.linalg.norm(r) - 1) > 1e-3 or abs(np.linalg.norm(c) - 1) > 1e-3 or abs(r @ c) > 1e-3:
        raise DicomGeometryError("ImageOrientationPatient vectors are not orthonormal")
    r = r / np.linalg.norm(r)
    c = c / np.linalg.norm(c)
    dr, dc = ps[0]  # spacing between rows (along c), between columns (along r)

    if _is_mosaic(files[0].ds):
        data, affine_lps, extra = _load_mosaic(files, r, c, dr, dc)
    else:
        data, affine_lps, extra = _load_stack(files, r, c, dr, dc)

    ds0 = files[0].ds
    meta = {k: _plain(ds0.get(k)) for k in SAFE_META if k in ds0}
    meta.update(extra)
    meta.update({
        "series_instance_uid": str(ds0.SeriesInstanceUID),
        "n_files": len(files),
        "affine_lps": affine_lps.tolist(),
        "frame": "scanner RAS (from DICOM LPS)",
    })
    affine_ras = T.LPS_TO_RAS @ affine_lps
    return Volume(data, affine_ras, meta)


# ---------------------------------------------------------------------------
# Regular slice stacks (one slice per file, possibly several time points)

def _load_stack(files, r, c, dr, dc):
    n = np.cross(r, c)
    ipp = np.array([[float(x) for x in f.ds.ImagePositionPatient] for f in files])
    dist = ipp @ n

    # Group files that share a slice position (within tolerance). Each group
    # holds the same slice at different time points.
    order = np.argsort(dist, kind="stable")
    groups: list[list[int]] = []
    for idx in order:
        if groups and abs(dist[idx] - dist[groups[-1][0]]) < POSITION_TOL_MM:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])

    counts = {len(g) for g in groups}
    if len(counts) != 1:
        raise DicomGeometryError(
            f"slice positions have different numbers of images ({sorted(counts)}): missing slices or a broken time series")
    n_t = counts.pop()

    # Within each position, order the time points by InstanceNumber, then
    # AcquisitionNumber, then file name. GE gives every file of an fMRI run the
    # same acquisition time, so the time stamp alone can't be trusted.
    def tkey(i):
        ds = files[i].ds
        return (_int(ds.get("InstanceNumber")), _int(ds.get("AcquisitionNumber")), files[i].path.name)
    groups = [sorted(g, key=tkey) for g in groups]

    # Every slice position must be the same point on one line: in-plane offsets
    # between slices mean a gantry tilt or a sheared stack.
    first = ipp[groups[0][0]]
    for g in groups:
        for i in g:
            off = ipp[i] - first
            inplane = off - (off @ n) * n
            if np.linalg.norm(inplane) > POSITION_TOL_MM:
                if len(groups) > 1 and i == g[0]:
                    tilt = math.degrees(math.atan2(np.linalg.norm(inplane), abs(off @ n)))
                    raise DicomGeometryError(
                        f"slices are not stacked along the slice normal (in-plane offset {np.linalg.norm(inplane):.2f} mm, "
                        f"{tilt:.2f} deg): gantry tilt or sheared stack")
                raise DicomGeometryError("a slice position changes between time points")

    nz = len(groups)
    if nz > 1:
        z = np.array([dist[g[0]] for g in groups])
        gaps = np.diff(z)
        step = float(np.median(gaps))
        if np.abs(gaps - step).max() > SPACING_TOL_MM:
            raise DicomGeometryError(
                f"uneven slice spacing: gaps from {gaps.min():.3f} to {gaps.max():.3f} mm")
        last = ipp[groups[-1][0]]
        span = last - first
        angle = math.degrees(math.acos(min(1.0, abs(span @ n) / np.linalg.norm(span))))
        if angle > TILT_TOL_DEG:
            raise DicomGeometryError(f"stack direction is {angle:.2f} deg off the slice normal")
        slice_vec = span / (nz - 1)
    else:
        ds = files[0].ds
        th = float(ds.get("SpacingBetweenSlices") or ds.get("SliceThickness") or 1.0)
        slice_vec = n * th
        step = th

    rows, cols = int(files[0].ds.Rows), int(files[0].ds.Columns)
    data = np.empty((cols, rows, nz, n_t), dtype=np.float32)
    for k, g in enumerate(groups):
        for t, i in enumerate(g):
            data[:, :, k, t] = _pixels(files[i].path).T  # [row, col] -> [col, row]

    A = np.eye(4)
    A[:3, 0] = r * dc
    A[:3, 1] = c * dr
    A[:3, 2] = slice_vec
    A[:3, 3] = first
    if n_t == 1:
        data = data[..., 0]
    thick = float(files[0].ds.get("SliceThickness") or 0)
    extra = {"n_slices": nz, "n_timepoints": n_t, "slice_spacing_mm": float(step),
             "slice_gap_mm": round(float(step) - thick, 4) if thick else None, "mosaic": False}
    return data, A, extra


# ---------------------------------------------------------------------------
# Siemens mosaics (one file per time point, slices tiled in a grid)

def _is_mosaic(ds: Dataset) -> bool:
    it = [str(x).upper() for x in (ds.get("ImageType") or [])]
    return "MOSAIC" in it and "SIEMENS" in str(ds.get("Manufacturer", "")).upper()


def _csa(ds: Dataset):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nibabel.nicom import csareader
    return csareader.get_csa_header(ds, "image")


def _load_mosaic(files, r, c, dr, dc):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nibabel.nicom import csareader

    files = sorted(files, key=lambda f: (_int(f.ds.get("AcquisitionNumber")), _int(f.ds.get("InstanceNumber")), f.path.name))
    ds0 = pydicom.dcmread(files[0].path)  # needs the private CSA element
    csa = _csa(ds0)
    if csa is None:
        raise DicomGeometryError("mosaic without a Siemens CSA image header")
    n_slices = int(csareader.get_scalar(csa, "NumberOfImagesInMosaic"))
    snv = csa["tags"].get("SliceNormalVector", {}).get("items")
    if not snv or len(snv) != 3:
        raise DicomGeometryError("mosaic CSA header has no SliceNormalVector")
    snv = np.array(snv, dtype=float)
    snv /= np.linalg.norm(snv)
    n = np.cross(r, c)
    if abs(abs(snv @ n) - 1) > 1e-3:
        raise DicomGeometryError("mosaic SliceNormalVector is not perpendicular to the image plane")

    tiles = math.ceil(math.sqrt(n_slices))
    rows, cols = int(ds0.Rows), int(ds0.Columns)
    if rows % tiles or cols % tiles:
        raise DicomGeometryError(f"mosaic {rows}x{cols} does not divide into {tiles}x{tiles} tiles")
    tr, tc = rows // tiles, cols // tiles

    # The stored IPP is the corner of the whole mosaic image, not of a slice.
    # Move it to the corner of one tile (the tile is centred in the mosaic's field of view).
    ipp = np.array([float(x) for x in ds0.ImagePositionPatient])
    first = ipp + r * dc * (cols - tc) / 2.0 + c * dr * (rows - tr) / 2.0

    spacing = float(ds0.get("SpacingBetweenSlices") or ds0.SliceThickness)

    for f in files[1:]:
        if np.abs(np.array([float(x) for x in f.ds.ImagePositionPatient]) - ipp).max() > POSITION_TOL_MM:
            raise DicomGeometryError("mosaic position changes between time points")

    data = np.empty((tc, tr, n_slices, len(files)), dtype=np.float32)
    for t, f in enumerate(files):
        img = _pixels(f.path)
        for s in range(n_slices):
            y0, x0 = (s // tiles) * tr, (s % tiles) * tc
            data[:, :, s, t] = img[y0:y0 + tr, x0:x0 + tc].T

    A = np.eye(4)
    A[:3, 0] = r * dc
    A[:3, 1] = c * dr
    A[:3, 2] = snv * spacing
    A[:3, 3] = first
    if len(files) == 1:
        data = data[..., 0]
    thick = float(ds0.get("SliceThickness") or 0)
    extra = {"n_slices": n_slices, "n_timepoints": len(files), "slice_spacing_mm": spacing,
             "slice_gap_mm": round(spacing - thick, 4) if thick else None, "mosaic": True,
             "mosaic_tiles": tiles}
    return data, A, extra


# ---------------------------------------------------------------------------

def _pixels(path: Path) -> np.ndarray:
    """Pixel data of one file with RescaleSlope/Intercept applied (per file:
    some scanners use a different scale on every slice)."""
    ds = pydicom.dcmread(path)
    if int(ds.get("NumberOfFrames", 1) or 1) > 1:
        raise DicomGeometryError(f"{path.name}: multi-frame (enhanced) DICOM is not supported yet")
    arr = ds.pixel_array.astype(np.float32)
    slope = float(ds.get("RescaleSlope", 1) or 1)
    inter = float(ds.get("RescaleIntercept", 0) or 0)
    if slope != 1 or inter != 0:
        arr = arr * slope + inter
    return arr


def _check_same(files, tag):
    vals = {int(f.ds.get(tag)) for f in files}
    if len(vals) != 1:
        raise DicomGeometryError(f"{tag} differs within the series: {sorted(vals)}")


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _plain(v):
    if v is None:
        return None
    if isinstance(v, (int, float, str)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)
