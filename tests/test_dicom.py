"""Bar B1b: synthetic DICOM series whose correct geometry is known by construction."""
import random
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from s2t import dicom
from s2t import transforms as T


def write_series(folder: Path, vol: np.ndarray, r, c, dr, dc, origin, slice_vec,
                 n_t=1, shuffle=True, slopes=None, ipp_jitter=None, iop_override=None,
                 drop=None, extra_files=True):
    """Write vol[i, j, k(, t)] (i = column, j = row) as one file per slice per time point."""
    folder.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    nz = vol.shape[2]
    jobs = [(k, t) for t in range(n_t) for k in range(nz)]
    if drop is not None:
        jobs = [j for j in jobs if j != drop]
    order = list(range(len(jobs)))
    if shuffle:
        random.Random(0).shuffle(order)
    for n, idx in enumerate(order):
        k, t = jobs[idx]
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"f{n:04d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = "1.2.3"
        ds.Modality = "MR"
        ds.Manufacturer = "Synthetic"
        ds.SeriesNumber = 1
        ds.PatientName = "Secret^Person"
        ds.PatientID = "MRN-0001"
        ds.InstanceNumber = 1 + t * nz + k
        ds.Rows, ds.Columns = vol.shape[1], vol.shape[0]
        ds.PixelSpacing = [dr, dc]
        ds.SliceThickness = float(np.linalg.norm(slice_vec))
        iop = list(r) + list(c) if iop_override is None or k != 1 else iop_override
        ds.ImageOrientationPatient = [float(x) for x in iop]
        pos = np.asarray(origin) + k * np.asarray(slice_vec)
        if ipp_jitter is not None and k in ipp_jitter:
            pos = pos + ipp_jitter[k]
        ds.ImagePositionPatient = [float(x) for x in pos]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        sl = vol[:, :, k, t] if vol.ndim == 4 else vol[:, :, k]
        if slopes is not None:
            ds.RescaleSlope = slopes[k]
            ds.RescaleIntercept = -10
            stored = np.round((sl + 10) / slopes[k]).astype(np.uint16)
        else:
            stored = sl.astype(np.uint16)
        ds.PixelData = stored.T.copy().tobytes()  # [col,row] -> [row,col]
        ds.save_as(folder / f"f{n:04d}.dcm", enforce_file_format=True)
    if extra_files:
        (folder / "README.txt").write_text("not dicom")
        (folder / ".DS_Store").write_bytes(b"\0\0\0\1Bud1")
    return series_uid


def oblique_geometry():
    R = T.rotation_xyz(np.deg2rad(20), np.deg2rad(-20), np.deg2rad(5))
    r, c = R[:, 0], R[:, 1]
    n = np.cross(r, c)
    return r, c, n


def test_oblique_series_matches_construction(tmp_path):
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 4000, size=(12, 10, 7)).astype(np.float32)
    r, c, n = oblique_geometry()
    dr, dc, dz = 1.2, 0.9, 2.5
    origin = np.array([-40.0, 12.0, 33.0])
    write_series(tmp_path, vol, r, c, dr, dc, origin, n * dz)
    v = dicom.load_series(tmp_path)
    expected_lps = np.eye(4)
    expected_lps[:3, 0], expected_lps[:3, 1], expected_lps[:3, 2], expected_lps[:3, 3] = r * dc, c * dr, n * dz, origin
    assert np.allclose(np.array(v.meta["affine_lps"]), expected_lps, atol=1e-5)
    assert np.allclose(v.affine, T.LPS_TO_RAS @ expected_lps, atol=1e-5)
    assert np.array_equal(v.data, vol)
    # A voxel's world position: LPS from DICOM, then RAS negates x and y.
    ijk = np.array([3, 4, 5])
    lps = origin + 3 * dc * r + 4 * dr * c + 5 * dz * n
    assert np.allclose(v.voxel_to_world(ijk), lps * [-1, -1, 1], atol=1e-5)


def test_reversed_stack_direction(tmp_path):
    """Slices written top-down (slice vector against r x c) still load in order."""
    vol = np.arange(6 * 5 * 4, dtype=np.float32).reshape(6, 5, 4)
    r, c, n = np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]
    write_series(tmp_path, vol, r, c, 1, 1, [0, 0, 30], -n * 3)
    v = dicom.load_series(tmp_path)
    # stored in ascending normal order: our k=0 is the file written last
    assert np.array_equal(v.data, vol[:, :, ::-1])
    assert np.allclose(v.voxel_to_world([0, 0, 0]), [0, 0, 30 - 9])


def test_time_series_split_by_instance_number(tmp_path):
    rng = np.random.default_rng(2)
    vol = rng.integers(0, 1000, size=(8, 8, 5, 3)).astype(np.float32)
    write_series(tmp_path, vol, np.eye(3)[0], np.eye(3)[1], 2, 2, [0, 0, 0], [0, 0, 4], n_t=3)
    v = dicom.load_series(tmp_path)
    assert v.data.shape == (8, 8, 5, 3)
    assert np.array_equal(v.data, vol)


def test_per_slice_rescale(tmp_path):
    rng = np.random.default_rng(3)
    vol = rng.integers(0, 500, size=(6, 6, 3)).astype(np.float32)
    write_series(tmp_path, vol, np.eye(3)[0], np.eye(3)[1], 1, 1, [0, 0, 0], [0, 0, 1], slopes=[1.0, 0.5, 2.0])
    v = dicom.load_series(tmp_path)
    assert np.allclose(v.data, vol, atol=1.0)  # quantised by the slope


def test_no_phi_in_meta(tmp_path):
    vol = np.ones((4, 4, 2), np.float32)
    write_series(tmp_path, vol, np.eye(3)[0], np.eye(3)[1], 1, 1, [0, 0, 0], [0, 0, 1])
    v = dicom.load_series(tmp_path)
    text = repr(v.meta)
    assert "Secret" not in text and "MRN-0001" not in text


@pytest.mark.parametrize("case,match", [
    ("gantry_tilt", "not stacked along the slice normal"),
    ("uneven", "uneven slice spacing"),
    ("missing", "different numbers of images"),
    ("mixed_orientation", "mixed slice orientations"),
])
def test_refusals(tmp_path, case, match):
    vol = np.ones((6, 6, 5, 2), np.float32)
    r, c, n = np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]
    kw = dict(n_t=2)
    sv = n * 3
    if case == "gantry_tilt":
        # each slice shifted 0.5 mm along y per slice: a 9.5 degree tilt
        sv = np.array([0, 0.5, 3.0])
    elif case == "uneven":
        kw["ipp_jitter"] = {3: np.array([0, 0, 0.8])}
    elif case == "missing":
        kw["drop"] = (2, 1)
    elif case == "mixed_orientation":
        R = T.rotation_xyz(0, 0, np.deg2rad(3))
        kw["iop_override"] = list(R[:, 0]) + list(R[:, 1])
    write_series(tmp_path, vol, r, c, 1, 1, [0, 0, 0], sv, **kw)
    with pytest.raises(dicom.DicomGeometryError, match=match):
        dicom.load_series(tmp_path)


def test_two_series_in_one_folder_needs_uid(tmp_path):
    vol = np.ones((4, 4, 2), np.float32)
    write_series(tmp_path / "a", vol, np.eye(3)[0], np.eye(3)[1], 1, 1, [0, 0, 0], [0, 0, 1])
    uid_b = write_series(tmp_path / "b", vol * 2, np.eye(3)[0], np.eye(3)[1], 1, 1, [0, 0, 0], [0, 0, 1])
    with pytest.raises(dicom.DicomGeometryError, match="2 series"):
        dicom.load_series(tmp_path)
    v = dicom.load_series(tmp_path, series_uid=uid_b)
    assert v.data.max() == 2
