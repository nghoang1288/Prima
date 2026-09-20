from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from tools.stage_dicom_study import scan


def write_dicom(
    path: Path,
    study_uid: str,
    series_uid: str,
    study_description: str = "MRI BRAIN",
):
    file_meta = Dataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()

    ds = FileDataset(
        str(path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.StudyDescription = study_description
    ds.save_as(str(path), enforce_file_format=True)


def test_recursive_scan_groups_one_study_into_series(tmp_path):
    study_uid = generate_uid()
    series_a = generate_uid()
    series_b = generate_uid()

    a = tmp_path / "nested" / "a"
    b = tmp_path / "other" / "b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)

    write_dicom(a / "1.dcm", study_uid, series_a)
    write_dicom(a / "2.dcm", study_uid, series_a)
    write_dicom(b / "3.dcm", study_uid, series_b)
    (tmp_path / "not_dicom.txt").write_text("ignore me")

    groups, found_uid, description, skipped = scan(tmp_path)

    assert found_uid == study_uid
    assert description == "MRI BRAIN"
    assert set(groups) == {series_a, series_b}
    assert len(groups[series_a]) == 2
    assert len(groups[series_b]) == 1
    assert skipped == {}


def test_scan_rejects_mixed_studies(tmp_path):
    folder = tmp_path / "mixed"
    folder.mkdir()
    write_dicom(folder / "a.dcm", generate_uid(), generate_uid())
    write_dicom(folder / "b.dcm", generate_uid(), generate_uid())

    with pytest.raises(RuntimeError, match="exactly one StudyInstanceUID"):
        scan(folder)


def test_scan_ignores_and_reports_non_mr_ancillary_objects(tmp_path):
    study_uid = generate_uid()
    mr_series = generate_uid()
    folder = tmp_path / "export"
    folder.mkdir()

    write_dicom(folder / "mr.dcm", study_uid, mr_series)

    sr_path = folder / "sr.dcm"
    write_dicom(sr_path, study_uid, generate_uid())
    ds = pydicom.dcmread(str(sr_path))
    ds.Modality = "SR"
    ds.save_as(str(sr_path), enforce_file_format=True)

    groups, found_uid, _, skipped = scan(folder)

    assert found_uid == study_uid
    assert set(groups) == {mr_series}
    assert skipped == {"SR": 1}
