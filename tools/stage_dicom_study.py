#!/usr/bin/env python3
"""Stage one local MR DICOM study into PRIMA's one-folder-per-series layout.

This tool does NOT de-identify or modify DICOM contents. It only copies files
locally into an anonymous directory layout so source paths/patient folder names
are not propagated into PRIMA logs or output filenames.

No PatientName/PatientID tags are requested or printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pydicom
from pydicom.errors import InvalidDicomError


TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "Modality",
    "StudyDescription",
    "SOPClassUID",
]

ANCILLARY_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.7": "SC",
    "1.2.840.10008.5.1.4.1.1.104.1": "DOC",
}


def safe_read(path: Path):
    kwargs = {
        "stop_before_pixels": True,
        "specific_tags": TAGS,
    }
    try:
        return pydicom.dcmread(str(path), force=False, **kwargs)
    except InvalidDicomError:
        # Some PACS exports omit the 128-byte preamble/DICM marker while still
        # carrying valid DICOM tags. Retry permissively, then require the UIDs
        # below before accepting the file.
        try:
            ds = pydicom.dcmread(str(path), force=True, **kwargs)
            if not getattr(ds, "StudyInstanceUID", None):
                return None
            if not getattr(ds, "SeriesInstanceUID", None):
                return None
            return ds
        except (OSError, ValueError):
            return None
    except (OSError, ValueError):
        return None


def uid_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def scan(source: Path):
    groups: Dict[str, List[Path]] = defaultdict(list)
    series_numbers: Dict[str, int | None] = {}
    study_uids = set()
    study_descriptions = set()
    skipped_modalities: Dict[str, int] = defaultdict(int)

    for path in source.rglob("*"):
        if not path.is_file():
            continue
        ds = safe_read(path)
        if ds is None:
            continue

        study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
        series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        modality = str(getattr(ds, "Modality", "") or "").upper()
        description = str(getattr(ds, "StudyDescription", "") or "").strip()
        raw_series_number = getattr(ds, "SeriesNumber", None)
        sop_class = str(getattr(ds, "SOPClassUID", "") or "")

        if not study_uid or not series_uid:
            continue
        if modality != "MR":
            skipped_modalities[modality or "<blank>"] += 1
            continue

        ancillary_label = ANCILLARY_SOP_CLASSES.get(sop_class)
        if not ancillary_label:
            if sop_class.startswith("1.2.840.10008.5.1.4.1.1.88."):
                ancillary_label = "SR"
            elif sop_class.startswith("1.2.840.10008.5.1.4.1.1.11."):
                ancillary_label = "PR"

        if ancillary_label:
            skipped_modalities[ancillary_label] += 1
            continue

        try:
            series_number = int(raw_series_number)
        except (TypeError, ValueError):
            series_number = None

        study_uids.add(study_uid)
        if description:
            study_descriptions.add(description)
        groups[series_uid].append(path)
        if series_uid not in series_numbers or series_numbers[series_uid] is None:
            series_numbers[series_uid] = series_number

    if not groups:
        raise RuntimeError("No readable MR DICOM series found")
    if len(study_uids) != 1:
        raise RuntimeError(
            f"Expected exactly one MR StudyInstanceUID, found {len(study_uids)}"
        )

    # A study should normally have one StudyDescription. If exporters copied
    # inconsistent values across instances, leave it blank rather than guessing.
    description = (
        next(iter(study_descriptions))
        if len(study_descriptions) == 1
        else ""
    )
    return (
        groups,
        next(iter(study_uids)),
        description,
        dict(skipped_modalities),
        series_numbers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--study-id", default="CASE001")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print study description or UID hashes to stdout.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError("Source study directory does not exist")

    if source == destination:
        raise ValueError("Source and destination must be different directories")
    if source in destination.parents or destination in source.parents:
        raise ValueError(
            "Source and destination must not be nested inside each other"
        )
    if destination == Path("/") or destination == Path.home().resolve():
        raise ValueError("Refusing unsafe staging destination")

    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(
                "Destination already exists; use --overwrite for a fresh local staging copy"
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, mode=0o700)
    destination.chmod(0o700)

    (
        groups,
        study_uid,
        study_description,
        skipped_modalities,
        series_numbers,
    ) = scan(source)

    series_rows = []
    ordered_series = sorted(
        groups.items(),
        key=lambda item: (
            series_numbers.get(item[0]) is None,
            series_numbers.get(item[0]) if series_numbers.get(item[0]) is not None else 0,
            item[0],
        ),
    )
    for series_index, (series_uid, files) in enumerate(
        ordered_series,
        start=1,
    ):
        series_dir = destination / f"series_{series_index:04d}_{uid_hash(series_uid)}"
        series_dir.mkdir(mode=0o700)
        for image_index, src in enumerate(sorted(files), start=1):
            suffix = src.suffix if src.suffix else ".dcm"
            dst = series_dir / f"image_{image_index:06d}{suffix}"
            shutil.copy2(src, dst)
            dst.chmod(0o600)

        series_rows.append(
            {
                "series_index": series_index,
                "series_uid_hash": uid_hash(series_uid),
                "series_number": series_numbers.get(series_uid),
                "file_count": len(files),
            }
        )

    manifest = {
        "study_id": args.study_id,
        "study_uid_hash": uid_hash(study_uid),
        "study_description": study_description,
        "series_count": len(series_rows),
        "series": series_rows,
        "skipped_non_mr_instances": skipped_modalities,
        "note": (
            "Local staging copy only. DICOM contents were not de-identified or modified."
        ),
    }
    (destination / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if args.quiet:
        print(
            json.dumps(
                {
                    "study_id": args.study_id,
                    "series_count": len(series_rows),
                    "skipped_non_mr_instances": skipped_modalities,
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
