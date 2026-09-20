"""Radiologist review-aid evidence panel for PRIMA reports.

Provides deterministic series ranking, representative slice selection,
and thumbnail generation for positive PRIMA labels.

IMPORTANT DISCLAIMER:
Thumbnails are review aids (ảnh gợi ý rà lại / lát cắt tiêu biểu) to assist
radiologists in surveying MRI studies. They are NOT proof, confirmation,
segmentation, or ground-truth lesion localization.
"""

from __future__ import annotations

import html
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Deterministic mapping from diagnosis groups / keywords to preferred MRI series keywords.
# Matches clinical protocol guidelines:
# - ischemic -> DWI/ADC/FLAIR
# - hemorrhagic -> SWI/T2*/SWAN/GRE/FLAIR
# - tumor -> T1 post/T1+C/T1 CE/T2/FLAIR/DWI
# - structural/ventricular -> T2/FLAIR/T1
# - infectious/inflammatory -> DWI/FLAIR/T2/T1 post
# - trauma -> SWI/T2*/SWAN/FLAIR/DWI
PREFERRED_KEYWORDS_BY_GROUP: Dict[str, List[str]] = {
    "ischemic": ["DWI", "ADC", "FLAIR"],
    "vascular_ischemic": ["DWI", "ADC", "FLAIR"],
    "hemorrhagic": ["SWI", "T2*", "SWAN", "GRE", "FLAIR"],
    "vascular_hemorrhagic": ["SWI", "T2*", "SWAN", "GRE", "FLAIR"],
    "tumor": ["T1 post", "T1+C", "T1 CE", "T2", "FLAIR", "DWI"],
    "structural": ["T2", "FLAIR", "T1"],
    "ventricular": ["T2", "FLAIR", "T1"],
    "structural_ventricular": ["T2", "FLAIR", "T1"],
    "infectious_inflammatory": ["DWI", "FLAIR", "T2", "T1 post"],
    "inflammatory": ["DWI", "FLAIR", "T2", "T1 post"],
    "infectious": ["DWI", "FLAIR", "T2", "T1 post"],
    "trauma": ["SWI", "T2*", "SWAN", "FLAIR", "DWI"],
    "vascular_malformation": ["TOF", "MRA", "SWI", "SWAN", "T2*", "T2", "FLAIR"],
    "cyst": ["T2", "FLAIR", "DWI"],
    "developmental": ["T2", "FLAIR", "T1"],
    "sellar": ["T1", "T2", "COR", "SAG"],
    "surgical": ["T2", "FLAIR", "T1"],
    "spine": ["T2", "T1", "SAG"],
    "other": ["T2", "FLAIR", "T1", "DWI"],
}

# Penalty keywords for non-diagnostic or localizer series
EXCLUDED_SERIES_KEYWORDS = [
    "LOCALIZER",
    "SCOUT",
    "CALIBRATION",
    "DERIVED",
    "SCREEN SAVE",
    "REPORT",
    "3-PLANE",
]


def get_preferred_keywords(group_or_code: str) -> List[str]:
    """Return ordered list of preferred MRI series keywords for a diagnosis group or code."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", group_or_code.lower()).strip("_")
    if cleaned in PREFERRED_KEYWORDS_BY_GROUP:
        return PREFERRED_KEYWORDS_BY_GROUP[cleaned]

    # Map prefixes and substrings
    if cleaned.startswith("vascular_ischemic") or "stroke" in cleaned or "ischemic" in cleaned:
        return PREFERRED_KEYWORDS_BY_GROUP["vascular_ischemic"]
    if cleaned.startswith("vascular_hemorrhagic") or "hemorrhage" in cleaned:
        return PREFERRED_KEYWORDS_BY_GROUP["vascular_hemorrhagic"]
    if cleaned.startswith("tumor_") or "tumor" in cleaned or "glioma" in cleaned or "metastasis" in cleaned:
        return PREFERRED_KEYWORDS_BY_GROUP["tumor"]
    if cleaned.startswith("infectious_") or cleaned.startswith("inflammatory_"):
        return PREFERRED_KEYWORDS_BY_GROUP["infectious_inflammatory"]
    if cleaned.startswith("trauma_"):
        return PREFERRED_KEYWORDS_BY_GROUP["trauma"]
    if cleaned.startswith("structural_") or cleaned.startswith("ventricular_"):
        return PREFERRED_KEYWORDS_BY_GROUP["structural"]
    if cleaned.startswith("vascular_malformation_") or "aneurysm" in cleaned or "moyamoya" in cleaned:
        return PREFERRED_KEYWORDS_BY_GROUP["vascular_malformation"]
    if cleaned.startswith("sellar_"):
        return PREFERRED_KEYWORDS_BY_GROUP["sellar"]
    if cleaned.startswith("cyst_"):
        return PREFERRED_KEYWORDS_BY_GROUP["cyst"]
    if cleaned.startswith("developmental_"):
        return PREFERRED_KEYWORDS_BY_GROUP["developmental"]
    return PREFERRED_KEYWORDS_BY_GROUP["other"]


def score_series_match(description: str, preferred_keywords: List[str], slice_count: int = 20) -> float:
    """Compute ranking score for a candidate series against preferred keywords."""
    desc_upper = description.upper()

    # Heavy penalty for localizers or scouts
    for exc in EXCLUDED_SERIES_KEYWORDS:
        if exc in desc_upper:
            return -10000.0

    if slice_count < 3:
        return -5000.0

    score = 0.0
    matched = False
    for rank, kw in enumerate(preferred_keywords):
        kw_norm = kw.upper().replace("+", r"\+").replace("*", r"\*")
        pattern = r"(?:\b|_)" + kw_norm + r"(?:\b|_)"
        if re.search(pattern, desc_upper) or kw.upper() in desc_upper:
            score += (len(preferred_keywords) - rank) * 100.0
            matched = True
            break

    if not matched:
        # Fallback minor credit for standard diagnostic sequences
        for fallback_kw in ["T2", "FLAIR", "T1", "DWI"]:
            if fallback_kw in desc_upper:
                score += 10.0
                break

    # Small tie-breaker preference for full volumes (> 15 slices)
    if slice_count >= 15:
        score += min(slice_count * 0.1, 20.0)

    return score


def rank_series_for_diagnosis(
    series_list: Sequence[Dict[str, Any]],
    diagnosis_code: str,
) -> List[Dict[str, Any]]:
    """Rank available MRI series by relevance to the given diagnosis code."""
    preferred = get_preferred_keywords(diagnosis_code)
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for s in series_list:
        desc = str(
            s.get("description")
            or s.get("series_description")
            or s.get("protocol_name")
            or s.get("name")
            or ""
        )
        slices = int(s.get("file_count") or s.get("instance_count") or s.get("slices") or 0)
        sc = score_series_match(desc, preferred, slice_count=slices)
        scored.append((sc, s))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored if item[0] > 0]


def select_representative_slices(
    volume: np.ndarray,
    count: int = 4,
    min_volume_fraction: float = 0.20,
    max_volume_fraction: float = 0.80,
) -> List[int]:
    """Select representative slice indices from the middle 60% of the volume.

    Uses high non-background brain content and intensity variation with stratified
    spatial separation across the depth axis. Does not use patient-derived model localization.

    Args:
        volume: 3D numpy array with shape (depth, height, width) or (height, width, depth).
        count: Desired number of representative slices (typically 3 to 6).
        min_volume_fraction: Lower bound of volume search range (default 0.20).
        max_volume_fraction: Upper bound of volume search range (default 0.80).

    Returns:
        List of integer slice indices along the primary slice dimension.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    # Standardize depth as axis 0
    # If axis 2 is depth (depth < height and depth < width), transpose to (D, H, W)
    if volume.shape[2] < volume.shape[0] and volume.shape[2] < volume.shape[1]:
        volume = np.transpose(volume, (2, 0, 1))

    total_slices = volume.shape[0]
    if total_slices <= count:
        return list(range(total_slices))

    # Constrain to middle 60% of volume (default 0.20 to 0.80)
    start_idx = max(0, int(math.floor(total_slices * min_volume_fraction)))
    end_idx = min(total_slices - 1, int(math.ceil(total_slices * max_volume_fraction)))

    if end_idx <= start_idx:
        start_idx = 0
        end_idx = total_slices - 1

    search_span = end_idx - start_idx + 1
    actual_count = min(count, search_span)

    # Score each slice within search range
    scores = {}
    for z in range(start_idx, end_idx + 1):
        slice_2d = volume[z].astype(np.float32)
        v_min, v_max = float(np.min(slice_2d)), float(np.max(slice_2d))
        if v_max <= v_min + 1e-6:
            scores[z] = 0.0
            continue

        norm = (slice_2d - v_min) / (v_max - v_min)
        threshold = max(float(np.mean(norm)) * 0.5, 0.10)
        mask = norm > threshold
        brain_ratio = float(np.sum(mask)) / float(slice_2d.size)

        if brain_ratio > 0.05:
            intensity_variation = float(np.std(norm[mask]))
        else:
            intensity_variation = float(np.std(norm))

        scores[z] = brain_ratio * (intensity_variation + 0.05)

    # Stratified selection: partition [start_idx, end_idx] into actual_count bins
    bin_size = search_span / float(actual_count)
    selected: List[int] = []

    for b in range(actual_count):
        bin_start = start_idx + int(math.floor(b * bin_size))
        bin_end = start_idx + int(math.floor((b + 1) * bin_size)) - 1
        bin_end = min(bin_end, end_idx)
        bin_end = max(bin_end, bin_start)

        best_z = max(range(bin_start, bin_end + 1), key=lambda idx: scores.get(idx, 0.0))
        selected.append(best_z)

    unique_selected = sorted(list(dict.fromkeys(selected)))
    return unique_selected


def normalize_slice_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Normalize a 2D image slice to uint8 using 1st-99th percentile windowing."""
    arr = slice_2d.astype(np.float32)
    p1, p99 = float(np.percentile(arr, 1.0)), float(np.percentile(arr, 99.0))
    if p99 <= p1 + 1e-6:
        p1, p99 = float(np.min(arr)), float(np.max(arr))
    if p99 <= p1 + 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)

    clipped = np.clip(arr, p1, p99)
    rescaled = ((clipped - p1) / (p99 - p1)) * 255.0
    return np.round(rescaled).astype(np.uint8)


def write_png(image_uint8: np.ndarray, target_path: Path) -> None:
    """Write 2D uint8 numpy array to a PNG file using SimpleITK."""
    import SimpleITK as sitk

    target_path.parent.mkdir(parents=True, exist_ok=True)
    sitk_img = sitk.GetImageFromArray(image_uint8)
    sitk.WriteImage(sitk_img, str(target_path))


def discover_study_series(study_dir: Path) -> List[Dict[str, Any]]:
    """Scan study directory to find staged MRI series and metadata."""
    import pydicom

    series_list: List[Dict[str, Any]] = []

    # Check for series_* subdirectories
    subdirs = sorted([d for d in study_dir.iterdir() if d.is_dir() and d.name.startswith("series_")])
    if not subdirs:
        # Check any subdirectories containing DICOM files
        subdirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])

    for sd in subdirs:
        dcm_files = sorted(list(sd.glob("*.dcm")) + list(sd.glob("*.DCM")))
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            desc = str(getattr(ds, "SeriesDescription", ""))
            proto = str(getattr(ds, "ProtocolName", ""))
            series_num = getattr(ds, "SeriesNumber", 0)
        except Exception:
            desc = sd.name
            proto = ""
            series_num = 0

        series_list.append({
            "directory": sd,
            "series_description": desc,
            "protocol_name": proto,
            "series_number": series_num,
            "file_count": len(dcm_files),
        })

    return series_list


def generate_study_evidence(
    study_dir: Path,
    output_dir: Path,
    positive_diagnoses: List[str],
    label_resolver: Optional[Any] = None,
    language: str = "vi",
    max_thumbnails_per_label: int = 4,
) -> List[Dict[str, Any]]:
    """Generate local PNG thumbnails and metadata for positive diagnoses."""
    import SimpleITK as sitk

    if not study_dir or not study_dir.exists():
        return []

    series_list = discover_study_series(study_dir)
    if not series_list:
        return []

    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    evidence_entries: List[Dict[str, Any]] = []
    # Cache loaded 3D volumes by directory to avoid re-reading the same series multiple times
    volume_cache: Dict[Path, Tuple[np.ndarray, str]] = {}

    for diag_code in positive_diagnoses:
        ranked = rank_series_for_diagnosis(series_list, diag_code)
        if not ranked:
            continue

        best = ranked[0]
        s_dir: Path = best["directory"]
        desc = best.get("series_description") or best.get("protocol_name") or s_dir.name

        if s_dir not in volume_cache:
            try:
                reader = sitk.ImageSeriesReader()
                dcm_files = reader.GetGDCMSeriesFileNames(str(s_dir))
                if not dcm_files:
                    dcm_files = sorted(str(p) for p in s_dir.glob("*.dcm"))
                if not dcm_files:
                    continue
                reader.SetFileNames(dcm_files)
                sitk_image = reader.Execute()
                vol = sitk.GetArrayFromImage(sitk_image)  # (depth, height, width)
                volume_cache[s_dir] = (vol, desc)
            except Exception:
                continue

        vol, series_desc = volume_cache[s_dir]
        if vol.ndim != 3 or vol.shape[0] < 1:
            continue

        slice_indices = select_representative_slices(vol, count=max_thumbnails_per_label)
        safe_code = re.sub(r"[^a-zA-Z0-9_]+", "_", diag_code).strip("_")

        thumbnails: List[Dict[str, Any]] = []
        for z in slice_indices:
            uint8_slice = normalize_slice_to_uint8(vol[z])
            png_name = f"{safe_code}_s{z+1:03d}.png"
            png_path = evidence_dir / png_name
            write_png(uint8_slice, png_path)

            thumbnails.append({
                "slice_index": z + 1,
                "total_slices": vol.shape[0],
                "relative_path": f"evidence/{png_name}",
            })

        display_label = (
            label_resolver(diag_code, language)
            if label_resolver
            else diag_code
        )

        evidence_entries.append({
            "code": diag_code,
            "label": display_label,
            "series_description": series_desc,
            "thumbnails": thumbnails,
        })

    return evidence_entries


def build_evidence_html(evidence_entries: List[Dict[str, Any]], language: str = "vi") -> str:
    """Generate HTML snippet for the radiologist review-aid evidence panel."""
    if not evidence_entries:
        return ""

    disclaimer_vi = (
        "<strong>Ảnh gợi ý rà lại / Lát cắt tiêu biểu:</strong> "
        "Các lát cắt dưới đây được trích xuất tự động từ chuỗi xung phù hợp nhằm hỗ trợ rà soát study. "
        "Đây <em>không phải</em> là bằng chứng khẳng định (proof/confirmation), không phải phân vùng tổn thương "
        "(segmentation), và không phải vị trí tổn thương chính xác (ground-truth localization)."
    )
    disclaimer_en = (
        "<strong>Review-Aid Representative Slices:</strong> "
        "The slices below are automatically sampled from preferred sequences to support study review. "
        "They are <em>not</em> proof, confirmation, segmentation, or ground-truth lesion localization."
    )
    disclaimer = disclaimer_vi if language == "vi" else disclaimer_en
    section_title = "Hình ảnh gợi ý rà soát" if language == "vi" else "Review-Aid Evidence Panel"

    html_parts = [
        '<section class="evidence-panel">',
        f'<h2>{html.escape(section_title)}</h2>',
        f'<div class="notice evidence-disclaimer">{disclaimer}</div>',
    ]

    for entry in evidence_entries:
        label = entry.get("label", "")
        series_desc = entry.get("series_description", "")
        thumbnails = entry.get("thumbnails", [])

        html_parts.append('<div class="evidence-item">')
        html_parts.append(f'<h3>{html.escape(label)}</h3>')
        if series_desc:
            series_label = "Chuỗi xung gợi ý:" if language == "vi" else "Preferred series:"
            html_parts.append(
                f'<p class="evidence-meta">{series_label} <strong>{html.escape(series_desc)}</strong></p>'
            )

        if thumbnails:
            html_parts.append('<div class="evidence-grid">')
            for thumb in thumbnails:
                rel_path = html.escape(str(thumb.get("relative_path", "")))
                slice_idx = thumb.get("slice_index", 0)
                total_slices = thumb.get("total_slices", 0)
                caption = (
                    f"Lát {slice_idx}/{total_slices}"
                    if language == "vi"
                    else f"Slice {slice_idx}/{total_slices}"
                )
                html_parts.append(
                    f'<div class="evidence-card">'
                    f'<img src="{rel_path}" alt="{html.escape(caption)}" loading="lazy" />'
                    f'<span>{html.escape(caption)}</span>'
                    f'</div>'
                )
            html_parts.append('</div>')
        else:
            no_series_msg = (
                "_Không tìm thấy chuỗi xung phù hợp để trích xuất ảnh tiêu biểu._"
                if language == "vi"
                else "_No suitable series found for representative slice extraction._"
            )
            html_parts.append(f'<p><em>{html.escape(no_series_msg)}</em></p>')

        html_parts.append('</div>')

    html_parts.append('</section>')
    return "\n".join(html_parts)
