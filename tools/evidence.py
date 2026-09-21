"""Radiologist review-aid evidence panel for PRIMA reports.

Provides deterministic sequence family mapping, series ranking, representative slice selection,
consistent LPS orientation, visible R/L markers, clickable thumbnails, and deduplication.

IMPORTANT CLINICAL DISCLAIMER:
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

# Diagnosis-specific sequence family overrides.
# Maps specific diagnosis codes to up to 3 clinically distinct sequence families:
# [("Family Name", [keywords...]), ...]
DIAGNOSIS_OVERRIDE_FAMILIES: Dict[str, List[Tuple[str, List[str]]]] = {
    # Moyamoya: MRA/TOF (stenosis/collaterals) + FLAIR/T2 (watershed) + DWI (acute infarct)
    "vascular_ischemic_moyamoya": [
        ("MRA / TOF", ["TOF", "MRA", "SPGR"]),
        ("T2 FLAIR", ["FLAIR", "T2"]),
        ("DWI", ["DWI", "DIFFUSION"]),
    ],
    # Aneurysm: 3D TOF/MRA + SWI/SWAN/T2* + T2
    "vascular_malformation_cerebral_aneurysm": [
        ("3D TOF / MRA", ["TOF", "MRA"]),
        ("SWI / SWAN / T2*", ["SWI", "SWAN", "T2*", "GRE"]),
        ("T2", ["T2"]),
    ],
    # AVM: MRA/TOF (feeding/draining vessels) + T2 (flow voids/nidus) + SWI/SWAN
    "vascular_malformation_arteriovenous_malformation": [
        ("MRA / TOF", ["TOF", "MRA"]),
        ("T2", ["T2"]),
        ("SWI / SWAN / T2*", ["SWI", "SWAN", "T2*", "GRE"]),
    ],
    # Cavernoma: SWI/SWAN/T2* (hemosiderin rim) + T2 (popcorn lesion) + FLAIR
    "vascular_malformation_cavernoma": [
        ("SWI / SWAN / T2*", ["SWI", "SWAN", "T2*", "GRE"]),
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
    ],
    # Sellar lesions: T1 contrast/coronal/sagittal + T2 + FLAIR
    "sellar_pituitary_adenoma": [
        ("T1 Coronal / Sagittal / Contrast", ["T1", "SAG", "COR", "CE", "+C"]),
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
    ],
    "sellar_craniopharyngioma": [
        ("T1 Coronal / Sagittal / Contrast", ["T1", "SAG", "COR", "CE", "+C"]),
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
    ],
    "sellar_rathkes_cleft_cyst": [
        ("T1 Coronal / Sagittal", ["T1", "SAG", "COR"]),
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
    ],
    # Ventriculomegaly / hydrocephalus: T2 + FLAIR + T1
    "ventricular_ventriculomegaly": [
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
        ("T1", ["T1"]),
    ],
    "ventricular_intracranial_hypotension": [
        ("T1 Sagittal / Contrast", ["T1", "SAG", "+C", "CE"]),
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
    ],
    # Cerebral atrophy: T2 Coronal/Axial + FLAIR + T1
    "structural_cerebral_atrophy": [
        ("T2 Coronal / Axial", ["T2", "COR"]),
        ("FLAIR", ["FLAIR"]),
        ("T1", ["T1", "SAG"]),
    ],
    # Encephalomalacia: FLAIR (gliosis) + T2 + T1
    "structural_encephalomalacia": [
        ("FLAIR", ["FLAIR"]),
        ("T2", ["T2"]),
        ("T1", ["T1"]),
    ],
    # Small vessel disease: FLAIR (WMH) + T2 + SWI/SWAN (microbleeds)
    "vascular_ischemic_small_vessel_disease": [
        ("FLAIR", ["FLAIR"]),
        ("T2", ["T2"]),
        ("SWI / SWAN / T2*", ["SWI", "SWAN", "T2*", "GRE"]),
    ],
    "vascular_ischemic_lacunar_stroke": [
        ("DWI", ["DWI", "DIFFUSION"]),
        ("ADC", ["ADC"]),
        ("FLAIR", ["FLAIR"]),
    ],
    # Large vessel ischemic stroke: DWI + ADC + FLAIR / MRA
    "vascular_ischemic_large_vessel_stroke": [
        ("DWI", ["DWI", "DIFFUSION"]),
        ("ADC", ["ADC"]),
        ("FLAIR / MRA", ["FLAIR", "TOF", "MRA"]),
    ],
    # Postoperative: T1 post/T1 + T2 + FLAIR
    "surgical_catheter": [
        ("T2 / CT", ["T2", "CT"]),
        ("T1", ["T1"]),
        ("FLAIR", ["FLAIR"]),
    ],
    "surgical_craniotomy": [
        ("T1 / CT", ["T1", "CT"]),
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
    ],
    "surgical_resection_cavity": [
        ("FLAIR", ["FLAIR"]),
        ("T2", ["T2"]),
        ("T1", ["T1"]),
    ],
}

# Group-level default sequence families
GROUP_DEFAULT_FAMILIES: Dict[str, List[Tuple[str, List[str]]]] = {
    "vascular_ischemic": [
        ("DWI", ["DWI", "DIFFUSION"]),
        ("ADC", ["ADC"]),
        ("FLAIR", ["FLAIR"]),
    ],
    "ischemic": [
        ("DWI", ["DWI", "DIFFUSION"]),
        ("ADC", ["ADC"]),
        ("FLAIR", ["FLAIR"]),
    ],
    "vascular_hemorrhagic": [
        ("SWI / SWAN / T2*", ["SWI", "SWAN", "T2*", "GRE"]),
        ("FLAIR", ["FLAIR"]),
        ("T2", ["T2"]),
    ],
    "hemorrhagic": [
        ("SWI / SWAN / T2*", ["SWI", "SWAN", "T2*", "GRE"]),
        ("FLAIR", ["FLAIR"]),
        ("T2", ["T2"]),
    ],
    "tumor": [
        ("T1 Post-contrast / T1", ["T1 POST", "T1+C", "T1 CE", "+C", "CE", "T1"]),
        ("FLAIR / T2", ["FLAIR", "T2"]),
        ("DWI", ["DWI", "DIFFUSION"]),
    ],
    "structural": [
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
        ("T1", ["T1"]),
    ],
    "ventricular": [
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
        ("T1", ["T1"]),
    ],
    "structural_ventricular": [
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
        ("T1", ["T1"]),
    ],
    "infectious_inflammatory": [
        ("DWI", ["DWI", "DIFFUSION"]),
        ("FLAIR / T2", ["FLAIR", "T2"]),
        ("T1 Post-contrast / T1", ["T1 POST", "T1+C", "+C", "CE", "T1"]),
    ],
    "inflammatory": [
        ("DWI", ["DWI", "DIFFUSION"]),
        ("FLAIR / T2", ["FLAIR", "T2"]),
        ("T1 Post-contrast / T1", ["T1 POST", "T1+C", "+C", "CE", "T1"]),
    ],
    "infectious": [
        ("DWI", ["DWI", "DIFFUSION"]),
        ("FLAIR / T2", ["FLAIR", "T2"]),
        ("T1 Post-contrast / T1", ["T1 POST", "T1+C", "+C", "CE", "T1"]),
    ],
    "trauma": [
        ("SWI / SWAN / T2*", ["SWI", "SWAN", "T2*", "GRE"]),
        ("FLAIR", ["FLAIR"]),
        ("DWI", ["DWI", "DIFFUSION"]),
    ],
    "vascular_malformation": [
        ("TOF / MRA", ["TOF", "MRA"]),
        ("SWI / SWAN / T2*", ["SWI", "SWAN", "T2*", "GRE"]),
        ("T2", ["T2"]),
    ],
    "cyst": [
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
        ("DWI", ["DWI", "DIFFUSION"]),
    ],
    "developmental": [
        ("T2", ["T2"]),
        ("T1", ["T1"]),
        ("FLAIR", ["FLAIR"]),
    ],
    "sellar": [
        ("T1 Sagittal / Coronal", ["T1", "SAG", "COR"]),
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
    ],
    "surgical": [
        ("T2", ["T2"]),
        ("T1", ["T1"]),
        ("FLAIR", ["FLAIR"]),
    ],
    "spine": [
        ("T2", ["T2"]),
        ("T1", ["T1"]),
        ("SAG", ["SAG"]),
    ],
    "other": [
        ("T2", ["T2"]),
        ("FLAIR", ["FLAIR"]),
    ],
}

# Backwards compatibility keywords mapping
PREFERRED_KEYWORDS_BY_GROUP: Dict[str, List[str]] = {
    group: [kw for _, kws in families for kw in kws]
    for group, families in GROUP_DEFAULT_FAMILIES.items()
}

EXCLUDED_SERIES_KEYWORDS = [
    "LOCALIZER",
    "SCOUT",
    "CALIBRATION",
    "DERIVED",
    "SCREEN SAVE",
    "REPORT",
    "3-PLANE",
]


def get_preferred_families(diagnosis_code: str) -> List[Tuple[str, List[str]]]:
    """Return ordered sequence families for a specific diagnosis code or group."""
    norm = re.sub(r"[^a-z0-9_]+", "_", diagnosis_code.lower()).strip("_")
    if norm in DIAGNOSIS_OVERRIDE_FAMILIES:
        return DIAGNOSIS_OVERRIDE_FAMILIES[norm]

    # Prefix match
    for group, families in GROUP_DEFAULT_FAMILIES.items():
        if norm.startswith(group + "_") or norm == group:
            return families

    if "ischemic" in norm or "stroke" in norm:
        return GROUP_DEFAULT_FAMILIES["vascular_ischemic"]
    if "hemorrhag" in norm:
        return GROUP_DEFAULT_FAMILIES["vascular_hemorrhagic"]
    if "tumor" in norm or "glioma" in norm or "metastasis" in norm:
        return GROUP_DEFAULT_FAMILIES["tumor"]
    if "infectious" in norm or "inflammatory" in norm:
        return GROUP_DEFAULT_FAMILIES["infectious_inflammatory"]
    if "trauma" in norm:
        return GROUP_DEFAULT_FAMILIES["trauma"]
    if "structural" in norm or "ventricular" in norm:
        return GROUP_DEFAULT_FAMILIES["structural"]
    if "aneurysm" in norm or "malformation" in norm or "moyamoya" in norm:
        return GROUP_DEFAULT_FAMILIES["vascular_malformation"]

    return GROUP_DEFAULT_FAMILIES["other"]


def get_preferred_keywords(group_or_code: str) -> List[str]:
    """Return flat list of preferred keywords for backwards compatibility."""
    families = get_preferred_families(group_or_code)
    seen = set()
    res = []
    for _, kws in families:
        for kw in kws:
            if kw not in seen:
                seen.add(kw)
                res.append(kw)
    return res


def score_series_match(description: str, preferred_keywords: List[str], slice_count: int = 20) -> float:
    """Compute ranking score for a candidate series against preferred keywords.

    Enforces STRICT NO-MISLEADING-FALLBACK: returns <= 0 if no preferred keyword matches.
    """
    desc_upper = description.upper()

    for exc in EXCLUDED_SERIES_KEYWORDS:
        if exc in desc_upper:
            return -10000.0

    if slice_count < 3:
        return -5000.0

    matched = False
    score = 0.0
    for rank, kw in enumerate(preferred_keywords):
        kw_norm = kw.upper().replace("+", r"\+").replace("*", r"\*")
        pattern = r"(?:\b|_)" + kw_norm + r"(?:\b|_)"
        if re.search(pattern, desc_upper) or kw.upper() in desc_upper:
            score += (len(preferred_keywords) - rank) * 100.0
            matched = True
            break

    # If no keyword matched, STRICTLY return 0.0 - NO arbitrary fallback series!
    if not matched:
        return 0.0

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


def select_multi_series_for_diagnosis(
    series_list: Sequence[Dict[str, Any]],
    diagnosis_code: str,
    max_families: int = 3,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Select up to `max_families` clinically distinct series for a diagnosis.

    Returns:
        List of tuples: (family_name, best_matching_series)
        If a family has no matching series, it is omitted (no misleading fallback).
    """
    families = get_preferred_families(diagnosis_code)[:max_families]
    selected_families: List[Tuple[str, Dict[str, Any]]] = []
    used_directories = set()

    for family_name, keywords in families:
        scored = []
        for s in series_list:
            s_dir = s.get("directory")
            desc = str(
                s.get("description")
                or s.get("series_description")
                or s.get("protocol_name")
                or s.get("name")
                or ""
            )
            slices = int(s.get("file_count") or s.get("instance_count") or s.get("slices") or 0)
            sc = score_series_match(desc, keywords, slice_count=slices)
            if sc > 0:
                # Slight penalty if series already used in another family of the same diagnosis
                if s_dir and s_dir in used_directories:
                    sc -= 5.0
                scored.append((sc, s))

        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            best_series = scored[0][1]
            if best_series.get("directory"):
                used_directories.add(best_series["directory"])
            selected_families.append((family_name, best_series))

    return selected_families


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
        count: Desired number of representative slices (typically 2 to 4 per series).
        min_volume_fraction: Lower bound of volume search range (default 0.20).
        max_volume_fraction: Upper bound of volume search range (default 0.80).

    Returns:
        List of integer slice indices along the primary slice dimension.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    # Standardize depth as axis 0
    if volume.shape[2] < volume.shape[0] and volume.shape[2] < volume.shape[1]:
        volume = np.transpose(volume, (2, 0, 1))

    total_slices = volume.shape[0]
    if total_slices <= count:
        return list(range(total_slices))

    start_idx = max(0, int(math.floor(total_slices * min_volume_fraction)))
    end_idx = min(total_slices - 1, int(math.ceil(total_slices * max_volume_fraction)))

    if end_idx <= start_idx:
        start_idx = 0
        end_idx = total_slices - 1

    search_span = end_idx - start_idx + 1
    actual_count = min(count, search_span)

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

    subdirs = sorted([d for d in study_dir.iterdir() if d.is_dir() and d.name.startswith("series_")])
    if not subdirs:
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


def orient_image_to_lps(sitk_image: Any) -> Any:
    """Standardize a SimpleITK image to LPS orientation if supported."""
    import SimpleITK as sitk
    if hasattr(sitk, "DICOMOrient"):
        try:
            return sitk.DICOMOrient(sitk_image, "LPS")
        except Exception:
            pass
    return sitk_image


def generate_study_evidence(
    study_dir: Path,
    output_dir: Path,
    positive_diagnoses: List[str],
    label_resolver: Optional[Any] = None,
    language: str = "vi",
    max_families_per_diagnosis: int = 3,
    slices_per_family: int = 3,
) -> List[Dict[str, Any]]:
    """Generate local PNG thumbnails and metadata for positive diagnoses with multi-series support and deduplication."""
    import SimpleITK as sitk

    if not study_dir or not study_dir.exists():
        return []

    series_list = discover_study_series(study_dir)
    if not series_list:
        return []

    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Volume cache: Path -> (np.ndarray, series_desc)
    volume_cache: Dict[Path, Tuple[np.ndarray, str]] = {}
    # Deduplication cache: (series_dir_name, slice_idx) -> relative_path
    thumbnail_cache: Dict[Tuple[str, int], str] = {}

    evidence_entries: List[Dict[str, Any]] = []

    for diag_code in positive_diagnoses:
        selected_families = select_multi_series_for_diagnosis(
            series_list, diag_code, max_families=max_families_per_diagnosis
        )
        display_label = (
            label_resolver(diag_code, language)
            if label_resolver
            else diag_code
        )

        if not selected_families:
            evidence_entries.append({
                "code": diag_code,
                "label": display_label,
                "sequence_groups": [],
            })
            continue

        seq_groups = []
        for family_name, s_info in selected_families:
            s_dir: Path = s_info["directory"]
            desc = s_info.get("series_description") or s_info.get("protocol_name") or s_dir.name

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

                    # Standardize to LPS orientation
                    sitk_image = orient_image_to_lps(sitk_image)

                    vol = sitk.GetArrayFromImage(sitk_image)  # (depth, height, width)
                    volume_cache[s_dir] = (vol, desc)
                except Exception:
                    continue

            vol, series_desc = volume_cache[s_dir]
            if vol.ndim != 3 or vol.shape[0] < 1:
                continue

            slice_indices = select_representative_slices(vol, count=slices_per_family)
            thumbnails: List[Dict[str, Any]] = []

            for z in slice_indices:
                cache_key = (s_dir.name, z)
                if cache_key in thumbnail_cache:
                    rel_path = thumbnail_cache[cache_key]
                else:
                    # Clean deduplicated filename based on series directory and slice index
                    png_name = f"thumb_{s_dir.name}_s{z+1:03d}.png"
                    png_path = evidence_dir / png_name
                    if not png_path.exists():
                        uint8_slice = normalize_slice_to_uint8(vol[z])
                        write_png(uint8_slice, png_path)
                    rel_path = f"evidence/{png_name}"
                    thumbnail_cache[cache_key] = rel_path

                thumbnails.append({
                    "slice_index": z + 1,
                    "total_slices": vol.shape[0],
                    "relative_path": rel_path,
                })

            seq_groups.append({
                "family_name": family_name,
                "series_description": series_desc,
                "thumbnails": thumbnails,
            })

        evidence_entries.append({
            "code": diag_code,
            "label": display_label,
            "sequence_groups": seq_groups,
        })

    return evidence_entries


def build_evidence_html(evidence_entries: List[Dict[str, Any]], language: str = "vi") -> str:
    """Generate radiologist-first HTML for evidence panels with LPS orientation, R/L markers, and clickable links."""
    if not evidence_entries:
        return ""

    disclaimer_vi = (
        "<strong>Ảnh gợi ý rà lại / Lát cắt tiêu biểu:</strong> "
        "Các lát cắt dưới đây được trích xuất tự động từ các chuỗi xung phù hợp nhằm hỗ trợ rà soát study. "
        "Đây <em>không phải</em> là bằng chứng khẳng định (proof/confirmation), không phải phân vùng tổn thương "
        "(segmentation), và không phải vị trí tổn thương chính xác (ground-truth localization). "
        "Hình ảnh được chuẩn hóa theo định dạng hiển thị chẩn đoán hình ảnh: bên trái ảnh là Bên Phải bệnh nhân (R), "
        "bên phải ảnh là Bên Trái bệnh nhân (L). Bấm vào ảnh để mở ảnh lớn."
    )
    disclaimer_en = (
        "<strong>Review-Aid Representative Slices:</strong> "
        "The slices below are automatically sampled from preferred sequences to support study review. "
        "They are <em>not</em> proof, confirmation, segmentation, or ground-truth lesion localization. "
        "Standard radiological view: patient Right on image left (R), patient Left on image right (L). Click image to enlarge."
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
        seq_groups = entry.get("sequence_groups", [])

        # Backwards compatibility for single-group format
        if not seq_groups and "thumbnails" in entry:
            seq_groups = [{
                "family_name": "",
                "series_description": entry.get("series_description", ""),
                "thumbnails": entry.get("thumbnails", []),
            }]

        html_parts.append('<div class="evidence-item">')
        html_parts.append(f'<h3>{html.escape(label)}</h3>')

        if seq_groups:
            for grp in seq_groups:
                fam = grp.get("family_name", "")
                s_desc = grp.get("series_description", "")
                thumbs = grp.get("thumbnails", [])

                meta_parts = []
                if fam:
                    meta_parts.append(f'<strong>{html.escape(fam)}</strong>')
                if s_desc:
                    meta_parts.append(f'<span>({html.escape(s_desc)})</span>')
                meta_line = " — ".join(meta_parts) if meta_parts else ""

                if meta_line:
                    html_parts.append(f'<p class="evidence-seq-meta">{meta_line}</p>')

                if thumbs:
                    html_parts.append('<div class="evidence-grid">')
                    for thumb in thumbs:
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
                            f'<div class="thumb-container">'
                            f'<span class="marker marker-r" title="Bên phải bệnh nhân (Patient Right)">R</span>'
                            f'<span class="marker marker-l" title="Bên trái bệnh nhân (Patient Left)">L</span>'
                            f'<a href="{rel_path}" target="_blank" title="Bấm để mở ảnh lớn">'
                            f'<img src="{rel_path}" alt="{html.escape(caption)}" loading="lazy" />'
                            f'</a>'
                            f'</div>'
                            f'<span class="thumb-caption">{html.escape(caption)}</span>'
                            f'</div>'
                        )
                    html_parts.append('</div>')
        else:
            no_series_msg = (
                "Không tìm thấy chuỗi xung phù hợp trong ca chụp để trích xuất ảnh tiêu biểu cho dấu hiệu này."
                if language == "vi"
                else "No suitable series found in this study for representative slice extraction."
            )
            html_parts.append(f'<p class="evidence-none"><em>{html.escape(no_series_msg)}</em></p>')

        html_parts.append('</div>')

    html_parts.append('</section>')
    return "\n".join(html_parts)
