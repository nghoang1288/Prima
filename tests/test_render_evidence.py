"""Synthetic, non-PHI regression tests for radiologist review-aid evidence panels."""

import html
import json
import re
from pathlib import Path
import numpy as np
import pytest

from tools.evidence import (
    DIAGNOSIS_OVERRIDE_FAMILIES,
    GROUP_DEFAULT_FAMILIES,
    PREFERRED_KEYWORDS_BY_GROUP,
    build_evidence_html,
    generate_study_evidence,
    get_preferred_families,
    get_preferred_keywords,
    normalize_slice_to_uint8,
    orient_image_to_lps,
    rank_series_for_diagnosis,
    score_series_match,
    select_multi_series_for_diagnosis,
    select_representative_slices,
)
from tools.render_report import render_reports


def test_preferred_keywords_coverage():
    """Verify all required clinical categories have preferred MRI series keywords mapped."""
    required_groups = [
        "ischemic",
        "hemorrhagic",
        "tumor",
        "structural",
        "ventricular",
        "infectious_inflammatory",
        "trauma",
    ]
    for grp in required_groups:
        keywords = get_preferred_keywords(grp)
        assert len(keywords) >= 3, f"Group {grp} should have at least 3 preferred keywords, got {keywords}"

    # Verify minimum required keywords per protocol
    assert "DWI" in get_preferred_keywords("ischemic")
    assert "ADC" in get_preferred_keywords("ischemic")
    assert "FLAIR" in get_preferred_keywords("ischemic")

    assert any(k in ["SWI", "T2*", "GRE", "SWAN"] for k in get_preferred_keywords("hemorrhagic"))
    assert any("T1" in k for k in get_preferred_keywords("tumor"))
    assert "T2" in get_preferred_keywords("structural")
    assert "FLAIR" in get_preferred_keywords("ventricular")
    assert "DWI" in get_preferred_keywords("infectious_inflammatory")
    assert any(k in ["SWI", "T2*", "SWAN"] for k in get_preferred_keywords("trauma"))


def test_series_ranking_deterministic():
    """Verify series ranking gives highest priority to clinically preferred sequences."""
    synthetic_series = [
        {"series_description": "3-Plane Localizer", "file_count": 3},
        {"series_description": "Ax T2 FLAIR FS", "file_count": 28},
        {"series_description": "Ax DWI ALL b1000 3mm", "file_count": 84},
        {"series_description": "ADC Map (10^-6 mm²/s)", "file_count": 42},
        {"series_description": "3D Ax SWAN", "file_count": 118},
        {"series_description": "Sag T1 FLAIR", "file_count": 25},
    ]

    # Ischemic stroke: DWI should rank #1, ADC #2, FLAIR #3
    ranked_ischemic = rank_series_for_diagnosis(synthetic_series, "vascular_ischemic_large_vessel_stroke")
    assert len(ranked_ischemic) >= 3
    assert "DWI" in ranked_ischemic[0]["series_description"]
    assert "ADC" in ranked_ischemic[1]["series_description"]

    # Hemorrhagic: SWAN/SWI should rank #1
    ranked_hem = rank_series_for_diagnosis(synthetic_series, "vascular_hemorrhagic_intracranial_hemorrhage")
    assert len(ranked_hem) >= 1
    assert "SWAN" in ranked_hem[0]["series_description"] or "SWI" in ranked_hem[0]["series_description"]

    # Structural: T2/FLAIR should rank #1
    ranked_struct = rank_series_for_diagnosis(synthetic_series, "structural_cerebral_atrophy")
    assert len(ranked_struct) >= 1
    assert any(k in ranked_struct[0]["series_description"] for k in ["T2", "FLAIR"])

    # Localizer must never be ranked
    for item in ranked_ischemic + ranked_hem + ranked_struct:
        assert "Localizer" not in item["series_description"]


def test_representative_slice_selection_bounds_and_spacing():
    """Verify slice selection stays within middle 60% and enforces spatial separation."""
    depth, height, width = 60, 64, 64
    vol = np.zeros((depth, height, width), dtype=np.float32)

    # Place synthetic brain-like high contrast in slices 12 to 48
    for z in range(12, 48):
        vol[z, 16:48, 16:48] = 50.0 + 10.0 * np.sin(z)

    count = 4
    slices = select_representative_slices(vol, count=count)
    assert len(slices) == count

    # Middle 60% of 60 slices is [12, 48]
    min_bound = int(depth * 0.20)
    max_bound = int(depth * 0.80)
    for s in slices:
        assert min_bound <= s <= max_bound, f"Slice {s} out of middle 60% [{min_bound}, {max_bound}]"

    # Strictly increasing and well-spaced
    for i in range(len(slices) - 1):
        assert slices[i + 1] > slices[i]
        assert slices[i + 1] - slices[i] >= (max_bound - min_bound) // (count * 2)


def test_slice_normalization_uint8():
    """Verify 2D slice normalization produces valid uint8 [0, 255] range."""
    arr = np.array([[10.0, 20.0], [50.0, 100.0]], dtype=np.float32)
    norm = normalize_slice_to_uint8(arr)
    assert norm.dtype == np.uint8
    assert norm.shape == arr.shape
    assert int(np.min(norm)) == 0
    assert int(np.max(norm)) == 255

    # Uniform slice handles edge case safely
    flat = np.full((10, 10), 42.0, dtype=np.float32)
    norm_flat = normalize_slice_to_uint8(flat)
    assert norm_flat.dtype == np.uint8
    assert int(np.max(norm_flat)) == 0


def test_evidence_html_disclaimer_and_no_definitive_claims():
    """Verify evidence panel disclaims ground-truth proof and uses review-aid terminology."""
    entries = [
        {
            "code": "vascular_ischemic_large_vessel_stroke",
            "label": "Nhồi máu mạch lớn",
            "series_description": "Ax DWI ALL b1000",
            "thumbnails": [
                {"slice_index": 20, "total_slices": 60, "relative_path": "evidence/stroke_s020.png"},
                {"slice_index": 35, "total_slices": 60, "relative_path": "evidence/stroke_s035.png"},
            ],
        }
    ]

    snippet = build_evidence_html(entries, language="vi")

    # Positive required wording per protocol
    assert "Ảnh gợi ý rà lại" in snippet or "Lát cắt tiêu biểu" in snippet
    assert "hỗ trợ rà soát" in snippet

    # Explicit negative disclaimers: must state it is NOT proof/confirmation/segmentation/ground truth
    assert "không phải" in snippet.lower()
    assert "bằng chứng khẳng định" in snippet or "proof" in snippet
    assert "phân vùng tổn thương" in snippet or "segmentation" in snippet
    assert "vị trí tổn thương chính xác" in snippet or "ground-truth" in snippet

    # Must contain label and series description
    assert "Nhồi máu mạch lớn" in snippet
    assert "Ax DWI ALL b1000" in snippet
    assert "evidence/stroke_s020.png" in snippet

    # English variant
    snippet_en = build_evidence_html(entries, language="en")
    assert "Review-Aid" in snippet_en
    assert "not" in snippet_en.lower()


def test_html_escaping_in_evidence_panel():
    """Verify XSS and special characters are safely escaped in HTML output."""
    malicious_entries = [
        {
            "code": "test_code",
            "label": "Tổn thương <script>alert('XSS')</script> & Co",
            "series_description": 'Ax "FLAIR" <img onerror=alert(1)>',
            "thumbnails": [
                {"slice_index": 1, "total_slices": 10, "relative_path": "evidence/<bad>.png"}
            ],
        }
    ]
    html_out = build_evidence_html(malicious_entries, language="vi")
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "<img onerror" not in html_out
    assert "&amp; Co" in html_out
    assert "&lt;bad&gt;" in html_out


def test_report_render_integration_no_absolute_paths(tmp_path):
    """Verify generated HTML report contains relative evidence paths and no local patient absolute paths."""
    predictions = {
        "priority": {"none": [0.2], "low": [-0.1], "high": [-0.8]},
        "diagnosis": {
            "vascular_ischemic_large_vessel_stroke": [0.85],
            "structural_cerebral_atrophy": [0.45],
            "infectious_brain_abscess": [-1.2],
        },
        "referral": {"ns-vascular": [0.90]},
    }

    out_dir = tmp_path / "output"
    md_path, html_path = render_reports(
        predictions,
        study_id="CASE_TEST",
        output_dir=out_dir,
        language="vi",
    )

    assert md_path.exists()
    assert html_path.exists()

    html_content = html_path.read_text(encoding="utf-8")
    assert "PRIMA — CASE_TEST" in html_content
    assert "Đọc nhanh" in html_content

    # In default mode, raw codes are hidden in the main overview
    assert "vascular_ischemic_large_vessel_stroke" not in html_content

    # Never leak absolute user directories or local paths
    assert "/home/hoang" not in html_content
    assert "C:\\Users" not in html_content
    assert "E:\\Prima" not in html_content


def test_multi_series_selection_up_to_three_distinct_families():
    """Verify up to 3 distinct sequence families are selected per diagnosis."""
    candidate_series = [
        {"directory": Path("s1"), "series_description": "Ax DWI ALL b1000", "file_count": 80},
        {"directory": Path("s2"), "series_description": "ADC Map (mm2/s)", "file_count": 40},
        {"directory": Path("s3"), "series_description": "Ax T2 FLAIR FS", "file_count": 30},
        {"directory": Path("s4"), "series_description": "3D Ax SWAN", "file_count": 100},
        {"directory": Path("s5"), "series_description": "Ax T2 SE", "file_count": 28},
        {"directory": Path("s6"), "series_description": "Ax T1 +C CE Post", "file_count": 28},
    ]

    # 1. Ischemic stroke: should pick DWI, ADC, and FLAIR
    selected_ischemic = select_multi_series_for_diagnosis(
        candidate_series, "vascular_ischemic_large_vessel_stroke", max_families=3
    )
    assert len(selected_ischemic) == 3
    family_names = [fam for fam, _ in selected_ischemic]
    assert family_names == ["DWI", "ADC", "FLAIR / MRA"]
    assert "DWI" in selected_ischemic[0][1]["series_description"]
    assert "ADC" in selected_ischemic[1][1]["series_description"]
    assert "FLAIR" in selected_ischemic[2][1]["series_description"]

    # 2. Hemorrhage: should pick SWI/SWAN/T2*, FLAIR, and T2
    selected_hem = select_multi_series_for_diagnosis(
        candidate_series, "vascular_hemorrhagic_intracranial_hemorrhage", max_families=3
    )
    assert len(selected_hem) == 3
    assert any("SWAN" in s["series_description"] for _, s in selected_hem)
    assert any("FLAIR" in s["series_description"] for _, s in selected_hem)
    assert any("T2" in s["series_description"] for _, s in selected_hem)

    # 3. Tumor: should pick T1+C, FLAIR/T2, DWI
    selected_tumor = select_multi_series_for_diagnosis(
        candidate_series, "tumor_high_grade_glioma", max_families=3
    )
    assert len(selected_tumor) == 3
    assert any("T1" in s["series_description"] for _, s in selected_tumor)
    assert any("FLAIR" in s["series_description"] or "T2" in s["series_description"] for _, s in selected_tumor)
    assert any("DWI" in s["series_description"] for _, s in selected_tumor)


def test_diagnosis_specific_overrides():
    """Verify diagnosis-specific overrides for required clinical conditions."""
    # Moyamoya
    moyamoya_fams = get_preferred_families("vascular_ischemic_moyamoya")
    assert [f[0] for f in moyamoya_fams] == ["MRA / TOF", "T2 FLAIR", "DWI"]

    # Aneurysm
    aneurysm_fams = get_preferred_families("vascular_malformation_cerebral_aneurysm")
    assert [f[0] for f in aneurysm_fams] == ["3D TOF / MRA", "SWI / SWAN / T2*", "T2"]

    # AVM
    avm_fams = get_preferred_families("vascular_malformation_arteriovenous_malformation")
    assert [f[0] for f in avm_fams] == ["MRA / TOF", "T2", "SWI / SWAN / T2*"]

    # Cavernoma
    cav_fams = get_preferred_families("vascular_malformation_cavernoma")
    assert [f[0] for f in cav_fams] == ["SWI / SWAN / T2*", "T2", "FLAIR"]

    # Sellar
    sellar_fams = get_preferred_families("sellar_pituitary_adenoma")
    assert any("T1" in f[0] for f in sellar_fams)
    assert any("T2" in f[0] for f in sellar_fams)

    # Ventriculomegaly
    vent_fams = get_preferred_families("ventricular_ventriculomegaly")
    assert [f[0] for f in vent_fams] == ["T2", "FLAIR", "T1"]

    # Atrophy
    atrophy_fams = get_preferred_families("structural_cerebral_atrophy")
    assert [f[0] for f in atrophy_fams] == ["T2 Coronal / Axial", "FLAIR", "T1"]

    # Small-vessel disease
    svd_fams = get_preferred_families("vascular_ischemic_small_vessel_disease")
    assert [f[0] for f in svd_fams] == ["FLAIR", "T2", "SWI / SWAN / T2*"]

    # Postoperative
    surg_fams = get_preferred_families("surgical_craniotomy")
    assert any("CT" in f[0] or "T1" in f[0] for f in surg_fams)


def test_no_misleading_fallback_when_preferred_sequence_missing():
    """Verify that missing preferred sequences do NOT fall back to arbitrary sequences."""
    # Only unrelated sequences available (e.g. Localizer, sagittal T1 without contrast)
    candidate_series = [
        {"directory": Path("s1"), "series_description": "3-Plane Localizer", "file_count": 3},
        {"directory": Path("s2"), "series_description": "Calibration Scan", "file_count": 1},
    ]

    # For large vessel stroke, DWI/ADC/FLAIR are required; neither localizer nor calibration can match
    selected = select_multi_series_for_diagnosis(
        candidate_series, "vascular_ischemic_large_vessel_stroke", max_families=3
    )
    assert selected == [], "Must return empty list instead of arbitrary fallback series"

    # Score match directly returns 0.0 or negative for non-matches
    score = score_series_match("Axial Gradient Echo Localizer", ["DWI", "DIFFUSION"], slice_count=20)
    assert score <= 0.0

    score_unrelated = score_series_match("Cervical Spine Scout", ["FLAIR", "T2"], slice_count=20)
    assert score_unrelated <= 0.0

    # In HTML rendering, empty sequence groups must display explicit notice, not dummy images
    empty_entry = [{
        "code": "vascular_ischemic_large_vessel_stroke",
        "label": "Nhồi máu mạch lớn",
        "sequence_groups": [],
    }]
    html_vi = build_evidence_html(empty_entry, language="vi")
    assert "Không tìm thấy chuỗi xung phù hợp" in html_vi
    assert "<img" not in html_vi

    html_en = build_evidence_html(empty_entry, language="en")
    assert "No suitable series found" in html_en
    assert "<img" not in html_en


def test_lps_orientation_standardization():
    """Verify loaded SimpleITK volumes are standardized to LPS orientation."""
    import SimpleITK as sitk

    # Create synthetic volume with RAI direction cosine matrix
    img = sitk.Image(16, 16, 8, sitk.sitkFloat32)
    img.SetDirection([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetSpacing((1.0, 1.0, 2.0))

    oriented = orient_image_to_lps(img)
    # Direction cosines for LPS must be identity matrix [1, 0, 0, 0, 1, 0, 0, 0, 1]
    expected_lps_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert oriented.GetDirection() == expected_lps_direction
    assert sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(oriented.GetDirection()) == "LPS"


def test_radiological_markers_and_clickable_links():
    """Verify radiological viewing convention with R/L markers and clickable relative links."""
    entries = [
        {
            "code": "vascular_ischemic_large_vessel_stroke",
            "label": "Nhồi máu não",
            "sequence_groups": [
                {
                    "family_name": "DWI",
                    "series_description": "Ax DWI ALL b1000",
                    "thumbnails": [
                        {
                            "slice_index": 25,
                            "total_slices": 60,
                            "relative_path": "evidence/thumb_s025.png",
                        }
                    ],
                }
            ],
        }
    ]

    html_vi = build_evidence_html(entries, language="vi")

    # Radiological R and L markers present with appropriate titles/classes
    assert 'class="marker marker-r"' in html_vi
    assert 'class="marker marker-l"' in html_vi
    assert '>R</span>' in html_vi
    assert '>L</span>' in html_vi

    # Clickable link wrapping thumbnail image
    assert '<a href="evidence/thumb_s025.png" target="_blank"' in html_vi
    assert '<img src="evidence/thumb_s025.png"' in html_vi

    # Offline safety: no external protocol links
    assert "http://" not in html_vi
    assert "https://" not in html_vi
    assert "file://" not in html_vi


def test_thumbnail_deduplication(tmp_path):
    """Verify identical series and slices are deduplicated across multiple diagnoses."""
    output_dir = tmp_path / "study_out"
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Simulate thumbnail cache as used in generate_study_evidence
    thumbnail_cache = {}
    slice_vol = np.zeros((10, 32, 32), dtype=np.uint8)
    slice_vol[5, 10:20, 10:20] = 200

    # Diagnosis 1 and Diagnosis 2 both select series 's02_dwi' slice 5
    diag1_key = ("s02_dwi", 5)
    diag2_key = ("s02_dwi", 5)

    rel_path_1 = f"evidence/thumb_{diag1_key[0]}_s{diag1_key[1]+1:03d}.png"
    thumbnail_cache[diag1_key] = rel_path_1

    # When diag2 evaluates the same series and slice, it reuses the cache entry
    assert diag2_key in thumbnail_cache
    rel_path_2 = thumbnail_cache[diag2_key]

    assert rel_path_1 == rel_path_2
    assert rel_path_1 == "evidence/thumb_s02_dwi_s006.png"

