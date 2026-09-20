import json
from pathlib import Path

from tools.render_report import (DIAGNOSIS_LABELS_VI, REFERRAL_LABELS_VI, build_markdown, markdown_to_html, render_reports)


def sample_predictions():
    return {
        "diagnosis": {
            "tumor_adult_glioma": [0.42],
            "vascular_hemorrhagic_intracranial_hemorrhage": [-0.10],
            "far_negative": [-2.0],
        },
        "referral": {
            "nl-stroke": [0.15],
            "nl-general": [-0.05],
        },
        "priority": {
            "high": [0.1],
            "low": [0.8],
            "none": [-0.2],
        },
        "clip_emb": [[123.456, 789.0]],
    }


def test_vietnamese_report_uses_threshold_semantics_and_priority_argmax():
    markdown = build_markdown(
        sample_predictions(),
        study_id="CASE001",
        language="vi",
        near_margin=0.25,
    )

    assert "Ưu tiên thấp" in markdown
    assert "U thần kinh đệm ở người lớn" in markdown
    assert "Xuất huyết nội sọ" in markdown
    assert "Mạch máu — xuất huyết" in markdown
    assert "Khối u" in markdown
    assert "margin `+0.420`" not in markdown
    assert "không phải xác suất bệnh" in markdown
    assert "clip_emb" not in markdown
    assert "123.456" not in markdown


def test_report_does_not_show_all_far_negative_scores_by_default():
    markdown = build_markdown(sample_predictions(), study_id="CASE001")
    assert "far_negative" not in markdown


def test_html_escapes_untrusted_model_label_text():
    predictions = sample_predictions()
    predictions["diagnosis"]["<script>alert(1)</script>"] = [0.5]
    markdown = build_markdown(predictions, study_id="CASE001")
    rendered = markdown_to_html(markdown, study_id="CASE001")

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_technical_metrics_are_rendered_without_source_paths():
    metrics = {
        "series": [{"name": "series_0000"}],
        "skipped_series": [],
        "gpu_name": "NVIDIA GeForce RTX 4060",
        "tokenizer_runtime_chunk_limit": 96,
        "stages": {
            "total": {"seconds": 37.0},
            "tokenizer_total": {"seconds": 20.0},
            "prima_inference": {"seconds": 3.0},
        },
        "config": {"study_dir": "<redacted>"},
    }
    markdown = build_markdown(
        sample_predictions(),
        study_id="CASE001",
        metrics=metrics,
    )
    assert "NVIDIA GeForce RTX 4060" in markdown
    assert "37.0 giây" in markdown
    assert "<redacted>" not in markdown


def test_vietnamese_map_covers_all_52_published_diagnoses():
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            repo_root
            / "Prima_training_and_evaluation"
            / "configs"
            / "jsons"
            / "prospective_eval.json"
        ).read_text()
    )
    assert len(config) == 52
    assert set(DIAGNOSIS_LABELS_VI) == set(config)


def test_vietnamese_map_covers_all_15_checkpoint_referrals():
    expected = {
        "ns-pediatric", "ns-skull base", "ns-general", "ns-trauma",
        "ns-tumor", "ns-vascular", "nl-pediatric", "nl-epilepsy",
        "nl-neurocritical", "nl-neuroimmunology", "nl-neurocognitive",
        "nl-oncology", "nl-trauma", "nl-stroke", "nl-general",
    }
    assert set(REFERRAL_LABELS_VI) == expected


def test_render_reports_writes_local_md_and_html(tmp_path):
    md_path, html_path = render_reports(
        sample_predictions(),
        study_id="CASE001",
        output_dir=tmp_path,
        language="vi",
    )

    assert md_path.name == "CASE001_report.md"
    assert html_path.name == "CASE001_report.html"
    assert md_path.exists()
    assert html_path.exists()
    assert "clip_emb" not in md_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html_text.lower()
    assert "CASE001" in html_text


def test_default_report_is_radiologist_first():
    markdown = build_markdown(sample_predictions(), study_id="CASE001")

    assert "## Đọc nhanh" in markdown
    assert "## Các nhãn PRIMA vượt ngưỡng" in markdown
    assert "## Các nhãn sát ngưỡng" in markdown
    assert "## Gợi ý hội chẩn / chuyên khoa" in markdown
    assert "## Giới hạn cần nhớ khi đọc kết quả" in markdown
    assert "tumor_adult_glioma" not in markdown
    assert "nl-stroke" not in markdown
    assert "margin `+0.420`" not in markdown


def test_show_all_scores_exposes_model_details_only_on_request():
    markdown = build_markdown(
        sample_predictions(),
        study_id="CASE001",
        show_all_scores=True,
    )

    assert "## Chi tiết mô hình" in markdown
    assert "tumor_adult_glioma" in markdown
    assert "margin `+0.420`" in markdown
