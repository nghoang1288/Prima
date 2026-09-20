import json
from pathlib import Path

from tools.render_report import LABELS_VI, build_markdown, markdown_to_html


def sample_predictions():
    return {
        "diagnosis": {
            "tumor_adult_glioma": [0.42],
            "intracranial_hemorrhage": [-0.10],
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
    assert "U thần kinh đệm người lớn" in markdown
    assert "Xuất huyết nội sọ" in markdown
    assert "margin `+0.420`" in markdown
    assert "không phải phần trăm xác suất bệnh" in markdown
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
    assert set(LABELS_VI) == set(config)
