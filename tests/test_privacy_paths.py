import importlib.util
from pathlib import Path


PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "end-to-end_inference_pipeline"
    / "pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("prima_runtime_pipeline", PIPELINE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_config(tmp_path, **overrides):
    data = {
        "study_dir": str(tmp_path / "NGUYEN THI HOA"),
        "output_dir": str(tmp_path / "output"),
        "tokenizer_model_config": "tokenizer.json",
        "prima_model_config": "prima.json",
        "study_description": "MRI BRAIN WO AND W CONTRAST",
    }
    data.update(overrides)
    return MODULE.PipelineConfig.from_dict(data)


def test_anonymous_fallback_id_never_uses_patient_folder_name(tmp_path):
    pipeline = MODULE.Pipeline.__new__(MODULE.Pipeline)
    pipeline.config = make_config(tmp_path)
    study_id = pipeline._resolve_study_id()

    assert study_id.startswith("study-")
    assert "NGUYEN" not in study_id.upper()
    assert "HOA" not in study_id.upper()


def test_explicit_case_id_is_used_and_source_path_redacted(tmp_path):
    pipeline = MODULE.Pipeline.__new__(MODULE.Pipeline)
    pipeline.config = make_config(tmp_path, study_id="CASE001")
    pipeline.study_id = pipeline._resolve_study_id()

    safe = pipeline._safe_config_dict()
    assert pipeline.study_id == "CASE001"
    assert safe["study_dir"] == "<redacted>"
    assert "NGUYEN THI HOA" not in str(safe)
