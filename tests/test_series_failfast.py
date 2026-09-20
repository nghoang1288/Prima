import importlib.util
import logging
from pathlib import Path

import pytest


PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "end-to-end_inference_pipeline"
    / "pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("prima_series_pipeline", PIPELINE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_pipeline(tmp_path, fail_on_series_error):
    pipeline = MODULE.Pipeline.__new__(MODULE.Pipeline)
    pipeline.config = MODULE.PipelineConfig.from_dict(
        {
            "study_dir": str(tmp_path / "CASE001"),
            "output_dir": str(tmp_path / "output"),
            "tokenizer_model_config": "tokenizer.json",
            "prima_model_config": "prima.json",
            "study_description": "MRI BRAIN",
            "device": "cpu",
            "fail_on_series_error": fail_on_series_error,
        }
    )
    pipeline.metrics = {"skipped_series": []}
    pipeline.logger = logging.getLogger("test-series-failfast")
    pipeline._reset_cuda_peak = lambda: None
    pipeline.load_tokenizer_model = lambda: object()
    pipeline._log_memory = lambda label: None
    pipeline._release_tokenizer = lambda: None
    pipeline._stage_done = lambda name, started: None

    def fail(*args, **kwargs):
        raise ValueError("synthetic series failure")

    pipeline._tokenize_series = fail
    return pipeline


def test_failfast_series_error_aborts_validation(tmp_path):
    pipeline = make_pipeline(tmp_path, True)
    with pytest.raises(RuntimeError, match="fail_on_series_error=true"):
        pipeline.run_tokenizer_model([object()], series_names=["T2"])


def test_legacy_skip_mode_records_skipped_series(tmp_path):
    pipeline = make_pipeline(tmp_path, False)
    embeddings, names, meta = pipeline.run_tokenizer_model(
        [object()], series_names=["T2"]
    )

    assert embeddings == []
    assert names == []
    assert meta == []
    assert len(pipeline.metrics["skipped_series"]) == 1
