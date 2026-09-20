"""Modern single-study end-to-end inference pipeline for PRIMA.

The runtime keeps the research/training code intact while adding a memory-aware
path intended for workstation GPUs (notably 8 GB Ada cards):

1. Stream DICOM series one at a time when requested.
2. Tokenize with VQ-VAE on GPU in bounded chunks.
3. Release VQ-VAE before PRIMA visual inference.
4. Keep the full checkpoint/task heads in CPU RAM in low-VRAM mode.
5. Move only the visual backbone to CUDA, optionally compile it.
6. Run diagnosis/referral/priority heads on CPU, optionally INT8 quantized.
7. Persist predictions plus runtime metrics for regression/benchmarking.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import SimpleITK as sitk
import torch
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Prima_training_and_evaluation.patchify import MedicalImagePatchifier
from tools.DicomUtils import DicomUtils
from tools.models import ModelLoader
from tools.mrcommondataset import MrVoxelDataset
from tools.utilities import chartovec, filtercoords


@dataclass
class PipelineConfig:
    """Runtime configuration.

    Defaults preserve the historical path. The supplied modern 4060 profile
    enables streaming + low-VRAM placement explicitly.
    """

    study_dir: str
    output_dir: str
    tokenizer_model_config: str
    prima_model_config: str
    study_description: str

    batch_size: int = 1
    num_workers: int = 0
    max_tokens_per_chunk: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    stream_dicom: bool = False
    low_vram: bool = False
    visual_dtype: str = "float16"
    quantize_cpu_heads: bool = False
    compile_visual: bool = False
    compile_mode: str = "default"
    attention_backend: str = "auto"

    otsu_percentage: int = 5
    log_cuda_memory: bool = True
    save_runtime_metrics: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        required = [
            "study_dir",
            "output_dir",
            "tokenizer_model_config",
            "prima_model_config",
            "study_description",
        ]
        missing = [key for key in required if key not in config_dict]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
        cfg = cls(**config_dict)
        if cfg.batch_size != 1:
            raise ValueError("Modern inference runtime currently supports batch_size=1")
        if cfg.attention_backend not in {"auto", "native", "flash", "sdpa"}:
            raise ValueError(
                "attention_backend must be one of: auto, native, flash, sdpa"
            )
        return cfg


class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = PipelineConfig.from_dict(config)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        self.logger.info("Initializing pipeline with config: %s", self.config)
        os.environ["PRIMA_ATTENTION_BACKEND"] = self.config.attention_backend

        self.tokenizer_model: Optional[torch.nn.Module] = None
        self.prima_model: Optional[torch.nn.Module] = None
        self.patchifier = MedicalImagePatchifier(in_dim=256)

        self._process = psutil.Process()
        self.metrics: Dict[str, Any] = {
            "config": asdict(self.config),
            "stages": {},
            "series": [],
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            self.metrics["gpu_name"] = torch.cuda.get_device_name(
                torch.device(self.config.device)
            )

    def _setup_logging(self) -> None:
        log_file = self.output_dir / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _device(self) -> torch.device:
        return torch.device(self.config.device)

    def _rss_gib(self) -> float:
        return self._process.memory_info().rss / (1024**3)

    def _cuda_snapshot(self) -> Dict[str, float]:
        if not torch.cuda.is_available() or self._device().type != "cuda":
            return {}
        dev = self._device()
        return {
            "allocated_gib": torch.cuda.memory_allocated(dev) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(dev) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(dev) / (1024**3),
        }

    def _log_memory(self, label: str) -> None:
        snap = self._cuda_snapshot()
        if snap and self.config.log_cuda_memory:
            self.logger.info(
                "CUDA memory [%s]: allocated=%.2f GiB reserved=%.2f GiB peak=%.2f GiB",
                label,
                snap["allocated_gib"],
                snap["reserved_gib"],
                snap["peak_allocated_gib"],
            )
        self.logger.info("CPU RSS [%s]: %.2f GiB", label, self._rss_gib())

    def _stage_done(self, name: str, started: float) -> None:
        entry: Dict[str, Any] = {
            "seconds": time.perf_counter() - started,
            "cpu_rss_gib": self._rss_gib(),
        }
        entry.update(self._cuda_snapshot())
        self.metrics["stages"][name] = entry
        self.logger.info("Stage %s finished in %.2fs", name, entry["seconds"])

    def _reset_cuda_peak(self) -> None:
        if torch.cuda.is_available() and self._device().type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device())

    def _release_tokenizer(self) -> None:
        if self.tokenizer_model is not None:
            try:
                self.tokenizer_model.cpu()
            except Exception:
                pass
            del self.tokenizer_model
            self.tokenizer_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _cleanup(self) -> None:
        self.logger.info("Cleaning up resources...")
        self._release_tokenizer()
        if self.prima_model is not None:
            del self.prima_model
            self.prima_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _load_config_file_or_dict(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, str):
            return value
        p = Path(value)
        with open(p, "r") as f:
            return yaml.safe_load(f) if p.suffix in (".yaml", ".yml") else json.load(f)

    def load_mri_study(self) -> Tuple[List[sitk.Image], List[str]]:
        self.logger.info("Loading MRI study into RAM")
        mri_study, series_list = DicomUtils.load_mri_study(self.config.study_dir)
        self.logger.info("Loaded %d series", len(mri_study))
        return mri_study, series_list

    def load_tokenizer_model(self) -> torch.nn.Module:
        if self.tokenizer_model is None:
            started = time.perf_counter()
            tokenizer_config = self._load_config_file_or_dict(
                self.config.tokenizer_model_config
            )
            self.tokenizer_model = ModelLoader.load_vqvae_model(tokenizer_config)
            self.tokenizer_model = self.tokenizer_model.to(self._device()).eval()
            self._stage_done("load_tokenizer", started)
            self._log_memory("VQ-VAE loaded")
        return self.tokenizer_model

    def load_full_prima_model(self) -> torch.nn.Module:
        if self.prima_model is not None:
            return self.prima_model

        started = time.perf_counter()
        if isinstance(self.config.prima_model_config, str):
            config_path = Path(self.config.prima_model_config)
            prima_config = self._load_config_file_or_dict(
                self.config.prima_model_config
            )
            config_dir = config_path.resolve().parent
            if "full_model_ckpt" in prima_config:
                p = Path(prima_config["full_model_ckpt"])
                if not p.is_absolute():
                    prima_config = {
                        **prima_config,
                        "full_model_ckpt": str(config_dir / p),
                    }
        else:
            prima_config = self.config.prima_model_config

        self.prima_model = ModelLoader.load_full_prima_model(
            prima_config,
            device=self.config.device,
            low_vram=self.config.low_vram,
            visual_dtype=self.config.visual_dtype,
            quantize_cpu_heads=self.config.quantize_cpu_heads,
            compile_visual=self.config.compile_visual,
            compile_mode=self.config.compile_mode,
        )
        self.prima_model.eval()
        self._stage_done("load_prima", started)
        self._log_memory("PRIMA ready")
        return self.prima_model

    def _encode_tokens(
        self, tokens: torch.Tensor, vqvae: torch.nn.Module
    ) -> torch.Tensor:
        if tokens.numel() == 0:
            raise RuntimeError("No tokens found")
        if tokens.shape[0] > 5000:
            raise RuntimeError(f"Too many tokens: {tokens.shape[0]} > 5000")

        embeddings: List[torch.Tensor] = []
        amp_enabled = self._device().type == "cuda"
        for start in range(0, tokens.shape[0], self.config.max_tokens_per_chunk):
            stop = min(start + self.config.max_tokens_per_chunk, tokens.shape[0])
            chunk = tokens[start:stop].unsqueeze(1).to(
                self._device(), non_blocking=True
            )
            with torch.inference_mode():
                if amp_enabled:
                    with torch.amp.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        emb = vqvae.encode(chunk)
                else:
                    emb = vqvae.encode(chunk)
            embeddings.append(emb.detach().cpu())
            del chunk, emb
        return torch.cat(embeddings, dim=0)

    def _tokenize_series(
        self,
        image: sitk.Image,
        series_name: str,
        vqvae: torch.nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        started = time.perf_counter()
        dataset = MrVoxelDataset([image])
        tokens, meta = dataset[0]
        embedding = self._encode_tokens(tokens, vqvae)
        elapsed = time.perf_counter() - started

        self.metrics["series"].append(
            {
                "name": series_name,
                "tokens": int(tokens.shape[0]),
                "seconds": elapsed,
                "cpu_rss_gib": self._rss_gib(),
                **self._cuda_snapshot(),
            }
        )
        self.logger.info(
            "Tokenized series %s: %d tokens in %.2fs",
            series_name,
            tokens.shape[0],
            elapsed,
        )
        del tokens
        return embedding, meta

    def run_tokenizer_model(
        self,
        mri_study: List[sitk.Image],
        series_names: Optional[List[str]] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[str]], List[Dict[str, Any]]]:
        self.logger.info("Running tokenizer over in-memory study")
        self._reset_cuda_peak()
        started = time.perf_counter()
        vqvae = self.load_tokenizer_model()

        series_embeddings: List[torch.Tensor] = []
        filtered_names: Optional[List[str]] = [] if series_names is not None else None
        all_meta: List[Dict[str, Any]] = []

        try:
            for idx, image in enumerate(mri_study):
                name = series_names[idx] if series_names is not None else f"series_{idx}"
                try:
                    emb, meta = self._tokenize_series(image, name, vqvae)
                except Exception as exc:
                    self.logger.warning(
                        "Skipping series index=%s name=%s: %s",
                        idx,
                        name,
                        exc,
                        exc_info=True,
                    )
                    continue
                series_embeddings.append(emb)
                all_meta.append(meta)
                if filtered_names is not None:
                    filtered_names.append(name)
        finally:
            self._log_memory("VQ-VAE finished")
            self._release_tokenizer()
            self._log_memory("VQ-VAE unloaded")

        self._stage_done("tokenizer_total", started)
        return series_embeddings, filtered_names, all_meta

    def run_streaming_tokenizer_study(
        self,
    ) -> Tuple[List[torch.Tensor], List[str], List[Dict[str, Any]]]:
        self.logger.info("Streaming DICOM series through tokenizer")
        self._reset_cuda_peak()
        started = time.perf_counter()
        vqvae = self.load_tokenizer_model()

        embeddings: List[torch.Tensor] = []
        names: List[str] = []
        all_meta: List[Dict[str, Any]] = []

        try:
            for idx, (image, name, _source) in enumerate(
                DicomUtils.iter_mri_study(self.config.study_dir)
            ):
                try:
                    emb, meta = self._tokenize_series(image, name, vqvae)
                except Exception as exc:
                    self.logger.warning(
                        "Skipping streamed series index=%s name=%s: %s",
                        idx,
                        name,
                        exc,
                        exc_info=True,
                    )
                    continue
                embeddings.append(emb)
                names.append(name)
                all_meta.append(meta)
                del image
                gc.collect()
        finally:
            self._log_memory("VQ-VAE finished")
            self._release_tokenizer()
            self._log_memory("VQ-VAE unloaded")

        self._stage_done("tokenizer_total", started)
        return embeddings, names, all_meta

    def prepare_prima_input(
        self,
        series_embeddings: List[torch.Tensor],
        series_names: List[str],
        all_ser_emb_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not series_embeddings:
            raise RuntimeError("No tokenized series available")
        if len(series_embeddings) != len(series_names):
            raise ValueError("series_embeddings and series_names are out of sync")

        coords = None
        if all_ser_emb_meta is not None:
            coords = []
            filtered_embeddings = []
            for i, meta in enumerate(all_ser_emb_meta):
                chosen = None
                for percent in range(self.config.otsu_percentage, -1, -1):
                    embs, embspos, _ = filtercoords(
                        meta, percent, series_embeddings[i]
                    )
                    self.logger.info(
                        "Otsu series=%s threshold=%d before=%d after=%d",
                        series_names[i],
                        percent,
                        len(series_embeddings[i]),
                        len(embs),
                    )
                    chosen = (embs, embspos)
                    if len(embspos) > 25:
                        break
                if chosen is None:
                    raise RuntimeError(f"Could not filter series {series_names[i]}")
                filtered_embeddings.append(chosen[0])
                coords.append(chosen[1])
            series_embeddings = filtered_embeddings

        study_lens = torch.tensor([len(series_embeddings)], dtype=torch.long)
        serie_lenss = torch.tensor(
            [len(v) for v in series_embeddings], dtype=torch.long
        ).unsqueeze(0)

        patched = self.patchifier(series_embeddings, coords=coords)
        max_len = int(serie_lenss.max().item())
        visuals: List[torch.Tensor] = []
        for img in patched:
            pad_len = max_len - len(img)
            if pad_len:
                img = torch.cat(
                    [
                        img,
                        torch.zeros(
                            (pad_len, *img.shape[1:]),
                            dtype=img.dtype,
                        ),
                    ],
                    dim=0,
                )
            visuals.append(img.unsqueeze(0))

        series_name_tensors = [chartovec(name) for name in series_names]
        max_name_len = max(len(t) for t in series_name_tensors)
        serienames_tensor = torch.zeros(
            len(series_name_tensors), max_name_len, dtype=torch.long
        )
        for i, tensor in enumerate(series_name_tensors):
            serienames_tensor[i, : len(tensor)] = tensor

        return {
            "visual": visuals,
            "lens": study_lens,
            "lenss": serie_lenss,
            "hash": ["study_0"],
            "serienames": serienames_tensor.unsqueeze(0),
            "studydescription": chartovec(self.config.study_description).unsqueeze(0),
        }

    @staticmethod
    def _move_to_device(obj: Any, device: torch.device) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, list):
            return [Pipeline._move_to_device(item, device) for item in obj]
        if isinstance(obj, dict):
            return {k: Pipeline._move_to_device(v, device) for k, v in obj.items()}
        return obj

    @staticmethod
    def _serializable(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {k: Pipeline._serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Pipeline._serializable(item) for item in obj]
        return obj

    def run_prima_model(
        self,
        series_embeddings: List[torch.Tensor],
        series_names: List[str],
        all_ser_emb_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.logger.info("Running PRIMA visual encoder + task heads")
        self._release_tokenizer()
        self._reset_cuda_peak()
        started = time.perf_counter()

        prima_input = self.prepare_prima_input(
            series_embeddings=series_embeddings,
            series_names=series_names,
            all_ser_emb_meta=all_ser_emb_meta,
        )
        prima_input = self._move_to_device(prima_input, self._device())
        model = self.load_full_prima_model()

        try:
            with torch.inference_mode():
                if self._device().type == "cuda":
                    with torch.amp.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        predictions = model(
                            prima_input,
                            inference_only_once=True,
                            heads_on_cpu=self.config.low_vram,
                        )
                else:
                    predictions = model(
                        prima_input,
                        inference_only_once=True,
                        heads_on_cpu=False,
                    )

            self._stage_done("prima_inference", started)
            self._log_memory("PRIMA inference complete")

            study_id = Path(self.config.study_dir).name or "study"
            output_path = (
                self.output_dir / f"{study_id}_predictions.json"
            ).resolve()
            with open(output_path, "w") as f:
                json.dump(self._serializable(predictions), f, indent=2)
            self.logger.info("Predictions saved to %s", output_path)
            return predictions
        finally:
            del prima_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_metrics(self) -> None:
        if not self.config.save_runtime_metrics:
            return
        path = self.output_dir / "runtime_metrics.json"
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info("Runtime metrics saved to %s", path)

    def run(self) -> Dict[str, Any]:
        total_started = time.perf_counter()
        try:
            if self.config.stream_dicom:
                embeddings, names, meta = self.run_streaming_tokenizer_study()
            else:
                mri_study, names = self.load_mri_study()
                embeddings, filtered_names, meta = self.run_tokenizer_model(
                    mri_study, series_names=names
                )
                if filtered_names is not None:
                    names = filtered_names

            if not embeddings:
                raise RuntimeError(
                    "No series could be tokenized; check pipeline.log for details"
                )

            predictions = self.run_prima_model(
                series_embeddings=embeddings,
                series_names=names,
                all_ser_emb_meta=meta,
            )
            self._stage_done("total", total_started)
            return predictions
        finally:
            self.save_metrics()
            self._cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="PRIMA end-to-end inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = (
            yaml.safe_load(f)
            if config_path.suffix in (".yaml", ".yml")
            else json.load(f)
        )

    pipeline = Pipeline(config)
    pipeline.logger.info("Starting pipeline execution")
    pipeline.run()
    pipeline.logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()
