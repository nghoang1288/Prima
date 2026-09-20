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
import contextlib
import gc
import hashlib
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
from tools.utilities import chartovec, filtercoords, select_otsu_indices


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
    study_id: Optional[str] = None
    redact_source_path: bool = True

    batch_size: int = 1
    num_workers: int = 0
    max_tokens_per_chunk: int = 128
    min_tokens_per_chunk: int = 16
    auto_reduce_tokenizer_chunk: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_encoder_only: bool = True
    tokenizer_dtype: str = "float16"
    stream_dicom: bool = False
    low_vram: bool = False
    visual_dtype: str = "float16"
    quantize_cpu_heads: bool = False
    head_quant_backend: str = "torch_dynamic"
    prune_inference_only: bool = True
    compile_visual: bool = False
    compile_mode: str = "default"
    attention_backend: str = "auto"

    otsu_percentage: int = 5
    prefilter_otsu_before_vq: bool = False
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
        if cfg.head_quant_backend not in {"torch_dynamic", "torchao"}:
            raise ValueError(
                "head_quant_backend must be one of: torch_dynamic, torchao"
            )
        if cfg.max_tokens_per_chunk < 1 or cfg.min_tokens_per_chunk < 1:
            raise ValueError("Tokenizer chunk sizes must be >= 1")
        if cfg.min_tokens_per_chunk > cfg.max_tokens_per_chunk:
            raise ValueError(
                "min_tokens_per_chunk cannot exceed max_tokens_per_chunk"
            )
        valid_dtypes = {"float16", "fp16", "bfloat16", "bf16", "float32", "fp32"}
        if cfg.tokenizer_dtype.lower() not in valid_dtypes:
            raise ValueError(f"Unsupported tokenizer_dtype: {cfg.tokenizer_dtype}")
        if cfg.visual_dtype.lower() not in valid_dtypes:
            raise ValueError(f"Unsupported visual_dtype: {cfg.visual_dtype}")
        device = torch.device(cfg.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but PyTorch cannot see CUDA. "
                    "On Windows use WSL2 with a current NVIDIA Windows driver; "
                    "do not install a Linux NVIDIA display driver inside WSL."
                )
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {device.index} is unavailable; "
                    f"visible device count is {torch.cuda.device_count()}"
                )
        return cfg


class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = PipelineConfig.from_dict(config)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        self.study_id = self._resolve_study_id()
        safe_config = self._safe_config_dict()
        self.logger.info("Initializing pipeline for study_id=%s", self.study_id)
        self.logger.info("Runtime config (source path redacted): %s", safe_config)
        os.environ["PRIMA_ATTENTION_BACKEND"] = self.config.attention_backend

        self.tokenizer_model: Optional[torch.nn.Module] = None
        self._tokenizer_encoder_only = False
        self.prima_model: Optional[torch.nn.Module] = None
        self.patchifier = MedicalImagePatchifier(in_dim=256)

        self._process = psutil.Process()
        self.metrics: Dict[str, Any] = {
            "study_id": self.study_id,
            "config": safe_config,
            "stages": {},
            "series": [],
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            dev = torch.device(self.config.device)
            self.metrics["gpu_name"] = torch.cuda.get_device_name(dev)
            total_vram_gib = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
            self.metrics["gpu_total_vram_gib"] = total_vram_gib
            if total_vram_gib <= 9.0 and not self.config.low_vram:
                self.logger.warning(
                    "GPU has %.1f GiB VRAM but low_vram=false; enable low_vram for 8 GB-class cards",
                    total_vram_gib,
                )

    def _resolve_study_id(self) -> str:
        if self.config.study_id:
            value = str(self.config.study_id).strip()
            if not value:
                raise ValueError("study_id cannot be blank")
            safe = "".join(
                ch if ch.isalnum() or ch in "-_." else "_"
                for ch in value
            ).strip("._")
            if not safe:
                raise ValueError("study_id contains no safe filename characters")
            return safe[:128]

        # Never derive output filenames from a potentially identifying folder
        # name. Use a deterministic anonymous ID instead.
        source = str(Path(self.config.study_dir).expanduser().resolve())
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        return f"study-{digest}"

    def _safe_config_dict(self) -> Dict[str, Any]:
        data = asdict(self.config)
        if self.config.redact_source_path:
            data["study_dir"] = "<redacted>"
        return data

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
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
        return {
            "allocated_gib": torch.cuda.memory_allocated(dev) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(dev) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(dev) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / (1024**3),
            "device_free_gib": free_bytes / (1024**3),
            "device_total_gib": total_bytes / (1024**3),
        }

    def _log_memory(self, label: str) -> None:
        snap = self._cuda_snapshot()
        if snap and self.config.log_cuda_memory:
            self.logger.info(
                "CUDA memory [%s]: allocated=%.2f GiB reserved=%.2f GiB "
                "peak_alloc=%.2f GiB peak_reserved=%.2f GiB free=%.2f/%.2f GiB",
                label,
                snap["allocated_gib"],
                snap["reserved_gib"],
                snap["peak_allocated_gib"],
                snap["peak_reserved_gib"],
                snap["device_free_gib"],
                snap["device_total_gib"],
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
            # The tokenizer is never reused after tokenization, so copying its
            # weights back to CPU before deletion only adds D2H traffic and a
            # transient RAM allocation.
            del self.tokenizer_model
            self.tokenizer_model = None
            self._tokenizer_encoder_only = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    @staticmethod
    def _dtype_from_name(name: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            return mapping[name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {name}") from exc

    def load_tokenizer_model(self) -> torch.nn.Module:
        if self.tokenizer_model is None:
            started = time.perf_counter()
            tokenizer_config = self._load_config_file_or_dict(
                self.config.tokenizer_model_config
            )
            if isinstance(self.config.tokenizer_model_config, str):
                cfg_path = Path(self.config.tokenizer_model_config).resolve()
                params = tokenizer_config.get("vqvae_config", {})
                ckpt = params.get("ckpt_path")
                if ckpt:
                    p = Path(ckpt)
                    if not p.is_absolute():
                        candidates = [
                            (Path.cwd() / p).resolve(),
                            (cfg_path.parent / p).resolve(),
                            (_REPO_ROOT / p).resolve(),
                        ]
                        resolved = next(
                            (candidate for candidate in candidates if candidate.exists()),
                            candidates[1],
                        )
                        params["ckpt_path"] = str(resolved)

            vqvae = ModelLoader.load_vqvae_model(tokenizer_config)

            use_encoder_only = (
                self.config.tokenizer_encoder_only
                and self._device().type == "cuda"
                and hasattr(vqvae, "encoder")
            )
            dtype = self._dtype_from_name(self.config.tokenizer_dtype)

            if use_encoder_only:
                encoder = vqvae.encoder
                del vqvae
                gc.collect()
                self.tokenizer_model = encoder.to(
                    device=self._device(), dtype=dtype
                ).eval()
                self._tokenizer_encoder_only = True
                self.logger.info(
                    "Loaded VQ-VAE encoder only on %s/%s",
                    self._device(), self.config.tokenizer_dtype,
                )
            else:
                self.tokenizer_model = vqvae.to(self._device()).eval()
                self._tokenizer_encoder_only = False

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
            head_quant_backend=self.config.head_quant_backend,
            prune_inference_only=self.config.prune_inference_only,
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
        chunk_size = min(self.config.max_tokens_per_chunk, int(tokens.shape[0]))
        cursor = 0

        while cursor < tokens.shape[0]:
            stop = min(cursor + chunk_size, tokens.shape[0])
            chunk = tokens[cursor:stop].unsqueeze(1).to(
                self._device(), non_blocking=True
            )
            try:
                with torch.inference_mode():
                    amp_dtype = self._dtype_from_name(
                        self.config.tokenizer_dtype
                    )
                    amp_context = (
                        torch.amp.autocast(
                            device_type="cuda",
                            dtype=amp_dtype,
                        )
                        if amp_enabled
                        and amp_dtype in (torch.float16, torch.bfloat16)
                        else contextlib.nullcontext()
                    )
                    with amp_context:
                        emb = (
                            vqvae(chunk)
                            if self._tokenizer_encoder_only
                            else vqvae.encode(chunk)
                        )
                embeddings.append(emb.detach().cpu())
                cursor = stop
                del emb, chunk
            except torch.OutOfMemoryError:
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                if (
                    not self.config.auto_reduce_tokenizer_chunk
                    or chunk_size <= self.config.min_tokens_per_chunk
                ):
                    raise

                new_chunk_size = max(
                    self.config.min_tokens_per_chunk,
                    chunk_size // 2,
                )
                if new_chunk_size == chunk_size:
                    raise

                self.logger.warning(
                    "VQ-VAE OOM at chunk=%d; retrying current tokens with chunk=%d",
                    chunk_size,
                    new_chunk_size,
                )
                chunk_size = new_chunk_size

        self.metrics["tokenizer_effective_chunk_size"] = chunk_size
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
        original_token_count = int(tokens.shape[0])
        # Preserve the upstream series-inclusion rule before any optimized
        # Otsu prefiltering. Otherwise optimized mode could include a >5000
        # patch series that baseline/stock would skip.
        if original_token_count > 5000:
            raise RuntimeError(
                f"Too many raw tokens for {series_name}: "
                f"{original_token_count} > 5000"
            )

        if self.config.prefilter_otsu_before_vq and original_token_count:
            useids, positions, selected_percent = select_otsu_indices(
                meta,
                original_token_count,
                start_percentage=self.config.otsu_percentage,
                min_count=25,
            )
            tokens = tokens[useids]
            meta = dict(meta)
            meta["_prima_prefiltered"] = True
            meta["_prima_selected_coords"] = positions
            meta["_prima_selected_percent"] = selected_percent
            self.logger.info(
                "Pre-VQ Otsu filter series=%s threshold=%d before=%d after=%d",
                series_name,
                selected_percent,
                original_token_count,
                len(tokens),
            )

        embedding = self._encode_tokens(tokens, vqvae)
        elapsed = time.perf_counter() - started

        self.metrics["series"].append(
            {
                "name": series_name,
                "tokens_before_otsu": original_token_count,
                "tokens_encoded": int(tokens.shape[0]),
                "seconds": elapsed,
                "cpu_rss_gib": self._rss_gib(),
                **self._cuda_snapshot(),
            }
        )
        self.logger.info(
            "Tokenized series %s: encoded %d/%d patches in %.2fs",
            series_name,
            tokens.shape[0],
            original_token_count,
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
                if meta.get("_prima_prefiltered", False):
                    filtered_embeddings.append(series_embeddings[i])
                    coords.append(meta["_prima_selected_coords"])
                    continue

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
        # Keep each series ragged. HierViT already receives the true per-series
        # lengths and performs its own packing, so padding every series to the
        # study maximum here only duplicates VRAM before the inner transformer.
        visuals: List[torch.Tensor] = [
            img.unsqueeze(0) for img in patched
        ]

        series_name_tensors = [
            chartovec(name, max_length=200) for name in series_names
        ]
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
            "studydescription": chartovec(
                self.config.study_description, max_length=200
            ).unsqueeze(0),
        }

    @staticmethod
    def _move_to_device(
        obj: Any,
        device: torch.device,
        floating_dtype: Optional[torch.dtype] = None,
    ) -> Any:
        if isinstance(obj, torch.Tensor):
            dtype = (
                floating_dtype
                if floating_dtype is not None and obj.is_floating_point()
                else obj.dtype
            )
            return obj.to(device=device, dtype=dtype, non_blocking=True)
        if isinstance(obj, list):
            return [
                Pipeline._move_to_device(item, device, floating_dtype)
                for item in obj
            ]
        if isinstance(obj, dict):
            return {
                k: Pipeline._move_to_device(v, device, floating_dtype)
                for k, v in obj.items()
            }
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

        prep_started = time.perf_counter()
        prima_input = self.prepare_prima_input(
            series_embeddings=series_embeddings,
            series_names=series_names,
            all_ser_emb_meta=all_ser_emb_meta,
        )
        input_dtype = (
            self._dtype_from_name(self.config.visual_dtype)
            if self.config.low_vram and self._device().type == "cuda"
            else None
        )
        prima_input = self._move_to_device(
            prima_input, self._device(), floating_dtype=input_dtype
        )
        self._stage_done("prepare_prima_input", prep_started)

        model = self.load_full_prima_model()
        self._reset_cuda_peak()
        inference_started = time.perf_counter()

        try:
            with torch.inference_mode():
                if self._device().type == "cuda":
                    amp_dtype = self._dtype_from_name(self.config.visual_dtype)
                    amp_context = (
                        torch.amp.autocast(
                            device_type="cuda",
                            dtype=amp_dtype,
                        )
                        if amp_dtype in (torch.float16, torch.bfloat16)
                        else contextlib.nullcontext()
                    )
                    with amp_context:
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

            self._stage_done("prima_inference", inference_started)
            self._log_memory("PRIMA inference complete")

            output_path = (
                self.output_dir / f"{self.study_id}_predictions.json"
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
