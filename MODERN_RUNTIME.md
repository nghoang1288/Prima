# Modern PRIMA Runtime — RTX 4060 8 GB

This branch modernizes the public PRIMA inference path while preserving the
research/training code and the original full-checkpoint format.

The primary target is **one MRI study at a time on an NVIDIA RTX 4060 8 GB**.

## What changed

- DICOM volumes can be streamed one series at a time instead of retaining the
  full resampled study in RAM.
- Preprocessing uses float32 rather than the previous float64 intermediate copy.
- 3D patch extraction is vectorized with PyTorch unfold operations.
- Only the VQ-VAE encoder is moved to CUDA; decoder/codebook stay off GPU.
- VQ-VAE runs in bounded chunks and is fully released before PRIMA.
- Tokenizer chunk size automatically backs off on CUDA OOM down to a safe minimum.
- The full checkpoint stays in CPU RAM in low-VRAM mode.
- Only the PRIMA visual backbone is moved to CUDA in FP16.
- Diagnosis, referral, and priority heads run on CPU.
- CPU heads can optionally use dynamic INT8 quantization; the conservative
  default is PyTorch dynamic INT8, with TorchAO 0.18 as an A/B backend.
- Attention selects between native PyTorch variable-length attention,
  external flash-attn, and PyTorch SDPA at runtime.
- External flash-attn is no longer a mandatory runtime dependency.
- Per-series visual inputs remain ragged instead of being padded twice on GPU.
- 3D positional encodings are generated only for used coordinates rather than
  allocating the historical ~120 MB fixed grid.
- Checkpoint loading uses mmap where supported; VQ-VAE also uses meta-device
  construction plus assign=True when possible.
- Every run writes timing, CPU RSS, CUDA allocated/reserved memory, and peak VRAM.
- Regression utilities compare optimized predictions to a baseline case.

## Recommended environment

Use Python 3.11 or 3.12 and a current NVIDIA driver.

The modern runtime is pinned to PyTorch 2.14 and MONAI 1.6:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-runtime.txt
```

For experimental TorchAO 0.18 A/B testing, optionally install:

```bash
pip install -r requirements-quant.txt
```

The default optimized profile does not require TorchAO. To test it, set
`head_quant_backend: "torchao"` after installing `requirements-quant.txt`.
The runtime fails explicitly if that backend is requested but unavailable,
rather than silently changing quantization during a medical inference run.

If PyTorch needs a CUDA-specific wheel on your platform, install the matching
PyTorch 2.14 build from the official PyTorch selector first, then install the
remaining requirements.

## 1. Download the official weights

Use the repository's existing setup helper or the official links in the root
README. Keep the original checkpoint unchanged.

Update:

```text
configs/sample_tokenizer_config.json
configs/sample_prima_config.json
```

with the local weight paths.

## 2. Run the 4060 baseline first

Edit:

```text
configs/rtx4060_8gb_baseline.yaml
```

and set:

- `study_dir`
- `output_dir`
- `study_description`

Then run:

```bash
python end-to-end_inference_pipeline/pipeline.py \
  --config configs/rtx4060_8gb_baseline.yaml
```

This profile already uses the important VRAM changes (streaming, FP16 visual
backbone, CPU head offload) but intentionally keeps heads FP32 and compile off.
Its prediction JSON is the optimized-runtime golden output.

## 3. Run the INT8-head profile on the same study

Edit the equivalent paths in:

```text
configs/rtx4060_8gb_optimized.yaml
```

then:

```bash
python end-to-end_inference_pipeline/pipeline.py \
  --config configs/rtx4060_8gb_optimized.yaml
```

The default optimized profile uses PyTorch dynamic INT8 on CPU. TorchAO 0.18 is
available only as an explicit A/B backend because current CPU performance is
workload-dependent.

## 4. Compare predictions

```bash
python tools/compare_predictions.py \
  output/4060-baseline/<study>_predictions.json \
  output/4060-optimized/<study>_predictions.json \
  --output output/comparison.json
```

The most important field during validation is:

```json
"sign_flips_at_zero"
```

because PRIMA's diagnostic/referral scores are emitted after subtracting each
head threshold. A sign flip can therefore change the binary decision.

Do not adopt a quantized profile for routine use until representative cases have
been checked.

## 5. Inspect actual memory and timing

Every run writes:

```text
output/.../runtime_metrics.json
```

including:

- per-series token count and tokenizer time
- per-stage latency
- CPU RSS
- CUDA allocated memory
- CUDA reserved memory
- peak CUDA allocated memory
- detected GPU name

The log also prints memory snapshots.

## 6. Audit the official checkpoint

To stop estimating model size:

```bash
python tools/checkpoint_audit.py /path/to/primafullmodel107.pt \
  --output output/checkpoint_audit.json
```

This loads the trusted official checkpoint on CPU and reports parameter counts
for the visual model, diagnosis heads, referral heads, and priority head, plus
FP32/FP16/INT8 memory estimates.

## 7. 8 GB tuning order

If VQ-VAE OOMs, reduce:

```yaml
max_tokens_per_chunk: 96
min_tokens_per_chunk: 4
auto_reduce_tokenizer_chunk: true
```

The runtime automatically retries 96 -> 48 -> 24 -> 12 -> 6 -> 4 when tokenizer OOM is
encountered. This auto-backoff is intentionally limited to the independently
chunkable VQ-VAE stage.

If PRIMA visual inference OOMs, tokenizer chunk size will not help because the
tokenizer has already been released. Keep:

```yaml
low_vram: true
visual_dtype: "float16"
compile_visual: false
attention_backend: "auto"
tokenizer_encoder_only: true
tokenizer_dtype: "float32"
prune_inference_only: true
```

and inspect the PRIMA-stage peak. You can force:

```yaml
attention_backend: "sdpa"
```

for a compatibility test.

Do not increase `otsu_percentage` or remove MRI series merely to fit VRAM
without validating the resulting prediction drift, because those changes alter
the model input distribution.

## 8. torch.compile

Only test compile after the eager baseline works:

```yaml
compile_visual: true
compile_mode: "max-autotune-no-cudagraphs"
```

The no-cudagraph mode is intentional for the 8 GB target. Compare both latency
and peak VRAM; retain compile only if it wins on representative studies.

## Attention backend

`attention_backend: "auto"` uses:

1. PyTorch native variable-length attention when supported by the installed
   PyTorch/CUDA/GPU combination.
2. External flash-attn if it is installed and native varlen cannot run.
3. PyTorch scaled-dot-product attention as the portable fallback.

You can force `native`, `flash`, or `sdpa` for debugging.

## Compatibility philosophy

Training/research dependencies remain in the original `requirements.txt`.
The modern inference environment lives in `requirements-runtime.txt`.

The old pipeline behavior is still reachable by disabling `stream_dicom` and
`low_vram`. This separation is intentional so that performance refactors can
be compared against the original implementation instead of silently replacing
it.

## Current validation boundary

The repository-level CPU tests verify the vectorized patch extraction and SDPA
attention math. The official weights and one representative real DICOM study have now been run on the target RTX 4060 8 GB workstation. Baseline inference completed reproducibly; multi-case validation is still required before merge or any clinical workflow integration.


## September 2026 audit

The detailed technology and code audit is maintained in
[`AUDIT_2026_09.md`](AUDIT_2026_09.md).

Important decisions for the RTX 4060 target:

- PyTorch 2.14 + MONAI 1.6 are the current baseline.
- Native PyTorch variable-length attention / SDPA are preferred over chasing
  FlashAttention 4, which is not the Ada-oriented target.
- NestedTensor/jagged is not used because it remains a prototype API.
- INT4 CPU heads and FP8/FP4 visual inference remain experimental follow-ups.
- `torch.compile` stays off until eager-mode VRAM and latency are measured.
- A safetensors-only full Prima checkpoint remains deferred until the official
  full-object checkpoint is audited exactly.

For WSL2, `PYTORCH_ALLOC_CONF=expandable_segments:True` is worth an A/B test
if allocator fragmentation is observed, but it is not hard-coded because the
option remains experimental.


## Windows / WSL2 RTX 4060 deployment

For the first hardware validation, use WSL2 rather than Windows-native Python.

- Keep the repo, model weights and staged DICOM under the Linux filesystem
  (for example `~/src/Prima`), not under `/mnt/c`.
- Install/update the NVIDIA display driver on Windows only. Do not install a
  Linux NVIDIA display driver inside WSL.
- PyTorch 2.14 PyPI Linux wheels use a CUDA 13.x runtime. For this release
  candidate, use a current Windows NVIDIA driver with major version >= 580.
- Run `python tools/preflight.py` before downloading/processing a study.
- WSL2 defaults to 50% of Windows host RAM. The full PRIMA checkpoint can be
  CPU-memory heavy, so on a host with >=32 GB RAM allocate roughly 75% to WSL
  for validation and keep swap enabled. Do not attempt routine validation on a
  host with <32 GB without reviewing checkpoint audit memory first.
- The PyTorch wheel already supplies the CUDA runtime needed for inference.
  Do not install a CUDA toolkit in WSL for the initial run; a toolkit is only
  needed later if compiling optional CUDA extensions.

### Official weights

Install `gdown` separately, then force a fresh official download for the first
validated deployment:

```bash
python tools/download_models_and_setup_test.py --force-download
```

The downloader writes `test/trained_models/model_manifest.json` containing
file size and SHA256. The full model must be the priority-corrected checkpoint
from 2026-02-19 or later.

### Local data safety

Use a de-identified staging case named with a neutral identifier such as
`CASE001`. Never commit local DICOM paths, patient identifiers, output JSON, or
runtime logs. Local deployment configs matching `configs/local_*.yaml` and
`output/` are git-ignored.

Do not connect this release candidate to PACS write-back or automated clinical
reporting. Complete baseline/optimized equivalence review on representative
de-identified cases first.


## Human-readable reports

The raw prediction JSON remains the authoritative machine output. By default,
RTX 4060 profiles also render two local presentation files after each run:

```text
CASE001_predictions.json
CASE001_report.md
CASE001_report.html
```

The Vietnamese report:

- selects priority by argmax;
- lists diagnosis/referral outputs with margin >= 0 as above-threshold;
- separately lists near-threshold negative margins;
- calls scores threshold margins, never disease probabilities;
- omits the raw CLIP embedding;
- keeps the original PRIMA label code beside any humanized label;
- includes non-PHI runtime metrics.

Set:

```yaml
render_human_report: true
report_language: "vi"
report_near_margin: 0.25
report_show_all_scores: false
```

The report files are git-ignored because they are patient-derived outputs.

For one-study-per-process use on the validated RTX 4060 workstation, start with:

```text
configs/rtx4060_8gb_fast.yaml
```

This keeps pre-VQ Otsu filtering but leaves task heads FP32, avoiding the large
one-time INT8 quantization startup cost observed during hardware validation.
The INT8 optimized profile remains useful to evaluate for a future persistent
multi-study service where quantization can be amortized.
