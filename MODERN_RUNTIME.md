# Modern PRIMA Runtime — RTX 4060 8 GB

This branch modernizes the public PRIMA inference path while preserving the
research/training code and the original full-checkpoint format.

The primary target is **one MRI study at a time on an NVIDIA RTX 4060 8 GB**.

## What changed

- DICOM volumes can be streamed one series at a time instead of retaining the
  full resampled study in RAM.
- Preprocessing uses float32 rather than the previous float64 intermediate copy.
- 3D patch extraction is vectorized with PyTorch unfold operations.
- VQ-VAE runs in bounded FP16-autocast chunks and is fully released before PRIMA.
- The full checkpoint stays in CPU RAM in low-VRAM mode.
- Only the PRIMA visual backbone is moved to CUDA in FP16.
- Diagnosis, referral, and priority heads run on CPU.
- CPU heads can optionally use dynamic INT8 quantization.
- Attention selects between native PyTorch variable-length attention,
  external flash-attn, and PyTorch SDPA at runtime.
- External flash-attn is no longer a mandatory runtime dependency.
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

For the preferred TorchAO INT8 backend, optionally install:

```bash
pip install -r requirements-quant.txt
```

If TorchAO is absent or incompatible, the optimized profile automatically falls
back to PyTorch dynamic INT8 rather than failing.

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

INT8 uses TorchAO when compatible and falls back to PyTorch dynamic INT8 on CPU.

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
```

to 64, then 48, then 32.

If PRIMA visual inference OOMs, tokenizer chunk size will not help because the
tokenizer has already been released. Keep:

```yaml
low_vram: true
visual_dtype: "float16"
compile_visual: false
attention_backend: "auto"
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
attention math. The full 4060 path cannot be considered hardware-validated until
the official weights and at least one real DICOM study have been run on the
target machine. Use the baseline/optimized comparison above for that validation.
