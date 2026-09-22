# RADAR-4060 Implementation Plan

## Mission

Create a reproducible, research-only RTX 4060 8GB inference port of Alibaba DAMO RADAR for contrast-enhanced abdominal CT.

Target repository: `nghoang1288/RADAR-4060`
Upstream: `alibaba-damo-academy/damo-radar`
Pinned upstream commit: `0dbf0ece209b77d722588e5e7943014d3db48f7f`

Do not silently update upstream during this implementation. Preserve Apache-2.0 LICENSE, THIRD_PARTY_LICENSES.md, copyright notices, and a clear upstream attribution.

## Hardware / runtime target

- Windows workstation with NVIDIA RTX 4060 8GB.
- Preferred runtime: WSL2 Ubuntu.
- Python 3.10.
- CUDA-enabled PyTorch compatible with the installed NVIDIA driver.
- Primary test data: upstream public demo NIfTI only.
- No patient-derived data may be committed to GitHub.

## Why a port is needed

The upstream inference path allocates the full CT volume and large 37-channel stitching tensors on CUDA. With a 3D UNet and ROI `(96, 256, 384)`, this is likely to exceed an 8GB GPU even though the checkpoint itself is much smaller.

The first implementation must preserve model semantics and ROI size. Do not reduce spatial resolution, quantize weights, change prompts, or alter thresholds merely to make memory fit unless later tasks explicitly request it.

## Architecture target

CPU:
- full preprocessed CT volume
- full-volume probability accumulator
- overlap/count accumulator
- final stitched probability volume / segmentation
- report/output serialization

GPU:
- model weights
- one sliding-window patch at a time
- temporary Conv3D activations
- temporary patch outputs

Memory changes:
1. Keep the full CT tensor on CPU.
2. Transfer only the current sliding-window patch to CUDA.
3. Use `torch.inference_mode()`.
4. Use CUDA autocast FP16 for inference, with a CLI switch to disable AMP.
5. Move each patch prediction back to CPU immediately.
6. Keep `full_mask` / probability accumulator on CPU.
7. Replace 37-channel `count_map` with a single-channel overlap map and rely on broadcasting.
8. Avoid unnecessary duplicate full-resolution 37-channel tensors.
9. Add explicit CUDA cache cleanup only at phase boundaries, not inside tight loops unless measurement proves useful.

## Repository layout

Keep upstream code recognizable. Add or modify only what is needed.

Recommended additions:

- `README_4060.md` — quick start and limitations.
- `docs/IMPLEMENTATION_PLAN.md` — copy of this plan.
- `docs/BENCHMARK.md` — measured versions, timings, RAM/VRAM.
- `requirements-inference.txt` — minimal inference-only dependencies.
- `scripts/setup_wsl.sh`
- `scripts/download_models.sh`
- `scripts/run_demo_4060.sh`
- `scripts/profile_demo_4060.sh`
- `tests/test_cpu_stitching.py`
- `tests/test_memory_path.py`
- `tests/test_demo_smoke.py`

Prefer a dedicated inference entry point such as:
- `RADAR_inference/inference_4060.py`

Avoid rewriting the entire upstream project.

## CLI requirements

The optimized entry point should expose at least:

- `--img-dir`
- `--save-dir`
- `--device cuda|cpu`
- `--amp fp16|off`
- `--stitch-device cpu|cuda` (default cpu)
- `--roi-size 96 256 384` with the upstream value as default
- `--profile-memory`
- `--checkpoint`

If a parameter already exists upstream, reuse its semantics.

## Phases

### Phase 0 — Bootstrap

1. Create private GitHub repository `nghoang1288/RADAR-4060`.
2. Clone upstream at the pinned SHA.
3. Push the pinned upstream state to `main` in the new repo.
4. Add upstream remote as `upstream`.
5. Create working branch `antigravity/radar-4060-bootstrap`.
6. Copy this plan to `docs/IMPLEMENTATION_PLAN.md`.
7. Add a short project README section explaining that this is an unofficial research port for 8GB GPUs.

### Phase 1 — Reproducible environment

1. Build a minimal Python 3.10 inference environment.
2. Do not install the full legacy requirements list blindly.
3. Install only dependencies actually imported by the inference path.
4. Record exact Python, PyTorch, CUDA runtime, NVIDIA driver, MONAI, SimpleITK, transformers and nibabel versions.
5. Verify `torch.cuda.is_available()` and report the RTX 4060 VRAM.
6. Download official RADAR checkpoints using the upstream helper.
7. Do not commit checkpoints to Git.

### Phase 2 — Baseline characterization

1. Use only the upstream public demo case.
2. Run the original inference path if it fits; otherwise capture the OOM cleanly.
3. Record:
   - peak allocated VRAM
   - peak reserved VRAM
   - CPU RAM peak if practical
   - elapsed time
   - failure location if OOM
4. Never change the model just to obtain a baseline.

### Phase 3 — 4060 memory implementation

Implement CPU-resident sliding-window stitching.

Required behavior:
- full volume remains CPU-resident;
- patch moves to CUDA only when needed;
- model runs under `torch.inference_mode()`;
- FP16 autocast default on CUDA;
- patch probability/logit needed for stitching returns to CPU immediately;
- accumulator stays on CPU;
- count map is single-channel;
- final normalization uses safe `clamp_min`;
- preserve ROI `96x256x384` unless upstream code requires padding;
- preserve class count and class order;
- preserve preprocessing behavior.

Do not use lower input resolution, lossy weight quantization, TensorRT conversion, or model surgery in this phase.

### Phase 4 — Tests and numerical checks

Add synthetic tests that prove:
1. single-channel count-map broadcasting equals a multi-channel reference implementation;
2. CPU stitching gives the same result as CUDA/reference stitching on a small synthetic volume;
3. patch order does not change the stitched result beyond floating-point tolerance;
4. AMP can be disabled;
5. CPU stitching does not leave full-volume 37-channel tensors on CUDA;
6. no patient data paths or files are used in tests.

For the public demo:
- compare optimized output with upstream expected/demo results where available;
- report mean and max absolute score differences;
- investigate any max score delta > 0.02 before calling the port validated;
- do not claim clinical equivalence from this numerical check.

### Phase 5 — 4060 benchmark gate

The optimized demo must:
- complete without CUDA OOM on RTX 4060 8GB;
- preserve the upstream ROI by default;
- keep peak CUDA reserved memory below 8GB, with a practical target <= 7.5GB;
- emit the normal inference result files;
- record wall-clock runtime;
- record exact environment versions.

If the demo still OOMs:
1. profile the peak;
2. first remove avoidable duplicate tensors / deep-supervision outputs during inference if they are not needed for model semantics;
3. consider chunking postprocessing;
4. only then propose a smaller ROI as an experimental fallback, and clearly label it non-reference.

### Phase 6 — Usability

After the 8GB demo gate passes:
- add a one-command WSL setup script;
- add a one-command model download script;
- add a one-command demo inference script;
- document Windows path mapping;
- keep NIfTI as the initial supported input.

Do not add DICOM/PACS integration until the optimized NIfTI path is stable.

### Phase 7 — Optional radiologist-facing output

Only after inference stability:
- transform CSV/scores into a local HTML summary;
- organize findings anatomically;
- allow configurable thresholds;
- label any displayed images as review aids, not proof/localization unless the model genuinely provides localization;
- keep generated patient reports/images local and Git-ignored.

## Safety / data handling

- GitHub contains code, synthetic tests, public demo artifacts if license permits, and aggregate benchmark numbers only.
- Never commit DICOM, PHI, patient NIfTI, patient screenshots, patient predictions, or patient HTML reports.
- Do not upload patient data to external services.
- Public demo data from upstream may be used for testing.
- This is research software, not a validated clinical device.

## Deliverables for the first Antigravity task

1. New private repository exists.
2. Upstream pinned SHA is imported and attributed.
3. Working branch exists.
4. Optimized inference path is implemented.
5. Minimal environment/scripts exist.
6. Synthetic tests pass.
7. Official public demo runs on RTX 4060 8GB without OOM, or a precise blocker report identifies the remaining peak allocation.
8. `docs/BENCHMARK.md` contains exact environment + peak VRAM + runtime + numerical delta.
9. A PR from the Antigravity working branch to `main` is opened but not merged.
