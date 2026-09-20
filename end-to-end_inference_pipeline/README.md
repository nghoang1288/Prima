# End-to-end Inference Pipeline

This folder contains code for an end-to-end inference pipeline for applying Prima directly on a raw MRI study. The pipeline is designed to be user-friendly and easy to use and can be run on a local machine or a server. It will load the study, minimally process the data, and then pass it forward to the Prima model to generate the predicted radiologic diagnoses, referral, and prioritization recommendations.

The input to the pipeline is a folder containing raw MRI scans in DICOM format. The output is a JSON file.

Expected raw mri study folder structure:
```
Study_dir/
    series1/
        image1.dcm
        image2.dcm
        ...
    series2/
        image1.dcm
        image2.dcm
        ...
    series3/
        image1.dcm
        image2.dcm
        ...
    series4/
        image1.dcm
        image2.dcm
        ...
    ...
```
Expected output JSON file structure:
```
{
    "diagnosis": "...",
    "referral": "...",
    "priority": "..."
}
```

To run the end-to-end pipeline, first you need to download both the [Prima model and head weights](https://drive.google.com/file/d/119kKMcdk1GPww69IQAf6JkXuNMIEEAIk/view) and the [VQ-VAE weights](https://drive.google.com/file/d/11EitVfPVXmdPSJviQQ5ZKasFNbQqD5Bt/view?usp=drive_link), then update `configs/pipeline_config.yaml` with and fill in `study_dir` (where you stored your study data), `output_dir` (where do you want the output json to be stored), `ckpt_dir` under `tokenizer_model_config` (or within `configs/sample_tokenizer_config.json`) to be where you stored the VQVAE checkpoint, and `full_model_ckpt` under `prima_model_config` (or within `configs/sample_prima_model_config.json`) to where you stored the Prima model and head weight pt file. Then, from the main repository directory, run
```
python /end-to-end_inference_pipeline/pipeline.py --config configs/pipeline_config.yaml
```



## Low-VRAM single-study inference

This fork adds an optional low-VRAM path intended for one MRI study at a time.

With `low_vram: true`:

1. The VQ-VAE tokenizer runs on the configured CUDA device.
2. Tokenizer embeddings are copied back to CPU and the tokenizer is unloaded.
3. The full PRIMA checkpoint is deserialized on CPU.
4. Only `clipvisualmodel` is moved to CUDA, using `visual_dtype`.
5. The study embedding is copied back to CPU once.
6. Diagnosis, referral, and priority heads run on CPU, so they do not consume GPU VRAM.
7. CUDA allocator usage is logged at major stages when `log_cuda_memory: true`.

Recommended starting config for an 8 GB RTX 4060:

```yaml
batch_size: 1
device: "cuda"
low_vram: true
visual_dtype: "float16"
log_cuda_memory: true
max_tokens_per_chunk: 128
```

If the VQ-VAE tokenizer still runs out of memory, reduce `max_tokens_per_chunk` to 96, 64, or 32.

The low-VRAM mode deliberately keeps task heads in system RAM. For this reason, 32 GB system RAM is a practical minimum target and 64 GB is more comfortable when loading the full checkpoint.

### Run

From the repository root:

```bash
python end-to-end_inference_pipeline/pipeline.py --config configs/pipeline_config.yaml
```

Watch `pipeline.log` for lines similar to:

```text
CUDA memory [VQ-VAE loaded]: ...
CUDA memory [VQ-VAE unloaded]: ...
CUDA memory [after PRIMA load]: ...
CUDA memory [PRIMA inference complete]: ...
```

Those values are the quickest way to determine whether the visual backbone itself fits a specific GPU.

### Notes

- This path is aimed at inference, not training.
- It processes one study at a time; `batch_size: 1` is recommended.
- The original full-GPU behavior is preserved by setting `low_vram: false`.
- If a particular study has many long/derived series, VRAM use can still increase because transformer activation memory depends on token count.
