# CodeT5+ Class Name Prediction

This repo scaffolds an end-to-end pipeline to fine-tune CodeT5+ to predict class names from code snippets. It includes dataset building from GitHub repos, training, and evaluation.

## Features

- ✅ **GPU-Accelerated Evaluation**: 10-20x faster inference with automatic GPU detection
- ✅ **Comprehensive Logging**: All scripts log to `logs/` with timestamps for history and monitoring
- ✅ **Detailed Results**: Per-sample predictions saved for error analysis
- ✅ **Progress Monitoring**: Use `tail -f` to monitor long-running processes
- ✅ **Multiple Models**: Support for CodeT5+ and CodeGen architectures

## Setup

Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build dataset from GitHub

1) Prepare a list of repos (one per line): `data/repos.txt`
```
https://github.com/pallets/flask
https://github.com/psf/requests
```

2) Run dataset builder (Python by default; or pass `--languages python,java`):
```bash
python scripts/build_dataset.py \
  --repos-file data/repos.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3
```

Outputs per language: `datasets/<language>/{train,valid,test}.jsonl` with fields `language, repo, path, class_span, source, target`.

## Train CodeT5+

Train on Python only (adjust path for Java):
```bash
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/python \
  --output model/checkpoints/run1-python \
  --batch-size 8 --grad-accum 2 --epochs 3 --fp16 
```

The prompt template is:
```
Predict class name:
{source}
Name:
```

## Train CodeGen

### Python Dataset (CPU - slower but universal)
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/python \
  --output model/checkpoints/run1-python-codegen \
  --batch-size 1 \
  --grad-accum 8 \
  --max-source-len 256 \
  --gradient-checkpointing \
  --cpu
```

### Java Dataset with GPU (recommended - much faster)
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 4 \
  --grad-accum 4 \
  --max-source-len 512 \
  --gradient-checkpointing \
  --fp16
```

**CodeGen Training Options:**
- `--cpu`: Force CPU training (slower but works everywhere)
- `--fp16`: Use FP16 mixed precision on CUDA GPUs (faster, less memory)
- `--bf16`: Use BF16 mixed precision for Apple Silicon/MPS
- `--gradient-checkpointing`: Save memory at cost of ~20% speed
- `--batch-size N`: Samples per GPU (reduce if OOM)
- `--grad-accum N`: Gradient accumulation steps (effective batch = batch-size × grad-accum)

## Evaluate

### Evaluate CodeT5+ (CPU - slower)
```bash
python scripts/eval.py \
  --ckpt model/checkpoints/run1-python \
  --data datasets/python \
  --k 5
```

### Evaluate CodeT5+ with GPU (recommended - 10-20x faster)
```bash
python scripts/eval_gpu.py \
  --ckpt model/checkpoints/run1-python \
  --data datasets/python \
  --k 5
```

**GPU Evaluation Options:**
- `--cpu`: Force CPU usage even if GPU is available
- `--batch-size N`: Custom batch size (default: 16 for GPU, 8 for CPU)

**Performance comparison:**
- CPU: ~60-80 minutes for 3256 samples (batch_size=8)
- GPU: ~2-3 minutes for 3256 samples (batch_size=16)

Example with custom settings:
```bash
python scripts/eval_gpu.py \
  --ckpt model/checkpoints/run1-java \
  --data datasets/java \
  --k 5 \
  --batch-size 32  # For high-end GPUs
```

## Evaluate CodeGen

### Python Dataset
```bash
python scripts/eval_codegen.py \
  --ckpt model/checkpoints/run1-python-codegen \
  --data datasets/python \
  --k 5
```

### Java Dataset (GPU auto-detected)
```bash
python scripts/eval_codegen.py \
  --ckpt model/checkpoints/run1-java-codegen \
  --data datasets/java \
  --k 5
```

CodeGen evaluation automatically uses GPU if available (CUDA or MPS), falling back to CPU otherwise.

Produces: `model/metrics/run1-python/metrics.json` with exact match, case-insensitive EM, top-k accuracy, and average Levenshtein distance.

Training artifacts (under your `--output`, e.g., `model/checkpoints/run1-python/`):
- `training_log.csv` — step-wise training/eval metrics.
- `training_curve.png` — training/eval loss plot.
- `checkpoints.txt` — discovered `checkpoint-*` directories and final model path.

## Notes
- Start with Python for best heuristic parsing; Java is supported with basic regex.
- Use `--mask` to replace the declared class identifier with `____` to avoid label leakage.
- Consider rate limits and licenses when mining GitHub; export `GITHUB_TOKEN` for higher clone limits.
- Large files and artifacts are ignored via `.gitignore`.

## Predict from a snippet

Use the trained checkpoint to predict a class name from a code snippet (file or stdin). By default, the header class name is masked to match training.

From a file (Python):
```bash
python scripts/predict.py \
  --ckpt model/checkpoints/run1-python \
  --language python \
  --file path/to/MyWrongClass.py \
  --k 5 \
  --mask-all \
  --out path/to/MyClass_fixed.py \
  --rename-all
```

From stdin (Java):
```bash
cat path/to/Foo.java | python scripts/predict.py --ckpt model/checkpoints/run1-java --language java
```

Examples included:
- `examples/wrong_name_lru_cache.py` — LRU cache with wrong class name.
- `examples/wrong_name_image.py` — image container with wrong class name.

## Predict CodeGen
```bash
python scripts/predict.py \
  --ckpt model/checkpoints/run1-python \
  --language python \
  --file path/to/MyWrongClass.py \
  --k 5 \
  --mask-all \
  --out path/to/MyClass_fixed.py \
  --rename-all
```

## GPU Selection and CUDA Devices

If you have multiple GPUs and want to select which one to use for training, set the `CUDA_VISIBLE_DEVICES` environment variable before running the training script. For example, to use GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/python \
  --output model/checkpoints/run1-python \
  --batch-size 8 --grad-accum 2 --epochs 3 --fp16 --cuda-device 0
```

- The script argument `--cuda-device` should be set to 0 when using `CUDA_VISIBLE_DEVICES`, as the visible device will be mapped to index 0.
- This ensures all CUDA allocations go to the correct GPU and avoids out-of-memory errors on the wrong device.

See [PyTorch CUDA documentation](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) for more details.

## Logging and Monitoring

All scripts automatically log to `logs/` directory with timestamps:

```
logs/
├── eval_gpu_20250127_143022.log
├── train_20250127_091530.log
└── build_dataset_20250127_083012.log
```

### Monitor Long-Running Processes

When you run a script, it will show the log file location:

```bash
python scripts/eval_gpu.py --ckpt model/checkpoints/run1-java --data datasets/java --k 5

# Output shows:
# Logging to: logs/eval_gpu_20250127_143022.log
# Monitor progress: tail -f logs/eval_gpu_20250127_143022.log
```

**Monitor in real-time** (open in separate terminal):

```bash
tail -f logs/eval_gpu_20250127_143022.log
```

### Running in Background

```bash
# Run in background
python scripts/eval_gpu.py --ckpt model/checkpoints/run1-java --data datasets/java --k 5 &

# Monitor the latest log
tail -f $(ls -t logs/eval_gpu_*.log | head -1)
```

### Recovery After Disconnection

If disconnected from remote server:

```bash
# Reconnect and find your log
ls -lt logs/

# Check progress
tail -50 logs/eval_gpu_20250127_143022.log

# Continue monitoring
tail -f logs/eval_gpu_20250127_143022.log

# Check if process still running
ps aux | grep eval_gpu
```

**See [LOGGING.md](LOGGING.md) for complete logging documentation.**