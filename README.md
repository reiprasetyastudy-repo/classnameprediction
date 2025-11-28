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

2) Run dataset builder:

**Python Dataset:**
```bash
python scripts/build_dataset.py \
  --repos-file data/repos.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3
```

**Java Dataset:**
```bash
python scripts/build_dataset.py \
  --repos-file data/repos_java.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3 \
  --languages java
```

**Multiple Languages:**
```bash
python scripts/build_dataset.py \
  --repos-file data/repos.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3 \
  --languages python,java
```

Outputs per language: `datasets/<language>/{train,valid,test}.jsonl` with fields `language, repo, path, class_span, source, target`.

## Fair Comparison Training (CodeT5+ vs CodeGen)

**Optimized for RTX 5090 (32GB VRAM) - GPU Only**

This section provides matched configurations for direct comparison between CodeT5+ and CodeGen models. Both models are trained with identical effective batch sizes, learning rates, and epochs to ensure fair evaluation.

### Configuration 1: Standard Training (5 Epochs)

**CodeT5+ - Java Dataset:**
```bash
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/java \
  --output model/checkpoints/run1-java \
  --batch-size 12 \
  --grad-accum 3 \
  --lr 5e-5 \
  --epochs 5 \
  --fp16
```

**CodeGen - Java Dataset:**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 9 \
  --grad-accum 4 \
  --lr 5e-5 \
  --epochs 5 \
  --max-length 1024
```

**Evaluation - CodeT5+:**
```bash
python scripts/eval_gpu.py \
  --ckpt model/checkpoints/run1-java \
  --data datasets/java \
  --k 5
```

**Evaluation - CodeGen:**
```bash
python scripts/eval_codegen.py \
  --model model/checkpoints/run1-java-codegen \
  --valid-data datasets/java/test.jsonl \
  --output-dir evaluation/run1-java-codegen
```

**Comparison:**
| Parameter | CodeT5+ | CodeGen |
|-----------|---------|---------|
| Batch Size | 12 | 9 |
| Gradient Accumulation | 3 | 4 |
| Effective Batch Size | 12 × 3 = 36 | 9 × 4 = 36 ✓ |
| Learning Rate | 5e-5 | 5e-5 ✓ |
| Epochs | 5 | 5 ✓ |
| Max Length | 1024 (source) + 32 (target) | 1024 (combined) ✓ |
| FP16 | Yes | Yes ✓ |
| Gradient Checkpointing | No | No ✓ |
| Seed | 42 | 42 ✓ |
| VRAM Usage | ~23GB | ~24GB |
| Training Time | ~2-2.5 hours | ~2-3 hours |
| Preprocessing | N/A | 3-4 minutes |
| Eval Frequency | Every 100 steps | Every 1000 steps |

### Configuration 2: Extended Training (10 Epochs)

**CodeT5+ - Java Dataset:**
```bash
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/java \
  --output model/checkpoints/run2-java \
  --batch-size 12 \
  --grad-accum 4 \
  --lr 2e-5 \
  --epochs 10 \
  --fp16
```

**CodeGen - Java Dataset:**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run2-java-codegen \
  --batch-size 8 \
  --grad-accum 6 \
  --lr 2e-5 \
  --epochs 10 \
  --max-length 1024
```

**Evaluation - CodeT5+:**
```bash
python scripts/eval_gpu.py \
  --ckpt model/checkpoints/run2-java \
  --data datasets/java \
  --k 5
```

**Evaluation - CodeGen:**
```bash
python scripts/eval_codegen.py \
  --model model/checkpoints/run2-java-codegen \
  --valid-data datasets/java/test.jsonl \
  --output-dir evaluation/run2-java-codegen
```

**Comparison:**
| Parameter | CodeT5+ | CodeGen |
|-----------|---------|---------|
| Batch Size | 12 | 8 |
| Gradient Accumulation | 4 | 6 |
| Effective Batch Size | 12 × 4 = 48 | 8 × 6 = 48 ✓ |
| Learning Rate | 2e-5 | 2e-5 ✓ |
| Epochs | 10 | 10 ✓ |
| Max Length | 1024 (source) + 32 (target) | 1024 (combined) ✓ |
| FP16 | Yes | Yes ✓ |
| Gradient Checkpointing | No | No ✓ |
| Seed | 42 | 42 ✓ |
| VRAM Usage | ~23GB | ~21GB |
| Training Time | ~3-3.5 hours | ~4-5 hours |
| Preprocessing | N/A | 3-4 minutes |
| Eval Frequency | Every 100 steps | Every 1000 steps |

**Notes:**
- **Configuration 1 (Standard):** 5 epochs, effective batch 36, ~2-3 hours (includes 3-4 min preprocessing) - Good baseline for comparison and initial experiments
- **Configuration 2 (Extended):** 10 epochs, effective batch 48, ~4-5 hours (includes 3-4 min preprocessing) - Full training for best performance and convergence
- Both configurations ensure **identical effective batch size** for fair model comparison (different per-device batch × gradient accumulation to achieve same total)
- **VRAM Optimized:** CodeT5+ (Seq2Seq) uses ~23GB with batch=12. CodeGen (Causal LM) without gradient checkpointing uses ~24GB with batch=9, ~21GB with batch=8
- **Batch Size Differences:** CodeGen uses smaller per-device batch (9/8 vs 12) with higher gradient accumulation (4/6 vs 3/4) to fit in VRAM without gradient checkpointing, while maintaining same effective batch size
- **Speed Optimized:** Gradient checkpointing is **disabled** for 10x faster training (2-3 hours vs 20-22 hours). This requires slightly lower per-device batch size for CodeGen to fit in VRAM
- **Optimized Preprocessing:** CodeGen preprocesses all samples once (~3-4 min), then training is fast. Uses batch tokenization (1000 samples/batch) for speed
- **Dynamic Padding:** Batch-wise dynamic padding (pads to longest in batch, not max_length)
- **Max Length:** CodeT5+ uses separate lengths for source (1024) and target (32). CodeGen uses combined length (1024) for prompt+target in single sequence
- Logs are saved to `logs/codet5/` and `logs/codegen/` respectively
- All configurations use seed 42 for reproducibility

---

## Train CodeT5+ (Other Configurations)

For Python dataset or custom configurations:
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

---

## Train CodeGen (Other VRAM Configurations)

### For 12GB VRAM (e.g., RTX 3060, RTX 4060 Ti)

**Python Dataset:**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/python \
  --output model/checkpoints/run1-python-codegen \
  --batch-size 2 \
  --grad-accum 16 \
  --lr 2e-5 \
  --epochs 5 \
  --gradient-checkpointing
```

**Java Dataset:**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 2 \
  --grad-accum 16 \
  --lr 2e-5 \
  --epochs 5 \
  --gradient-checkpointing
```

- Effective batch size: 2 × 16 = 32
- Training speed: ~3-4 hours for 275K samples (5 epochs)

### For 24GB VRAM (e.g., RTX 3090, RTX 4090)

**Recommended (balanced speed and stability):**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 6 \
  --grad-accum 8 \
  --lr 2e-5 \
  --epochs 5
```

- Effective batch size: 6 × 8 = 48
- Training speed: ~1.5-2 hours for 275K samples (5 epochs)
- Better gradient stability than 12GB setup

### For 32GB+ VRAM (e.g., RTX 5090, A6000, A100)

**Option 1: Maximum Throughput (fastest training):**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 12 \
  --grad-accum 4 \
  --lr 2e-5 \
  --epochs 5
```

- Effective batch size: 12 × 4 = 48
- Training speed: ~45-60 minutes for 275K samples (5 epochs)
- **Best for**: Fast iteration, rapid experimentation

**Option 2: Larger Effective Batch (best convergence):**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 8 \
  --grad-accum 8 \
  --lr 2e-5 \
  --epochs 5
```

- Effective batch size: 8 × 8 = 64
- Training speed: ~1-1.5 hours for 275K samples (5 epochs)
- **Best for**: Smooth convergence, production models

**Option 3: Maximum Batch Size (ultra-stable gradients):**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 10 \
  --grad-accum 10 \
  --lr 2e-5 \
  --epochs 5
```

- Effective batch size: 10 × 10 = 100
- Training speed: ~1.5 hours for 275K samples (5 epochs)
- **Best for**: Most stable training, minimal noise in loss curve

### Monitoring GPU Memory

To monitor GPU usage during training:
```bash
watch -n 0.5 nvidia-smi
```

Or in another terminal:
```bash
nvidia-smi dmon -s mu
```

**CodeGen Training Options:**
- `--batch-size N`: Per-device batch size (adjust based on VRAM: 2 for 12GB, 6-8 for 24GB, 8-12 for 32GB+)
- `--grad-accum N`: Gradient accumulation steps (effective batch = batch-size × grad-accum)
- `--lr`: Learning rate (default: 2e-5)
- `--epochs`: Number of training epochs (default: 5)
- `--max-length`: Maximum sequence length for prompt+target combined (default: 1024, matches CodeT5+ max_source_len)
- `--max-steps`: Maximum training steps (overrides epochs if set, default: -1)
- `--seed`: Random seed for reproducibility (default: 42)

**Features:**
- Automatic FP16 mixed precision for CUDA GPUs (~50% memory savings)
- Gradient checkpointing enabled by default (~40% memory savings)
- Memory-optimized evaluation with `eval_accumulation_steps=1`
- Preprocessing logits for metrics (saves GBs during eval)
- Evaluation every 100 steps, checkpoint save every 200 steps
- Scalable from 12GB to 80GB+ VRAM without code changes

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

Evaluate trained CodeGen model on validation set with comprehensive metrics:

```bash
python scripts/eval_codegen.py \
  --model model/checkpoints/run1-python-codegen \
  --valid-data datasets/python/valid.jsonl \
  --output-dir evaluation/results
```

**Evaluation Options:**
- `--model`: Path to trained model checkpoint
- `--valid-data`: Path to validation JSONL file
- `--max-length`: Maximum sequence length (default: 512)
- `--num-samples`: Number of samples to evaluate (default: None = all)
- `--output-dir`: Directory to save evaluation results (default: ./evaluation_results)

**Output Files:**
- `evaluation_predictions_{timestamp}.jsonl`: Detailed per-sample predictions
- `evaluation_summary_{timestamp}.json`: Summary metrics in JSON format
- `evaluation_report_{timestamp}.txt`: Human-readable report with examples

**Metrics Computed:**
- Token-level accuracy (based on forward pass logits)
- Exact match accuracy (based on generated predictions)
- Per-sample predictions with match status
- Language and repository statistics

## Test CodeGen

Run comprehensive testing on test set with full metrics including Top-K accuracy and edit distance:

```bash
python scripts/test_codegen.py \
  --model model/checkpoints/run1-python-codegen \
  --test-data datasets/python/test.jsonl \
  --output-dir test/results
```

**Test with Top-K Predictions** (slower but more comprehensive):
```bash
python scripts/test_codegen.py \
  --model model/checkpoints/run1-python-codegen \
  --test-data datasets/python/test.jsonl \
  --enable-topk \
  --k 5 \
  --output-dir test/results
```

**Test Options:**
- `--model`: Path to trained model checkpoint
- `--test-data`: Path to test JSONL file
- `--max-length`: Maximum sequence length (default: 512)
- `--num-samples`: Number of samples to test (default: None = all)
- `--output-dir`: Directory to save test results (default: ./test_results)
- `--batch-size`: Batch size for inference (default: 1)
- `--k`: Top-k for accuracy calculation (default: 5)
- `--enable-topk`: Enable top-k predictions generation (slower)

**Output Files:**
- `test_predictions_{timestamp}.jsonl`: Detailed predictions with source/target/predicted
- `test_summary_{timestamp}.json`: Comprehensive metrics in JSON format
- `test_report_{timestamp}.txt`: Human-readable report with statistics

**Comprehensive Metrics:**
- **Basic Metrics**: Exact Match, Exact Match (Case Insensitive), Top-K Accuracy
- **Edit Distance**: Average, Min, Max, Std Dev, Median Levenshtein distance
- **Traditional ML Metrics**: Precision, Recall, F1-Score (sklearn)
- **Statistical Analysis**: Prediction length, target length, class distribution
- **Language-wise Performance**: Per-language accuracy breakdown
- **Performance Stats**: Elapsed time, samples per second

CodeGen evaluation and testing automatically use GPU if available (CUDA), falling back to CPU otherwise.

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