# CodeT5+ Class Name Prediction

This repo scaffolds an end-to-end pipeline to fine-tune CodeT5+ and CodeGen to predict class names from code snippets. It includes dataset building from GitHub repos, training, and evaluation.

## Setup

Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For HuggingFace Hub features (optional):

```bash
pip install -r requirements_hf.txt
```

For HuggingFace Hub, setup token:

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN from https://huggingface.co/settings/tokens
```


## Dataset

### Quick Start: Download from HuggingFace Hub (Recommended)

Download pre-built datasets in 1-2 minutes (vs 30-60 minutes building from scratch):

**Java Dataset:**
```bash
python scripts/download_dataset_from_hf.py \
  --dataset-id reiprasetya-study/java-class-names \
  --output datasets/java
```

**Python Dataset:**
```bash
python scripts/download_dataset_from_hf.py \
  --dataset-id reiprasetya-study/python-class-names \
  --output datasets/python
```

**Load directly in Python:**
```python
from datasets import load_dataset
dataset = load_dataset("reiprasetya-study/java-class-names")
```

### Alternative: Build from GitHub (Optional)

Build dataset from scratch (30-60 minutes, only needed once).

Prepare a list of repos (one per line): `data/repos.txt`

```
https://github.com/pallets/flask
https://github.com/psf/requests
```

Run dataset builder:

**Python Dataset (300 repositories):**
```bash
python scripts/build_dataset.py \
  --repos-file data/repos_python.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3
```

**Java Dataset (99 repositories):**
```bash
python scripts/build_dataset.py \
  --repos-file data/repos_java.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3 \
  --languages java
```

Outputs per language: `datasets/<language>/{train,valid,test}.jsonl` with fields `language, repo, path, class_span, source, target`.

**Upload to HuggingFace Hub (optional, for reuse):**
```bash
python scripts/upload_dataset_to_hf.py \
  --dataset-dir datasets/java \
  --dataset-id reiprasetya-study/java-class-names \
  --language java \
  --private
```


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
  --batch-size 12 \
  --grad-accum 3 \
  --lr 5e-5 \
  --epochs 5 \
  --max-length 1024 \
  --gradient-checkpointing
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
| Batch Size | 12 | 12 ✓ |
| Gradient Accumulation | 3 | 3 ✓ |
| Effective Batch Size | 12 × 3 = 36 | 12 × 3 = 36 ✓ |
| Learning Rate | 5e-5 | 5e-5 ✓ |
| Epochs | 5 | 5 ✓ |
| Max Length | 1024 (source) + 32 (target) | 1024 (combined) ✓ |
| FP16 | Yes | Yes ✓ |
| Gradient Checkpointing | No | Yes (required) |
| Seed | 42 | 42 ✓ |
| VRAM Usage | ~23GB | ~23GB |
| Training Time | ~2-2.5 hours | ~18-20 hours |
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
  --batch-size 12 \
  --grad-accum 4 \
  --lr 2e-5 \
  --epochs 10 \
  --max-length 1024 \
  --gradient-checkpointing
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
| Batch Size | 12 | 12 ✓ |
| Gradient Accumulation | 4 | 4 ✓ |
| Effective Batch Size | 12 × 4 = 48 | 12 × 4 = 48 ✓ |
| Learning Rate | 2e-5 | 2e-5 ✓ |
| Epochs | 10 | 10 ✓ |
| Max Length | 1024 (source) + 32 (target) | 1024 (combined) ✓ |
| FP16 | Yes | Yes ✓ |
| Gradient Checkpointing | No | Yes (required) |
| Seed | 42 | 42 ✓ |
| VRAM Usage | ~23GB | ~23GB |
| Training Time | ~3-3.5 hours | ~36-40 hours |
| Preprocessing | N/A | 3-4 minutes |
| Eval Frequency | Every 100 steps | Every 1000 steps |

**Notes:**
- **Configuration 1 (Standard):** 5 epochs, CodeT5+ ~2-2.5 hours, CodeGen ~18-20 hours - Good baseline for comparison
- **Configuration 2 (Extended):** 10 epochs, CodeT5+ ~3-3.5 hours, CodeGen ~36-40 hours - Full training for best performance
- **Apple to Apple Comparison:** ALL parameters are identical (batch size, gradient accumulation, learning rate, epochs, seed) to ensure fair comparison
- **Gradient Checkpointing Trade-off:** CodeGen requires gradient checkpointing to fit batch=12 in VRAM, making it ~10x slower than CodeT5+. This is an architectural difference between Causal LM (CodeGen) and Seq2Seq (CodeT5+)
- **VRAM Usage:** Both use ~23GB with these settings
- **Training Time Difference:** CodeT5+ is much faster due to not requiring gradient checkpointing. This is expected and part of the comparison
- **Optimized Preprocessing:** CodeGen preprocesses all samples once (~3-4 min), then training is fast. Uses batch tokenization (1000 samples/batch) for speed
- **Dynamic Padding:** Batch-wise dynamic padding (pads to longest in batch, not max_length)
- **Max Length:** CodeT5+ uses separate lengths for source (1024) and target (32). CodeGen uses combined length (1024) for prompt+target in single sequence
- Logs are saved to `logs/codet5/` and `logs/codegen/` respectively
- All configurations use seed 42 for reproducibility


## Resume Training from Checkpoint

If training crashes or is interrupted, resume from the latest checkpoint:

**Auto-detect latest checkpoint:**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 12 \
  --grad-accum 3 \
  --lr 5e-5 \
  --epochs 5 \
  --max-length 1024 \
  --gradient-checkpointing \
  --resume-from-checkpoint auto
```

**Resume from specific checkpoint:**
```bash
python scripts/train_codegen.py \
  --resume-from-checkpoint model/checkpoints/run1-java-codegen/checkpoint-2000 \
  ...
```

**Upload checkpoint to HuggingFace (for VM migration):**
```bash
hf upload reiprasetya-study/codegen-java-checkpoint2000 \
  model/checkpoints/run1-java-codegen/checkpoint-2000 \
  --repo-type model \
  --private
```

**Download checkpoint on new VM:**
```bash
hf download reiprasetya-study/codegen-java-checkpoint2000 \
  --local-dir model/checkpoints/run1-java-codegen/checkpoint-2000 \
  --repo-type model
```

Checkpoints are saved every 2000 steps and contain: model weights, optimizer state, scheduler state, and training progress. Training will resume from the exact step with the same loss and learning rate.


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


## Train CodeGen (Other VRAM Configurations)

### For 12GB VRAM (e.g., RTX 3060, RTX 4060 Ti)

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

Effective batch size: 2 × 16 = 32

### For 24GB VRAM (e.g., RTX 3090, RTX 4090)

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

Effective batch size: 6 × 8 = 48


## Evaluate

### Evaluate CodeT5+ with GPU (recommended - 10-20x faster)
```bash
python scripts/eval_gpu.py \
  --ckpt model/checkpoints/run1-python \
  --data datasets/python \
  --k 5
```

### Evaluate CodeGen
```bash
python scripts/eval_codegen.py \
  --model model/checkpoints/run1-python-codegen \
  --valid-data datasets/python/test.jsonl \
  --output-dir evaluation/results
```

Produces: `model/metrics/<run>/metrics.json` with exact match, case-insensitive EM, top-k accuracy, and average Levenshtein distance.

Training artifacts (under your `--output`):
- `training_log.csv` — step-wise training/eval metrics
- `training_curve.png` — training/eval loss plot
- `checkpoints.txt` — discovered checkpoint directories and final model path


## HuggingFace Hub Integration (Models)

Upload trained models to HuggingFace Hub for persistent storage and easy sharing.

### Auto-Upload During Training

```bash
python scripts/train_codegen.py \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 12 --grad-accum 3 --lr 5e-5 --epochs 5 \
  --gradient-checkpointing \
  --push-to-hub \
  --hub-model-id reiprasetya-study/codegen-java-run1 \
  --language java \
  --private
```

### Manual Upload After Training

```bash
python scripts/upload_to_hf.py \
  --ckpt model/checkpoints/run1-java-codegen \
  --hub-model-id reiprasetya-study/codegen-java-run1 \
  --metrics model/metrics/run1-java-codegen/metrics.json \
  --language java \
  --private
```

### View Results Without Downloading Model

```bash
python scripts/view_hf_results.py \
  --hub-model-id reiprasetya-study/codegen-java-run1
```


## Predict from a Snippet

Use the trained checkpoint to predict a class name from a code snippet (file or stdin). By default, the header class name is masked to match training.

From a file (Python):
```bash
python scripts/predict.py \
  --ckpt model/checkpoints/run1-python \
  --language python \
  --file path/to/MyWrongClass.py \
  --k 5
```

From stdin (Java):
```bash
cat path/to/Foo.java | python scripts/predict.py --ckpt model/checkpoints/run1-java --language java
```


## Predict CodeGen

```bash
python scripts/predict_codegen.py \
  --ckpt model/checkpoints/run1-python-codegen \
  --language python \
  --file path/to/MyWrongClass.py \
  --k 5
```


## GPU Selection and CUDA Devices

If you have multiple GPUs and want to select which one to use for training, set the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/python \
  --output model/checkpoints/run1-python \
  --batch-size 8 --grad-accum 2 --epochs 3 --fp16 --cuda-device 0
```

The script argument `--cuda-device` should be set to 0 when using `CUDA_VISIBLE_DEVICES`, as the visible device will be mapped to index 0.


## Notes

- Start with Python for best heuristic parsing; Java is supported with basic regex
- Use `--mask` to replace the declared class identifier with `____` to avoid label leakage
- Consider rate limits and licenses when mining GitHub; export `GITHUB_TOKEN` for higher clone limits
- Large files and artifacts are ignored via `.gitignore`
- For logging details, see [LOGGING.md](LOGGING.md)
- For Claude Code instructions, see [CLAUDE.md](CLAUDE.md)
