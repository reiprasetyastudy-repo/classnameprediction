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

**Python Dataset (450 repositories):**
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


## Training (Optimized for RTX 5090 - 32GB VRAM)

These configurations are optimized for best performance, training time, and GPU utilization on RTX 5090.

### CodeT5+ Training

**Java Dataset (275k samples):**
```bash
# Set environment variable to reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/java \
  --output model/checkpoints/run1-java \
  --batch-size 10 \
  --grad-accum 4 \
  --lr 5e-5 \
  --epochs 5 \
  --fp16
```

**Python Dataset (155k samples):**
```bash
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/python \
  --output model/checkpoints/run1-python \
  --batch-size 12 \
  --grad-accum 3 \
  --lr 5e-5 \
  --epochs 5 \
  --fp16
```

**Performance:**
| Dataset | Batch | Grad Accum | Effective Batch | VRAM | Time | Accuracy |
|---------|-------|------------|-----------------|------|------|----------|
| Java | 10 | 4 | 40 | ~26GB | ~7h | ~85.7% |
| Python | 12 | 3 | 36 | ~23GB | ~2.5h | ~85.5% |

### CodeGen Training

**Java Dataset:**
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

**Python Dataset:**
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/python \
  --output model/checkpoints/run1-python-codegen \
  --batch-size 12 \
  --grad-accum 3 \
  --lr 5e-5 \
  --epochs 5 \
  --max-length 1024 \
  --gradient-checkpointing
```

**Performance:**
| Dataset | Time | Preprocessing | Notes |
|---------|------|---------------|-------|
| Java | ~18-20h | 3-4 min | Requires gradient checkpointing (~10x slower than CodeT5+) |
| Python | ~10-12h | 3-4 min | Same effective batch as CodeT5+ for fair comparison |

### Key Differences: CodeT5+ vs CodeGen

| Aspect | CodeT5+ | CodeGen |
|--------|---------|---------|
| Architecture | Seq2Seq (Encoder-Decoder) | Causal LM (Decoder-only) |
| Training Speed | Fast (2.5-7h) | Slow (10-20h) |
| Gradient Checkpointing | Not needed | Required (causes 10x slowdown) |
| VRAM Efficiency | High | Lower (needs checkpointing) |
| Preprocessing | Cached automatically | Custom (~3-4 min once) |

**Notes:**
- CodeT5+ is recommended for faster iteration
- CodeGen may have slightly different prediction patterns due to architecture
- Both achieve similar accuracy (~85-86%)
- All configs use seed 42 for reproducibility
- Logs saved to `logs/codet5/` and `logs/codegen/`


## Resume Training from Checkpoint

Both `train.py` (CodeT5+) and `train_codegen.py` (CodeGen) support checkpoint resume. If training crashes or is interrupted, resume from the latest checkpoint:

**CodeT5+ - Auto-detect latest checkpoint:**
```bash
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/java \
  --output model/checkpoints/run1-java \
  --batch-size 10 \
  --grad-accum 4 \
  --lr 5e-5 \
  --epochs 5 \
  --fp16 \
  --resume-from-checkpoint auto
```

**CodeGen - Auto-detect latest checkpoint:**
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
# Add to any training command:
--resume-from-checkpoint model/checkpoints/run1-java/checkpoint-15000
```

**Checkpoint Management:**
- CodeT5+: Saved every **1000 steps**, keeps last 2 checkpoints
- CodeGen: Saved every **2000 steps**, keeps last 2 checkpoints
- Contains: model weights, optimizer state, scheduler state, RNG state, training progress
- Resume behavior: Continues from exact step with same loss and learning rate
- **PyTorch 2.6 compatible**: Uses monkey-patched `torch.load` for checkpoint resume

**Upload/Download Checkpoints (for VM migration):**
```bash
# Upload to HuggingFace Hub
hf upload username/checkpoint-name model/checkpoints/run1-java/checkpoint-15000 --repo-type model --private

# Download on new VM
hf download username/checkpoint-name --local-dir model/checkpoints/run1-java/checkpoint-15000 --repo-type model
```


## Alternative VRAM Configurations

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
