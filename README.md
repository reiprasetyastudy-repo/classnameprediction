# CodeT5+ Class Name Prediction

Fine-tune CodeT5+ and CodeGen to predict class names from code snippets. Includes dataset building, training, and evaluation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For HuggingFace Hub (optional)
pip install -r requirements_hf.txt
cp .env.example .env  # Add HF_TOKEN from https://huggingface.co/settings/tokens
```


## Dataset

### Download from HuggingFace (Recommended)

```bash
# Java
python scripts/download_dataset_from_hf.py \
  --dataset-id reiprasetya-study/java-class-names \
  --output datasets/java

# Python  
python scripts/download_dataset_from_hf.py \
  --dataset-id reiprasetya-study/python-class-names \
  --output datasets/python

# C#
python scripts/download_dataset_from_hf.py \
  --dataset-id reiprasetya-study/csharp-class-names \
  --output datasets/csharp
```

### Build from GitHub (Optional)

```bash
# Python (450 repos)
python scripts/build_dataset.py \
  --repos-file data/repos_python.txt \
  --in data --out datasets \
  --mask --min-lines 3

# Java (99 repos)
python scripts/build_dataset.py \
  --repos-file data/repos_java.txt \
  --in data --out datasets \
  --mask --min-lines 3 \
  --languages java

# C# (134 repos)
python scripts/build_dataset.py \
  --repos-file data/repos_csharp.txt \
  --in data --out datasets \
  --mask --min-lines 3 \
  --languages csharp
```


## Training

### CodeT5+

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Java (275k samples)
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/java \
  --output model/checkpoints/run1-java-codet5 \
  --batch-size 10 --grad-accum 4 --lr 5e-5 --epochs 5 --fp16
  # --resume-from-checkpoint auto  # uncomment to resume from latest checkpoint

# Python (155k samples)
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/python \
  --output model/checkpoints/run1-python-codet5 \
  --batch-size 10 --grad-accum 4 --lr 5e-5 --epochs 5 --fp16

# C# (226k samples)
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/csharp \
  --output model/checkpoints/run1-csharp-codet5 \
  --batch-size 10 --grad-accum 4 --lr 5e-5 --epochs 5 --fp16
```

| Dataset | Samples | VRAM | Time | Accuracy |
|---------|---------|------|------|----------|
| Java | 275k | ~26GB | ~7h | ~85.7% |
| Python | 155k | ~24GB | ~2.5h | ~85.5% |
| C# | 226k | ~26GB | ~5h | TBD |

### CodeGen

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Java
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 10 --grad-accum 4 --lr 5e-5 --epochs 5 \
  --max-length 1024 --gradient-checkpointing --fp16
  # --resume-from-checkpoint auto  # uncomment to resume

# Python
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/python \
  --output model/checkpoints/run1-python-codegen \
  --batch-size 10 --grad-accum 4 --lr 5e-5 --epochs 5 \
  --max-length 1024 --gradient-checkpointing --fp16

# C#
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/csharp \
  --output model/checkpoints/run1-csharp-codegen \
  --batch-size 10 --grad-accum 4 --lr 5e-5 --epochs 5 \
  --max-length 1024 --gradient-checkpointing --fp16
```

| Dataset | Samples | Time | Notes |
|---------|---------|------|-------|
| Java | 275k | ~18-20h | Same params as CodeT5+ |
| Python | 155k | ~10-12h | Same params as CodeT5+ |
| C# | 226k | ~15-17h | Same params as CodeT5+ |

### CodeT5+ vs CodeGen

| Aspect | CodeT5+ | CodeGen |
|--------|---------|---------|
| Architecture | Seq2Seq (Encoder-Decoder) | Causal LM (Decoder-only) |
| Training Speed | Fast (2.5-7h) | Slow (10-20h) |
| VRAM Efficiency | High | Lower (needs gradient checkpointing) |

### Alternative VRAM Configurations

```bash
# 12GB VRAM (RTX 3060, RTX 4060 Ti) - batch 2, accum 16
python scripts/train_codegen.py ... --batch-size 2 --grad-accum 16 --gradient-checkpointing

# 24GB VRAM (RTX 3090, RTX 4090) - batch 6, accum 8  
python scripts/train_codegen.py ... --batch-size 6 --grad-accum 8
```


## Evaluate

```bash
# CodeT5+
python scripts/eval_gpu.py \
  --ckpt model/checkpoints/run1-java-codet5 \
  --data datasets/java \
  --k 5

# CodeGen
python scripts/eval_codegen.py \
  --model model/checkpoints/run1-java-codegen \
  --valid-data datasets/java/test.jsonl \
  --output-dir model/metrics/run1-java-codegen
```

Output: `model/metrics/<run>/metrics.json` with exact match, top-k accuracy, Levenshtein distance.


## Predict

```bash
# CodeT5+
python scripts/predict.py \
  --ckpt model/checkpoints/run1-python-codet5 \
  --language python \
  --file path/to/MyClass.py \
  --k 5

# CodeGen
python scripts/predict_codegen.py \
  --ckpt model/checkpoints/run1-python-codegen \
  --language python \
  --file path/to/MyClass.py \
  --k 5

# From stdin
cat MyClass.java | python scripts/predict.py \
  --ckpt model/checkpoints/run1-java-codet5 \
  --language java
```


## HuggingFace Hub

```bash
# Upload model
python scripts/upload_to_hf.py \
  --ckpt model/checkpoints/run1-java-codet5 \
  --hub-model-id reiprasetya-study/codet5-java-run1 \
  --metrics model/metrics/run1-java-codet5/metrics.json \
  --model-name CodeT5+ \
  --language java

# Upload model.safetensors manually (if failed)
huggingface-cli upload reiprasetya-study/codet5-java-run1 \
  model/checkpoints/run1-java-codet5/model.safetensors \
  model.safetensors \
  --repo-type model

# Upload checkpoints (optional, for resume on different machine)
huggingface-cli upload reiprasetya-study/codet5-java-run1 \
  model/checkpoints/run1-java-codet5/checkpoint-34495 \
  checkpoints/checkpoint-34495 \
  --repo-type model

# Download checkpoints
huggingface-cli download reiprasetya-study/codet5-java-run1 \
  --include "checkpoints/*" \
  --local-dir model/checkpoints/run1-java-codet5

# View results without downloading
python scripts/view_hf_results.py \
  --hub-model-id reiprasetya-study/codet5-java-run1
```


## GPU Selection

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=1 python scripts/train.py ... --cuda-device 0
```


## Notes

- Use `--mask` to replace class names with `____` (prevents label leakage)
- Export `GITHUB_TOKEN` for higher GitHub clone limits
- Checkpoints saved every 1000 steps (CodeT5+) or 2000 steps (CodeGen)
- See [LOGGING.md](LOGGING.md) and [CLAUDE.md](CLAUDE.md) for more details
