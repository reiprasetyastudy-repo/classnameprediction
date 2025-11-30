# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning pipeline for fine-tuning transformer models (CodeT5+ and CodeGen) to predict class names from code snippets. The pipeline includes dataset building from GitHub repositories, model training, evaluation, and inference.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GitHub cloning during dataset building, set `GITHUB_TOKEN` environment variable for higher rate limits.

## Common Commands

### Build Dataset

Python dataset (60 repositories):
```bash
python scripts/build_dataset.py \
  --repos-file data/repos_python.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3
```

Java dataset (99 repositories):
```bash
python scripts/build_dataset.py \
  --repos-file data/repos_java.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3 \
  --languages java
```

C# dataset (134 repositories):
```bash
python scripts/build_dataset.py \
  --repos-file data/repos_csharp.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3 \
  --languages csharp
```

Outputs: `datasets/<language>/{train,valid,test}.jsonl` with fields: `language`, `repo`, `path`, `class_span`, `source`, `target`.

### Train CodeT5+
```bash
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/python \
  --output model/checkpoints/run1-python \
  --batch-size 8 --grad-accum 2 --epochs 3 --fp16
```

For multi-GPU systems, use `CUDA_VISIBLE_DEVICES`:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --cuda-device 0 ...
```

### Train CodeGen
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/python \
  --output model/checkpoints/run1-python-codegen \
  --batch-size 1 --grad-accum 8 \
  --max-source-len 256 \
  --gradient-checkpointing \
  --cpu
```

Use `--cpu` for CPU training, `--bf16` for Apple Silicon/MPS, `--fp16` for CUDA.

### Train C# Models

CodeT5+ for C#:
```bash
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/csharp \
  --output model/checkpoints/run1-csharp-codet5 \
  --batch-size 10 --grad-accum 4 --lr 5e-5 --epochs 5 --fp16
```

CodeGen for C#:
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/csharp \
  --output model/checkpoints/run1-csharp-codegen \
  --batch-size 10 --grad-accum 4 --lr 5e-5 --epochs 5 \
  --max-length 1024 --gradient-checkpointing --fp16
```

### Resume Training from Checkpoint
Auto-detect latest checkpoint:
```bash
python scripts/train_codegen.py \
  --model Salesforce/codegen-350M-mono \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --batch-size 12 --grad-accum 3 --lr 5e-5 --epochs 5 \
  --max-length 1024 --gradient-checkpointing \
  --resume-from-checkpoint auto
```

Resume from specific checkpoint:
```bash
python scripts/train_codegen.py \
  --resume-from-checkpoint model/checkpoints/run1/checkpoint-2000 \
  ...
```

Checkpoints saved every 2000 steps. Contains model, optimizer, scheduler state for exact resume.

### Evaluate Models
CodeT5+:
```bash
python scripts/eval.py \
  --ckpt model/checkpoints/run1-python \
  --data datasets/python \
  --k 5
```

CodeGen:
```bash
python scripts/eval_codegen.py \
  --ckpt model/checkpoints/run1-python-codegen \
  --data datasets/python \
  --k 5
```

Outputs: `model/metrics/<run>/metrics.json` with exact match, case-insensitive EM, top-k accuracy, and average Levenshtein distance.

### Predict Class Names
From file:
```bash
python scripts/predict.py \
  --ckpt model/checkpoints/run1-python \
  --language python \
  --file path/to/MyWrongClass.py \
  --k 5
```

From stdin:
```bash
cat path/to/Foo.java | python scripts/predict.py \
  --ckpt model/checkpoints/run1-java \
  --language java
```

CodeGen prediction uses `scripts/predict_codegen.py` with same arguments.

### Plot Training Curves
```bash
python scripts/plot_training_log.py
```

### HuggingFace Hub Integration

**Dataset Management:**
Upload datasets to HuggingFace Datasets Hub to avoid rebuilding on cloud instances (30-60 min build → 1-2 min download).

Upload dataset (once):
```bash
python scripts/upload_dataset_to_hf.py \
  --dataset-dir datasets/java \
  --dataset-id reiprasetya-study/java-class-names \
  --language java \
  --private
```

Download on new instance (fast):
```bash
# Java dataset
python scripts/download_dataset_from_hf.py \
  --dataset-id reiprasetya-study/java-class-names \
  --output datasets/java

# C# dataset
python scripts/download_dataset_from_hf.py \
  --dataset-id reiprasetya-study/csharp-class-names \
  --output datasets/csharp
```

Load directly in Python:
```python
from datasets import load_dataset
dataset = load_dataset("reiprasetya-study/java-class-names")
# or
dataset = load_dataset("reiprasetya-study/csharp-class-names")
```

**Model Management:**
Upload trained models for persistent storage and easy sharing.

Auto-upload during training:
```bash
python scripts/train_codegen.py \
  --data datasets/java \
  --output model/checkpoints/run1-java-codegen \
  --push-to-hub \
  --hub-model-id reiprasetya-study/codegen-java-run1 \
  --language java \
  --private
```

Manual upload after training:
```bash
python scripts/upload_to_hf.py \
  --ckpt model/checkpoints/run1-java-codegen \
  --hub-model-id reiprasetya-study/codegen-java-run1 \
  --metrics model/metrics/run1-java-codegen/metrics.json \
  --language java
```

View results without downloading model:
```bash
python scripts/view_hf_results.py \
  --hub-model-id reiprasetya-study/codegen-java-run1
```

Setup: Install `requirements_hf.txt` and configure `HF_TOKEN` in `.env` file.

## Architecture

### Data Pipeline (build_dataset.py)
- Clones GitHub repos into `data/repos/<owner>__<repo>`
- Parses classes using regex: Python (`PY_CLASS_RE`) and Java (`JAVA_CLASS_RE`)
- Masking: Replaces class identifier with `____` in declaration (controlled by `--mask`)
- Deduplication: Normalizes whitespace, removes duplicates per language
- Splits: 80/10/10 train/valid/test by default

### Training Pipeline
**CodeT5+ (train.py):**
- Seq2Seq model from HuggingFace transformers
- Prompt template: `"Predict class name:\n{source}\nName:"`
- Uses `Seq2SeqTrainer` with custom `CSVLoggerCallback`
- Early GPU device setting to avoid CUDA allocation issues
- Training artifacts: `training_log.csv`, `training_curve.png`, `checkpoints.txt`

**CodeGen (train_codegen.py):**
- Causal LM (GPT-style) model
- Same prompt template but formatted as causal completion: `prompt + " " + target`
- Custom `CustomDataCollator`: Pads with `-100` for labels (ignore index)
- MPS optimizations: `MPSMemoryCallback` for Apple Silicon memory management
- Optimizer selection: `adamw_torch` for MPS, `adafactor` for CUDA
- Gradient checkpointing available via `--gradient-checkpointing`

### Evaluation (eval.py, eval_codegen.py)
- Beam search with `num_beams=k` for top-k predictions
- Metrics: exact match, case-insensitive match, top-k accuracy, Levenshtein distance
- Normalizes predictions by taking first token

### Prediction (predict.py, predict_codegen.py)
- Masks class header by default (disable with `--no-mask`)
- Returns top-k predictions with deduplication
- Supports both file and stdin input

### Utilities
- `csv_logger.py`: Custom HuggingFace callback for step-wise CSV logging
- `diagnose.py`: Diagnostic utilities
- `plot_training_log.py`: Visualizes training curves from CSV logs
- `hf_utils.py`: HuggingFace Hub utility functions (upload, download, token management)
- `upload_dataset_to_hf.py`: Upload datasets to HuggingFace Datasets Hub
- `download_dataset_from_hf.py`: Download datasets from HuggingFace Datasets Hub
- `upload_to_hf.py`: Upload trained models to HuggingFace Model Hub
- `view_hf_results.py`: View model results without downloading full model
- `generate_readme.py`: Auto-generate README for HuggingFace model repositories

## Important Details

### Dataset Source Repositories
High-quality repository lists for building datasets:
- **Java**: `data/repos_java.txt` (99 repositories)
  - Covers popular Java projects: Spring, Elasticsearch, Kafka, etc.
  - Focus on enterprise applications and frameworks
- **Python**: `data/repos_python.txt` (450 repositories)
  - Diverse categories: web frameworks, ML/AI, scientific computing, algorithms, educational projects, Django ecosystem, testing, code quality
  - Examples: Django, Flask, PyTorch, TensorFlow, pandas, NumPy, FastAPI, Streamlit, TheAlgorithms/Python, etc.
  - All projects follow PEP 8 naming conventions
  - Includes 150+ educational/algorithmic repos with simple, well-named classes (Circle, Stack, Queue, BubbleSort, etc.)
- **C#**: `data/repos_csharp.txt` (134 repositories)
  - Covers popular .NET projects: ASP.NET Core, EF Core, MAUI, Orleans, etc.
  - Categories: web frameworks, ORM, testing (xUnit, NUnit, Moq), UI (Avalonia, MahApps), messaging (MassTransit, RabbitMQ), e-commerce (nopCommerce, Smartstore)
  - Examples: Newtonsoft.Json, AutoMapper, Serilog, Polly, FluentValidation, MediatR, etc.
  - All projects follow C# PascalCase naming conventions
  - Excludes generated SDK code (Azure, AWS, Google Cloud) for quality naming patterns
- All repositories are open source, well-maintained, and have good class naming patterns

### Language Support
- Python: Regex-based class extraction (`class <Name>:`)
- Java: Supports class, interface, enum keywords
- C#: Supports class, interface, struct, enum, record keywords with access modifiers
- Heuristic parsing only—no AST-based extraction

### Masking Behavior
Training uses `--mask` to replace declared class names with `____` to prevent label leakage. Prediction scripts mask by default to match training data distribution.

### Device Management
- CodeT5+: Early `torch.cuda.set_device()` before model load (train.py:32-35)
- CodeGen: Supports CPU (`--cpu`), CUDA (`--fp16`), and MPS (`--bf16`)
- Multi-GPU: Use `CUDA_VISIBLE_DEVICES` environment variable + `--cuda-device 0`

### Memory Optimization
- CodeGen on MPS: Custom memory callback, no pin memory, AdamW optimizer
- Gradient checkpointing available for both models
- Gradient accumulation for effective larger batch sizes

### Output Locations
- Checkpoints: `model/checkpoints/<run>/`
- Metrics: `model/metrics/<run>/metrics.json`
- Training logs: `<output>/training_log.csv`, `training_curve.png`, `checkpoints.txt`
- Datasets: `datasets/<language>/{train,valid,test}.jsonl`
- HuggingFace Datasets: `https://huggingface.co/datasets/<username>/<dataset-name>`
- HuggingFace Models: `https://huggingface.co/<username>/<model-name>`

## Research Documentation

### Paper Drafts
The `docs/draft/` directory contains draft sections for the research paper on class name generation:

**Evaluation and Comparison Methodology:**
- `evaluation_methodology.md`: Detailed methodology draft with bullet points and structured sections
  - Evaluation metrics: EM, EM_CI, Top-K Accuracy, Levenshtein Distance, Sample Count
  - Evaluation procedure: Test set preparation, prediction normalization, metric computation
  - Comprehensive comparison between CodeT5+ and CodeGen across 9 dimensions:
    1. Architecture comparison (Encoder-Decoder vs Causal LM)
    2. Training configuration differences
    3. Hyperparameter and optimization settings
    4. Computational efficiency (training time, memory, hardware support)
    5. Performance analysis on prediction metrics
    6. Strengths and limitations of each model
    7. Language-specific analysis (Python vs Java)
    8. Statistical significance testing
    9. Use case recommendations
  - Fair comparison guidelines to ensure objective evaluation

- `evaluation_methodology_ieee.md`: IEEE conference paper format version
  - Same content as above but formatted in dense paragraphs
  - Suitable for inclusion in conference paper submissions
  - Follows IEEE conference paper writing style with narrative flow
  - Uses bold for sub-topics within paragraphs (e.g., **Architecture Comparison.**)

**Reference Papers:**
- `our draft.pdf`: Current research paper draft
- `example 1 - Handling Out-of-Vocabulary in Indonesian POS Tagging A Comparative Study.pdf`: Reference paper example
- `example 2 - Towards Better HS Code Prediction A Comparative Study of Machine Learning and NLP Approaches.pdf`: Reference paper example

These drafts follow the structure and style observed in the reference papers, using formal Indonesian language for methodology sections and English for the IEEE format version.
