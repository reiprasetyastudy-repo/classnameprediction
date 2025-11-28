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
```bash
python scripts/build_dataset.py \
  --repos-file data/repos.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3
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

## Important Details

### Language Support
- Python: Regex-based class extraction (`class <Name>:`)
- Java: Supports class, interface, enum keywords
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
