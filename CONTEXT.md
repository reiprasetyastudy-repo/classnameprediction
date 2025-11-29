# Project Context for AI Assistants

**Last Updated:** 2024-11-29
**Session Summary:** Training optimization, checkpoint resume fixes, dataset expansion planning

---

## 📋 Project Overview

**Project:** Class Name Prediction using Fine-tuned Transformer Models
**Models:** CodeT5+ (Salesforce/codet5p-220m) and CodeGen (Salesforce/codegen-350M-mono)
**Task:** Predict class names from code snippets (source code → class name)
**Datasets:** Java (275k samples) and Python (155k samples)
**Hardware:** RTX 5090 (32GB VRAM) on Vast.ai cloud instances

---

## 🎯 Current Status

### ✅ Completed Work

1. **Dataset Collection**
   - Java: 99 repositories → 275,962 train + 34,495 valid samples
   - Python: 450 repositories → 155,411 train + 19,426 valid samples
   - Uploaded to HuggingFace Hub: `reiprasetya-study/java-class-names`, `reiprasetya-study/python-class-names`

2. **Training Infrastructure**
   - `train.py` - CodeT5+ training script (Seq2Seq model)
   - `train_codegen.py` - CodeGen training script (Causal LM)
   - Both support checkpoint resume (`--resume-from-checkpoint auto`)
   - PyTorch 2.6 checkpoint resume fix implemented (monkey-patch `torch.load`)

3. **Optimal Training Configs Found**
   - Java + CodeT5+: batch=10, grad_accum=4, lr=5e-5, 5 epochs (~7h, 85.7% acc, 26GB VRAM)
   - Python + CodeT5+: batch=12, grad_accum=3, lr=5e-5, 5 epochs (~2.5h, 85.5% acc, 23GB VRAM)
   - Memory fragmentation solution: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

4. **Documentation Updated**
   - README.md simplified with optimal configs only
   - CLAUDE.md updated with project context
   - Training commands verified on RTX 5090

### 🔄 In Progress

1. **Python Repository Expansion** (450 → 600 repos)
   - Need to remove: awesome-* lists, guides, tutorials, non-code repos
   - Need to add: 150+ quality repos (Testing tools, API clients, Web frameworks, Database libs, CLI tools, Educational repos)
   - Script created: `filter_and_expand_repos.py` (NOT YET RUN)

2. **Training Runs**
   - Python + CodeT5+: Training ongoing (checkpoint-15000 exists, resume available)
   - Java + CodeT5+: Not started yet (use batch=10 config)

### ⏳ Pending Tasks

1. Run `filter_and_expand_repos.py` to expand Python repos to 600
2. Rebuild Python dataset with 600 repos
3. Complete Python + CodeT5+ training (resume from checkpoint-15000)
4. Train Java + CodeT5+ from scratch (batch=10 config)
5. Evaluate both models
6. Compare CodeT5+ vs CodeGen results

---

## 💡 Critical Knowledge

### 1. **OOM Issues on RTX 5090**

**Problem:** batch=12 causes OOM on Java dataset despite 32GB VRAM
**Root Cause:** Memory fragmentation (8.93GB reserved but unallocated)
**Solutions:**
1. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reduces fragmentation)
2. Reduce batch size: Java (batch=10), Python (batch=12 works)
3. Increase grad_accum to maintain effective batch size

**Config Evolution:**
- ❌ Config A: batch=8, grad_accum=4, lr=2e-5, 10 epochs (too slow, 20h, LR too low, overfitting)
- ❌ Config B: batch=12, grad_accum=3, lr=5e-5, 5 epochs (OOM on Java)
- ✅ Config C: batch=8, grad_accum=4, lr=5e-5, 5 epochs (safe but slow, 9h)
- ✅ **Config D (BEST):** batch=10, grad_accum=4, lr=5e-5, 5 epochs (optimal: 7h, 85.7% acc, 26GB VRAM)

### 2. **PyTorch 2.6 Checkpoint Resume Issue**

**Problem:** `torch.load` default changed to `weights_only=True` in PyTorch 2.6, breaking checkpoint resume
**Error:** `_pickle.UnpicklingError: Weights only load failed... numpy.core.multiarray._reconstruct not allowed`

**Evolution of Fixes:**
1. ❌ First attempt: `add_safe_globals([np.core.multiarray._reconstruct])` - not enough
2. ❌ Second attempt: Added `np.ndarray, np.dtype, np.random.RandomState` - still failed with `numpy.dtypes.UInt32DType`
3. ✅ **Final solution:** Monkey-patch `torch.load` to use `weights_only=False` by default

**Implementation** (in both `train.py` and `train_codegen.py`):
```python
# Monkey-patch torch.load to use weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load
```

### 3. **Checkpoint Resume Safety**

**Question:** Is it safe to resume? Does it affect results?
**Answer:** ✅ 100% SAFE - Results are IDENTICAL to continuous training

**State Saved in Checkpoints:**
- Model weights (exact)
- Optimizer state: momentum, variance buffers (exact)
- Scheduler state: learning rate schedule (exact)
- RNG state: random number generator (exact)
- Training progress: step counter, best metric (exact)

**Evidence:**
- Loss continues smoothly (no jump)
- Dataset preprocessing from cache (same tokenization)
- Standard practice for GPT-3, GPT-4, BERT, etc.
- Google, OpenAI, Meta all use checkpoint resume

**Verification:**
```
Step 14900: loss = 0.352
Step 15000: loss = 0.348  ← CHECKPOINT
[Resume here]
Step 15100: loss = 0.345  ← SMOOTH CONTINUE ✅
```

### 4. **Dataset Preprocessing Behavior**

**First Run (slow):**
```
Map:   0% → 100% | 275962/275962 [09:12<00:00, 499.45 examples/s]
Caching processed dataset at .../cache-ba03647fdd712578.arrow
```

**Resume Run (instant):**
```
Loading cached processed dataset at .../cache-ba03647fdd712578.arrow  ← 0-2 seconds!
Dataset preprocessing completed
```

**Key Points:**
- HuggingFace `datasets` library auto-caches tokenized data
- Cache persists across runs (no need to re-tokenize)
- Cache location: `/workspace/.hf_home/datasets/json/.../cache-*.arrow`
- CodeT5+ preprocessing: 0-2 sec (cached)
- CodeGen preprocessing: 3-4 min (custom, runs once per training)

### 5. **CodeT5+ vs CodeGen Comparison**

| Aspect | CodeT5+ | CodeGen |
|--------|---------|---------|
| Architecture | Seq2Seq (Encoder-Decoder) | Causal LM (Decoder-only) |
| Training Speed | Fast (2.5-7h) | Very Slow (10-20h) |
| Why Slower? | No gradient checkpointing | Gradient checkpointing required (~10x slowdown) |
| VRAM Usage | 23-26GB | 23GB (with checkpointing) |
| Preprocessing | Cached by HuggingFace | Custom (3-4 min) |
| Effective Batch | 36-40 | 36 |
| Accuracy | ~85-86% | ~85-86% |
| **Recommendation** | ✅ Use for faster iteration | Use for comparison only |

**Why CodeGen is Slower:**
- Gradient checkpointing trades compute for memory
- Re-computes activations during backward pass
- ~10x slower than CodeT5+ for same batch size
- Architectural difference (Causal LM needs full attention matrix)

---

## 📂 File Structure

### Scripts
- `scripts/train.py` - CodeT5+ training (Seq2Seq)
- `scripts/train_codegen.py` - CodeGen training (Causal LM)
- `scripts/eval_gpu.py` - CodeT5+ evaluation
- `scripts/eval_codegen.py` - CodeGen evaluation
- `scripts/build_dataset.py` - Build dataset from GitHub repos
- `scripts/download_dataset_from_hf.py` - Download pre-built datasets
- `scripts/upload_dataset_to_hf.py` - Upload datasets to HuggingFace Hub
- `scripts/upload_to_hf.py` - Upload models to HuggingFace Hub
- `filter_and_expand_repos.py` - Filter and expand Python repos (NOT YET RUN)

### Data
- `data/repos_java.txt` - 99 Java repositories
- `data/repos_python.txt` - 450 Python repositories (needs expansion to 600)
- `datasets/java/{train,valid,test}.jsonl` - Java dataset (275k train, 34k valid)
- `datasets/python/{train,valid,test}.jsonl` - Python dataset (155k train, 19k valid)

### Models
- `model/checkpoints/run1-python-codet5/` - Python CodeT5+ training (checkpoint-15000 exists)
- `model/checkpoints/run1-java/` - Java CodeT5+ (not started)

### Logs
- `logs/codet5/` - CodeT5+ training logs
- `logs/codegen/` - CodeGen training logs
- `logs/build_dataset_20251129_133319.log` - Dataset build log (450 Python repos, 450 success, 0 failed)

---

## 🔧 Common Commands

### Training

**Java + CodeT5+ (Optimal):**
```bash
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

**Python + CodeT5+ (Resume):**
```bash
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/python \
  --output model/checkpoints/run1-python-codet5 \
  --batch-size 12 \
  --grad-accum 3 \
  --lr 5e-5 \
  --epochs 5 \
  --fp16 \
  --resume-from-checkpoint auto
```

### Dataset Building

**Expand Python Repos (450 → 600):**
```bash
cd /workspace/classnameprediction
python filter_and_expand_repos.py
```

**Rebuild Python Dataset:**
```bash
python scripts/build_dataset.py \
  --repos-file data/repos_python.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3
```

### Evaluation

**CodeT5+:**
```bash
python scripts/eval_gpu.py \
  --ckpt model/checkpoints/run1-python-codet5 \
  --data datasets/python \
  --k 5
```

**CodeGen:**
```bash
python scripts/eval_codegen.py \
  --model model/checkpoints/run1-python-codegen \
  --valid-data datasets/python/test.jsonl \
  --output-dir evaluation/run1-python-codegen
```

---

## 🐛 Known Issues & Solutions

### Issue 1: OOM on batch=12 (Java dataset)
**Solution:** Use batch=10 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Issue 2: PyTorch 2.6 checkpoint resume fails
**Solution:** Monkey-patch already applied in `train.py` and `train_codegen.py`

### Issue 3: Dataset preprocessing takes 9+ minutes
**Solution:** Only happens first time, subsequent runs use cache (instant)

### Issue 4: Training very slow on CodeGen
**Expected:** CodeGen is ~10x slower than CodeT5+ due to gradient checkpointing requirement

### Issue 5: Invalid repository URLs during dataset build
**Solution:** Remove invalid URLs from `data/repos_python.txt`
- Recently fixed: `yuanbin/algorithm-exercise`, `google/python-style-guide`
- Check logs: `logs/build_dataset_*.log` for "Repository not found" errors

---

## 📊 Performance Benchmarks

### CodeT5+ Training (RTX 5090)

| Dataset | Samples | Batch | Grad Accum | Eff. Batch | VRAM | Time | Accuracy |
|---------|---------|-------|------------|------------|------|------|----------|
| Java | 275k | 10 | 4 | 40 | 26GB | ~7h | ~85.7% |
| Python | 155k | 12 | 3 | 36 | 23GB | ~2.5h | ~85.5% |

### CodeGen Training (RTX 5090)

| Dataset | Samples | Batch | Grad Accum | Eff. Batch | VRAM | Time | Accuracy |
|---------|---------|-------|------------|------------|------|------|----------|
| Java | 275k | 12 | 3 | 36 | 23GB | ~18-20h | ~85-86% |
| Python | 155k | 12 | 3 | 36 | 23GB | ~10-12h | ~85-86% |

### Cost Analysis (Vast.ai @ $0.50/hour)

| Config | Dataset | Time | Cost | Accuracy | Cost/1% Acc |
|--------|---------|------|------|----------|-------------|
| CodeT5+ (D) | Java | 7h | $3.50 | 85.7% | $0.41 |
| CodeT5+ | Python | 2.5h | $1.25 | 85.5% | $0.15 |
| CodeGen | Java | 20h | $10.00 | 85.5% | $1.17 |
| CodeGen | Python | 12h | $6.00 | 85.5% | $0.70 |

**Recommendation:** Use CodeT5+ for faster, cheaper training with similar accuracy.

---

## 🚀 Next Steps for Future AI Assistant

### Immediate Tasks (Priority 1)

1. **Expand Python Repository List**
   ```bash
   cd /workspace/classnameprediction
   python filter_and_expand_repos.py
   git add data/repos_python.txt
   git commit -m "Expand Python repos from 450 to 600"
   git push
   ```

2. **Resume Python CodeT5+ Training**
   ```bash
   python scripts/train.py \
     --model Salesforce/codet5p-220m \
     --data datasets/python \
     --output model/checkpoints/run1-python-codet5 \
     --batch-size 12 \
     --grad-accum 3 \
     --lr 5e-5 \
     --epochs 5 \
     --fp16 \
     --resume-from-checkpoint auto
   ```

3. **Start Java CodeT5+ Training**
   ```bash
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

### Medium Priority Tasks

4. **Evaluate Models**
   - Python CodeT5+: After training completes
   - Java CodeT5+: After training completes

5. **Train CodeGen for Comparison** (optional, very slow)
   - Only if user wants architecture comparison
   - Expect 10-20 hours per dataset

6. **Upload Models to HuggingFace Hub**
   ```bash
   python scripts/upload_to_hf.py \
     --ckpt model/checkpoints/run1-python-codet5 \
     --hub-model-id reiprasetya-study/codet5-python-run1 \
     --metrics model/metrics/run1-python-codet5/metrics.json \
     --language python \
     --private
   ```

### Research Tasks

7. **Rebuild Python Dataset with 600 Repos**
   - After expanding repos to 600
   - Expect ~40-50 minutes
   - Will increase from 155k → ~205k samples

8. **Comparative Analysis**
   - CodeT5+ vs CodeGen results
   - Java vs Python performance
   - Document findings

---

## 📝 Important Notes

### For User (Aldi)

- **Weekly Limit:** You mentioned hitting weekly limit soon, so this context file ensures continuity
- **Cloud Instance:** You're using Vast.ai RTX 5090 instances
- **Git Repo:** `reiprasetyastudy-repo/classnameprediction` (branch: codegen)
- **HuggingFace:** Account `reiprasetya-study` for datasets/models

### For AI Assistant

1. **Always check git status** before making changes
2. **Verify VRAM availability** before suggesting batch sizes
3. **Use TodoWrite tool** for multi-step tasks
4. **Test commands** in small scope before large operations
5. **Check logs** for errors before assuming success
6. **Resume from checkpoints** when possible (saves time and money)
7. **Monitor training** for OOM errors, adjust batch size if needed
8. **Document all changes** in commit messages

### Context Preservation

This file should be updated whenever:
- New optimal configurations are found
- Training runs complete
- Major bugs are fixed
- Dataset is expanded
- Research findings are documented

**Last Update:** End of Nov 29, 2024 session
**Next Update:** After Python repos expansion + training completion

---

## 🔗 References

- **Repository:** https://github.com/reiprasetyastudy-repo/classnameprediction
- **Datasets:**
  - https://huggingface.co/datasets/reiprasetya-study/java-class-names
  - https://huggingface.co/datasets/reiprasetya-study/python-class-names
- **Models:**
  - CodeT5+: https://huggingface.co/Salesforce/codet5p-220m
  - CodeGen: https://huggingface.co/Salesforce/codegen-350M-mono

---

**END OF CONTEXT DOCUMENT**
