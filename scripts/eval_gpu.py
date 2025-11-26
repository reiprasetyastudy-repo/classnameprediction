#!/usr/bin/env python
"""
Evaluate a fine-tuned CodeT5+ checkpoint on class name prediction with GPU support.

Inputs:
  --ckpt: Path to model checkpoint directory (e.g., model/checkpoints/run1)
  --data: Path to JSONL file or directory containing test.jsonl
  --k: Top-k to compute. Default: 5
  --cpu: Force CPU usage (default: use GPU if available)
  --batch-size: Batch size for evaluation (default: 16 for GPU, 8 for CPU)

Outputs:
  model/metrics/<run>/metrics.json with EM, EM_ci, avg_levenshtein, topk accuracy
  logs/eval_gpu_YYYYMMDD_HHMMSS.log with detailed execution log
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import time

import datasets as hfds
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein

from logger_utils import setup_logger_with_tqdm, log_section, log_config, log_metrics


PROMPT = "Predict class name:\n{source}\nName:"


def load_test(path: str):
    if os.path.isdir(path):
        file = os.path.join(path, 'test.jsonl')
    else:
        file = path
    ds = hfds.load_dataset('json', data_files={'test': file})
    return ds['test']


def generate(model, tokenizer, sources: List[str], max_new_tokens=16, num_return_sequences=5):
    inputs = tokenizer(sources, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)
    input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_return_sequences,
            num_return_sequences=num_return_sequences,
            do_sample=False,
        )
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # group by sequences per input
    grouped = [texts[i:i+num_return_sequences] for i in range(0, len(texts), num_return_sequences)]
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--cpu', action='store_true', help='Force CPU usage')
    ap.add_argument('--batch-size', type=int, default=None, help='Batch size (default: 16 for GPU, 8 for CPU)')
    args = ap.parse_args()

    # Setup logger
    logger = setup_logger_with_tqdm('eval_gpu')
    start_time = time.time()

    log_section(logger, "CodeT5+ Evaluation with GPU Support")

    # Determine device
    if args.cpu:
        device = torch.device('cpu')
        print("Using CPU for evaluation")
        logger.info("Using CPU for evaluation (forced)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using GPU: {gpu_name}")
            logger.info(f"Using GPU: {gpu_name}")
        else:
            print("GPU not available, using CPU")
            logger.info("GPU not available, using CPU")

    # Set batch size based on device if not specified
    if args.batch_size is None:
        batch_size = 16 if device.type == 'cuda' else 8
    else:
        batch_size = args.batch_size

    print(f"Batch size: {batch_size}")

    # Log configuration
    config = {
        'checkpoint': args.ckpt,
        'data': args.data,
        'k': args.k,
        'device': str(device),
        'batch_size': batch_size,
    }
    log_config(logger, config)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt)
    logger.info("Model and tokenizer loaded")

    # Move model to device
    model = model.to(device)
    model.eval()
    logger.info(f"Model moved to {device} and set to eval mode")

    ds = load_test(args.data)
    print(f"Loaded {len(ds)} test examples")
    logger.info(f"Loaded {len(ds)} test examples")

    k = args.k
    em = 0
    em_ci = 0
    topk = 0
    lev_sum = 0.0
    n = 0
    all_results = []

    metrics_dir = Path('model/metrics') / Path(args.ckpt).name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evaluation...")
    logger.info(f"Total batches: {len(ds) // batch_size + (1 if len(ds) % batch_size else 0)}")

    for i in tqdm(range(0, len(ds), batch_size), desc="Evaluating"):
        batch = ds[i:i+batch_size]
        prompts = [PROMPT.format(source=s) for s in batch['source']]
        g = generate(model, tokenizer, prompts, num_return_sequences=k)
        for j, (preds, gold) in enumerate(zip(g, batch['target'])):
            n += 1
            # Normalize predictions: take first token-ish segment (strip spaces, split non-word)
            norm_preds = [p.strip().split()[0] if p.strip() else '' for p in preds]

            # Store detailed results
            result = {
                'source': batch['source'][j],
                'target': gold,
                'predictions': norm_preds,
                'top_prediction': norm_preds[0] if norm_preds else "",
                'exact_match': norm_preds[0] == gold if norm_preds else False,
                'in_topk': gold in norm_preds[:k],
            }
            all_results.append(result)

            if norm_preds and norm_preds[0] == gold:
                em += 1
            if norm_preds and norm_preds[0].lower() == gold.lower():
                em_ci += 1
            if gold in norm_preds[:k]:
                topk += 1
            lev = Levenshtein.distance(norm_preds[0], gold) if norm_preds else len(gold)
            lev_sum += float(lev)

    metrics = {
        'n': n,
        'exact_match': em / n if n else 0.0,
        'exact_match_case_insensitive': em_ci / n if n else 0.0,
        'topk_accuracy': topk / n if n else 0.0,
        'avg_levenshtein': lev_sum / n if n else 0.0,
        'k': k,
        'device': str(device),
        'batch_size': batch_size,
    }

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    metrics['elapsed_time_seconds'] = elapsed_time
    metrics['samples_per_second'] = n / elapsed_time if elapsed_time > 0 else 0

    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Average speed: {metrics['samples_per_second']:.2f} samples/second")

    # Save metrics
    metrics_file = metrics_dir / 'metrics.json'
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    logger.info(f"Metrics saved to: {metrics_file}")

    # Save detailed results for analysis
    results_file = metrics_dir / 'detailed_results.jsonl'
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"Detailed results saved to: {results_file}")

    # Print results to console (original behavior)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    print(f"\nDetailed results saved to: {results_file}")

    # Also log to file
    log_section(logger, "EVALUATION RESULTS")
    log_metrics(logger, metrics)
    logger.info(f"Total evaluation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == '__main__':
    main()
