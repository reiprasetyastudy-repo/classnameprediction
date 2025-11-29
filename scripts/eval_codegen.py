#!/usr/bin/env python
"""
Evaluate a fine-tuned CodeGen checkpoint on class name prediction.

Designed to match eval_gpu.py (CodeT5+) output format for fair comparison.

Inputs:
  --ckpt: Path to model checkpoint directory
  --data: Path to JSONL file or directory containing test.jsonl
  --k: Top-k to compute (default: 5)
  --batch-size: Batch size for evaluation (default: 8)

Outputs:
  model/metrics/<run>/metrics.json with EM, EM_ci, avg_levenshtein, topk accuracy
  model/metrics/<run>/detailed_results.jsonl with per-sample predictions
"""

import argparse
import json
import os
from pathlib import Path
from typing import List
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein

from logger_utils import setup_logger, log_section, log_config, log_metrics


PROMPT = "{source}\nClass name:"


def load_test(path: str):
    """Load test data from JSONL file or directory"""
    if os.path.isdir(path):
        file = os.path.join(path, 'test.jsonl')
    else:
        file = path
    
    samples = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples


def generate_batch(model, tokenizer, sources: List[str], device, max_new_tokens=20, num_return_sequences=5):
    """Generate predictions for a batch of sources using beam search"""
    prompts = [PROMPT.format(source=s) for s in sources]
    
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=1024
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_return_sequences,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode all outputs
    all_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
    # Group by input (each input has num_return_sequences outputs)
    grouped = []
    for i in range(len(sources)):
        start_idx = i * num_return_sequences
        end_idx = start_idx + num_return_sequences
        batch_texts = all_texts[start_idx:end_idx]
        
        # Extract predictions from each generated text
        predictions = []
        for text in batch_texts:
            pred = extract_prediction(text, prompts[i])
            predictions.append(pred)
        
        grouped.append(predictions)
    
    return grouped


def extract_prediction(generated_text: str, prompt: str) -> str:
    """Extract the predicted class name from generated text"""
    # Remove the prompt part
    if "Class name:" in generated_text:
        prediction_part = generated_text.split("Class name:")[-1]
    else:
        prediction_part = generated_text.replace(prompt, "")
    
    # Clean up
    prediction = prediction_part.split('<|endoftext|>')[0].strip()
    prediction = prediction.split('\n')[0].strip()  # Take first line only
    prediction = prediction.rstrip('.:;,').strip()
    
    # Take first word/token only (normalize like CodeT5+ eval)
    if prediction:
        prediction = prediction.split()[0] if prediction.split() else prediction
    
    return prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    ap.add_argument('--data', type=str, required=True, help='Path to test data (JSONL or directory)')
    ap.add_argument('--k', type=int, default=5, help='Top-k for accuracy calculation')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    ap.add_argument('--cpu', action='store_true', help='Force CPU usage')
    args = ap.parse_args()

    # Setup logger
    logger = setup_logger('eval_codegen', log_dir='logs/codegen')
    start_time = time.time()

    log_section(logger, "CodeGen Evaluation")

    # Determine device
    if args.cpu:
        device = torch.device('cpu')
        logger.info("Using CPU for evaluation (forced)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name}")
        else:
            logger.info("GPU not available, using CPU")

    # Log configuration
    config = {
        'checkpoint': args.ckpt,
        'data': args.data,
        'k': args.k,
        'batch_size': args.batch_size,
        'device': str(device),
    }
    log_config(logger, config)

    # Load model and tokenizer
    logger.info(f"Loading model from {args.ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for batch generation with causal LM
    
    model = model.to(device)
    model.eval()
    logger.info("Model loaded and set to eval mode")

    # Load test data
    test_data = load_test(args.data)
    logger.info(f"Loaded {len(test_data)} test examples")

    k = args.k
    batch_size = args.batch_size
    
    # Metrics counters
    em = 0           # Exact match
    em_ci = 0        # Exact match case-insensitive
    topk = 0         # Top-k accuracy
    lev_sum = 0.0    # Levenshtein distance sum
    n = 0
    all_results = []

    # Create metrics directory
    metrics_dir = Path('model/metrics') / Path(args.ckpt).name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evaluation with batch_size={batch_size}, k={k}")
    total_batches = (len(test_data) + batch_size - 1) // batch_size
    logger.info(f"Total batches: {total_batches}")

    # Process in batches
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i:i+batch_size]
        sources = [item['source'] for item in batch]
        targets = [item['target'] for item in batch]
        
        # Generate predictions (returns list of k predictions per sample)
        try:
            predictions_batch = generate_batch(model, tokenizer, sources, device, num_return_sequences=k)
        except Exception as e:
            logger.error(f"Error generating batch {i}: {e}")
            # Fallback to empty predictions
            predictions_batch = [[""] * k for _ in range(len(batch))]
        
        # Compute metrics for each sample in batch
        for j, (preds, gold) in enumerate(zip(predictions_batch, targets)):
            n += 1
            
            # Normalize predictions
            norm_preds = [p.strip() for p in preds]
            top_pred = norm_preds[0] if norm_preds else ""
            
            # Store detailed results
            result = {
                'source': sources[j],
                'target': gold,
                'predictions': norm_preds,
                'top_prediction': top_pred,
                'exact_match': top_pred == gold,
                'exact_match_ci': top_pred.lower() == gold.lower(),
                'in_topk': gold in norm_preds[:k],
            }
            all_results.append(result)
            
            # Exact match
            if top_pred == gold:
                em += 1
            
            # Case-insensitive exact match
            if top_pred.lower() == gold.lower():
                em_ci += 1
            
            # Top-k accuracy
            if gold in norm_preds[:k]:
                topk += 1
            
            # Levenshtein distance
            lev = Levenshtein.distance(top_pred, gold) if top_pred else len(gold)
            lev_sum += float(lev)

    # Calculate final metrics
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

    # Save metrics (same format as eval_gpu.py)
    metrics_file = metrics_dir / 'metrics.json'
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    logger.info(f"Metrics saved to: {metrics_file}")

    # Save detailed results
    results_file = metrics_dir / 'detailed_results.jsonl'
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"Detailed results saved to: {results_file}")

    # Print results to console
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Detailed results saved to: {results_file}")

    # Log metrics
    log_section(logger, "EVALUATION RESULTS")
    log_metrics(logger, metrics)
    logger.info(f"Total evaluation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == '__main__':
    main()
