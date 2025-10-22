#!/usr/bin/env python
"""
Evaluate a fine-tuned CodeGen checkpoint on class name prediction.

Inputs:
  --ckpt: Path to model checkpoint directory (e.g., run1-python-codegen/checkpoint-5000)
  --data: Path to JSONL file or directory containing test.jsonl
  --k: Top-k to compute. Default: 5

Outputs:
  model/metrics/<run>/metrics.json with EM, EM_ci, avg_levenshtein, topk accuracy
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import datasets as hfds
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein


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
            pad_token_id=tokenizer.eos_token_id,  # Important for CodeGen
        )
    
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # group by sequences per input
    grouped = [texts[i:i+num_return_sequences] for i in range(0, len(texts), num_return_sequences)]
    return grouped


def extract_prediction(text: str, original_prompt: str) -> str:
    """
    Extract the prediction from generated text by removing the original prompt.
    For CodeGen models, we need to handle this carefully.
    """
    # Remove the original prompt to get just the generated part
    if text.startswith(original_prompt):
        prediction = text[len(original_prompt):].strip()
    else:
        # If for some reason the prompt isn't at start, try to find the last "Name:" part
        if "Name:" in text:
            parts = text.split("Name:")
            if len(parts) > 1:
                prediction = parts[-1].strip()
            else:
                prediction = text.strip()
        else:
            prediction = text.strip()
    
    # Take only the first token/word as the class name prediction
    prediction = prediction.split()[0] if prediction else ""
    return prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, 
                   help='Path to model checkpoint directory (e.g., run1-python-codegen/checkpoint-5000)')
    ap.add_argument('--data', type=str, required=True,
                   help='Path to JSONL file or directory containing test.jsonl')
    ap.add_argument('--k', type=int, default=5,
                   help='Top-k to compute')
    args = ap.parse_args()

    print(f"Loading model from: {args.ckpt}")
    
    # Load tokenizer and model for CodeGen
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    ds = load_test(args.data)
    print(f"Loaded test dataset with {len(ds)} examples")

    k = args.k
    em = 0
    em_ci = 0
    topk = 0
    lev_sum = 0.0
    n = 0

    # Create metrics directory
    ckpt_name = Path(args.ckpt).name
    parent_dir = Path(args.ckpt).parent.name
    metrics_dir = Path('model/metrics') / parent_dir / ckpt_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    print(f"Metrics will be saved to: {metrics_dir}")

    batch_size = 4  # Reduced for CodeGen-350M if GPU memory is limited
    all_results = []
    
    for i in tqdm(range(0, len(ds), batch_size), desc="Evaluating"):
        batch = ds[i:i+batch_size]
        prompts = [PROMPT.format(source=s) for s in batch['source']]
        
        try:
            g = generate(model, tokenizer, prompts, num_return_sequences=k)
            
            for j, (preds, gold, original_prompt) in enumerate(zip(g, batch['target'], prompts)):
                n += 1
                
                # Extract predictions by removing the original prompt
                norm_preds = []
                for pred in preds:
                    extracted_pred = extract_prediction(pred, original_prompt)
                    norm_preds.append(extracted_pred)
                
                # Store results for debugging
                result = {
                    'source': batch['source'][j],
                    'target': gold,
                    'predictions': norm_preds,
                    'top_prediction': norm_preds[0] if norm_preds else ""
                }
                all_results.append(result)
                
                # Calculate metrics
                if norm_preds and norm_preds[0] == gold:
                    em += 1
                if norm_preds and norm_preds[0].lower() == gold.lower():
                    em_ci += 1
                if gold in norm_preds[:k]:
                    topk += 1
                
                lev = Levenshtein.distance(norm_preds[0], gold) if norm_preds and norm_preds[0] else len(gold)
                lev_sum += float(lev)
                
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            continue

    # Calculate final metrics
    metrics = {
        'n': n,
        'exact_match': em / n if n else 0.0,
        'exact_match_case_insensitive': em_ci / n if n else 0.0,
        'topk_accuracy': topk / n if n else 0.0,
        'avg_levenshtein': lev_sum / n if n else 0.0,
        'k': k,
        'model': args.ckpt,
        'dataset': args.data,
    }

    # Save metrics
    metrics_file = metrics_dir / 'metrics.json'
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    
    # Save detailed results for analysis
    results_file = metrics_dir / 'detailed_results.jsonl'
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print("\n=== Evaluation Results ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == '__main__':
    main()