#!/usr/bin/env python
"""
Batch predict class names from test.jsonl using fine-tuned CodeGen-350M-Mono.

Usage:
  python scripts/predict_batch.py \
    --ckpt model/checkpoints/run1-python/run1-python-codegen/checkpoint-5000 \
    --test_file datasets/python/test.jsonl \
    --output_file predictions.jsonl \
    --k 3
"""
import argparse
import json
import os
import sys
import re
from tqdm import tqdm
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT = "Predict class name:\n{source}\nName:"

# Regex patterns
PY_CLASS_RE = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(?", re.MULTILINE)
JAVA_CLASS_RE = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def mask_header(src: str, language: str) -> str:
    """Replace only the declared identifier in the class header with '____'."""
    if language == 'python':
        m = PY_CLASS_RE.search(src)
        if not m:
            return src
        name = m.group(1)
        return re.sub(rf"(class\s+){re.escape(name)}(\s*\(?)", r"\1____\2", src, count=1)
    elif language == 'java':
        m = JAVA_CLASS_RE.search(src)
        if not m:
            return src
        name = m.group(2)
        return re.sub(rf"(\b(class|interface|enum)\s+){re.escape(name)}\b", r"\1____", src, count=1)
    else:
        return src


def extract_prediction_from_completion(completion: str) -> str:
    """Extract the first valid identifier from completion text."""
    if "Name:" in completion:
        parts = completion.split("Name:", 1)
        if len(parts) > 1:
            prediction_part = parts[1].strip()
            match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)', prediction_part)
            if match:
                return match.group(1)
    
    match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)', completion.strip())
    return match.group(1) if match else ''


class BatchPredictor:
    def __init__(self, model_path: str):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict_batch(self, sources: List[str], language: str, k: int = 1, max_new_tokens: int = 16) -> List[List[str]]:
        """Predict for multiple sources at once."""
        all_predictions = []
        
        for source in tqdm(sources, desc="Predicting"):
            masked = mask_header(source, language)
            prompt = PROMPT.format(source=masked)

            inputs = self.tokenizer([prompt], return_tensors='pt', padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=max(k, 1),
                    num_return_sequences=max(k, 1),
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[:, input_length:]
            texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            preds = [extract_prediction_from_completion(t) for t in texts]
            
            # de-dup while preserving order
            seen = set()
            uniq = []
            for p in preds:
                if p and p not in seen:
                    seen.add(p)
                    uniq.append(p)
            all_predictions.append(uniq[:k])
        
        return all_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to fine-tuned checkpoint')
    parser.add_argument('--test_file', required=True, help='Path to test.jsonl file')
    parser.add_argument('--output_file', required=True, help='Path to output predictions file')
    parser.add_argument('--k', type=int, default=1, help='Top-k predictions to return')
    parser.add_argument('--max_new_tokens', type=int, default=16, help='Max new tokens to generate')
    parser.add_argument('--language', default='python', choices=['python', 'java'], help='Source language')
    
    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from: {args.test_file}")
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Extract sources
    sources = [item['source'] for item in test_data]
    
    # Initialize predictor
    predictor = BatchPredictor(args.ckpt)
    
    # Get predictions
    predictions = predictor.predict_batch(
        sources=sources,
        language=args.language,
        k=args.k,
        max_new_tokens=args.max_new_tokens
    )
    
    # Combine with original data and save
    output_data = []
    for i, (item, preds) in enumerate(zip(test_data, predictions)):
        output_item = item.copy()
        output_item['predictions'] = preds
        output_item['predicted_name'] = preds[0] if preds else ""
        output_item['correct'] = output_item.get('class_name', '') == output_item['predicted_name']
        output_data.append(output_item)
    
    # Save predictions
    print(f"Saving predictions to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Calculate and print accuracy
    correct_count = sum(1 for item in output_data if item['correct'])
    accuracy = correct_count / len(output_data) if output_data else 0
    print(f"\nResults:")
    print(f"Total examples: {len(output_data)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Show some examples
    print(f"\nSample predictions:")
    for i, item in enumerate(output_data[:5]):
        status = "✓" if item['correct'] else "✗"
        true_name = item.get('class_name', 'N/A')
        pred_name = item['predicted_name']
        print(f"  {status} {i+1}. True: {true_name} -> Pred: {pred_name}")


if __name__ == '__main__':
    import re
    main()