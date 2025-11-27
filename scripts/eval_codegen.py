#!/usr/bin/env python
"""
Evaluation Script - FIXED VERSION with Output Saving and Logging
Compatible with Training Format
"""

import json
import torch
import numpy as np
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
from tqdm import tqdm
from logger_utils import setup_logger, log_section, log_config, log_metrics

class EvalDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, logger=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if logger:
            logger.info(f"Preprocessing {len(dataset)} evaluation samples...")

        for item in dataset:
            source = item['source']
            target = item['target']

            if not source or not target:
                continue

            # Format yang sama dengan training
            prompt_text = f"{source}\nClass name:"
            full_text = prompt_text + f" {target}<|endoftext|>"

            encoding = tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            prompt_encoding = tokenizer(
                prompt_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt"
            )
            prompt_len = prompt_encoding['input_ids'].shape[1]

            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]
            labels = input_ids.clone()

            # Mask prompt part seperti di training
            if prompt_len < len(labels):
                labels[:prompt_len] = -100

            labels[attention_mask == 0] = -100

            sample = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'source_text': source,
                'target_text': target,
                'language': item.get('language', 'unknown'),
                'repo': item.get('repo', 'unknown')
            }
            self.samples.append(sample)

        if logger:
            logger.info(f"Preprocessing complete: {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def load_validation_data(valid_data_path, logger):
    """Load validation data from JSONL"""
    logger.info(f"Loading validation data from {valid_data_path}")
    samples = []
    with open(valid_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(samples)} validation samples")
    return samples

def generate_prediction(model, tokenizer, source_text, device, max_length=512):
    """Generate prediction for a single sample - FIXED VERSION"""
    prompt = f"{source_text}\nClass name:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
        add_special_tokens=True
    )

    # FIX: Ensure attention mask is properly set
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

    with torch.no_grad():
        # FIX: Use simpler generation without problematic parameters
        outputs = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            max_new_tokens=20,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

    # FIX: outputs is already a tensor, not a dictionary
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract prediction after "Class name:"
    if "Class name:" in generated:
        prediction_part = generated.split("Class name:")[-1]
        # Clean up the prediction
        prediction = prediction_part.split('<|endoftext|>')[0].strip()
        prediction = prediction.rstrip('.:;,\n\t').strip()

        # If prediction contains newlines, take only the first line
        if '\n' in prediction:
            prediction = prediction.split('\n')[0].strip()
    else:
        # Fallback: try to extract from the end
        prediction = generated.replace(prompt, "").strip()
        prediction = prediction.split('<|endoftext|>')[0].strip()
        prediction = prediction.rstrip('.:;,\n\t').strip()

    return prediction

def save_evaluation_results(results, output_dir, model_path, valid_data_path, logger):
    """Save evaluation results to files"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed predictions
    predictions_file = os.path.join(output_dir, f"evaluation_predictions_{timestamp}.jsonl")
    logger.info(f"Saving detailed predictions to {predictions_file}")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in results['predictions']:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    # Save summary results
    summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
    summary = {
        'timestamp': timestamp,
        'model_path': model_path,
        'validation_data_path': valid_data_path,
        'token_accuracy': results['token_accuracy'],
        'exact_match_accuracy': results['exact_match_accuracy'],
        'total_samples': results['total_samples'],
        'correct_exact_matches': results['correct_exact_matches'],
        'evaluation_details': {
            'token_accuracy_percentage': f"{results['token_accuracy']:.2%}",
            'exact_match_accuracy_percentage': f"{results['exact_match_accuracy']:.2%}",
            'correct_vs_total': f"{results['correct_exact_matches']}/{results['total_samples']}"
        }
    }

    logger.info(f"Saving summary to {summary_file}")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save human-readable report
    report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
    logger.info(f"Saving report to {report_file}")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Validation Data: {valid_data_path}\n")
        f.write(f"Total Samples: {results['total_samples']}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Token-level Accuracy: {results['token_accuracy']:.4f} ({results['token_accuracy']:.2%})\n")
        f.write(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f} ({results['exact_match_accuracy']:.2%})\n")
        f.write(f"Correct/Total: {results['correct_exact_matches']}/{results['total_samples']}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("SAMPLE PREDICTIONS\n")
        f.write("=" * 70 + "\n")

        for i, pred in enumerate(results['predictions'][:10]):  # Show first 10
            f.write(f"\n--- Sample {i+1} ---\n")
            f.write(f"Expected: '{pred['target']}'\n")
            f.write(f"Predicted: '{pred['predicted']}'\n")
            f.write(f"Match: {pred['match']}\n")
            f.write(f"Language: {pred.get('language', 'unknown')}\n")
            f.write(f"Repo: {pred.get('repo', 'unknown')}\n")

            source_preview = pred['source'][:200] + '...' if len(pred['source']) > 200 else pred['source']
            f.write(f"Source preview: {source_preview}\n")

    logger.info("Evaluation results saved successfully")

    return {
        'predictions_file': predictions_file,
        'summary_file': summary_file,
        'report_file': report_file
    }

def compute_validation_metrics(model, eval_dataset, device, tokenizer, logger):
    """Compute metrics on validation set"""
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    exact_matches = 0
    all_predictions = []

    logger.info("Running validation evaluation...")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(eval_dataset, desc="Evaluating")):
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            labels = batch['labels'].unsqueeze(0).to(device)

            # Forward pass untuk token accuracy
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Token-level accuracy
            logits = outputs.logits
            pred_ids = logits.argmax(dim=-1)

            mask = labels != -100
            valid_labels = labels[mask]
            valid_preds = pred_ids[mask]

            correct_tokens += (valid_preds == valid_labels).sum().item()
            total_tokens += mask.sum().item()

            # Exact match prediction menggunakan generation
            source_text = batch['source_text']
            target_text = batch['target_text']

            predicted_text = generate_prediction(model, tokenizer, source_text, device)

            all_predictions.append({
                'source': source_text,
                'target': target_text,
                'predicted': predicted_text,
                'match': predicted_text.strip().lower() == target_text.strip().lower(),
                'language': batch.get('language', 'unknown'),
                'repo': batch.get('repo', 'unknown'),
                'timestamp': datetime.now().isoformat()
            })

            if predicted_text.strip().lower() == target_text.strip().lower():
                exact_matches += 1

            if (idx + 1) % 1000 == 0:
                logger.info(f"Evaluated {idx + 1}/{len(eval_dataset)} samples")

    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    exact_match_accuracy = exact_matches / len(eval_dataset) if len(eval_dataset) > 0 else 0

    logger.info(f"Token accuracy: {token_accuracy:.4f}")
    logger.info(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    logger.info(f"Exact matches: {exact_matches}/{len(eval_dataset)}")

    return {
        'token_accuracy': token_accuracy,
        'exact_match_accuracy': exact_match_accuracy,
        'total_samples': len(eval_dataset),
        'correct_exact_matches': exact_matches,
        'predictions': all_predictions
    }

def evaluate_model(model_path, valid_data_path, max_length=512, num_samples=None, output_dir="./evaluation_results"):
    """Main evaluation function"""

    # Setup logger - save to logs/ directory
    logger = setup_logger('eval_codegen')

    log_section(logger, "Evaluation Configuration")
    log_config(logger, {
        'model_path': model_path,
        'valid_data_path': valid_data_path,
        'max_length': max_length,
        'num_samples': num_samples if num_samples else 'all',
        'output_dir': output_dir
    })

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info(f"Using device: {device}")

    # Load validation data
    valid_data = load_validation_data(valid_data_path, logger)

    if num_samples and num_samples < len(valid_data):
        valid_data = valid_data[:num_samples]
        logger.info(f"Using {num_samples} samples for evaluation")
    else:
        logger.info(f"Using all {len(valid_data)} samples for evaluation")

    # Create eval dataset
    log_section(logger, "Dataset Preprocessing")
    eval_dataset = EvalDataset(valid_data, tokenizer, max_length, logger)

    # Compute metrics
    log_section(logger, "Running Evaluation")
    results = compute_validation_metrics(model, eval_dataset, device, tokenizer, logger)

    # Print results to console
    log_section(logger, "Evaluation Results")
    logger.info(f"Token-level Accuracy: {results['token_accuracy']:.4f} ({results['token_accuracy']:.2%})")
    logger.info(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f} ({results['exact_match_accuracy']:.2%})")
    logger.info(f"Correct/Total: {results['correct_exact_matches']}/{results['total_samples']}")

    # Show some examples
    log_section(logger, "Sample Predictions")
    for i, pred in enumerate(results['predictions'][:5]):  # Show first 5
        logger.info(f"\n--- Sample {i+1} ---")
        logger.info(f"Expected: '{pred['target']}'")
        logger.info(f"Predicted: '{pred['predicted']}'")
        logger.info(f"Match: {pred['match']}")
        logger.info(f"Language: {pred.get('language', 'unknown')}")
        logger.info(f"Repo: {pred.get('repo', 'unknown')}")

        source_preview = pred['source'][:150] + '...' if len(pred['source']) > 150 else pred['source']
        logger.info(f"Source preview: {source_preview}")

    # Save results to files
    log_section(logger, "Saving Results")
    saved_files = save_evaluation_results(results, output_dir, model_path, valid_data_path, logger)

    logger.info(f"Evaluation complete!")

    return {**results, 'saved_files': saved_files}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained model on validation set')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--valid-data', type=str, required=True, help='Path to validation JSONL file')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to evaluate (None = all)')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', help='Directory to save evaluation results')

    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        valid_data_path=args.valid_data,
        max_length=args.max_length,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
