#!/usr/bin/env python
"""
Test Script with Comprehensive Metrics
Run after evaluation - ENHANCED VERSION (No Levenshtein dependency)
"""

import json
import torch
import numpy as np
import os
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class TestDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Preprocessing {len(dataset)} test samples...")
        
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
            
            sample = {
                'input_ids': encoding['input_ids'][0],
                'attention_mask': encoding['attention_mask'][0],
                'source_text': source,
                'target_text': target,
                'language': item.get('language', 'unknown'),
                'repo': item.get('repo', 'unknown')
            }
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def load_test_data(test_data_path):
    """Load test data from JSONL"""
    samples = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

def generate_prediction(model, tokenizer, source_text, device, max_length=512):
    """Generate prediction for a single sample"""
    prompt = f"{source_text}\nClass name:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
        add_special_tokens=True
    )
    
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    
    with torch.no_grad():
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
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract prediction after "Class name:"
    if "Class name:" in generated:
        prediction_part = generated.split("Class name:")[-1]
        prediction = prediction_part.split('<|endoftext|>')[0].strip()
        prediction = prediction.rstrip('.:;,\n\t').strip()
        
        if '\n' in prediction:
            prediction = prediction.split('\n')[0].strip()
    else:
        prediction = generated.replace(prompt, "").strip()
        prediction = prediction.split('<|endoftext|>')[0].strip()
        prediction = prediction.rstrip('.:;,\n\t').strip()
    
    return prediction

def generate_topk_predictions(model, tokenizer, source_text, device, max_length=512, k=5):
    """Generate top-k predictions with probabilities"""
    prompt = f"{source_text}\nClass name:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
        add_special_tokens=True
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            max_new_tokens=20,
            num_return_sequences=k,
            do_sample=True,
            num_beams=k,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    topk_predictions = []
    for i in range(k):
        if i < len(outputs.sequences):
            generated = tokenizer.decode(outputs.sequences[i], skip_special_tokens=False)
            
            if "Class name:" in generated:
                prediction_part = generated.split("Class name:")[-1]
                prediction = prediction_part.split('<|endoftext|>')[0].strip()
                prediction = prediction.rstrip('.:;,\n\t').strip()
                
                if '\n' in prediction:
                    prediction = prediction.split('\n')[0].strip()
            else:
                prediction = generated.replace(prompt, "").strip()
                prediction = prediction.split('<|endoftext|>')[0].strip()
                prediction = prediction.rstrip('.:;,\n\t').strip()
            
            topk_predictions.append(prediction)
    
    return topk_predictions

def calculate_edit_distance(str1, str2):
    """Calculate simple edit distance without external dependencies"""
    if str1 == str2:
        return 0
    
    len1, len2 = len(str1), len(str2)
    
    # Create a matrix to store distances
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize the matrix
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # deletion
                    dp[i][j-1] + 1,    # insertion
                    dp[i-1][j-1] + 1   # substitution
                )
    
    return dp[len1][len2]

def calculate_advanced_metrics(all_predictions, k=5):
    """Calculate advanced evaluation metrics including edit distance and top-k accuracy"""
    
    # Prepare data for basic metrics
    y_true = [pred['target'].strip().lower() for pred in all_predictions]
    y_pred = [pred['predicted'].strip().lower() for pred in all_predictions]
    
    # Exact match metrics
    exact_matches = [true == pred for true, pred in zip(y_true, y_pred)]
    exact_accuracy = sum(exact_matches) / len(exact_matches)
    
    # Case insensitive exact match
    exact_matches_ci = [true.lower() == pred.lower() for true, pred in 
                       zip([pred['target'].strip() for pred in all_predictions],
                           [pred['predicted'].strip() for pred in all_predictions])]
    exact_accuracy_ci = sum(exact_matches_ci) / len(exact_matches_ci)
    
    # Edit distance (simple implementation)
    edit_distances = []
    for pred in all_predictions:
        target = pred['target'].strip()
        predicted = pred['predicted'].strip()
        distance = calculate_edit_distance(target, predicted)
        edit_distances.append(distance)
    
    avg_edit_distance = np.mean(edit_distances) if edit_distances else 0
    
    # Top-k accuracy (if available)
    topk_accuracy = 0
    if 'topk_predictions' in all_predictions[0]:
        topk_correct = 0
        for pred in all_predictions:
            target = pred['target'].strip().lower()
            topk_preds = [p.lower() for p in pred.get('topk_predictions', [])[:k]]
            if target in topk_preds:
                topk_correct += 1
        topk_accuracy = topk_correct / len(all_predictions)
    
    # Traditional sklearn metrics
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    except Exception as e:
        print(f"Warning: Error calculating some metrics: {e}")
        accuracy = exact_accuracy
        precision = 0
        recall = 0
        f1 = 0
        class_report = {}
    
    # Additional metrics
    avg_pred_length = np.mean([len(pred['predicted']) for pred in all_predictions])
    avg_target_length = np.mean([len(pred['target']) for pred in all_predictions])
    
    # Language-wise performance
    language_stats = {}
    for pred in all_predictions:
        lang = pred.get('language', 'unknown')
        if lang not in language_stats:
            language_stats[lang] = {'total': 0, 'correct': 0, 'correct_ci': 0}
        language_stats[lang]['total'] += 1
        if pred['target'].strip().lower() == pred['predicted'].strip().lower():
            language_stats[lang]['correct'] += 1
        if pred['target'].strip().lower() == pred['predicted'].strip().lower():
            language_stats[lang]['correct_ci'] += 1
    
    for lang in language_stats:
        language_stats[lang]['accuracy'] = language_stats[lang]['correct'] / language_stats[lang]['total']
        language_stats[lang]['accuracy_ci'] = language_stats[lang]['correct_ci'] / language_stats[lang]['total']
    
    return {
        # Basic metrics
        'n': len(all_predictions),
        'exact_match': exact_accuracy,
        'exact_match_case_insensitive': exact_accuracy_ci,
        'topk_accuracy': topk_accuracy,
        'avg_edit_distance': avg_edit_distance,
        'k': k,
        
        # Traditional metrics
        'sklearn_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        
        # Additional metrics
        'avg_prediction_length': avg_pred_length,
        'avg_target_length': avg_target_length,
        'correct_predictions': sum(exact_matches),
        'class_distribution': len(set(y_true + y_pred)),
        
        # Statistical metrics
        'edit_distance_stats': {
            'min': float(np.min(edit_distances)) if edit_distances else 0,
            'max': float(np.max(edit_distances)) if edit_distances else 0,
            'mean': float(avg_edit_distance),
            'std': float(np.std(edit_distances)) if edit_distances else 0,
            'median': float(np.median(edit_distances)) if edit_distances else 0
        },
        
        'language_stats': language_stats,
        'classification_report': class_report,
        'predictions': all_predictions
    }

def save_test_results(results, output_dir, model_path, test_data_path, elapsed_time, batch_size, device_type):
    """Save test results to files"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed predictions
    predictions_file = os.path.join(output_dir, f"test_predictions_{timestamp}.jsonl")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in results['predictions']:
            # Remove large data before saving
            save_pred = pred.copy()
            if 'input_ids' in save_pred:
                del save_pred['input_ids']
            if 'attention_mask' in save_pred:
                del save_pred['attention_mask']
            f.write(json.dumps(save_pred, ensure_ascii=False) + '\n')
    
    # Save comprehensive summary with all metrics
    summary_file = os.path.join(output_dir, f"test_summary_{timestamp}.json")
    summary = {
        'timestamp': timestamp,
        'model_path': model_path,
        'test_data_path': test_data_path,
        'performance_metrics': {
            'n': results['n'],
            'exact_match': results['exact_match'],
            'exact_match_case_insensitive': results['exact_match_case_insensitive'],
            'topk_accuracy': results['topk_accuracy'],
            'avg_edit_distance': results['avg_edit_distance'],
            'k': results['k'],
            'device': device_type,
            'batch_size': batch_size,
            'elapsed_time_seconds': elapsed_time,
            'samples_per_second': results['n'] / elapsed_time if elapsed_time > 0 else 0
        },
        'traditional_metrics': {
            'sklearn_accuracy': results['sklearn_accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'avg_prediction_length': results['avg_prediction_length'],
            'avg_target_length': results['avg_target_length'],
            'correct_predictions': results['correct_predictions'],
            'class_distribution': results['class_distribution']
        },
        'edit_distance_stats': results['edit_distance_stats'],
        'language_stats': results['language_stats'],
        'classification_report': results['classification_report']
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save human-readable report
    report_file = os.path.join(output_dir, f"test_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE TEST REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Data: {test_data_path}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Samples (n): {results['n']}\n")
        f.write(f"Exact Match: {results['exact_match']:.4f} ({results['exact_match']:.2%})\n")
        f.write(f"Exact Match (Case Insensitive): {results['exact_match_case_insensitive']:.4f} ({results['exact_match_case_insensitive']:.2%})\n")
        f.write(f"Top-{results['k']} Accuracy: {results['topk_accuracy']:.4f} ({results['topk_accuracy']:.2%})\n")
        f.write(f"Avg Edit Distance: {results['avg_edit_distance']:.4f}\n")
        f.write(f"Device: {device_type}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Samples/Second: {results['n'] / elapsed_time:.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TRADITIONAL METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"SkLearn Accuracy: {results['sklearn_accuracy']:.4f} ({results['sklearn_accuracy']:.2%})\n")
        f.write(f"Precision: {results['precision']:.4f} ({results['precision']:.2%})\n")
        f.write(f"Recall: {results['recall']:.4f} ({results['recall']:.2%})\n")
        f.write(f"F1-Score: {results['f1_score']:.4f} ({results['f1_score']:.2%})\n")
        f.write(f"Correct/Total: {results['correct_predictions']}/{results['n']}\n")
        f.write(f"Unique Classes: {results['class_distribution']}\n")
        
        f.write(f"\nAverage Prediction Length: {results['avg_prediction_length']:.2f} chars\n")
        f.write(f"Average Target Length: {results['avg_target_length']:.2f} chars\n")
        
        # Edit distance statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("EDIT DISTANCE STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Minimum: {results['edit_distance_stats']['min']:.2f}\n")
        f.write(f"Maximum: {results['edit_distance_stats']['max']:.2f}\n")
        f.write(f"Mean: {results['edit_distance_stats']['mean']:.2f}\n")
        f.write(f"Std Dev: {results['edit_distance_stats']['std']:.2f}\n")
        f.write(f"Median: {results['edit_distance_stats']['median']:.2f}\n")
        
        # Language-wise performance
        f.write("\n" + "=" * 80 + "\n")
        f.write("LANGUAGE-WISE PERFORMANCE\n")
        f.write("=" * 80 + "\n")
        for lang, stats in results['language_stats'].items():
            f.write(f"{lang:15}: {stats['correct']:3d}/{stats['total']:3d} = {stats['accuracy']:.2%} (CI: {stats['accuracy_ci']:.2%})\n")
    
    print(f"\nSaved test results to:")
    print(f"  - Predictions: {predictions_file}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Report: {report_file}")
    
    return {
        'predictions_file': predictions_file,
        'summary_file': summary_file,
        'report_file': report_file
    }

def run_comprehensive_test(model_path, test_data_path, max_length=512, num_samples=None, 
                         output_dir="./test_results", batch_size=1, k=5, enable_topk=False):
    """Main test function with comprehensive metrics"""
    
    start_time = time.time()
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_data = load_test_data(test_data_path)
    
    if num_samples and num_samples < len(test_data):
        test_data = test_data[:num_samples]
        print(f"Using {num_samples} samples for testing")
    else:
        print(f"Using all {len(test_data)} samples for testing")
    
    # Create test dataset
    test_dataset = TestDataset(test_data, tokenizer, max_length)
    
    # Run predictions
    print("Running comprehensive testing...")
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataset, desc="Testing"):
            source_text = batch['source_text']
            target_text = batch['target_text']
            
            predicted_text = generate_prediction(model, tokenizer, source_text, device)
            
            prediction_data = {
                'source': source_text,
                'target': target_text,
                'predicted': predicted_text,
                'language': batch.get('language', 'unknown'),
                'repo': batch.get('repo', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add top-k predictions if enabled
            if enable_topk:
                topk_preds = generate_topk_predictions(model, tokenizer, source_text, device, max_length, k)
                prediction_data['topk_predictions'] = topk_preds
            
            all_predictions.append(prediction_data)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Calculate comprehensive metrics
    print("Calculating comprehensive metrics...")
    results = calculate_advanced_metrics(all_predictions, k)
    
    # Print results to console
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*80)
    print(f"Total Samples (n): {results['n']}")
    print(f"Exact Match: {results['exact_match']:.4f} ({results['exact_match']:.2%})")
    print(f"Exact Match (CI): {results['exact_match_case_insensitive']:.4f} ({results['exact_match_case_insensitive']:.2%})")
    print(f"Top-{k} Accuracy: {results['topk_accuracy']:.4f} ({results['topk_accuracy']:.2%})")
    print(f"Avg Edit Distance: {results['avg_edit_distance']:.4f}")
    print(f"Elapsed Time: {elapsed_time:.2f}s")
    print(f"Samples/Second: {results['n']/elapsed_time:.2f}")
    
    print(f"\nTraditional Metrics:")
    print(f"SkLearn Accuracy: {results['sklearn_accuracy']:.4f} ({results['sklearn_accuracy']:.2%})")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    # Save results to files
    saved_files = save_test_results(results, output_dir, model_path, test_data_path, 
                                  elapsed_time, batch_size, device_type)
    
    return {**results, 'saved_files': saved_files, 'elapsed_time': elapsed_time}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive testing with full metrics')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test JSONL file')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to test (None = all)')
    parser.add_argument('--output-dir', type=str, default='./test_results', help='Directory to save test results')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--k', type=int, default=5, help='Top-k for accuracy calculation')
    parser.add_argument('--enable-topk', action='store_true', help='Enable top-k predictions (slower)')
    
    args = parser.parse_args()
    
    results = run_comprehensive_test(
        model_path=args.model,
        test_data_path=args.test_data,
        max_length=args.max_length,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        k=args.k,
        enable_topk=args.enable_topk
    )