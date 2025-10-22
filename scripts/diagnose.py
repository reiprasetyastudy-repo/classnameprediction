#!/usr/bin/env python
"""
Diagnose with correct checkpoint path
"""
import json
import re
import os
from collections import Counter

def diagnose_with_correct_path():
    print("🔍 DIAGNOSING WITH CORRECT CHECKPOINT PATH")
    print("=" * 60)
    
    # Check if files exist
    train_path = "datasets/python/train.jsonl"
    test_path = "datasets/python/test.jsonl"
    ckpt_path = "model/checkpoints/run1-python/run1-python-codegen/checkpoint-5000"
    
    print(f"📁 PATHS:")
    print(f"  Train: {train_path} → {os.path.exists(train_path)}")
    print(f"  Test: {test_path} → {os.path.exists(test_path)}")
    print(f"  Checkpoint: {ckpt_path} → {os.path.exists(ckpt_path)}")
    
    # 1. Check training data structure
    try:
        with open(train_path, "r") as f:
            train_data = [json.loads(line) for line in f]
        
        with open(test_path, "r") as f:
            test_data = [json.loads(line) for line in f]
        
        print(f"\n✅ DATA FILES LOADED SUCCESSFULLY")
        print(f"📊 DATA STATS:")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")
        
        # Check data format
        if train_data:
            sample = train_data[0]
            print(f"\n📝 CURRENT DATA STRUCTURE:")
            print(f"  Keys: {list(sample.keys())}")
            
            # Show what we have
            if 'source' in sample:
                source_preview = sample['source'][:100] + "..." if len(sample['source']) > 100 else sample['source']
                print(f"  Source preview: {source_preview}")
            
            # Check for class name in different keys
            class_name_keys = ['class_name', 'target', 'label', 'className']
            found_class_key = None
            for key in class_name_keys:
                if key in sample:
                    found_class_key = key
                    print(f"  🏷️  Class name key found: '{key}' = '{sample[key]}'")
                    break
            
            if not found_class_key:
                print(f"  ❌ NO CLASS NAME KEY FOUND!")
                print(f"  Available keys: {list(sample.keys())}")
        
        # 2. Check what the model was trained on
        print(f"\n🤖 WHAT WAS THE MODEL TRAINED ON?")
        print(f"  Current keys in data: {list(train_data[0].keys())}")
        
        # If 'target' exists, that's probably the class name
        if 'target' in train_data[0]:
            train_classes = [item['target'] for item in train_data if 'target' in item]
            test_classes = [item['target'] for item in test_data if 'target' in item]
            
            print(f"  Using 'target' as class name:")
            print(f"  Unique classes in train: {len(set(train_classes))}")
            print(f"  Unique classes in test: {len(set(test_classes))}")
            print(f"  Top 5 train classes: {Counter(train_classes).most_common(5)}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

def check_model_predictions_actual():
    """Test actual predictions with the model"""
    print(f"\n🎯 TESTING ACTUAL PREDICTIONS:")
    print("-" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        ckpt_path = "model/checkpoints/run1-python/run1-python-codegen/checkpoint-5000"
        
        print(f"Loading model from: {ckpt_path}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model = AutoModelForCausalLM.from_pretrained(ckpt_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Test prediction
        test_code = """
class ____:
    def __init__(self, value):
        self.value = value
    
    def calculate(self, x):
        return self.value * x
"""
        prompt = f"Predict class name:\n{test_code}\nName:"
        
        inputs = tokenizer([prompt], return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                num_beams=1,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        completion = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        print(f"🧪 TEST PREDICTION:")
        print(f"  Input: {test_code[:50]}...")
        print(f"  Model output: '{completion}'")
        
        # Extract prediction
        if "Name:" in completion:
            parts = completion.split("Name:", 1)
            if len(parts) > 1:
                prediction = parts[1].strip().split()[0] if parts[1].strip() else "EMPTY"
                print(f"  Extracted prediction: '{prediction}'")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == '__main__':
    diagnose_with_correct_path()
    check_model_predictions_actual()