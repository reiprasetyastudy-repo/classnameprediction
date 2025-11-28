#!/usr/bin/env python
"""
Fine-tune CodeGen - 12GB VRAM Optimized with Logging
"""

import argparse
import os
import sys
import torch
import numpy as np
import datasets as hfds
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
from logger_utils import setup_logger, log_section, log_config

# Import HuggingFace upload utilities
try:
    from upload_to_hf import upload_model_to_hub
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Bersihkan cache memori
torch.cuda.empty_cache()

def load_dataset(data_dir: str, logger):
    logger.info(f"Loading dataset from {data_dir}")
    ds = hfds.load_dataset('json', data_files={
        'train': os.path.join(data_dir, 'train.jsonl'),
        'validation': os.path.join(data_dir, 'valid.jsonl'),
    })
    logger.info(f"Train samples: {len(ds['train'])}")
    logger.info(f"Validation samples: {len(ds['validation'])}")
    return ds

class MaskedClassNameDataset(Dataset):
    """Eager preprocessing with optimized memory - preprocess once, train fast"""
    def __init__(self, dataset, tokenizer, max_length, logger):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        logger.info(f"Preprocessing {len(dataset)} samples (optimized eager loading)...")

        # Batch processing for speed
        batch_size = 1000
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:min(i+batch_size, len(dataset))]

            # Prepare texts
            full_texts = []
            prompt_texts = []

            # HuggingFace dataset batch is dict of lists, not list of dicts
            sources = batch['source']
            targets = batch['target']

            for source, target in zip(sources, targets):
                prompt_text = f"{source}\nClass name:"
                full_text = prompt_text + f" {target}<|endoftext|>"
                full_texts.append(full_text)
                prompt_texts.append(prompt_text)

            # Batch tokenization (much faster!)
            full_encodings = tokenizer(
                full_texts,
                max_length=max_length,
                truncation=True,
                # No padding - store dynamic length
            )

            prompt_encodings = tokenizer(
                prompt_texts,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
            )

            # Process each sample
            for j in range(len(full_texts)):
                input_ids = full_encodings['input_ids'][j]
                prompt_len = len(prompt_encodings['input_ids'][j])

                # Create labels: mask prompt part
                labels = input_ids.copy()
                if prompt_len < len(labels):
                    labels[:prompt_len] = [-100] * prompt_len

                # Store as lists (not tensors) to save RAM
                self.samples.append({
                    'input_ids': input_ids,
                    'labels': labels
                })

            if (i + batch_size) % 10000 == 0 or (i + batch_size) >= len(dataset):
                logger.info(f"Preprocessed {min(i + batch_size, len(dataset))}/{len(dataset)} samples")

        logger.info(f"Preprocessing complete: {len(self.samples)} samples ready")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Custom DataCollator for dynamic padding
class CustomDataCollator:
    """Pad batch dynamically to longest sequence in batch"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Get max length in this batch
        max_len = max(len(f['input_ids']) for f in features)

        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }

        for f in features:
            input_ids = f['input_ids']
            labels = f['labels']

            # Calculate padding length for this sample
            pad_len = max_len - len(input_ids)

            # Pad input_ids with tokenizer.pad_token_id
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

            # Create attention_mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(input_ids) + [0] * pad_len

            # Pad labels with -100 (ignored in loss)
            padded_labels = labels + [-100] * pad_len

            batch['input_ids'].append(padded_input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(padded_labels)

        # Convert to tensors
        import torch
        return {
            'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(batch['labels'], dtype=torch.long)
        }

# FUNGSI BARU: Menghemat Memori GPU saat Evaluasi
def preprocess_logits_for_metrics(logits, labels):
    """
    Original logits shape: [batch, seq_len, vocab_size] -> Sangat Besar (GBs)
    Optimized shape: [batch, seq_len] -> Sangat Kecil (MBs)
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    # Ambil index dengan probabilitas tertinggi saja (argmax)
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    # Karena sudah di-argmax di preprocess, preds sekarang isinya integer, bukan float logits
    pred_ids, labels = eval_preds

    mask = labels != -100

    correct = (pred_ids[mask] == labels[mask]).sum()
    total = mask.sum()

    accuracy = correct / total if total > 0 else 0
    return {"token_accuracy": accuracy}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default="Salesforce/codegen-350M-mono")
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    # Batch size 2 aman untuk 12GB VRAM
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument('--max-steps', type=int, default=-1, help='Maximum training steps (overrides epochs if set)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing (slower but uses less VRAM)')

    # HuggingFace Hub integration
    ap.add_argument('--push-to-hub', action='store_true', help='Upload model to HuggingFace Hub after training')
    ap.add_argument('--hub-model-id', type=str, default=None, help='HuggingFace model ID (e.g., username/model-name)')
    ap.add_argument('--model-name', type=str, default='CodeGen', help='Model name for README (default: CodeGen)')
    ap.add_argument('--language', type=str, default='java', help='Programming language for README (default: java)')
    ap.add_argument('--private', action='store_true', help='Make HuggingFace repository private')

    args = ap.parse_args()

    # Validate HuggingFace arguments
    if args.push_to_hub:
        if not HF_AVAILABLE:
            print("❌ Error: HuggingFace Hub utilities not available")
            print("   Install dependencies: pip install -r requirements_hf.txt")
            sys.exit(1)
        if args.hub_model_id is None:
            print("❌ Error: --hub-model-id is required when using --push-to-hub")
            sys.exit(1)

    # Setup logger - save to logs/codegen/ directory
    logger = setup_logger('train_codegen', log_dir='logs/codegen')

    log_section(logger, "CodeGen Training")

    # Log GPU info if available
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: 0")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available, using CPU")

    log_config(logger, {
        'model': args.model,
        'data': args.data,
        'output': args.output,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.grad_accum,
        'effective_batch_size': args.batch_size * args.grad_accum,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'max_length': args.max_length,
        'max_steps': args.max_steps,
        'fp16': True,
        'gradient_checkpointing': args.gradient_checkpointing,
        'seed': args.seed,
    })

    logger.info(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.gradient_checkpointing:
        logger.info("Loading model with gradient checkpointing enabled")
        model.gradient_checkpointing_enable()
    else:
        logger.info("Loading model without gradient checkpointing (faster training)")

    logger.info("Loading dataset...")
    ds = load_dataset(args.data, logger)

    log_section(logger, "Dataset Preprocessing")
    train_dataset = MaskedClassNameDataset(ds['train'], tokenizer, args.max_length, logger)
    val_dataset = MaskedClassNameDataset(ds['validation'], tokenizer, args.max_length, logger)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    log_section(logger, "Training Arguments")
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy='steps',
        eval_steps=1000,  # Evaluate less frequently (was 100)
        save_strategy='steps',
        save_steps=2000,  # Save less frequently (was 200)
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # 2x larger for eval (no gradients)
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        seed=args.seed,
        fp16=True,
        dataloader_num_workers=0,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # SOLUSI MEMORI EVALUASI:
        eval_accumulation_steps=4,  # Process 4 batches before moving to CPU (was 1)
    )

    # Create custom data collator for dynamic padding
    data_collator = CustomDataCollator(tokenizer)

    log_section(logger, "Training Strategy")
    logger.info(f"Evaluation every {training_args.eval_steps} steps (optimized for speed)")
    logger.info(f"Eval batch size: {training_args.per_device_eval_batch_size} (2x train batch)")
    logger.info(f"Eval accumulation steps: {training_args.eval_accumulation_steps}")
    logger.info(f"Save checkpoint every {training_args.save_steps} steps")
    if args.gradient_checkpointing:
        logger.info("Gradient checkpointing: ENABLED (saves VRAM, slower training)")
    else:
        logger.info("Gradient checkpointing: DISABLED (faster training, uses more VRAM)")
    logger.info("FP16 mixed precision enabled")
    logger.info("Dynamic padding per batch (10-20x faster than max_length padding)")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,  # Dynamic padding
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, # Inject fungsi hemat memori
    )

    log_section(logger, "Starting Training")
    logger.info(f"Total training samples: {len(train_dataset)}")
    logger.info(f"Total validation samples: {len(val_dataset)}")

    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    log_section(logger, "Saving Final Model")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info(f"Model and tokenizer saved to {args.output}")

    log_section(logger, "Training Summary")
    logger.info(f"Total steps: {trainer.state.global_step}")
    logger.info(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
    logger.info(f"Best eval loss: {trainer.state.best_metric}")
    logger.info("Done.")

    # Upload to HuggingFace Hub if requested
    if args.push_to_hub:
        log_section(logger, "HuggingFace Hub Upload")
        logger.info(f"Uploading model to: {args.hub_model_id}")

        # Try to find metrics file
        metrics_file = None
        possible_metrics = [
            Path(args.output) / "metrics.json",
            Path("model/metrics") / Path(args.output).name / "metrics.json",
        ]
        for possible in possible_metrics:
            if possible.exists():
                metrics_file = str(possible)
                logger.info(f"Found metrics file: {metrics_file}")
                break

        success = upload_model_to_hub(
            checkpoint_path=args.output,
            hub_model_id=args.hub_model_id,
            metrics_file=metrics_file,
            model_name=args.model_name,
            language=args.language,
            private=args.private
        )

        if success:
            logger.info(f"✅ Model uploaded successfully to HuggingFace Hub")
            logger.info(f"🔗 View at: https://huggingface.co/{args.hub_model_id}")
        else:
            logger.error("❌ Failed to upload model to HuggingFace Hub")

if __name__ == '__main__':
    main()
