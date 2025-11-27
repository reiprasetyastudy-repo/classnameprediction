#!/usr/bin/env python
"""
Fine-tune CodeGen - 12GB VRAM Optimized with Logging
"""

import argparse
import os
import torch
import numpy as np
import datasets as hfds
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
from logger_utils import setup_logger, log_section, log_config

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
    def __init__(self, dataset, tokenizer, max_length, logger):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        logger.info(f"Preprocessing {len(dataset)} samples...")

        for i, item in enumerate(dataset):
            source = item['source']
            target = item['target']

            if not source or not target:
                continue

            prompt_text = f"{source}\nClass name:"
            full_text = prompt_text + f" {target}<|endoftext|>"

            full_encoding = tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = full_encoding['input_ids'][0]
            attention_mask = full_encoding['attention_mask'][0]

            prompt_encoding = tokenizer(
                prompt_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt"
            )
            prompt_len = prompt_encoding['input_ids'].shape[1]

            labels = input_ids.clone()

            if prompt_len < len(labels):
                labels[:prompt_len] = -100

            labels[attention_mask == 0] = -100

            sample = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            self.samples.append(sample)

            if (i + 1) % 10000 == 0:
                logger.info(f"Preprocessed {i + 1}/{len(dataset)} samples")

        logger.info(f"Preprocessing complete: {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

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

    args = ap.parse_args()

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
        'seed': args.seed,
    })

    logger.info(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model with gradient checkpointing enabled")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.gradient_checkpointing_enable()

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
        eval_steps=100,
        save_strategy='steps',
        save_steps=200,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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
        eval_accumulation_steps=1, # Pindahkan ke CPU setiap 1 step
    )

    log_section(logger, "Training Strategy")
    logger.info(f"Evaluation every {training_args.eval_steps} steps")
    logger.info(f"Save checkpoint every {training_args.save_steps} steps")
    logger.info("Gradient checkpointing enabled for memory efficiency")
    logger.info("FP16 mixed precision enabled")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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

if __name__ == '__main__':
    main()
