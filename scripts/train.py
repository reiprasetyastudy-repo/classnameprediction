#!/usr/bin/env python
"""
Fine-tune CodeT5+ for class name prediction.

Inputs:
  --model: Pretrained model name or path (e.g., Salesforce/codet5p-220m)
  --data: Path to datasets directory containing train.jsonl/valid.jsonl
  --output: Output directory for checkpoints (model/checkpoints/<run>)

Optional:
  --batch-size, --grad-accum, --lr, --epochs, --max-source-len, --max-target-len, --fp16
  --wandb: Enable Weights & Biases logging (WANDB_PROJECT env var recommended)
  --seed: Random seed

Mapping:
  Prompt template: "Predict class name:\n{source}\nName:"
  Label: target (string)
"""

import argparse
import os
import sys

def parse_cuda_device():
    # Parse only --cuda-device from sys.argv
    import argparse as _argparse
    ap = _argparse.ArgumentParser(add_help=False)
    ap.add_argument('--cuda-device', type=int, default=0)
    args, _ = ap.parse_known_args()
    return args.cuda_device

cuda_device = parse_cuda_device()
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(cuda_device)

# Fix PyTorch 2.6 weights_only issue for checkpoint resume
import numpy as np
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.random.RandomState,
    ])

from dataclasses import dataclass
from typing import Dict, List
import time
import datasets as hfds
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from logger_utils import setup_logger, log_section, log_config


PROMPT = "Predict class name:\n{source}\nName:"


def load_dataset(data_dir: str):
    ds = hfds.load_dataset('json', data_files={
        'train': os.path.join(data_dir, 'train.jsonl'),
        'validation': os.path.join(data_dir, 'valid.jsonl'),
    })
    return ds


@dataclass
class Preprocessor:
    tokenizer: AutoTokenizer
    max_source_len: int
    max_target_len: int

    def __call__(self, batch: Dict[str, List[str]]):
        sources = [PROMPT.format(source=s) for s in batch['source']]
        targets = batch['target']
        model_inputs = self.tokenizer(
            sources, max_length=self.max_source_len, truncation=True
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, max_length=self.max_target_len, truncation=True
            )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs


def main():
    # Import CSVLoggerCallback
    from csv_logger import CSVLoggerCallback
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--max-source-len', type=int, default=1024)
    ap.add_argument('--max-target-len', type=int, default=32)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--cuda-device', type=int, default=0, help='CUDA device id (default: 0)')
    ap.add_argument('--resume-from-checkpoint', type=str, default=None, help='Path to checkpoint to resume from, or "auto" to auto-detect latest checkpoint')
    args = ap.parse_args()

    # Setup logger - save to logs/codet5/ directory
    logger = setup_logger('train', log_dir='logs/codet5')
    start_time = time.time()

    log_section(logger, "CodeT5+ Training")

    hfds.logging.set_verbosity_info()

    # Set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        logger.info(f"Using CUDA device: {args.cuda_device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(args.cuda_device)}")
    else:
        logger.info("CUDA not available, using CPU")

    # Log configuration
    config = {
        'model': args.model,
        'data': args.data,
        'output': args.output,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.grad_accum,
        'effective_batch_size': args.batch_size * args.grad_accum,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'max_source_len': args.max_source_len,
        'max_target_len': args.max_target_len,
        'fp16': args.fp16,
        'seed': args.seed,
    }
    log_config(logger, config)

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    logger.info(f"Model loaded: {args.model}")

    logger.info("Loading and preprocessing dataset...")
    ds = load_dataset(args.data)
    logger.info(f"Train examples: {len(ds['train'])}")
    logger.info(f"Validation examples: {len(ds['validation'])}")

    proc = Preprocessor(tokenizer, args.max_source_len, args.max_target_len)
    cols = ds['train'].column_names
    ds = ds.map(proc, batched=True, remove_columns=cols)
    logger.info("Dataset preprocessing completed")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        evaluation_strategy='steps',
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        logging_steps=50,
        report_to=['wandb'] if args.wandb else [],
        seed=args.seed,
    )

    csv_log_path = os.path.join(args.output, 'training_log.csv')
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[CSVLoggerCallback(csv_log_path)],
    )

    logger.info("Starting training...")
    logger.info(f"Total training steps: {len(ds['train']) // (args.batch_size * args.grad_accum) * args.epochs}")

    # Handle checkpoint resume
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint == "auto":
        # Auto-detect latest checkpoint
        last_checkpoint = get_last_checkpoint(args.output)
        if last_checkpoint is not None:
            logger.info(f"Auto-detected checkpoint: {last_checkpoint}")
            resume_from_checkpoint = last_checkpoint
        else:
            logger.info("No checkpoint found for auto-resume, starting from scratch")
            resume_from_checkpoint = None
    elif resume_from_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    # Start training
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")

    logger.info(f"Saving model to {args.output}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    logger.info("Model and tokenizer saved successfully")

    log_section(logger, "Training Summary")
    logger.info(f"Total time: {training_time/3600:.2f} hours")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Training log: {csv_log_path}")


if __name__ == '__main__':
    main()
