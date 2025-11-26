#!/usr/bin/env python
"""
Fine-tune CodeGen for class name prediction.

Inputs:
  --model: Pretrained model name or path (e.g., Salesforce/codegen-350M-mono)
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

# Set MPS memory management
if torch.backends.mps.is_available():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

from dataclasses import dataclass
from typing import Dict, List, Any
import time
import datasets as hfds
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from logger_utils import setup_logger, log_section, log_config


PROMPT = "Predict class name:\n{source}\nName:"


class MPSMemoryCallback(TrainerCallback):
    """Callback to clear MPS cache periodically during training."""
    
    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available() and state.global_step % 10 == 0:
            torch.mps.empty_cache()
    
    def on_evaluate(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def load_dataset(data_dir: str):
    ds = hfds.load_dataset('json', data_files={
        'train': os.path.join(data_dir, 'train.jsonl'),
        'validation': os.path.join(data_dir, 'valid.jsonl'),
    })
    return ds


@dataclass
class CustomDataCollator:
    """Custom data collator for causal LM that properly handles labels."""
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract input_ids and labels
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Find max length in batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            padding_length = max_length - len(ids)
            
            # Pad input_ids and attention_mask
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
            attention_mask.append([1] * len(ids) + [0] * padding_length)
            
            # Pad labels with -100 (ignore index)
            padded_labels.append(lbls + [-100] * padding_length)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
        }


@dataclass
class Preprocessor:
    tokenizer: AutoTokenizer
    max_source_len: int
    max_target_len: int

    def __call__(self, batch: Dict[str, List[str]]):
        # Format as "prompt + target" for causal LM
        texts = [
            PROMPT.format(source=s) + " " + t
            for s, t in zip(batch['source'], batch['target'])
        ]
        
        # Tokenize the full text
        model_inputs = self.tokenizer(
            texts,
            max_length=self.max_source_len + self.max_target_len,
            truncation=True,
            padding=False,
        )
        
        # For causal LM, labels are the same as input_ids
        # Simply assign the input_ids list to labels
        model_inputs['labels'] = model_inputs['input_ids']
        
        return model_inputs


def main():
    # Import CSVLoggerCallback
    from csv_logger import CSVLoggerCallback
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=4)  # Reduced default from 8
    ap.add_argument('--grad-accum', type=int, default=4)  # Increased default from 2
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--max-source-len', type=int, default=1024)
    ap.add_argument('--max-target-len', type=int, default=32)
    ap.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision (CUDA only)')
    ap.add_argument('--bf16', action='store_true', help='Use BF16 mixed precision (better for MPS/Apple Silicon)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing to save memory')
    ap.add_argument('--cpu', action='store_true', help='Force CPU training')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--cuda-device', type=int, default=0, help='CUDA device id (default: 0)')
    args = ap.parse_args()

    # Setup logger
    logger = setup_logger('train_codegen')
    start_time = time.time()

    log_section(logger, "CodeGen Training")

    hfds.logging.set_verbosity_info()

    # Set device
    if args.cpu:
        # Force CPU training
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
        print("Training on CPU (this will be slow)")
        logger.info("Training on CPU (forced)")
    elif torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        device = f'cuda:{args.cuda_device}'
        print(f"Training on {device}")
        logger.info(f"Training on {device} - {torch.cuda.get_device_name(args.cuda_device)}")
    else:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Training on {device}")
        logger.info(f"Training on {device}")
        if device == 'mps':
            print("MPS detected - using memory-optimized settings")
            logger.info("MPS detected - using memory-optimized settings")

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
        'bf16': args.bf16,
        'gradient_checkpointing': args.gradient_checkpointing,
        'seed': args.seed,
    }
    log_config(logger, config)

    # Load CodeGen model and tokenizer
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    model = AutoModelForCausalLM.from_pretrained(args.model)
    logger.info(f"Model loaded: {args.model}")

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
        logger.info("Gradient checkpointing enabled")

    logger.info("Loading and preprocessing dataset...")
    ds = load_dataset(args.data)
    logger.info(f"Train examples: {len(ds['train'])}")
    logger.info(f"Validation examples: {len(ds['validation'])}")

    proc = Preprocessor(tokenizer, args.max_source_len, args.max_target_len)
    cols = ds['train'].column_names
    ds = ds.map(proc, batched=True, remove_columns=cols)
    logger.info("Dataset preprocessing completed")

    # Use custom data collator
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    # Choose optimizer based on device
    # Adafactor doesn't work well with MPS, use AdamW instead
    use_mps = torch.backends.mps.is_available() and not args.cpu and not torch.cuda.is_available()
    optimizer = 'adamw_torch' if use_mps else 'adafactor'

    training_args = TrainingArguments(
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
        fp16=args.fp16 and not args.cpu and not use_mps,  # Disable fp16 on CPU and MPS
        bf16=args.bf16 and not args.cpu,  # Disable bf16 on CPU
        logging_steps=50,
        report_to=['wandb'] if args.wandb else [],
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=optimizer,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,  # Save memory
        use_cpu=args.cpu,
    )

    print(f"Using optimizer: {optimizer}")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    logger.info(f"Using optimizer: {optimizer}")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")

    csv_log_path = os.path.join(args.output, 'training_log.csv')
    
    # Add MPS memory callback if using MPS
    callbacks = [CSVLoggerCallback(csv_log_path)]
    if use_mps:
        callbacks.append(MPSMemoryCallback())
        print("MPS memory management callback enabled")
        logger.info("MPS memory management callback enabled")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Clear cache before training
    if use_mps:
        torch.mps.empty_cache()
        logger.info("MPS cache cleared")

    logger.info("Starting training...")
    logger.info(f"Total training steps: {len(ds['train']) // (args.batch_size * args.grad_accum) * args.epochs}")

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