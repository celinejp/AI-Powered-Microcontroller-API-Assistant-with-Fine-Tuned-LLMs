#!/usr/bin/env python3
"""
Simple Training Script for Microcontroller API Assistant
Compatible with current transformers version
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Simple training function."""
    logger.info("Starting simple fine-tuning...")
    
    # Configuration
    model_name = "microsoft/DialoGPT-medium"
    dataset_path = "./data/micro_api_dataset"
    output_dir = "./models/fine-tuned"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for compatibility
    )
    
    # Load dataset
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation")
    
    # Prepare dataset
    def tokenize_function(examples):
        texts = []
        for instruction, output in zip(examples['instruction'], examples['output']):
            text = f"Instruction: {instruction}\nOutput: {output}\n"
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Tokenize datasets
    train_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    val_dataset = dataset["validation"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # Small batch for memory
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=1,  # Quick training
        max_steps=20,  # Very quick for testing
        warmup_steps=5,
        save_steps=10,
        eval_steps=10,
        logging_steps=5,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        fp16=False,  # Disable fp16 for compatibility
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Training completed! Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
