#!/usr/bin/env python3
"""
Real Fine-tuning Script for Microcontroller API Assistant

This script actually performs fine-tuning using PEFT/LoRA and saves
real model artifacts (checkpoints, logs, metrics).
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from huggingface_hub import login, HfApi
import wandb
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
import numpy as np
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for real fine-tuning."""
    model_name: str
    dataset_path: str
    output_dir: str
    hub_repo_id: str
    max_length: int = 2048
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_steps: Optional[int] = None  # New: for quick runs
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_4bit: bool = True
    use_8bit: bool = False
    use_wandb: bool = True
    push_to_hub: bool = True
    save_total_limit: int = 3

class RealModelLoader:
    """Handles model and tokenizer loading with quantization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cpu":
            logger.warning("Training on CPU is not recommended. Consider using a GPU.")
    
    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration."""
        # Disable quantization on CPU/MPS to avoid CUDA requirement
        if self.device in ["cpu", "mps"]:
            logger.info(f"Disabling quantization for {self.device} training")
            return None
        
        if self.config.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.use_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    
    def _check_flash_attention(self) -> bool:
        """Check if FlashAttention is available."""
        try:
            import flash_attn
            logger.info("✅ FlashAttention is available - using for performance optimization")
            return True
        except ImportError:
            logger.info("⚠️ FlashAttention not available - using standard attention")
            return False
    
    def _check_triton(self) -> bool:
        """Check if Triton is available."""
        try:
            import triton
            logger.info("✅ Triton is available - using for performance optimization")
            return True
        except ImportError:
            logger.info("⚠️ Triton not available - using standard kernels")
            return False
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Check for performance optimizations
        flash_attention_available = self._check_flash_attention()
        triton_available = self._check_triton()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        # Add quantization if available
        quantization_config = self.get_quantization_config()
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Add FlashAttention if available (CUDA only)
        if flash_attention_available and torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using FlashAttention 2 for improved performance")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Prepare for training
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info(f"Model loaded successfully with optimizations: FlashAttention={flash_attention_available}, Triton={triton_available}")
        return self.model, self.tokenizer

class RealDatasetLoader:
    """Handles dataset loading and preprocessing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def load_dataset(self) -> DatasetDict:
        """Load and preprocess the dataset."""
        logger.info(f"Loading dataset from: {self.config.dataset_path}")
        
        try:
            # Try to load from Hugging Face format first
            dataset_dict = load_from_disk(self.config.dataset_path)
            logger.info("Dataset loaded from Hugging Face format")
        except Exception as e:
            logger.warning(f"Could not load from Hugging Face format: {e}")
            # Fallback to JSONL format
            dataset_dict = self._load_from_jsonl()
        
        logger.info(f"Dataset loaded: {len(dataset_dict['train'])} train, {len(dataset_dict['validation'])} validation")
        return dataset_dict
    
    def _load_from_jsonl(self) -> DatasetDict:
        """Load dataset from JSONL format."""
        # Handle both file path and directory path
        jsonl_path = Path(self.config.dataset_path)
        
        # If it's a directory, look for the JSONL file inside
        if jsonl_path.is_dir():
            jsonl_path = jsonl_path / "micro_api_dataset.jsonl"
        
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
        
        logger.info(f"Loading dataset from JSONL: {jsonl_path}")
        
        examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(examples)} examples from JSONL")
        
        # Split into train/validation (80/20)
        split_idx = int(len(examples) * 0.8)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Convert to datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def preprocess_function(self, examples, tokenizer):
        """Preprocess examples for training."""
        # Combine instruction and output
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}\n\n### End\n"
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Set labels to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized

class RealTrainingManager:
    """Manages the real training process."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_loader = RealModelLoader(config)
        self.dataset_loader = RealDatasetLoader(config)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
    
    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        if self.config.use_wandb:
            try:
                wandb.init(
                    project="microcontroller-api-assistant",
                    name=f"real-finetune-{self.config.model_name.split('/')[-1]}",
                    config={
                        "model_name": self.config.model_name,
                        "max_length": self.config.max_length,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                        "num_epochs": self.config.num_epochs,
                        "lora_r": self.config.lora_r,
                        "lora_alpha": self.config.lora_alpha,
                    }
                )
                logger.info("Weights & Biases initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Weights & Biases: {e}")
                self.config.use_wandb = False
    
    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration for the model."""
        # Different target modules for different model architectures
        if "dialo" in self.config.model_name.lower():
            # DialoGPT uses different module names
            target_modules = ["c_attn", "c_proj", "c_fc", "c_proj"]
        elif "llama" in self.config.model_name.lower() or "mistral" in self.config.model_name.lower():
            # Llama/Mistral models
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # Generic transformer modules
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "c_proj", "c_attn"]
        
        logger.info(f"Using LoRA target modules: {target_modules}")
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
    
    def train(self) -> Path:
        """Run the complete training process with timing."""
        logger.info("Starting real fine-tuning process...")
        overall_start = time.time()
        
        try:
            # Step 1: Load model and tokenizer
            step_start = time.time()
            logger.info("Step 1: Loading model and tokenizer...")
            model, tokenizer = self.model_loader.load_model_and_tokenizer()
            logger.info(f"⏱ Model loading took: {time.time() - step_start:.2f}s")
            
            # Step 2: Load dataset
            step_start = time.time()
            logger.info("Step 2: Loading dataset...")
            dataset_dict = self.dataset_loader.load_dataset()
            logger.info(f"⏱ Dataset loading took: {time.time() - step_start:.2f}s")
            
            # Step 2.5: Prepare dataset for training
            step_start = time.time()
            logger.info("Step 2.5: Preparing dataset for training...")
            dataset_dict = self._prepare_dataset(dataset_dict, tokenizer)
            logger.info(f"⏱ Dataset preparation took: {time.time() - step_start:.2f}s")
            
            # Step 3: Prepare training
            step_start = time.time()
            logger.info("Step 3: Preparing training...")
            trainer = self._prepare_trainer(model, tokenizer, dataset_dict)
            logger.info(f"⏱ Training preparation took: {time.time() - step_start:.2f}s")
            
            # Step 4: Run training
            step_start = time.time()
            logger.info("Step 4: Running training...")
            trainer.train()
            logger.info(f"⏱ Training took: {time.time() - step_start:.2f}s")
            
            # Step 5: Save model
            step_start = time.time()
            logger.info("Step 5: Saving model...")
            final_model_path = self._save_final_model(model, tokenizer)
            logger.info(f"⏱ Model saving took: {time.time() - step_start:.2f}s")
            
            # Step 6: Push to hub (if enabled)
            if self.config.push_to_hub:
                step_start = time.time()
                logger.info("Step 6: Pushing to Hub...")
                self.push_to_hub(final_model_path)
                logger.info(f"⏱ Hub push took: {time.time() - step_start:.2f}s")
            
            logger.info(f"⏱ Total training time: {time.time() - overall_start:.2f}s")
            return final_model_path
            
        except Exception as e:
            logger.error(f"Training failed after {time.time() - overall_start:.2f}s: {e}")
            raise
    
    def _prepare_trainer(self, model, tokenizer, dataset_dict):
        """Helper to prepare the Trainer object."""
        try:
            lora_config = self.create_lora_config()
            model = get_peft_model(model, lora_config)
            logger.info("✅ LoRA adapter applied successfully")
        except ValueError as e:
            if "Target modules" in str(e):
                logger.warning("⚠️ LoRA target modules not found, proceeding without LoRA")
                logger.warning("This will perform full fine-tuning (slower but functional)")
            else:
                raise e
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,  # Use max_steps for quick runs
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,  # Use save_steps for checkpointing
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",  # Fixed parameter name
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=self.config.save_total_limit,
            remove_unused_columns=False,
            push_to_hub=False,  # We'll handle this separately
            report_to="wandb" if self.config.use_wandb else "none",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,  # Reduce for CPU training
            fp16=False,  # Disable fp16 for CPU training
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if self.config.use_wandb else None,
        )
        return trainer
    
    def _save_final_model(self, model, tokenizer):
        """Helper to save the final model and tokenizer."""
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        return final_model_path
    
    def save_training_metrics(self, trainer, start_time):
        """Save real training metrics and plots."""
        metrics_path = self.output_dir / "metrics" / "training_metrics.json"
        metrics_path.parent.mkdir(exist_ok=True)
        
        # Get training history
        history = trainer.state.log_history
        
        # Calculate additional metrics
        training_time = time.time() - start_time
        total_steps = len(history)
        
        metrics = {
            "training_time_seconds": training_time,
            "total_steps": total_steps,
            "steps_per_second": total_steps / training_time if training_time > 0 else 0,
            "history": history,
            "model_name": self.config.model_name,
            "dataset_size": len(trainer.train_dataset),
            "validation_size": len(trainer.eval_dataset),
            "final_train_loss": history[-1].get("loss", 0) if history else 0,
            "final_eval_loss": history[-1].get("eval_loss", 0) if history else 0,
        }
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create plots
        self.create_training_plots(history)
        
        logger.info(f"Training metrics saved to {metrics_path}")
    
    def create_training_plots(self, history):
        """Create real training plots."""
        if not history:
            return
        
        plots_dir = self.output_dir / "metrics" / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Extract metrics
        steps = [log.get('step', i) for i, log in enumerate(history)]
        train_loss = [log.get('loss', 0) for log in history if 'loss' in log]
        eval_loss = [log.get('eval_loss', 0) for log in history if 'eval_loss' in log]
        
        # Create loss plot
        plt.figure(figsize=(12, 8))
        
        if train_loss:
            plt.subplot(2, 1, 1)
            plt.plot(steps[:len(train_loss)], train_loss, label='Training Loss', color='blue')
            plt.title('Real Training Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        if eval_loss:
            plt.subplot(2, 1, 2)
            plt.plot(steps[:len(eval_loss)], eval_loss, label='Validation Loss', color='red')
            plt.title('Real Validation Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "training_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training plots saved")
    
    def push_to_hub(self, model_path: Path):
        """Push model to Hugging Face Hub."""
        try:
            logger.info(f"Pushing model to Hub: {self.config.hub_repo_id}")
            
            # Login to Hugging Face
            if os.getenv("HF_TOKEN"):
                login(token=os.getenv("HF_TOKEN"))
            
            # Push model
            api = HfApi()
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=self.config.hub_repo_id,
                repo_type="model"
            )
            
            logger.info(f"Model pushed to: https://huggingface.co/{self.config.hub_repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")

    def _prepare_dataset(self, dataset_dict: DatasetDict, tokenizer) -> DatasetDict:
        """Prepare dataset for training by tokenizing the data."""
        logger.info("Preparing dataset for training...")
        
        def tokenize_function(examples):
            # Combine instruction and output for training
            texts = []
            for instruction, output in zip(examples['instruction'], examples['output']):
                # Create a training prompt
                text = f"Instruction: {instruction}\nOutput: {output}\n"
                texts.append(text)
            
            # Tokenize the texts
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None,
            )
            
            # Set labels to input_ids for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization to both train and validation datasets
        tokenized_datasets = {}
        for split in dataset_dict.keys():
            tokenized_datasets[split] = dataset_dict[split].map(
                tokenize_function,
                batched=True,
                remove_columns=dataset_dict[split].column_names,
                desc=f"Tokenizing {split} split",
            )
        
        logger.info("Dataset preparation completed")
        return DatasetDict(tokenized_datasets)

def main():
    """Main function for real fine-tuning."""
    parser = argparse.ArgumentParser(description="Real fine-tune language model on MCU API dataset")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="Model to fine-tune")
    parser.add_argument("--dataset-path", default="data/micro_api_dataset", help="Path to dataset")
    parser.add_argument("--output-dir", default="real_checkpoints", help="Output directory")
    parser.add_argument("--hub-repo-id", required=True, help="Hugging Face Hub repository ID")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum training steps (for quick runs)")
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases")
    parser.add_argument("--no-push", action="store_true", help="Don't push to Hub")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        hub_repo_id=args.hub_repo_id,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_steps=args.max_steps,  # Use max_steps for quick runs
        save_steps=args.save_steps,  # Use save_steps for checkpointing
        eval_steps=args.eval_steps, # Use eval_steps from args
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb,
        push_to_hub=not args.no_push,
    )
    
    # Create training manager and run real training
    manager = RealTrainingManager(config)
    final_model_path = manager.train()
    
    logger.info(f"Real training completed! Model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
