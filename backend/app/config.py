"""
Configuration settings for the Microcontroller API Assistant backend.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

# Configure logger
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings."""
    
    # Model configuration
    model_path: str = os.getenv("MODEL_PATH", "./checkpoints/mcu-llm/final_model")
    model_name: str = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
    active_model_path: Optional[str] = None  # Will be set dynamically
    
    # Fine-tuned model detection
    checkpoint_dir: str = "./checkpoints/mcu-llm"
    
    # vLLM configuration
    enable_vllm: bool = os.getenv("ENABLE_VLLM", "false").lower() == "true"
    vllm_trust_remote_code: bool = True
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 2048
    
    # Hugging Face configuration
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    hf_cache_dir: Optional[str] = os.getenv("HF_CACHE_DIR")
    
    # Performance configuration
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_title: str = "Microcontroller API Assistant"
    api_version: str = "1.0.0"
    api_description: str = "LLM-powered API for generating microcontroller code"
    
    # CORS configuration
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    # Logging configuration
    log_level: str = "INFO"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._detect_active_model()
    
    def _detect_active_model(self):
        """Automatically detect and set the active model path."""
        # First, check for fine-tuned model in the expected location
        fine_tuned_path = Path(self.model_path)
        checkpoint_dir = Path(self.checkpoint_dir)
        
        # Look for the latest checkpoint in the checkpoint directory
        if checkpoint_dir.exists():
            # Find the most recent checkpoint
            checkpoints = []
            for item in checkpoint_dir.iterdir():
                if item.is_dir() and (
                    (item / "pytorch_model.bin").exists()
                    or (item / "adapter_model.safetensors").exists()
                    or (item / "model.safetensors").exists()
                ):
                    checkpoints.append(item)
            
            if checkpoints:
                # Sort by modification time (newest first)
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                self.active_model_path = str(latest_checkpoint)
                logger.info(f"[USING FINETUNED MODEL] Loaded checkpoint: {self.active_model_path}")
                return
        
        # Check if the final model exists
        if fine_tuned_path.exists() and (
            (fine_tuned_path / "pytorch_model.bin").exists()
            or (fine_tuned_path / "adapter_model.safetensors").exists()
            or (fine_tuned_path / "model.safetensors").exists()
        ):
            self.active_model_path = self.model_path
            logger.info(f"[USING FINETUNED MODEL] Loaded final model: {self.active_model_path}")
            return
        
        # Fallback to pre-trained model
        self.active_model_path = self.model_name
        logger.warning(f"[USING TEMPLATE FALLBACK] No fine-tuned model found, using pre-trained: {self.active_model_path}")
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env

# Global settings instance
settings = Settings()
