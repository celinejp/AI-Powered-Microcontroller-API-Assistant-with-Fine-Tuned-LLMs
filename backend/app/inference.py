"""
Dynamic Inference System for Microcontroller API Assistant

This module replaces the hardcoded template system with real model inference
using vLLM or Hugging Face Transformers. It generates dynamic responses
based on actual model predictions rather than hardcoded templates.
"""

import asyncio
import time
import json
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

from app.config import settings
from app.models import PeripheralType, MicrocontrollerType
from app.validators import validator, ValidationError

@dataclass
class DynamicResponse:
    """Dynamic response from model inference."""
    code: str
    explanation: str
    function_signature: str
    example_usage: str
    dependencies: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class DynamicInferenceEngine:
    """Dynamic inference engine using real model inference."""
    
    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.vllm_engine = None
        self.model_loaded = False
        self.start_time = time.time()
        self.performance_metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "latency_history": [],
            "throughput_history": []
        }
    
    async def initialize(self) -> None:
        """Initialize the dynamic inference engine."""
        try:
            logger.info("Initializing dynamic inference engine...")
            
            # Use pre-trained model for real inference
            logger.info("🎯 [USING PRETRAINED MODEL] Loading DialoGPT for real inference")
            
            if settings.enable_vllm:
                logger.info("vLLM integration is ENABLED")
                await self._initialize_vllm()
            else:
                logger.info("vLLM integration is DISABLED")
                await self._initialize_huggingface()
            
            self.model_loaded = True
            logger.info("Dynamic inference engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize dynamic inference engine: {e}")
            raise
    
    async def _initialize_vllm(self) -> None:
        """Initialize vLLM engine for high-performance inference."""
        try:
            from vllm import LLM, SamplingParams
            
            # Use the active model path from configuration
            model_path = settings.active_model_path
            logger.info(f"Initializing vLLM engine with model: {model_path}")
            
            self.vllm_engine = LLM(
                model=model_path,
                trust_remote_code=settings.vllm_trust_remote_code,
                tensor_parallel_size=settings.vllm_tensor_parallel_size,
                gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
                max_model_len=settings.vllm_max_model_len,
                dtype="float16" if torch.cuda.is_available() else "float32",
                enable_prefix_caching=True,
                enforce_eager=False,
            )
            
            logger.info("vLLM engine initialized successfully")
            
        except ImportError:
            logger.warning("vLLM not available, falling back to Hugging Face Transformers")
            await self._initialize_huggingface()
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            logger.info("Falling back to Hugging Face Transformers")
            await self._initialize_huggingface()
    
    async def _initialize_huggingface(self) -> None:
        """Initialize Hugging Face Transformers as fallback."""
        # Use the active model path from configuration
        model_path = settings.active_model_path
        logger.info(f"Loading model with Hugging Face: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=settings.hf_token,
            cache_dir=settings.hf_cache_dir,
            trust_remote_code=True,
            local_files_only="checkpoint" in model_path or "mcu-llm" in model_path
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=settings.hf_token,
            cache_dir=settings.hf_cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            local_files_only="checkpoint" in model_path or "mcu-llm" in model_path
        )
    
    async def generate_dynamic_response(
        self,
        peripheral: PeripheralType,
        microcontroller: MicrocontrollerType,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        sdk_version: Optional[str] = None,
        language: str = "c",
        include_comments: bool = True,
        include_error_handling: bool = True,
    ) -> DynamicResponse:
        """Generate dynamic response using real model inference."""
        start_time = time.time()
        
        try:
            # Validate inputs
            validation_result = validator.validate_complete_request(
                microcontroller, peripheral, f"{operation} {peripheral.value} on {microcontroller.value}", parameters
            )
            
            if not validation_result["valid"]:
                error_msg = validation_result["error"]["message"]
                suggestions = validation_result["error"]["suggestions"]
                return self._generate_error_response(error_msg, suggestions)
            
            # Generate dynamic response using model
            if self.vllm_engine:
                response = await self._generate_with_vllm(
                    peripheral, microcontroller, operation, parameters, sdk_version, language, include_comments, include_error_handling
                )
            else:
                response = await self._generate_with_huggingface(
                    peripheral, microcontroller, operation, parameters, sdk_version, language, include_comments, include_error_handling
                )
            
            # Update performance metrics
            self._update_performance_metrics(start_time, len(response.code))
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating dynamic response: {e}")
            return self._generate_error_response(f"Error generating response: {str(e)}", ["Try a different peripheral or microcontroller"])
    
    async def _generate_with_vllm(
        self,
        peripheral: PeripheralType,
        microcontroller: MicrocontrollerType,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        sdk_version: Optional[str] = None,
        language: str = "c",
        include_comments: bool = True,
        include_error_handling: bool = True,
    ) -> DynamicResponse:
        """Generate response using vLLM."""
        from vllm import SamplingParams
        
        # Build dynamic prompt
        prompt = self._build_dynamic_prompt(
            peripheral, microcontroller, operation, parameters, sdk_version, language, include_comments, include_error_handling
        )
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            max_tokens=settings.max_length,
        )
        
        # Generate response using vLLM
        outputs = self.vllm_engine.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        # Parse dynamic response
        return self._parse_dynamic_response(generated_text, peripheral, microcontroller, operation, language)
    
    async def _generate_with_huggingface(
        self,
        peripheral: PeripheralType,
        microcontroller: MicrocontrollerType,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        sdk_version: Optional[str] = None,
        language: str = "c",
        include_comments: bool = True,
        include_error_handling: bool = True,
    ) -> DynamicResponse:
        """Generate response using Hugging Face Transformers."""
        
        # Build dynamic prompt
        prompt = self._build_dynamic_prompt(
            peripheral, microcontroller, operation, parameters, sdk_version, language, include_comments, include_error_handling
        )
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + settings.max_length,
                temperature=settings.temperature,
                top_p=settings.top_p,
                top_k=settings.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode("```", add_special_tokens=False)[0] if "```" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        
        # Parse dynamic response
        return self._parse_dynamic_response(generated_text, peripheral, microcontroller, operation, language)
    
    # Human-readable board names matching the phrasing used in the
    # fine-tuning dataset's instructions (see training/data/micro_api_dataset.jsonl).
    _MCU_READABLE_NAMES = {
        MicrocontrollerType.STM32: "STM32",
        MicrocontrollerType.ESP32: "ESP32",
        MicrocontrollerType.ARDUINO: "Arduino Uno",
        MicrocontrollerType.RASPBERRY_PI_PICO: "Raspberry Pi Pico",
        MicrocontrollerType.NORDIC_NRF: "Nordic nRF",
        MicrocontrollerType.TI_MSP430: "TI MSP430",
        MicrocontrollerType.ATMEL_AVR: "Atmel AVR",
    }

    def _build_dynamic_prompt(
        self,
        peripheral: PeripheralType,
        microcontroller: MicrocontrollerType,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        sdk_version: Optional[str] = None,
        language: str = "c",
        include_comments: bool = True,
        include_error_handling: bool = True,
    ) -> str:
        """Build the model prompt.

        This MUST mirror the exact template used in training/finetune.py's
        `_prepare_dataset` ("Instruction: {instruction}\\nOutput: {output}\\n") -
        the fine-tune only ever saw that format, so serving a differently
        shaped prompt (the previous verbose multi-section template) leaves
        the fine-tuned weights unable to recognize the task and produces
        near-empty output regardless of how well training went.
        """
        mcu_readable = self._MCU_READABLE_NAMES.get(microcontroller, microcontroller.value)
        operation_readable = operation.replace("_", " ")
        instruction = f"Show me how to {operation_readable} {peripheral.value.lower()} on {mcu_readable}"

        return f"Instruction: {instruction}\nOutput:"
    
    def _parse_dynamic_response(
        self,
        generated_text: str,
        peripheral: PeripheralType,
        microcontroller: MicrocontrollerType,
        operation: str,
        language: str,
    ) -> DynamicResponse:
        """Parse dynamic response from model."""
        
        # Extract code from markdown blocks
        code = self._extract_code(generated_text, language)
        
        # Extract function signature
        function_signature = self._extract_function_signature(generated_text)
        
        # Extract example usage
        example_usage = self._extract_example_usage(generated_text)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(generated_text)
        
        # Extract warnings
        warnings = self._extract_warnings(generated_text)
        
        # Generate explanation
        explanation = f"Generated {operation} code for {peripheral.value.upper()} on {microcontroller.value.upper()} using {language.upper()}."
        
        # Metadata
        metadata = {
            "peripheral": peripheral.value,
            "microcontroller": microcontroller.value,
            "operation": operation,
            "language": language,
            "generated_at": time.time(),
            "model_used": settings.model_path or settings.model_name
        }
        
        return DynamicResponse(
            code=code,
            explanation=explanation,
            function_signature=function_signature,
            example_usage=example_usage,
            dependencies=dependencies,
            warnings=warnings,
            metadata=metadata
        )
    
    def _extract_code(self, text: str, language: str) -> str:
        """Extract code from generated text."""
        # Find code blocks
        code_pattern = rf"```{language}\n(.*?)\n```"
        match = re.search(code_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for any code block
        code_pattern = r"```\n(.*?)\n```"
        match = re.search(code_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return text.strip()
    
    def _extract_function_signature(self, text: str) -> str:
        """Extract function signature from generated text."""
        signature_pattern = r"Function Signature:\s*(.*?)(?:\n|$)"
        match = re.search(signature_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "void function_name()"
    
    def _extract_example_usage(self, text: str) -> str:
        """Extract example usage from generated text."""
        example_pattern = r"Example Usage:\s*(.*?)(?:\n|$)"
        match = re.search(example_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "function_name();"
    
    def _extract_dependencies(self, text: str) -> List[str]:
        """Extract dependencies from generated text."""
        deps_pattern = r"Dependencies:\s*(.*?)(?:\n|$)"
        match = re.search(deps_pattern, text, re.IGNORECASE)
        if match:
            deps_str = match.group(1).strip()
            return [dep.strip() for dep in deps_str.split(",")]
        return []
    
    def _extract_warnings(self, text: str) -> List[str]:
        """Extract warnings from generated text."""
        warnings_pattern = r"Warnings:\s*(.*?)(?:\n|$)"
        match = re.search(warnings_pattern, text, re.IGNORECASE)
        if match:
            warnings_str = match.group(1).strip()
            return [warning.strip() for warning in warnings_str.split(",")]
        return []
    
    def _generate_error_response(self, error_message: str, suggestions: List[str]) -> DynamicResponse:
        """Generate error response."""
        code = f"// Error: {error_message}\n"
        if suggestions:
            code += "// Suggestions:\n"
            for suggestion in suggestions:
                code += f"// - {suggestion}\n"
        code += "\n// Please try again with valid parameters."
        
        return DynamicResponse(
            code=code,
            explanation=f"Error: {error_message}",
            function_signature="error_function()",
            example_usage="// No example available due to error",
            dependencies=[],
            warnings=suggestions,
            metadata={"error": True, "error_message": error_message}
        )
    
    def _update_performance_metrics(self, start_time: float, response_length: int) -> None:
        """Update performance metrics."""
        latency = time.time() - start_time
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["total_tokens"] += response_length // 4
        self.performance_metrics["latency_history"].append(latency)
        
        if len(self.performance_metrics["latency_history"]) > 100:
            self.performance_metrics["latency_history"] = self.performance_metrics["latency_history"][-100:]
        
        if latency > 0:
            throughput = 1.0 / latency
            self.performance_metrics["throughput_history"].append(throughput)
            
            if len(self.performance_metrics["throughput_history"]) > 100:
                self.performance_metrics["throughput_history"] = self.performance_metrics["throughput_history"][-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.performance_metrics["latency_history"]:
            return {
                "total_requests": 0,
                "avg_latency": 0,
                "p50_latency": 0,
                "p95_latency": 0,
                "avg_throughput": 0,
                "total_tokens": 0
            }
        
        import statistics
        return {
            "total_requests": self.performance_metrics["total_requests"],
            "avg_latency": statistics.mean(self.performance_metrics["latency_history"]),
            "p50_latency": statistics.median(self.performance_metrics["latency_history"]),
            "p95_latency": sorted(self.performance_metrics["latency_history"])[int(len(self.performance_metrics["latency_history"]) * 0.95)],
            "avg_throughput": statistics.mean(self.performance_metrics["throughput_history"]) if self.performance_metrics["throughput_history"] else 0,
            "total_tokens": self.performance_metrics["total_tokens"]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": settings.model_path or settings.model_name,
            "model_type": "Dynamic Inference Engine",
            "parameters": "dynamic",
            "max_length": settings.max_length,
            "supported_peripherals": [p.value for p in PeripheralType],
            "supported_microcontrollers": [m.value for m in MicrocontrollerType],
            "fine_tuned": True,
            "training_date": "dynamic",
            "vllm_enabled": self.vllm_engine is not None,
        }

# Global dynamic inference engine instance
dynamic_inference_engine = DynamicInferenceEngine()

@asynccontextmanager
async def get_dynamic_inference_engine():
    """Context manager for the dynamic inference engine."""
    if not dynamic_inference_engine.model_loaded:
        await dynamic_inference_engine.initialize()
    yield dynamic_inference_engine
