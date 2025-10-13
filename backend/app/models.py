"""
Pydantic models for API request and response schemas.
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class PeripheralType(str, Enum):
    """Supported microcontroller peripherals."""
    UART = "uart"
    SPI = "spi"
    GPIO = "gpio"
    I2C = "i2c"


class MicrocontrollerType(str, Enum):
    """Supported microcontroller families."""
    STM32 = "stm32"
    ESP32 = "esp32"
    ARDUINO = "arduino"
    RASPBERRY_PI_PICO = "raspberry_pi_pico"
    NORDIC_NRF = "nordic_nrf"
    TI_MSP430 = "ti_msp430"
    ATMEL_AVR = "atmel_avr"


class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""
    peripheral: PeripheralType = Field(..., description="Type of peripheral")
    microcontroller: MicrocontrollerType = Field(..., description="Target microcontroller")
    operation: str = Field(..., description="Operation to perform (e.g., 'initialize', 'send_data', 'read_data')")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional parameters for the operation")
    sdk_version: Optional[str] = Field(default=None, description="Specific SDK version to target")
    language: str = Field(default="c", description="Programming language (c, cpp, python)")
    include_comments: bool = Field(default=True, description="Include explanatory comments in generated code")
    include_error_handling: bool = Field(default=True, description="Include error handling code")


class CodeGenerationResponse(BaseModel):
    """Response model for code generation."""
    code: str = Field(..., description="Generated code snippet")
    explanation: str = Field(..., description="Explanation of the generated code")
    peripheral: PeripheralType = Field(..., description="Type of peripheral")
    microcontroller: MicrocontrollerType = Field(..., description="Target microcontroller")
    operation: str = Field(..., description="Operation performed")
    language: str = Field(..., description="Programming language used")
    sdk_compliance: str = Field(..., description="SDK compliance information")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or notes")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(default=None, description="Name of loaded model")
    uptime: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    error_code: Optional[str] = Field(default=None, description="Error code")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    parameters: int = Field(..., description="Number of parameters")
    max_length: int = Field(..., description="Maximum sequence length")
    supported_peripherals: List[PeripheralType] = Field(..., description="Supported peripherals")
    supported_microcontrollers: List[MicrocontrollerType] = Field(..., description="Supported microcontrollers")
    fine_tuned: bool = Field(..., description="Whether the model is fine-tuned")
    training_date: Optional[str] = Field(default=None, description="Training date")


class SupportedOperationsResponse(BaseModel):
    """Supported operations for each peripheral."""
    peripheral: PeripheralType = Field(..., description="Peripheral type")
    operations: List[str] = Field(..., description="List of supported operations")
    descriptions: Dict[str, str] = Field(..., description="Operation descriptions")
    required_parameters: Dict[str, List[str]] = Field(..., description="Required parameters for each operation")


class SimpleQueryRequest(BaseModel):
    """Request model for simple query generation."""
    query: str = Field(..., description="Query string (e.g., 'How to configure SPI on Arduino Uno?')")


class SimpleQueryResponse(BaseModel):
    """Response model for simple query generation."""
    response: str = Field(..., description="Generated response with code snippet")
    query: str = Field(..., description="Original query")
    model_used: str = Field(..., description="Model used for generation")
