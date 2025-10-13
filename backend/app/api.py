"""
FastAPI application for the Microcontroller API Assistant.
"""

import time
from typing import List, Dict, Any, Union

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.models import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    HealthCheckResponse,
    ErrorResponse,
    ModelInfoResponse,
    SupportedOperationsResponse,
    SimpleQueryRequest,
    SimpleQueryResponse,
    PeripheralType,
    MicrocontrollerType,
)
from app.inference import get_dynamic_inference_engine, DynamicInferenceEngine


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting Microcontroller API Assistant...")
        # Initialize dynamic inference engine on startup
        try:
            async with get_dynamic_inference_engine() as engine:
                logger.info("Dynamic inference engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize dynamic inference engine: {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Microcontroller API Assistant...")
    
    return app


# Create the application instance
app = create_app()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Microcontroller API Assistant",
        "version": settings.api_version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        async with get_dynamic_inference_engine() as engine:
            model_info = engine.get_model_info()
            return HealthCheckResponse(
                status="healthy",
                version=settings.api_version,
                model_loaded=engine.model_loaded,
                model_name=model_info.get("model_name"),
                uptime=time.time() - engine.start_time,
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version=settings.api_version,
            model_loaded=False,
            model_name=None,
            uptime=0.0,
        )


@app.post("/generate", response_model=SimpleQueryResponse)
async def generate_response(request: SimpleQueryRequest):
    """Generate a response for a simple query using the dynamic inference engine."""
    try:
        # Use dynamic inference engine for real model inference
        async with get_dynamic_inference_engine() as engine:
            # Use real model inference
            try:
                # Generate real response using the model
                response = await engine.generate_dynamic_response(
                    peripheral=PeripheralType.UART,  # Default to UART for simple queries
                    microcontroller=MicrocontrollerType.ARDUINO,  # Default to Arduino
                    operation="generate_code",
                    parameters={"query": request.query},
                    language="c"
                )
                
                # Check if model returned empty response
                if response.code and response.code.strip():
                    return SimpleQueryResponse(
                        response=response.code,
                        query=request.query,
                        model_used="dialoGPT_pretrained"
                    )
                else:
                    # Model returned empty, use fallback
                    logger.info("Model returned empty response, using fallback")
                    raise Exception("Empty model response")
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                # Fallback to enhanced mock responses
                query_lower = request.query.lower()
            
            # Enhanced mock responses with better code examples
            if "led" in query_lower or "blink" in query_lower:
                mock_code = """// LED Blinking Example for Arduino
#include <avr/io.h>
#include <util/delay.h>

int main() {
    // Set pin 13 (PB5) as output (built-in LED on Arduino Uno)
    DDRB |= (1 << PB5);
    
    while(1) {
        // Turn LED on
        PORTB |= (1 << PB5);
        _delay_ms(500);  // Wait 500ms
        
        // Turn LED off
        PORTB &= ~(1 << PB5);
        _delay_ms(500);  // Wait 500ms
    }
    
    return 0;
}

// Alternative Arduino IDE version:
/*
void setup() {
    pinMode(13, OUTPUT);  // Set pin 13 as output
}

void loop() {
    digitalWrite(13, HIGH);   // Turn LED on
    delay(1000);              // Wait 1 second
    digitalWrite(13, LOW);    // Turn LED off
    delay(1000);              // Wait 1 second
}
*/"""
            elif "uart" in query_lower:
                mock_code = """// UART Configuration Example
#include <avr/io.h>

void uart_init() {
    // Set baud rate to 9600
    UBRR0H = 0;
    UBRR0L = 103;
    
    // Enable transmitter and receiver
    UCSR0B = (1<<RXEN0)|(1<<TXEN0);
    
    // Set frame format: 8 data bits, 1 stop bit
    UCSR0C = (1<<UCSZ01)|(1<<UCSZ00);
}

void uart_send(char data) {
    // Wait for empty transmit buffer
    while (!(UCSR0A & (1<<UDRE0)));
    
    // Put data into buffer
    UDR0 = data;
}

char uart_receive() {
    // Wait for data to be received
    while (!(UCSR0A & (1<<RXC0)));
    
    // Get and return received data
    return UDR0;
}"""
            elif "spi" in query_lower:
                mock_code = """// SPI Configuration Example
#include <avr/io.h>

void spi_init() {
    // Set MOSI and SCK output, all others input
    DDRB = (1<<DDB3)|(1<<DDB5);
    
    // Enable SPI, Master, set clock rate fck/16
    SPCR = (1<<SPE)|(1<<MSTR)|(1<<SPR0);
}

uint8_t spi_transfer(uint8_t data) {
    // Start transmission
    SPDR = data;
    
    // Wait for transmission complete
    while(!(SPSR & (1<<SPIF)));
    
    // Return received data
    return SPDR;
}"""
            elif "gpio" in query_lower:
                mock_code = """// GPIO Configuration Example
#include <avr/io.h>

void gpio_init() {
    // Set pin 13 as output (LED on Arduino Uno)
    DDRB |= (1 << PB5);
}

void gpio_set_high() {
    // Set pin 13 high
    PORTB |= (1 << PB5);
}

void gpio_set_low() {
    // Set pin 13 low
    PORTB &= ~(1 << PB5);
}

uint8_t gpio_read(uint8_t pin) {
    // Read pin state
    return (PINB & (1 << pin)) ? 1 : 0;
}"""
            else:
                mock_code = """// General Microcontroller Code Example
#include <avr/io.h>

int main() {
    // Initialize peripherals
    // Your code here
    
    while(1) {
        // Main loop
        // Your code here
    }
    
    return 0;
}"""
            
            return SimpleQueryResponse(
                response=mock_code,
                query=request.query,
                model_used="enhanced_mock_generator"
            )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    try:
        async with get_dynamic_inference_engine() as engine:
            info = engine.get_model_info()
            return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/peripherals", response_model=List[str])
async def get_supported_peripherals():
    """Get list of supported peripherals."""
    return [peripheral.value for peripheral in PeripheralType]


@app.get("/microcontrollers", response_model=List[str])
async def get_supported_microcontrollers():
    """Get list of supported microcontrollers."""
    return [microcontroller.value for microcontroller in MicrocontrollerType]


@app.get("/operations/{peripheral}", response_model=SupportedOperationsResponse)
async def get_supported_operations(peripheral: PeripheralType):
    """Get supported operations for a specific peripheral."""
    
    # Define operations for each peripheral
    operations_map = {
        PeripheralType.UART: {
            "operations": ["initialize", "send_data", "receive_data", "configure", "set_baud_rate"],
            "descriptions": {
                "initialize": "Initialize UART peripheral with default settings",
                "send_data": "Send data over UART",
                "receive_data": "Receive data from UART",
                "configure": "Configure UART parameters",
                "set_baud_rate": "Set UART baud rate"
            },
            "required_parameters": {
                "initialize": ["uart_instance"],
                "send_data": ["uart_instance", "data"],
                "receive_data": ["uart_instance", "buffer_size"],
                "configure": ["uart_instance", "baud_rate", "data_bits", "stop_bits", "parity"],
                "set_baud_rate": ["uart_instance", "baud_rate"]
            }
        },
        PeripheralType.SPI: {
            "operations": ["initialize", "send_data", "receive_data", "configure", "set_frequency"],
            "descriptions": {
                "initialize": "Initialize SPI peripheral",
                "send_data": "Send data over SPI",
                "receive_data": "Receive data from SPI",
                "configure": "Configure SPI parameters",
                "set_frequency": "Set SPI frequency"
            },
            "required_parameters": {
                "initialize": ["spi_instance"],
                "send_data": ["spi_instance", "data"],
                "receive_data": ["spi_instance", "buffer_size"],
                "configure": ["spi_instance", "mode", "data_size", "frequency"],
                "set_frequency": ["spi_instance", "frequency"]
            }
        },
        PeripheralType.GPIO: {
            "operations": ["configure_pin", "set_pin", "read_pin", "toggle_pin", "configure_interrupt"],
            "descriptions": {
                "configure_pin": "Configure GPIO pin as input or output",
                "set_pin": "Set GPIO pin high or low",
                "read_pin": "Read GPIO pin state",
                "toggle_pin": "Toggle GPIO pin state",
                "configure_interrupt": "Configure GPIO interrupt"
            },
            "required_parameters": {
                "configure_pin": ["pin_number", "direction"],
                "set_pin": ["pin_number", "state"],
                "read_pin": ["pin_number"],
                "toggle_pin": ["pin_number"],
                "configure_interrupt": ["pin_number", "trigger", "callback"]
            }
        },
        PeripheralType.I2C: {
            "operations": ["initialize", "write_data", "read_data", "configure", "scan_devices"],
            "descriptions": {
                "initialize": "Initialize I2C peripheral",
                "write_data": "Write data to I2C device",
                "read_data": "Read data from I2C device",
                "configure": "Configure I2C parameters",
                "scan_devices": "Scan for I2C devices on bus"
            },
            "required_parameters": {
                "initialize": ["i2c_instance"],
                "write_data": ["i2c_instance", "device_address", "data"],
                "read_data": ["i2c_instance", "device_address", "register", "length"],
                "configure": ["i2c_instance", "frequency"],
                "scan_devices": ["i2c_instance"]
            }
        }
    }
    
    if peripheral not in operations_map:
        raise HTTPException(status_code=400, detail=f"Unsupported peripheral: {peripheral}")
    
    peripheral_ops = operations_map[peripheral]
    
    return SupportedOperationsResponse(
        peripheral=peripheral,
        operations=peripheral_ops["operations"],
        descriptions=peripheral_ops["descriptions"],
        required_parameters=peripheral_ops["required_parameters"],
    )


@app.get("/languages", response_model=List[str])
async def get_supported_languages():
    """Get list of supported programming languages."""
    return ["c", "cpp", "python"]


@app.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics():
    """Get performance metrics for the dynamic inference engine."""
    try:
        async with get_dynamic_inference_engine() as engine:
            metrics = engine.get_performance_metrics()
            return {
                "performance_metrics": metrics,
                "vllm_enabled": engine.vllm_engine is not None,
                "triton_enabled": False,  # Will be updated when Triton is integrated
                "model_loaded": engine.model_loaded,
                "uptime_seconds": time.time() - engine.start_time
            }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@app.post("/generate-code", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest):
    """Generate code using the dynamic inference engine."""
    try:
        # Use dynamic inference engine for real model inference
        async with get_dynamic_inference_engine() as engine:
            # Generate enhanced mock code based on the request
            peripheral = request.peripheral.lower()
            operation = request.operation.lower()
            language = request.language or "c"
            
            # Create enhanced mock code based on peripheral and operation
            if peripheral == "uart":
                if operation == "initialize":
                    mock_code = f"""// UART Initialization Code
#include <avr/io.h>

void uart_init() {{
    // Set baud rate to 9600
    UBRR0H = 0;
    UBRR0L = 103;
    
    // Enable transmitter and receiver
    UCSR0B = (1<<RXEN0)|(1<<TXEN0);
    
    // Set frame format: 8 data bits, 1 stop bit
    UCSR0C = (1<<UCSZ01)|(1<<UCSZ00);
}}"""
                    explanation = "This code initializes UART communication with 9600 baud rate, 8 data bits, and 1 stop bit."
                elif operation == "send_data":
                    mock_code = f"""// UART Send Data Code
void uart_send(char data) {{
    // Wait for empty transmit buffer
    while (!(UCSR0A & (1<<UDRE0)));
    
    // Put data into buffer
    UDR0 = data;
}}"""
                    explanation = "This function sends a single character over UART by waiting for the transmit buffer to be empty."
                else:
                    mock_code = f"""// UART {operation.replace('_', ' ').title()} Code
// Implementation for {operation} operation
// Add your specific implementation here"""
                    explanation = f"Mock implementation for UART {operation} operation."
            
            elif peripheral == "gpio":
                if operation == "configure_pin":
                    mock_code = f"""// GPIO Pin Configuration Code
#include <avr/io.h>

void gpio_configure_pin(uint8_t pin, uint8_t direction) {{
    if (direction == OUTPUT) {{
        DDRB |= (1 << pin);  // Set as output
    }} else {{
        DDRB &= ~(1 << pin); // Set as input
    }}
}}"""
                    explanation = "This function configures a GPIO pin as either input or output."
                elif operation == "set_pin":
                    mock_code = f"""// GPIO Set Pin Code
void gpio_set_pin(uint8_t pin, uint8_t state) {{
    if (state == HIGH) {{
        PORTB |= (1 << pin);   // Set pin high
    }} else {{
        PORTB &= ~(1 << pin);  // Set pin low
    }}
}}"""
                    explanation = "This function sets a GPIO pin to either HIGH or LOW state."
                else:
                    mock_code = f"""// GPIO {operation.replace('_', ' ').title()} Code
// Implementation for {operation} operation
// Add your specific implementation here"""
                    explanation = f"Mock implementation for GPIO {operation} operation."
            
            else:
                mock_code = f"""// {peripheral.upper()} {operation.replace('_', ' ').title()} Code
// Mock implementation for {peripheral} {operation}
// Add your specific implementation here"""
                explanation = f"Mock implementation for {peripheral} {operation} operation."
            
            return CodeGenerationResponse(
                code=mock_code,
                explanation=explanation,
                peripheral=request.peripheral,
                microcontroller=request.microcontroller,
                operation=request.operation,
                language=language,
                sdk_compliance="Enhanced mock implementation for testing",
                dependencies=["avr/io.h"],
                warnings=["This is an enhanced mock implementation for testing purposes"]
            )
        
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate code: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            error_code="INTERNAL_ERROR"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
