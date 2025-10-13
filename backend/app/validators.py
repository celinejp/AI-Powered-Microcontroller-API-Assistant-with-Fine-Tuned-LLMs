"""
Validation module for the Microcontroller API Assistant.

This module provides comprehensive input validation and error handling for:
- Microcontroller types and capabilities
- Peripheral types and compatibility
- Query validation and sanitization
- Error message generation and suggestions
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from app.models import MicrocontrollerType, PeripheralType


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, suggestions: List[str] = None, error_code: str = None):
        self.message = message
        self.suggestions = suggestions or []
        self.error_code = error_code
        super().__init__(self.message)


class Validator:
    """Comprehensive validator for Microcontroller API Assistant inputs."""
    
    def __init__(self):
        self.microcontroller_capabilities = {
            MicrocontrollerType.ARDUINO: {
                "gpio_pins": list(range(0, 20)),  # Digital pins 0-19
                "analog_pins": [14, 15, 16, 17, 18, 19],  # Analog pins A0-A5
                "pwm_pins": [3, 5, 6, 9, 10, 11],
                "uart_ports": ["Serial"],  # Hardware UART
                "spi_ports": ["SPI"],  # Hardware SPI
                "i2c_ports": ["Wire"],  # Hardware I2C
                "supported_peripherals": [
                    PeripheralType.GPIO, PeripheralType.UART, 
                    PeripheralType.SPI, PeripheralType.I2C
                ],
                "max_clock_speed": 16000000,  # 16 MHz
                "memory_ram": 2048,  # 2KB SRAM
                "memory_flash": 32768,  # 32KB Flash
            },
            MicrocontrollerType.ESP32: {
                "gpio_pins": list(range(0, 40)),  # GPIO 0-39
                "analog_pins": list(range(32, 40)),  # ADC1 channels
                "pwm_pins": list(range(0, 40)),  # All GPIO pins support PWM
                "uart_ports": ["UART0", "UART1", "UART2"],
                "spi_ports": ["SPI1", "SPI2", "SPI3"],
                "i2c_ports": ["I2C0", "I2C1"],
                "supported_peripherals": [
                    PeripheralType.GPIO, PeripheralType.UART, 
                    PeripheralType.SPI, PeripheralType.I2C
                ],
                "max_clock_speed": 240000000,  # 240 MHz
                "memory_ram": 520000,  # 520KB SRAM
                "memory_flash": 4194304,  # 4MB Flash
            },
            MicrocontrollerType.STM32: {
                "gpio_pins": list(range(0, 16)),  # GPIO A-P, 16 pins each
                "analog_pins": list(range(0, 16)),  # ADC channels
                "pwm_pins": list(range(0, 16)),  # Timer channels
                "uart_ports": ["USART1", "USART2", "USART3"],
                "spi_ports": ["SPI1", "SPI2", "SPI3"],
                "i2c_ports": ["I2C1", "I2C2"],
                "supported_peripherals": [
                    PeripheralType.GPIO, PeripheralType.UART, 
                    PeripheralType.SPI, PeripheralType.I2C
                ],
                "max_clock_speed": 168000000,  # 168 MHz
                "memory_ram": 196608,  # 192KB SRAM
                "memory_flash": 1048576,  # 1MB Flash
            },

        }
        self.peripheral_compatibility = self._initialize_peripheral_compatibility()
        self.query_patterns = self._initialize_query_patterns()
    

    
    def _initialize_peripheral_compatibility(self) -> Dict[PeripheralType, Dict]:
        """Initialize peripheral compatibility matrix."""
        return {
            PeripheralType.GPIO: {
                "supported_mcus": [
                    MicrocontrollerType.ARDUINO,
                    MicrocontrollerType.ESP32,
                    MicrocontrollerType.STM32
                ],
                "required_pins": 1,
                "max_pins": 50,
                "operations": ["read", "write", "toggle", "pwm"]
            },
            PeripheralType.UART: {
                "supported_mcus": [
                    MicrocontrollerType.ARDUINO,
                    MicrocontrollerType.ESP32,
                    MicrocontrollerType.STM32
                ],
                "required_pins": 2,  # TX, RX
                "max_pins": 4,  # TX, RX, CTS, RTS
                "operations": ["initialize", "send", "receive", "configure"]
            },
            PeripheralType.SPI: {
                "supported_mcus": [
                    MicrocontrollerType.ARDUINO,
                    MicrocontrollerType.ESP32,
                    MicrocontrollerType.STM32
                ],
                "required_pins": 4,  # MOSI, MISO, SCK, CS
                "max_pins": 5,  # MOSI, MISO, SCK, CS, RESET
                "operations": ["initialize", "transfer", "configure"]
            },
            PeripheralType.I2C: {
                "supported_mcus": [
                    MicrocontrollerType.ARDUINO,
                    MicrocontrollerType.ESP32,
                    MicrocontrollerType.STM32
                ],
                "required_pins": 2,  # SDA, SCL
                "max_pins": 2,
                "operations": ["initialize", "read", "write", "scan"]
            },

        }
    
    def _initialize_query_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for query validation."""
        return {
            "pin_number": re.compile(r'pin\s+(\d+)', re.IGNORECASE),
            "baud_rate": re.compile(r'(\d+)\s*baud', re.IGNORECASE),
            "i2c_address": re.compile(r'0x[0-9a-fA-F]{2}', re.IGNORECASE),
            "spi_mode": re.compile(r'mode\s+(\d)', re.IGNORECASE),
            "frequency": re.compile(r'(\d+)\s*(?:MHz|kHz|Hz)', re.IGNORECASE),
            "voltage": re.compile(r'(\d+(?:\.\d+)?)\s*(?:V|volts)', re.IGNORECASE),
        }
    
    def validate_microcontroller(self, mcu: MicrocontrollerType) -> None:
        """Validate microcontroller type."""
        if mcu not in self.microcontroller_capabilities:
            suggestions = [f"Supported MCUs: {', '.join([m.value for m in self.microcontroller_capabilities.keys()])}"]
            raise ValidationError(
                f"Unsupported microcontroller: {mcu}",
                suggestions=suggestions,
                error_code="INVALID_MCU"
            )
    
    def validate_peripheral(self, peripheral: PeripheralType) -> None:
        """Validate peripheral type."""
        if peripheral not in self.peripheral_compatibility:
            suggestions = [f"Supported peripherals: {', '.join([p.value for p in self.peripheral_compatibility.keys()])}"]
            raise ValidationError(
                f"Unsupported peripheral: {peripheral}",
                suggestions=suggestions,
                error_code="INVALID_PERIPHERAL"
            )
    
    def validate_mcu_peripheral_compatibility(
        self, 
        mcu: MicrocontrollerType, 
        peripheral: PeripheralType
    ) -> None:
        """Validate MCU-peripheral compatibility."""
        self.validate_microcontroller(mcu)
        self.validate_peripheral(peripheral)
        
        if peripheral not in self.peripheral_compatibility:
            raise ValidationError(f"Unknown peripheral: {peripheral}")
        
        if mcu not in self.peripheral_compatibility[peripheral]["supported_mcus"]:
            supported_mcus = [m.value for m in self.peripheral_compatibility[peripheral]["supported_mcus"]]
            suggestions = [
                f"Use one of these MCUs for {peripheral.value}: {', '.join(supported_mcus)}",
                f"Or use a different peripheral for {mcu.value}"
            ]
            raise ValidationError(
                f"{peripheral.value} is not supported on {mcu.value}",
                suggestions=suggestions,
                error_code="INCOMPATIBLE_MCU_PERIPHERAL"
            )
    
    def validate_pin_number(self, pin: int, mcu: MicrocontrollerType) -> None:
        """Validate pin number for specific microcontroller."""
        self.validate_microcontroller(mcu)
        
        capabilities = self.microcontroller_capabilities[mcu]
        valid_pins = capabilities["gpio_pins"]
        
        if pin not in valid_pins:
            suggestions = [
                f"Valid pins for {mcu.value}: {', '.join(map(str, valid_pins[:10]))}{'...' if len(valid_pins) > 10 else ''}",
                f"Pin {pin} is out of range for {mcu.value}"
            ]
            raise ValidationError(
                f"Invalid pin number {pin} for {mcu.value}",
                suggestions=suggestions,
                error_code="INVALID_PIN"
            )
    
    def validate_query(self, query: str) -> Tuple[str, List[str]]:
        """Validate and sanitize user query."""
        if not query or not query.strip():
            raise ValidationError(
                "Query cannot be empty",
                suggestions=["Provide a specific question about microcontroller programming"],
                error_code="EMPTY_QUERY"
            )
        
        query = query.strip()
        
        # Check for minimum query length
        if len(query) < 10:
            suggestions = [
                "Be more specific about what you want to do",
                "Include the microcontroller type and peripheral",
                "Example: 'How to control an LED on Arduino Uno pin 13?'"
            ]
            raise ValidationError(
                "Query is too short. Please provide more details.",
                suggestions=suggestions,
                error_code="QUERY_TOO_SHORT"
            )
        
        # Check for maximum query length
        if len(query) > 500:
            suggestions = [
                "Break down your question into smaller parts",
                "Focus on one specific peripheral or operation"
            ]
            raise ValidationError(
                "Query is too long. Please be more concise.",
                suggestions=suggestions,
                error_code="QUERY_TOO_LONG"
            )
        
        # Extract potential parameters from query
        extracted_params = self._extract_parameters_from_query(query)
        
        return query, extracted_params
    
    def _extract_parameters_from_query(self, query: str) -> List[str]:
        """Extract potential parameters from query for validation."""
        params = []
        
        for param_name, pattern in self.query_patterns.items():
            matches = pattern.findall(query)
            if matches:
                params.extend(matches)
        
        return params
    
    def validate_parameters(self, parameters: Dict[str, Any], mcu: MicrocontrollerType, peripheral: PeripheralType) -> None:
        """Validate operation parameters."""
        if not parameters:
            return
        
        # Validate pin numbers if present
        if "pins" in parameters:
            pins = parameters["pins"]
            if isinstance(pins, list):
                for pin in pins:
                    self.validate_pin_number(pin, mcu)
            elif isinstance(pins, int):
                self.validate_pin_number(pins, mcu)
        
        # Validate baud rate for UART
        if peripheral == PeripheralType.UART and "baud_rate" in parameters:
            baud_rate = parameters["baud_rate"]
            if not isinstance(baud_rate, int) or baud_rate <= 0:
                raise ValidationError(
                    "Invalid baud rate. Must be a positive integer.",
                    suggestions=["Common baud rates: 9600, 115200, 230400"],
                    error_code="INVALID_BAUD_RATE"
                )
        
        # Validate I2C address
        if peripheral == PeripheralType.I2C and "address" in parameters:
            address = parameters["address"]
            if isinstance(address, str):
                if not re.match(r'0x[0-9a-fA-F]{2}', address):
                    raise ValidationError(
                        "Invalid I2C address format. Use 0xXX format.",
                        suggestions=["Example: 0x48, 0x3C, 0x68"],
                        error_code="INVALID_I2C_ADDRESS"
                    )
            elif isinstance(address, int):
                if address < 0 or address > 127:
                    raise ValidationError(
                        "Invalid I2C address. Must be between 0 and 127.",
                        suggestions=["Use 7-bit addresses (0-127)"],
                        error_code="INVALID_I2C_ADDRESS"
                    )
    
    def get_fallback_suggestions(self, mcu: MicrocontrollerType, peripheral: PeripheralType) -> List[str]:
        """Get fallback suggestions for failed validation."""
        suggestions = []
        
        # Check if peripheral is supported on any MCU
        if peripheral in self.peripheral_compatibility:
            supported_mcus = self.peripheral_compatibility[peripheral]["supported_mcus"]
            if mcu not in supported_mcus:
                suggestions.append(f"Try using {peripheral.value} on: {', '.join([m.value for m in supported_mcus])}")
        
        # Suggest alternative peripherals for the MCU
        if mcu in self.microcontroller_capabilities:
            supported_peripherals = self.microcontroller_capabilities[mcu]["supported_peripherals"]
            suggestions.append(f"Available peripherals for {mcu.value}: {', '.join([p.value for p in supported_peripherals])}")
        
        # Generic suggestions
        suggestions.extend([
            "Check the microcontroller datasheet for pin assignments",
            "Verify peripheral compatibility with your MCU",
            "Consider using a different pin or peripheral"
        ])
        
        return suggestions
    
    def validate_complete_request(
        self, 
        mcu: MicrocontrollerType, 
        peripheral: PeripheralType, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Complete validation of a code generation request."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "suggestions": [],
            "extracted_params": []
        }
        
        try:
            # Validate basic components
            self.validate_microcontroller(mcu)
            self.validate_peripheral(peripheral)
            self.validate_mcu_peripheral_compatibility(mcu, peripheral)
            
            # Validate query
            sanitized_query, extracted_params = self.validate_query(query)
            validation_result["extracted_params"] = extracted_params
            
            # Validate parameters
            if parameters:
                self.validate_parameters(parameters, mcu, peripheral)
            
            # Add helpful suggestions
            validation_result["suggestions"] = self._get_helpful_suggestions(mcu, peripheral, query)
            
        except ValidationError as e:
            validation_result["valid"] = False
            validation_result["error"] = {
                "message": e.message,
                "suggestions": e.suggestions,
                "error_code": e.error_code
            }
        
        return validation_result
    
    def _get_helpful_suggestions(self, mcu: MicrocontrollerType, peripheral: PeripheralType, query: str) -> List[str]:
        """Get helpful suggestions based on the request."""
        suggestions = []
        
        # MCU-specific suggestions
        if mcu == MicrocontrollerType.ARDUINO:
            suggestions.append("Arduino Uno uses 5V logic levels")
            suggestions.append("Built-in LED is on pin 13")
        elif mcu == MicrocontrollerType.ESP32:
            suggestions.append("ESP32 uses 3.3V logic levels")
            suggestions.append("Built-in WiFi and Bluetooth available")
        elif mcu == MicrocontrollerType.STM32:
            suggestions.append("STM32 uses 3.3V logic levels")
            suggestions.append("Multiple UART and SPI ports available")
        
        # Peripheral-specific suggestions
        if peripheral == PeripheralType.UART:
            suggestions.append("Common baud rates: 9600, 115200, 230400")
        elif peripheral == PeripheralType.I2C:
            suggestions.append("I2C uses pull-up resistors (typically 4.7kΩ)")
        elif peripheral == PeripheralType.SPI:
            suggestions.append("SPI requires CS (Chip Select) pin")
        
        return suggestions


# Global validator instance
validator = Validator()
