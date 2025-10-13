"""
Code Validators for Microcontroller API Assistant QA
===================================================

This module provides validation functions to check generated code for:
- Expected functions and keywords
- Correct library usage
- Edge case handling
- Graceful error responses
"""

import re
from typing import Dict, List, Set
from cases import MicrocontrollerType


class CodeValidator:
    """Validates generated code against expected patterns and handles edge cases."""
    
    def __init__(self):
        # Define critical keywords that must be present for valid code
        self.critical_keywords = {
            "setup": ["void setup()", "setup()"],
            "loop": ["void loop()", "loop()"],
            "semicolon": [";"],
            "braces": ["{", "}"],
        }
        
        # Board-specific pin maps and API expectations
        self.board_configs = {
            "arduino_uno": {
                "valid_pins": {
                    "digital": list(range(0, 14)),  # 0-13
                    "analog": [f"A{i}" for i in range(6)],  # A0-A5
                    "pwm": [3, 5, 6, 9, 10, 11]
                },
                "expected_api": ["pinMode", "digitalWrite", "digitalRead", "analogRead", "analogWrite"],
                "allowed_libraries": ["SPI.h", "Wire.h", "Servo.h", "DHT.h", "LiquidCrystal.h", "LiquidCrystal_I2C.h"],
                "forbidden_api": ["gpio_config", "gpio_set_level", "HAL_GPIO_WritePin"]
            },
            "esp32": {
                "valid_pins": {
                    "digital": [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33, 34, 35, 36, 39],
                    "analog": [32, 33, 34, 35, 36, 39],
                    "pwm": [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27]
                },
                "expected_api": ["gpio_set_level", "gpio_get_level", "gpio_config", "GPIO_NUM_"],
                "allowed_libraries": ["driver/gpio.h", "driver/spi_master.h", "driver/i2c.h", "WiFi.h", "DHT.h"],
                "forbidden_api": ["HAL_GPIO_WritePin", "HAL_GPIO_ReadPin"]
            },
            "stm32": {
                "valid_pins": {
                    "digital": list(range(0, 16)),  # PA0-PA15, PB0-PB15, etc.
                    "analog": list(range(0, 16)),
                    "pwm": list(range(0, 16))
                },
                "expected_api": ["HAL_GPIO_WritePin", "HAL_GPIO_ReadPin", "HAL_UART_Transmit", "HAL_SPI_Transmit"],
                "allowed_libraries": ["stm32f4xx_hal.h", "stm32f1xx_hal.h"],
                "forbidden_api": ["pinMode", "digitalWrite", "gpio_set_level"]
            }
        }
        
        # API response schema validation
        self.expected_schema = {
            "response": str,
            "model_used": str,
            "latency_ms": (int, float)
        }
        
        # Define warning patterns for edge cases
        self.warning_patterns = {
            "invalid_pin": [
                r"pin\s+1[0-9][0-9]",  # Pin 100+
                r"GPIO_NUM_1[0-9][0-9]",  # ESP32 pin 100+
                r"pin\s+[0-9]{3,}",  # Any 3+ digit pin
            ],
            "invalid_i2c": [
                r"SDA\s*=\s*SCL",
                r"SCL\s*=\s*SDA",
                r"i2c.*SDA.*SCL.*same",
            ],
            "ambiguous_query": [
                r"coffee",
                r"make.*drink",
                r"cook.*food",
            ],
            "vague_request": [
                r"do something",
                r"something with",
                r"work with",
            ]
        }
        
        # Define error patterns that indicate invalid code
        self.error_patterns = {
            "syntax_error": [
                r"void setup\(\)\s*\{[^}]*$",  # Unclosed setup function
                r"void loop\(\)\s*\{[^}]*$",   # Unclosed loop function
                r"[^;]\s*$",  # Line without semicolon (basic check)
            ],
            "missing_critical": [
                r"(?<!void\s)setup\(",  # setup() without void
                r"(?<!void\s)loop\(",   # loop() without void
            ]
        }
        
        # Define graceful handling indicators
        self.graceful_indicators = {
            "fallback_template": [
                r"Microcontroller code template",
                r"Based on your query",
                r"Add your specific",
            ],
            "warning_comments": [
                r"//.*warning",
                r"//.*note",
                r"//.*check",
            ],
            "error_handling": [
                r"if.*error",
                r"if.*fail",
                r"try.*catch",
            ]
        }
    
    def validate_code(self, 
                     generated_code: str, 
                     expected_keywords: List[str], 
                     expected_libraries: List[str],
                     microcontroller_type: MicrocontrollerType,
                     api_response: Dict = None) -> Dict[str, bool]:
        """
        Validate generated code against multiple criteria.
        
        Args:
            generated_code: The generated code string
            expected_keywords: List of expected keywords/functions
            expected_libraries: List of expected libraries
            microcontroller_type: Type of microcontroller
            
        Returns:
            Dictionary with validation results
        """
        if not generated_code or generated_code.strip() == "":
            return {
                "syntax": False,
                "critical_keywords": False,
                "expected_keywords": False,
                "expected_libraries": False,
                "edge_case_handling": False,
                "graceful_fallback": False,
            }
        
        # Basic syntax validation
        syntax_valid = self._validate_syntax(generated_code)
        
        # Critical keywords validation
        critical_valid = self._validate_critical_keywords(generated_code)
        
        # Expected keywords validation
        expected_keywords_valid = self._validate_expected_keywords(generated_code, expected_keywords)
        
        # Expected libraries validation
        expected_libraries_valid = self._validate_expected_libraries(generated_code, expected_libraries)
        
        # Edge case handling validation
        edge_case_valid = self._validate_edge_case_handling(generated_code)
        
        # Graceful fallback validation
        graceful_fallback_valid = self._validate_graceful_fallback(generated_code, expected_keywords)
        
        # Board-specific validation
        board_validation = self._validate_board_specific(generated_code, microcontroller_type)
        
        # API schema validation
        schema_valid = self._validate_api_schema(api_response) if api_response else True
        
        return {
            "syntax": syntax_valid,
            "critical_keywords": critical_valid,
            "expected_keywords": expected_keywords_valid,
            "expected_libraries": expected_libraries_valid,
            "edge_case_handling": edge_case_valid,
            "graceful_fallback": graceful_fallback_valid,
            "board_specific": board_validation,
            "api_schema": schema_valid,
        }
    
    def _validate_syntax(self, code: str) -> bool:
        """Validate basic syntax structure."""
        # Check for code block markers
        if not ("```cpp" in code or "```c++" in code):
            return False
        
        # Check for basic function structure
        has_setup = "void setup()" in code
        has_loop = "void loop()" in code
        
        # Check for proper braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        braces_balanced = open_braces == close_braces and open_braces >= 2
        
        # Check for semicolons (basic indicator of proper syntax)
        has_semicolons = ';' in code
        
        return has_setup and has_loop and braces_balanced and has_semicolons
    
    def _validate_critical_keywords(self, code: str) -> bool:
        """Validate presence of critical keywords."""
        code_lower = code.lower()
        
        # Check for setup and loop functions
        has_setup = any(pattern.lower() in code_lower for pattern in self.critical_keywords["setup"])
        has_loop = any(pattern.lower() in code_lower for pattern in self.critical_keywords["loop"])
        
        # Check for basic syntax elements
        has_semicolons = any(pattern in code for pattern in self.critical_keywords["semicolon"])
        has_braces = all(pattern in code for pattern in self.critical_keywords["braces"])
        
        return has_setup and has_loop and has_semicolons and has_braces
    
    def _validate_expected_keywords(self, code: str, expected_keywords: List[str]) -> bool:
        """Validate presence of expected keywords."""
        if not expected_keywords:
            return True  # No expectations means pass
        
        code_lower = code.lower()
        found_keywords = 0
        
        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            if (keyword_lower in code_lower or 
                keyword in code or  # Original case
                keyword_lower.replace('_', '') in code_lower.replace('_', '')):
                found_keywords += 1
        
        # Require at least 50% of expected keywords
        return found_keywords >= len(expected_keywords) * 0.5
    
    def _validate_expected_libraries(self, code: str, expected_libraries: List[str]) -> bool:
        """Validate presence of expected libraries."""
        if not expected_libraries:
            return True  # No expectations means pass
        
        found_libraries = 0
        
        for library in expected_libraries:
            pattern = f"#include\\s+<{library}>"
            if re.search(pattern, code, re.IGNORECASE):
                found_libraries += 1
        
        # Require at least 50% of expected libraries
        return found_libraries >= len(expected_libraries) * 0.5
    
    def _validate_edge_case_handling(self, code: str) -> bool:
        """Validate handling of edge cases."""
        code_lower = code.lower()
        
        # Check for invalid patterns
        has_invalid_patterns = False
        for pattern_type, patterns in self.warning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_lower, re.IGNORECASE):
                    has_invalid_patterns = True
                    break
        
        # If invalid patterns are found, check for graceful handling
        if has_invalid_patterns:
            return self._has_graceful_handling(code)
        
        return True
    
    def _validate_graceful_fallback(self, code: str, expected_keywords: List[str]) -> bool:
        """Validate graceful fallback for edge cases."""
        # If no expected keywords, check for generic template
        if not expected_keywords:
            return self._has_graceful_handling(code)
        
        # Check if code provides reasonable fallback
        return self._has_graceful_handling(code)
    
    def _has_graceful_handling(self, code: str) -> bool:
        """Check if code shows graceful handling of edge cases."""
        code_lower = code.lower()
        
        # Check for fallback template indicators
        has_fallback = any(pattern.lower() in code_lower for pattern in self.graceful_indicators["fallback_template"])
        
        # Check for warning comments
        has_warnings = any(re.search(pattern, code, re.IGNORECASE) for pattern in self.graceful_indicators["warning_comments"])
        
        # Check for error handling
        has_error_handling = any(re.search(pattern, code, re.IGNORECASE) for pattern in self.graceful_indicators["error_handling"])
        
        return has_fallback or has_warnings or has_error_handling
    
    def _validate_board_specific(self, code: str, microcontroller_type: MicrocontrollerType) -> bool:
        """Validate board-specific patterns and pin usage."""
        board_name = microcontroller_type.value
        
        if board_name not in self.board_configs:
            return True  # Unknown board, skip validation
        
        config = self.board_configs[board_name]
        code_lower = code.lower()
        
        # Check for forbidden APIs
        for forbidden_api in config["forbidden_api"]:
            if forbidden_api.lower() in code_lower:
                return False
        
        # Check pin validity (basic check)
        pin_pattern = r'pin\s+(\d+)'
        pins = re.findall(pin_pattern, code_lower)
        
        for pin in pins:
            try:
                pin_num = int(pin)
                if pin_num not in config["valid_pins"]["digital"]:
                    return False
            except ValueError:
                continue
        
        return True
    
    def _validate_api_schema(self, api_response: Dict) -> bool:
        """Validate API response schema."""
        if not api_response:
            return False
        
        # Check required fields
        for field, expected_type in self.expected_schema.items():
            if field not in api_response:
                return False
            
            if isinstance(expected_type, tuple):
                if not isinstance(api_response[field], expected_type):
                    return False
            else:
                if not isinstance(api_response[field], expected_type):
                    return False
        
        return True
    
    def get_validation_details(self, code: str, expected_keywords: List[str], 
                             expected_libraries: List[str]) -> Dict:
        """Get detailed validation analysis."""
        code_lower = code.lower()
        
        # Check for specific issues
        issues = []
        warnings = []
        
        # Check for invalid patterns
        for pattern_type, patterns in self.warning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_lower, re.IGNORECASE):
                    warnings.append(f"Potential {pattern_type}: {pattern}")
        
        # Check for syntax errors
        for pattern_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.MULTILINE):
                    issues.append(f"Syntax issue: {pattern_type}")
        
        # Check for missing expected elements
        if expected_keywords:
            missing_keywords = []
            for keyword in expected_keywords:
                keyword_lower = keyword.lower()
                if not (keyword_lower in code_lower or 
                       keyword in code or 
                       keyword_lower.replace('_', '') in code_lower.replace('_', '')):
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                issues.append(f"Missing keywords: {', '.join(missing_keywords)}")
        
        if expected_libraries:
            missing_libraries = []
            for library in expected_libraries:
                pattern = f"#include\\s+<{library}>"
                if not re.search(pattern, code, re.IGNORECASE):
                    missing_libraries.append(library)
            
            if missing_libraries:
                issues.append(f"Missing libraries: {', '.join(missing_libraries)}")
        
        # Check for graceful handling indicators
        graceful_indicators = []
        for indicator_type, patterns in self.graceful_indicators.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    graceful_indicators.append(f"{indicator_type}: {pattern}")
        
        return {
            "issues": issues,
            "warnings": warnings,
            "graceful_indicators": graceful_indicators,
            "code_length": len(code),
            "has_setup": "void setup()" in code,
            "has_loop": "void loop()" in code,
            "has_comments": "//" in code or "/*" in code,
            "has_error_handling": any(re.search(pattern, code, re.IGNORECASE) 
                                    for pattern in self.graceful_indicators["error_handling"]),
        }


if __name__ == "__main__":
    # Test the validator
    validator = CodeValidator()
    
    # Test case 1: Valid code
    valid_code = """
    #include <SPI.h>
    
    void setup() {
      Serial.begin(9600);
      SPI.begin();
      Serial.println("SPI initialized!");
    }
    
    void loop() {
      // Send data over SPI
      byte data = SPI.transfer(0x55);
      Serial.println(data);
      delay(1000);
    }
    """
    
    validation = validator.validate_code(
        valid_code,
        ["SPI.begin", "SPI.transfer", "Serial.begin"],
        ["SPI.h"],
        MicrocontrollerType.ARDUINO_UNO
    )
    
    print("Valid Code Validation:", validation)
    
    # Test case 2: Edge case with graceful handling
    edge_case_code = """
    // Microcontroller code template
    // Query: How to make coffee with a microcontroller?
    // This is a basic template for microcontroller programming
    
    void setup() {
      Serial.begin(9600);
      Serial.println("Microcontroller initialized!");
    }
    
    void loop() {
      Serial.println("Hello World!");
      delay(1000);
    }
    """
    
    validation = validator.validate_code(
        edge_case_code,
        [],  # No expected keywords for edge case
        [],
        MicrocontrollerType.GENERIC
    )
    
    print("Edge Case Validation:", validation)
    
    details = validator.get_validation_details(
        edge_case_code,
        [],
        []
    )
    
    print("Validation Details:", details)
