"""
Code Scoring System for Microcontroller API Assistant QA
=======================================================

This module provides a lightweight scoring mechanism to evaluate generated code
based on syntax correctness, peripheral coverage, and library usage.
"""

import re
from typing import Dict, List, Set
from cases import MicrocontrollerType


class CodeScorer:
    """Scores generated code based on multiple criteria."""
    
    def __init__(self):
        # Define scoring weights
        self.weights = {
            "syntax": 0.4,      # Basic syntax and structure
            "keywords": 0.4,    # Expected keywords and functions
            "libraries": 0.2,   # Correct library usage
        }
        
        # Define microcontroller-specific patterns
        self.microcontroller_patterns = {
            MicrocontrollerType.ARDUINO_UNO: {
                "gpio": ["pinMode", "digitalWrite", "digitalRead"],
                "uart": ["Serial.begin", "Serial.print", "Serial.println"],
                "spi": ["SPI.begin", "SPI.transfer", "SPISettings"],
                "i2c": ["Wire.begin", "Wire.requestFrom", "Wire.read"],
                "servo": ["Servo", "attach", "write"],
                "wifi": ["WiFi.begin", "WiFi.status", "WL_CONNECTED"],
                "sensors": ["DHT", "dht.begin", "readTemperature", "readHumidity"],
                "lcd": ["LiquidCrystal", "lcd.begin", "lcd.print"],
            },
            MicrocontrollerType.ESP32: {
                "gpio": ["gpio_set_level", "gpio_get_level", "gpio_config", "GPIO_NUM_"],
                "uart": ["uart_config_t", "uart_driver_install", "uart_write_bytes"],
                "spi": ["spi_bus_config_t", "spi_device_interface_config_t", "spi_bus_initialize"],
                "i2c": ["i2c_config_t", "i2c_driver_install", "i2c_master_write"],
                "wifi": ["WiFi.begin", "WiFi.status", "WL_CONNECTED"],
                "sensors": ["DHT", "dht.begin", "readTemperature", "readHumidity"],
                "lcd": ["LiquidCrystal", "lcd.begin", "lcd.print"],
            },
            MicrocontrollerType.STM32: {
                "gpio": ["pinMode", "digitalWrite", "digitalRead"],
                "uart": ["Serial.begin", "Serial.print", "Serial.println"],
                "spi": ["SPI.begin", "SPI.transfer", "SPISettings"],
                "i2c": ["Wire.begin", "Wire.requestFrom", "Wire.read"],
                "servo": ["Servo", "attach", "write"],
                "pwm": ["analogWrite", "PWM", "frequency"],
            },
            MicrocontrollerType.GENERIC: {
                "gpio": ["pinMode", "digitalWrite", "digitalRead"],
                "uart": ["Serial.begin", "Serial.print", "Serial.println"],
                "spi": ["SPI.begin", "SPI.transfer"],
                "i2c": ["Wire.begin", "Wire.requestFrom", "Wire.read"],
                "servo": ["Servo", "attach", "write"],
            }
        }
        
        # Define library patterns
        self.library_patterns = {
            "SPI.h": r"#include\s+<SPI\.h>",
            "Wire.h": r"#include\s+<Wire\.h>",
            "Servo.h": r"#include\s+<Servo\.h>",
            "WiFi.h": r"#include\s+<WiFi\.h>",
            "DHT.h": r"#include\s+<DHT\.h>",
            "LiquidCrystal.h": r"#include\s+<LiquidCrystal\.h>",
            "EEPROM.h": r"#include\s+<EEPROM\.h>",
            "driver/gpio.h": r"#include\s+<driver/gpio\.h>",
            "driver/spi_master.h": r"#include\s+<driver/spi_master\.h>",
        }
    
    def score_code(self, 
                   generated_code: str, 
                   expected_keywords: List[str], 
                   expected_libraries: List[str],
                   microcontroller_type: MicrocontrollerType) -> Dict[str, float]:
        """
        Score generated code based on multiple criteria.
        
        Args:
            generated_code: The generated code string
            expected_keywords: List of expected keywords/functions
            expected_libraries: List of expected libraries
            microcontroller_type: Type of microcontroller
            
        Returns:
            Dictionary with scoring breakdown
        """
        if not generated_code or generated_code.strip() == "":
            return {
                "syntax": 0.0,
                "keywords": 0.0,
                "libraries": 0.0,
                "total": 0.0
            }
        
        # Normalize code for analysis
        code_lower = generated_code.lower()
        code_lines = generated_code.split('\n')
        
        # Score syntax
        syntax_score = self._score_syntax(generated_code, code_lines)
        
        # Score keywords
        keyword_score = self._score_keywords(code_lower, expected_keywords, microcontroller_type)
        
        # Score libraries
        library_score = self._score_libraries(generated_code, expected_libraries)
        
        # Calculate weighted total (normalized to 0-2.0 range)
        total_score = (
            syntax_score * self.weights["syntax"] +
            keyword_score * self.weights["keywords"] +
            library_score * self.weights["libraries"]
        ) * 2.0  # Scale to 0-2.0 range
        
        return {
            "syntax": round(syntax_score * 2.0, 2),
            "keywords": round(keyword_score * 2.0, 2),
            "libraries": round(library_score * 2.0, 2),
            "total": round(total_score, 2)
        }
    
    def _score_syntax(self, code: str, code_lines: List[str]) -> float:
        """Score basic syntax and structure."""
        score = 0.0
        
        # Check for code block markers
        if "```cpp" in code or "```c++" in code:
            score += 0.3
        
        # Check for basic structure
        if "void setup()" in code:
            score += 0.2
        if "void loop()" in code:
            score += 0.2
        
        # Check for proper semicolons and braces
        semicolon_count = code.count(';')
        brace_count = code.count('{') + code.count('}')
        
        if semicolon_count > 0:
            score += 0.1
        if brace_count >= 2:  # At least setup and loop functions
            score += 0.1
        
        # Check for comments
        comment_lines = sum(1 for line in code_lines if '//' in line or '/*' in line)
        if comment_lines > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_keywords(self, code_lower: str, expected_keywords: List[str], 
                       microcontroller_type: MicrocontrollerType) -> float:
        """Score keyword and function usage."""
        if not expected_keywords:
            # For edge cases where no keywords are expected, check if code is reasonable
            return self._score_generic_keywords(code_lower, microcontroller_type)
        
        found_keywords = 0
        total_keywords = len(expected_keywords)
        
        for keyword in expected_keywords:
            # Check for keyword in various forms
            keyword_lower = keyword.lower()
            if (keyword_lower in code_lower or 
                keyword in code_lower or  # Original case
                keyword_lower.replace('_', '') in code_lower.replace('_', '')):
                found_keywords += 1
        
        # Bonus for microcontroller-specific patterns
        bonus = self._calculate_microcontroller_bonus(code_lower, microcontroller_type)
        
        base_score = found_keywords / total_keywords if total_keywords > 0 else 0.0
        total_score = min(base_score + bonus, 1.0)
        
        return total_score
    
    def _score_generic_keywords(self, code_lower: str, microcontroller_type: MicrocontrollerType) -> float:
        """Score generic keywords for edge cases."""
        patterns = self.microcontroller_patterns.get(microcontroller_type, {})
        
        found_patterns = 0
        total_patterns = 0
        
        for category, keywords in patterns.items():
            for keyword in keywords:
                total_patterns += 1
                if keyword.lower() in code_lower:
                    found_patterns += 1
        
        return found_patterns / total_patterns if total_patterns > 0 else 0.5
    
    def _calculate_microcontroller_bonus(self, code_lower: str, microcontroller_type: MicrocontrollerType) -> float:
        """Calculate bonus for microcontroller-specific patterns."""
        patterns = self.microcontroller_patterns.get(microcontroller_type, {})
        bonus = 0.0
        
        for category, keywords in patterns.items():
            category_found = any(keyword.lower() in code_lower for keyword in keywords)
            if category_found:
                bonus += 0.1
        
        return min(bonus, 0.3)  # Cap bonus at 0.3
    
    def _score_libraries(self, code: str, expected_libraries: List[str]) -> float:
        """Score library usage."""
        if not expected_libraries:
            return 0.5  # Neutral score for no expected libraries
        
        found_libraries = 0
        total_libraries = len(expected_libraries)
        
        for library in expected_libraries:
            pattern = self.library_patterns.get(library, f"#include\\s+<{library}>")
            if re.search(pattern, code, re.IGNORECASE):
                found_libraries += 1
        
        return found_libraries / total_libraries if total_libraries > 0 else 0.0
    
    def get_detailed_analysis(self, generated_code: str, expected_keywords: List[str], 
                            expected_libraries: List[str], microcontroller_type: MicrocontrollerType) -> Dict:
        """Get detailed analysis of the generated code."""
        code_lower = generated_code.lower()
        
        # Find missing keywords
        missing_keywords = []
        found_keywords = []
        
        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            if (keyword_lower in code_lower or 
                keyword in generated_code or 
                keyword_lower.replace('_', '') in code_lower.replace('_', '')):
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        # Find missing libraries
        missing_libraries = []
        found_libraries = []
        
        for library in expected_libraries:
            pattern = self.library_patterns.get(library, f"#include\\s+<{library}>")
            if re.search(pattern, generated_code, re.IGNORECASE):
                found_libraries.append(library)
            else:
                missing_libraries.append(library)
        
        # Check for unexpected patterns
        unexpected_patterns = []
        patterns = self.microcontroller_patterns.get(microcontroller_type, {})
        
        for category, keywords in patterns.items():
            for keyword in keywords:
                if keyword.lower() in code_lower and keyword not in expected_keywords:
                    unexpected_patterns.append(f"{category}: {keyword}")
        
        return {
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "found_libraries": found_libraries,
            "missing_libraries": missing_libraries,
            "unexpected_patterns": unexpected_patterns,
            "code_length": len(generated_code),
            "has_setup": "void setup()" in generated_code,
            "has_loop": "void loop()" in generated_code,
            "has_comments": "//" in generated_code or "/*" in generated_code,
        }


if __name__ == "__main__":
    # Test the scorer
    scorer = CodeScorer()
    
    test_code = """
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
    
    score = scorer.score_code(
        test_code,
        ["SPI.begin", "SPI.transfer", "Serial.begin"],
        ["SPI.h"],
        MicrocontrollerType.ARDUINO_UNO
    )
    
    print("Test Score:", score)
    
    analysis = scorer.get_detailed_analysis(
        test_code,
        ["SPI.begin", "SPI.transfer", "Serial.begin"],
        ["SPI.h"],
        MicrocontrollerType.ARDUINO_UNO
    )
    
    print("Analysis:", analysis)
