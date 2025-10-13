"""
Test Cases for Microcontroller API Assistant QA System
=====================================================

This module defines test cases for various microcontroller scenarios including:
- Basic functionality tests
- Edge cases and error handling
- Stress testing
- Regression testing
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class MicrocontrollerType(Enum):
    """Supported microcontroller types."""
    ARDUINO_UNO = "arduino_uno"
    ESP32 = "esp32"
    STM32 = "stm32"
    GENERIC = "generic"


@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    prompt: str
    microcontroller_type: MicrocontrollerType
    expected_keywords: List[str]
    expected_libraries: List[str]
    description: str
    category: str
    difficulty: str  # 'easy', 'medium', 'hard'


@dataclass
class TestSuite:
    """Represents a collection of test cases."""
    name: str
    description: str
    test_cases: List[TestCase]


def create_basic_test_suite() -> TestSuite:
    """Create basic functionality test cases."""
    return TestSuite(
        name="basic",
        description="Basic functionality tests for common microcontroller operations",
        test_cases=[
            # Arduino Uno Tests
            TestCase(
                name="Arduino_LED_Control",
                prompt="How to control an LED on Arduino Uno pin 13?",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["pinMode", "digitalWrite", "OUTPUT", "HIGH", "LOW"],
                expected_libraries=[],
                description="Basic LED control on Arduino Uno",
                category="gpio",
                difficulty="easy"
            ),
            TestCase(
                name="Arduino_Button_Read",
                prompt="How to read a push button on Arduino Uno pin 2?",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["pinMode", "digitalRead", "INPUT_PULLUP"],
                expected_libraries=[],
                description="Button reading with internal pull-up",
                category="gpio",
                difficulty="easy"
            ),
            TestCase(
                name="Arduino_UART_Communication",
                prompt="How to initialize UART on Arduino Uno at 9600 baud?",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["Serial.begin", "Serial.print", "Serial.println"],
                expected_libraries=[],
                description="UART communication setup",
                category="communication",
                difficulty="easy"
            ),
            TestCase(
                name="Arduino_SPI_Master",
                prompt="How to configure SPI as master on Arduino Uno?",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["SPI.begin", "SPI.transfer", "SPISettings"],
                expected_libraries=["SPI.h"],
                description="SPI master configuration",
                category="communication",
                difficulty="medium"
            ),
            TestCase(
                name="Arduino_I2C_Read",
                prompt="How to read from I2C device at address 0x50 on Arduino Uno?",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["Wire.begin", "Wire.requestFrom", "Wire.read"],
                expected_libraries=["Wire.h"],
                description="I2C device reading",
                category="communication",
                difficulty="medium"
            ),
            
            # ESP32 Tests
            TestCase(
                name="ESP32_LED_Control",
                prompt="How to control an LED on ESP32 pin 2?",
                microcontroller_type=MicrocontrollerType.ESP32,
                expected_keywords=["gpio_set_level", "GPIO_NUM_2", "gpio_config"],
                expected_libraries=["driver/gpio.h"],
                description="LED control using ESP32 GPIO API",
                category="gpio",
                difficulty="easy"
            ),
            TestCase(
                name="ESP32_WiFi_Connect",
                prompt="How to connect ESP32 to WiFi network?",
                microcontroller_type=MicrocontrollerType.ESP32,
                expected_keywords=["WiFi.begin", "WiFi.status", "WL_CONNECTED"],
                expected_libraries=["WiFi.h"],
                description="WiFi connection setup",
                category="communication",
                difficulty="medium"
            ),
            TestCase(
                name="ESP32_SPI_Configuration",
                prompt="How to configure SPI on ESP32?",
                microcontroller_type=MicrocontrollerType.ESP32,
                expected_keywords=["spi_bus_config_t", "spi_device_interface_config_t", "spi_bus_initialize"],
                expected_libraries=["driver/spi_master.h"],
                description="ESP32 SPI configuration",
                category="communication",
                difficulty="hard"
            ),
            
            # STM32 Tests
            TestCase(
                name="STM32_Servo_Control",
                prompt="How to control a servo motor on STM32?",
                microcontroller_type=MicrocontrollerType.STM32,
                expected_keywords=["Servo", "attach", "write"],
                expected_libraries=["Servo.h"],
                description="Servo motor control",
                category="actuators",
                difficulty="medium"
            ),
            TestCase(
                name="STM32_UART_Setup",
                prompt="How to set up UART communication on STM32?",
                microcontroller_type=MicrocontrollerType.STM32,
                expected_keywords=["Serial.begin", "Serial.print"],
                expected_libraries=[],
                description="STM32 UART setup",
                category="communication",
                difficulty="medium"
            ),
            
            # Generic Tests
            TestCase(
                name="Generic_GPIO_Control",
                prompt="How to control GPIO pins?",
                microcontroller_type=MicrocontrollerType.GENERIC,
                expected_keywords=["pinMode", "digitalWrite", "digitalRead"],
                expected_libraries=[],
                description="Generic GPIO control",
                category="gpio",
                difficulty="easy"
            ),
            TestCase(
                name="Generic_UART_Communication",
                prompt="Send data over UART",
                microcontroller_type=MicrocontrollerType.GENERIC,
                expected_keywords=["Serial.begin", "Serial.print", "Serial.println"],
                expected_libraries=[],
                description="Generic UART communication",
                category="communication",
                difficulty="easy"
            ),
        ]
    )


def create_edge_case_test_suite() -> TestSuite:
    """Create edge case and error handling test cases."""
    return TestSuite(
        name="edge",
        description="Edge cases and error handling tests",
        test_cases=[
            # Invalid pin numbers
            TestCase(
                name="Invalid_Pin_Number",
                prompt="Blink LED on ESP32 pin 100",
                microcontroller_type=MicrocontrollerType.ESP32,
                expected_keywords=[],  # Should handle gracefully
                expected_libraries=[],
                description="Test handling of invalid pin numbers",
                category="error_handling",
                difficulty="easy"
            ),
            TestCase(
                name="Invalid_I2C_Configuration",
                prompt="Use I2C on Arduino Uno with SDA=SCL",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=[],  # Should provide guidance, not code
                expected_libraries=[],
                description="Test handling of invalid I2C configuration",
                category="error_handling",
                difficulty="medium"
            ),
            TestCase(
                name="Multi_Peripheral_Setup",
                prompt="Initialize SPI and UART together on Arduino Uno",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["Serial.begin", "SPI.begin"],
                expected_libraries=["SPI.h"],
                description="Multiple peripheral initialization",
                category="complex",
                difficulty="hard"
            ),
            TestCase(
                name="Sensor_Display_Integration",
                prompt="Read DHT11 sensor and display on LCD with Arduino Uno",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["DHT", "lcd", "begin", "print"],
                expected_libraries=["DHT.h", "LiquidCrystal.h"],
                description="Sensor and display integration",
                category="complex",
                difficulty="hard"
            ),
            TestCase(
                name="DHT11_LCD_I2C_Integration",
                prompt="Read DHT11 and show on 16x2 LCD (I2C) on Uno",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["DHT", "LiquidCrystal_I2C", "Wire.begin"],
                expected_libraries=["DHT.h", "LiquidCrystal_I2C.h", "Wire.h"],
                description="DHT11 sensor with I2C LCD display",
                category="complex",
                difficulty="hard"
            ),
            # Remove duplicate test case
            
            # Remove duplicate test cases (already added above)
            
            # Ambiguous queries
            TestCase(
                name="Ambiguous_Query",
                prompt="How to make coffee with a microcontroller?",
                microcontroller_type=MicrocontrollerType.GENERIC,
                expected_keywords=[],  # Should fallback gracefully
                expected_libraries=[],
                description="Test handling of ambiguous queries",
                category="error_handling",
                difficulty="easy"
            ),
            TestCase(
                name="Vague_Request",
                prompt="Do something with pins",
                microcontroller_type=MicrocontrollerType.GENERIC,
                expected_keywords=["pinMode", "digitalWrite"],  # Should provide generic template
                expected_libraries=[],
                description="Test handling of vague requests",
                category="error_handling",
                difficulty="easy"
            ),
        ]
    )


def create_stress_test_suite() -> TestSuite:
    """Create stress testing scenarios."""
    return TestSuite(
        name="stress",
        description="Stress testing for complex scenarios and performance",
        test_cases=[
            # Complex multi-peripheral scenarios
            TestCase(
                name="Full_System_Integration",
                prompt="Create a complete IoT system with ESP32: WiFi connection, temperature sensor reading, LCD display, and servo control",
                microcontroller_type=MicrocontrollerType.ESP32,
                expected_keywords=["WiFi.begin", "DHT", "lcd", "Servo"],
                expected_libraries=["WiFi.h", "DHT.h", "LiquidCrystal.h", "Servo.h"],
                description="Complex IoT system integration",
                category="integration",
                difficulty="hard"
            ),
            TestCase(
                name="Multi_Communication_Protocols",
                prompt="Set up Arduino Uno with UART, SPI, and I2C communication simultaneously",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["Serial.begin", "SPI.begin", "Wire.begin"],
                expected_libraries=["SPI.h", "Wire.h"],
                description="Multiple communication protocols",
                category="integration",
                difficulty="hard"
            ),
            
            # Performance edge cases
            TestCase(
                name="High_Frequency_Operations",
                prompt="Generate code for high-frequency PWM output on STM32 at 1MHz",
                microcontroller_type=MicrocontrollerType.STM32,
                expected_keywords=["analogWrite", "PWM", "frequency"],
                expected_libraries=[],
                description="High-frequency PWM generation",
                category="performance",
                difficulty="hard"
            ),
            TestCase(
                name="Real_Time_Processing",
                prompt="Create real-time data processing with interrupts on ESP32",
                microcontroller_type=MicrocontrollerType.ESP32,
                expected_keywords=["attachInterrupt", "interrupt", "ISR"],
                expected_libraries=[],
                description="Real-time interrupt handling",
                category="performance",
                difficulty="hard"
            ),
            
            # Memory and resource intensive
            TestCase(
                name="Large_Data_Processing",
                prompt="Process large sensor data arrays with filtering and storage on Arduino Uno",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["array", "filter", "EEPROM"],
                expected_libraries=["EEPROM.h"],
                description="Large data processing and storage",
                category="performance",
                difficulty="hard"
            ),
        ]
    )


def create_regression_test_suite() -> TestSuite:
    """Create regression testing scenarios based on previous issues."""
    return TestSuite(
        name="regression",
        description="Regression tests for previously identified issues",
        test_cases=[
            # Previously problematic scenarios
            TestCase(
                name="Generic_UART_Fallback",
                prompt="Send data over UART at 9600 baud",
                microcontroller_type=MicrocontrollerType.GENERIC,
                expected_keywords=["Serial.begin", "Serial.print", "9600"],
                expected_libraries=[],
                description="Generic UART fallback template",
                category="regression",
                difficulty="easy"
            ),
            TestCase(
                name="Generic_SPI_Fallback",
                prompt="Initialize SPI as master",
                microcontroller_type=MicrocontrollerType.GENERIC,
                expected_keywords=["SPI.begin", "SPI.transfer"],
                expected_libraries=["SPI.h"],
                description="Generic SPI fallback template",
                category="regression",
                difficulty="easy"
            ),
            TestCase(
                name="Generic_GPIO_Fallback",
                prompt="Set pin 13 as output and write high",
                microcontroller_type=MicrocontrollerType.GENERIC,
                expected_keywords=["pinMode", "digitalWrite", "OUTPUT", "HIGH"],
                expected_libraries=[],
                description="Generic GPIO fallback template",
                category="regression",
                difficulty="easy"
            ),
            TestCase(
                name="Generic_I2C_Fallback",
                prompt="Read from device at address 0x50",
                microcontroller_type=MicrocontrollerType.GENERIC,
                expected_keywords=["Wire.begin", "Wire.requestFrom", "Wire.read"],
                expected_libraries=["Wire.h"],
                description="Generic I2C fallback template",
                category="regression",
                difficulty="easy"
            ),
            
            # Specific microcontroller edge cases
            TestCase(
                name="ESP32_GPIO_API",
                prompt="How to read a push button on ESP32 pin 2?",
                microcontroller_type=MicrocontrollerType.ESP32,
                expected_keywords=["gpio_get_level", "GPIO_NUM_2", "gpio_config"],
                expected_libraries=["driver/gpio.h"],
                description="ESP32 GPIO API usage",
                category="regression",
                difficulty="medium"
            ),
            TestCase(
                name="Arduino_Servo_Control",
                prompt="How to control a servo motor on Arduino Uno?",
                microcontroller_type=MicrocontrollerType.ARDUINO_UNO,
                expected_keywords=["Servo", "attach", "write"],
                expected_libraries=["Servo.h"],
                description="Arduino servo control",
                category="regression",
                difficulty="medium"
            ),
        ]
    )


def load_test_suites() -> Dict[str, TestSuite]:
    """Load all test suites."""
    return {
        "basic": create_basic_test_suite(),
        "edge": create_edge_case_test_suite(),
        "stress": create_stress_test_suite(),
        "regression": create_regression_test_suite(),
    }


if __name__ == "__main__":
    # Test the test cases
    suites = load_test_suites()
    for name, suite in suites.items():
        print(f"\nTest Suite: {name}")
        print(f"Description: {suite.description}")
        print(f"Number of test cases: {len(suite.test_cases)}")
        for test_case in suite.test_cases:
            print(f"  - {test_case.name}: {test_case.description}")
