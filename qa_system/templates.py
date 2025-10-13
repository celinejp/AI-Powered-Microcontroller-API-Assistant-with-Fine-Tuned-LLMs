#!/usr/bin/env python3
"""
Test script for the new modular template system.
This script tests the template system to ensure it correctly generates code for different platforms.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.template_system import template_system, PlatformType

def test_template_system():
    """Test the template system with various queries."""
    
    test_cases = [
        # Arduino Uno tests
        ("How to blink an LED on Arduino Uno?", "arduino_uno_led_blink"),
        ("How to read a button on Arduino Uno?", "arduino_uno_button_read"),
        ("How to use UART on Arduino Uno?", "arduino_uno_uart_basic"),
        ("How to use I2C on Arduino Uno?", "arduino_uno_i2c_basic"),
        
        # ESP32 tests
        ("How to blink an LED on ESP32?", "esp32_led_blink"),
        ("How to read a button on ESP32?", "esp32_button_read"),
        ("How to use UART on ESP32?", "esp32_uart_basic"),
        ("How to use I2C on ESP32?", "esp32_i2c_basic"),
        ("How to connect to WiFi on ESP32?", "esp32_wifi_connect"),
        ("How to use Bluetooth on ESP32?", "esp32_bluetooth_basic"),
        
        # STM32 tests
        ("How to blink an LED on STM32?", "stm32_led_blink"),
        ("How to read a button on STM32?", "stm32_button_read"),
        ("How to use UART on STM32?", "stm32_uart_basic"),
        ("How to use I2C on STM32?", "stm32_i2c_basic"),
        
        # Raspberry Pi Pico tests
        ("How to blink an LED on Raspberry Pi Pico?", "raspberry_pi_pico_led_blink"),
        ("How to read a button on Raspberry Pi Pico?", "raspberry_pi_pico_button_read"),
        ("How to use UART on Raspberry Pi Pico?", "raspberry_pi_pico_uart_basic"),
        ("How to use I2C on Raspberry Pi Pico?", "raspberry_pi_pico_i2c_basic"),
    ]
    
    print("🧪 Testing Modular Template System")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for query, expected_template in test_cases:
        print(f"\n📝 Testing: {query}")
        
        # Get the template
        template = template_system.get_template(query)
        
        if template:
            template_key = f"{template.platform.value}_{template.peripheral}_{template.task}"
            if template_key == expected_template:
                print(f"✅ PASS: Found template '{template_key}'")
                print(f"   Platform: {template.platform.value}")
                print(f"   Peripheral: {template.peripheral}")
                print(f"   Task: {template.task}")
                print(f"   Libraries: {template.libraries}")
                print(f"   Keywords: {template.keywords[:3]}...")  # Show first 3 keywords
                passed += 1
            else:
                print(f"❌ FAIL: Expected '{expected_template}', got '{template_key}'")
                failed += 1
        else:
            print(f"❌ FAIL: No template found for query")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Template system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the template system.")
        return False

def test_platform_detection():
    """Test platform detection functionality."""
    
    print("\n🔍 Testing Platform Detection")
    print("=" * 30)
    
    test_queries = [
        ("How to blink LED on Arduino Uno?", PlatformType.ARDUINO_UNO),
        ("ESP32 button read", PlatformType.ESP32),
        ("STM32 UART communication", PlatformType.STM32),
        ("Raspberry Pi Pico I2C", PlatformType.RASPBERRY_PI_PICO),
        ("Pico LED blink", PlatformType.RASPBERRY_PI_PICO),
        ("Generic query without platform", PlatformType.ARDUINO_UNO),  # Should default to Arduino
    ]
    
    for query, expected_platform in test_queries:
        detected_platform = template_system.detect_platform(query)
        if detected_platform == expected_platform:
            print(f"✅ {query} -> {detected_platform.value}")
        else:
            print(f"❌ {query} -> Expected {expected_platform.value}, got {detected_platform.value}")

def test_peripheral_detection():
    """Test peripheral detection functionality."""
    
    print("\n🔧 Testing Peripheral Detection")
    print("=" * 30)
    
    test_queries = [
        ("LED blink", "led"),
        ("Button read", "button"),
        ("UART communication", "uart"),
        ("SPI device", "spi"),
        ("I2C sensor", "i2c"),
        ("WiFi connection", "wifi"),
        ("Bluetooth module", "bluetooth"),
        ("Generic query", "gpio"),  # Should default to gpio
    ]
    
    for query, expected_peripheral in test_queries:
        detected_peripheral = template_system.detect_peripheral(query)
        if detected_peripheral == expected_peripheral:
            print(f"✅ {query} -> {detected_peripheral}")
        else:
            print(f"❌ {query} -> Expected {expected_peripheral}, got {detected_peripheral}")

def test_task_detection():
    """Test task detection functionality."""
    
    print("\n📋 Testing Task Detection")
    print("=" * 30)
    
    test_queries = [
        ("Blink LED", "blink"),
        ("Read button", "read"),
        ("Write data", "write"),
        ("Initialize UART", "init"),
        ("Transmit data", "transmit"),
        ("Connect to WiFi", "connect"),
        ("Generic task", "basic"),  # Should default to basic
    ]
    
    for query, expected_task in test_queries:
        detected_task = template_system.detect_task(query)
        if detected_task == expected_task:
            print(f"✅ {query} -> {detected_task}")
        else:
            print(f"❌ {query} -> Expected {expected_task}, got {detected_task}")

def test_code_generation():
    """Test actual code generation."""
    
    print("\n💻 Testing Code Generation")
    print("=" * 30)
    
    test_queries = [
        "How to blink an LED on Arduino Uno?",
        "How to read a button on ESP32?",
        "How to use UART on STM32?",
        "How to use I2C on Raspberry Pi Pico?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Generating code for: {query}")
        code = template_system.generate_code(query)
        
        # Check if code was generated
        if code and "```cpp" in code:
            print(f"✅ Code generated successfully")
            print(f"   Length: {len(code)} characters")
            print(f"   Contains 'setup()': {'setup()' in code}")
            print(f"   Contains 'loop()': {'loop()' in code or 'while' in code}")
        else:
            print(f"❌ Failed to generate code")

if __name__ == "__main__":
    print("🚀 Starting Template System Tests")
    print("=" * 50)
    
    # Run all tests
    test_platform_detection()
    test_peripheral_detection()
    test_task_detection()
    test_code_generation()
    success = test_template_system()
    
    if success:
        print("\n🎉 All template system tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some template system tests failed!")
        sys.exit(1)
