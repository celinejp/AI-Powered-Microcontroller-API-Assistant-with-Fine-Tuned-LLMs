#!/usr/bin/env python3
"""
Dynamic SDK Ingestion Pipeline for Microcontroller API Assistant
"""

import os
import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    instruction: str
    output: str
    metadata: Dict[str, Any]
    source_url: str
    tags: List[str]
    title: str

class SDKDocParser:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def parse_arduino_docs(self) -> List[TrainingExample]:
        """Parse Arduino documentation dynamically."""
        logger.info("Parsing Arduino documentation...")
        examples = []
        
        # Dynamic Arduino API discovery
        arduino_apis = [
            "https://www.arduino.cc/reference/en/language/functions/",
            "https://www.arduino.cc/reference/en/language/functions/digital-io/",
            "https://www.arduino.cc/reference/en/language/functions/communication/",
        ]
        
        for api_url in arduino_apis:
            try:
                response = self.session.get(api_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract functions dynamically
                functions = soup.find_all(['h2', 'h3'], class_='function-name')
                for func in functions:
                    func_name = func.get_text().strip()
                    if func_name:
                        instruction = f"Show me how to use {func_name} in Arduino"
                        output = self._generate_arduino_example(func_name)
                        
                        examples.append(TrainingExample(
                            instruction=instruction,
                            output=output,
                            metadata={
                                "peripheral": self._classify_peripheral(func_name),
                                "microcontroller": "Arduino Uno",
                                "language": "cpp",
                                "function": func_name
                            },
                            source_url=api_url,
                            tags=["arduino", "tutorial"],
                            title=f"Arduino {func_name} Function"
                        ))
                        
            except Exception as e:
                logger.error(f"Error parsing Arduino docs from {api_url}: {e}")
        
        return examples
    
    def _generate_arduino_example(self, func_name: str) -> str:
        """Generate dynamic Arduino example based on function name."""
        if 'pinMode' in func_name:
            return """```cpp
const int ledPin = 13;
pinMode(ledPin, OUTPUT);
```"""
        elif 'digitalWrite' in func_name:
            return """```cpp
const int ledPin = 13;
digitalWrite(ledPin, HIGH);
```"""
        elif 'Serial' in func_name:
            return """```cpp
void setup() {
  Serial.begin(9600);
}
void loop() {
  Serial.println("Hello World!");
}
```"""
        else:
            return f"```cpp\n{func_name}();\n```"
    
    def _classify_peripheral(self, func_name: str) -> str:
        """Dynamically classify function by peripheral type."""
        func_lower = func_name.lower()
        
        if any(x in func_lower for x in ['pin', 'digital', 'analog', 'gpio']):
            return "GPIO"
        elif any(x in func_lower for x in ['serial', 'uart', 'usart']):
            return "UART"
        elif any(x in func_lower for x in ['spi', 'spi_']):
            return "SPI"
        elif any(x in func_lower for x in ['i2c', 'wire']):
            return "I2C"
        elif any(x in func_lower for x in ['wifi', 'wlan']):
            return "WiFi"
        else:
            return "GPIO"

class DynamicDatasetGenerator:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.parser = SDKDocParser()
    
    def generate_dataset(self) -> str:
        """Generate dynamic training dataset."""
        logger.info("Generating dynamic training dataset...")
        
        all_examples = []
        
        # Parse SDK documentation dynamically
        all_examples.extend(self.parser.parse_arduino_docs())
        
        # Generate additional examples dynamically
        all_examples.extend(self._generate_dynamic_examples())
        
        # Save to JSONL file
        output_file = self.output_dir / "micro_api_dataset.jsonl"
        self._save_to_jsonl(all_examples, output_file)
        
        logger.info(f"Generated {len(all_examples)} training examples")
        return str(output_file)
    
    def _generate_dynamic_examples(self) -> List[TrainingExample]:
        """Generate additional examples dynamically."""
        examples = []
        
        # Dynamic peripheral configurations
        peripherals = ["GPIO", "UART", "SPI", "I2C", "WiFi"]
        microcontrollers = ["Arduino Uno", "ESP32", "STM32F4", "Raspberry Pi Pico"]
        
        for peripheral in peripherals:
            for mcu in microcontrollers:
                instruction = f"Show me how to configure {peripheral} on {mcu}"
                output = self._generate_dynamic_example(peripheral, mcu)
                
                examples.append(TrainingExample(
                    instruction=instruction,
                    output=output,
                    metadata={
                        "peripheral": peripheral,
                        "microcontroller": mcu,
                        "language": "python" if "Pico" in mcu else "cpp",
                        "function": f"configure_{peripheral.lower()}"
                    },
                    source_url="dynamic_generation",
                    tags=[mcu.lower().replace(" ", "_"), peripheral.lower()],
                    title=f"{mcu} {peripheral} Configuration"
                ))
        
        return examples
    
    def _generate_dynamic_example(self, peripheral: str, mcu: str) -> str:
        """Generate dynamic example based on peripheral and microcontroller."""
        if mcu == "Arduino Uno":
            return self._generate_arduino_example(peripheral)
        elif mcu == "ESP32":
            return self._generate_esp32_example(peripheral)
        elif mcu == "STM32F4":
            return self._generate_stm32_example(peripheral)
        elif mcu == "Raspberry Pi Pico":
            return self._generate_pico_example(peripheral)
        else:
            return f"// {peripheral} configuration for {mcu}"
    
    def _generate_arduino_example(self, peripheral: str) -> str:
        if peripheral == "GPIO":
            return """```cpp
const int ledPin = 13;
pinMode(ledPin, OUTPUT);
digitalWrite(ledPin, HIGH);
```"""
        elif peripheral == "UART":
            return """```cpp
void setup() {
  Serial.begin(9600);
}
void loop() {
  Serial.println("Hello World!");
}
```"""
        elif peripheral == "I2C":
            return """```cpp
#include <Wire.h>
void setup() {
  Wire.begin();
}
```"""
        elif peripheral == "SPI":
            return """```cpp
#include <SPI.h>
void setup() {
  SPI.begin();
}
```"""
        else:
            return f"```cpp\n// {peripheral} configuration\n```"
    
    def _generate_esp32_example(self, peripheral: str) -> str:
        if peripheral == "GPIO":
            return """```cpp
#include "driver/gpio.h"
gpio_config_t io_conf = {};
io_conf.mode = GPIO_MODE_OUTPUT;
io_conf.pin_bit_mask = (1ULL << GPIO_NUM_2);
gpio_config(&io_conf);
```"""
        elif peripheral == "UART":
            return """```cpp
#include "driver/uart.h"
uart_config_t uart_config = {
    .baud_rate = 115200,
    .data_bits = UART_DATA_8_BITS,
    .parity = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
};
uart_param_config(UART_NUM_0, &uart_config);
```"""
        else:
            return f"```cpp\n// ESP32 {peripheral} configuration\n```"
    
    def _generate_stm32_example(self, peripheral: str) -> str:
        if peripheral == "GPIO":
            return """```cpp
#include "stm32f4xx_hal.h"
GPIO_InitTypeDef GPIO_InitStruct = {0};
__HAL_RCC_GPIOD_CLK_ENABLE();
GPIO_InitStruct.Pin = GPIO_PIN_13;
GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);
```"""
        elif peripheral == "UART":
            return """```cpp
#include "stm32f4xx_hal.h"
UART_HandleTypeDef huart2;
huart2.Instance = USART2;
huart2.Init.BaudRate = 115200;
huart2.Init.WordLength = UART_WORDLENGTH_8B;
HAL_UART_Init(&huart2);
```"""
        else:
            return f"```cpp\n// STM32 {peripheral} configuration\n```"
    
    def _generate_pico_example(self, peripheral: str) -> str:
        if peripheral == "GPIO":
            return """```python
from machine import Pin
led = Pin(25, Pin.OUT)
led.value(1)
```"""
        elif peripheral == "UART":
            return """```python
from machine import UART
uart = UART(0, baudrate=115200)
uart.write("Hello World!")
```"""
        elif peripheral == "I2C":
            return """```python
from machine import I2C, Pin
i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=100000)
```"""
        else:
            return f"```python\n# Pico {peripheral} configuration\n```"
    
    def _save_to_jsonl(self, examples: List[TrainingExample], output_file: Path):
        """Save examples to JSONL file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(asdict(example), f, ensure_ascii=False)
                f.write('\n')

def main():
    """Generate dynamic dataset."""
    generator = DynamicDatasetGenerator()
    output_file = generator.generate_dataset()
    print(f"Dataset generated: {output_file}")

if __name__ == "__main__":
    main()
