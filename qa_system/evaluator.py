#!/usr/bin/env python3
"""
Project Evaluator - Microcontroller API Assistant
================================================

Systematic evaluation of the embedded systems code generation project
by auto-generating diverse test cases across microcontrollers, peripherals, and tasks.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
import random

from runner import QATestRunner
from cases import TestCase, MicrocontrollerType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestScenario:
    """Represents a test scenario with microcontroller, peripheral, and task."""
    mcu: str
    peripheral: str
    task: str
    prompt: str
    expected_keywords: List[str]
    expected_libraries: List[str]
    difficulty: str

class ProjectEvaluation:
    """Comprehensive project evaluator for embedded systems code generation."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.runner = QATestRunner(api_url, "evaluation_results")
        
        # Define test spaces
        self.microcontrollers = [
            "Arduino Uno", "ESP32", "Raspberry Pi Pico", "STM32"
        ]
        
        self.peripherals = [
            "LED", "Button", "I2C Sensor", "SPI Device", "UART", "WiFi", "Bluetooth"
        ]
        
        self.tasks = [
            "Blink", "Read input", "Transmit data", "Initialize connection", 
            "Multi-peripheral setup", "Sensor reading", "Data logging"
        ]
        
        # MCU-specific configurations
        self.mcu_configs = {
            "Arduino Uno": {
                "api_style": "arduino",
                "keywords": ["pinMode", "digitalWrite", "digitalRead", "Serial", "Wire", "SPI"],
                "libraries": ["Arduino.h", "Wire.h", "SPI.h", "Servo.h"]
            },
            "ESP32": {
                "api_style": "esp32",
                "keywords": ["gpio_set_level", "gpio_get_level", "gpio_config", "WiFi", "Bluetooth"],
                "libraries": ["driver/gpio.h", "driver/spi_master.h", "driver/i2c.h", "WiFi.h"]
            },
            "Raspberry Pi Pico": {
                "api_style": "pico",
                "keywords": ["gpio_put", "gpio_get", "gpio_init", "i2c_write_blocking", "spi_write_blocking"],
                "libraries": ["hardware/gpio.h", "hardware/i2c.h", "hardware/spi.h", "pico/stdlib.h"]
            },
            "STM32": {
                "api_style": "stm32",
                "keywords": ["HAL_GPIO_WritePin", "HAL_GPIO_ReadPin", "HAL_UART_Transmit", "HAL_SPI_Transmit"],
                "libraries": ["stm32f4xx_hal.h", "stm32f1xx_hal.h"]
            }
        }
        
        # Peripheral-specific expectations
        self.peripheral_configs = {
            "LED": {
                "keywords": ["pinMode", "digitalWrite", "gpio_set_level", "HAL_GPIO_WritePin"],
                "libraries": ["Arduino.h", "driver/gpio.h", "hardware/gpio.h", "stm32f4xx_hal.h"]
            },
            "Button": {
                "keywords": ["pinMode", "digitalRead", "gpio_get_level", "HAL_GPIO_ReadPin"],
                "libraries": ["Arduino.h", "driver/gpio.h", "hardware/gpio.h", "stm32f4xx_hal.h"]
            },
            "I2C Sensor": {
                "keywords": ["Wire.begin", "Wire.beginTransmission", "Wire.requestFrom", "i2c_write_blocking"],
                "libraries": ["Wire.h", "driver/i2c.h", "hardware/i2c.h", "stm32f4xx_hal.h"]
            },
            "SPI Device": {
                "keywords": ["SPI.begin", "SPI.transfer", "spi_write_blocking", "HAL_SPI_Transmit"],
                "libraries": ["SPI.h", "driver/spi_master.h", "hardware/spi.h", "stm32f4xx_hal.h"]
            },
            "UART": {
                "keywords": ["Serial.begin", "Serial.print", "uart_write_blocking", "HAL_UART_Transmit"],
                "libraries": ["Arduino.h", "driver/uart.h", "hardware/uart.h", "stm32f4xx_hal.h"]
            },
            "WiFi": {
                "keywords": ["WiFi.begin", "WiFi.connect", "wifi_init", "HAL_WiFi_Connect"],
                "libraries": ["WiFi.h", "esp_wifi.h", "pico/wifi.h", "stm32f4xx_hal_wifi.h"]
            },
            "Bluetooth": {
                "keywords": ["Bluetooth.begin", "Bluetooth.available", "bt_init", "HAL_BT_Init"],
                "libraries": ["Bluetooth.h", "esp_bt.h", "pico/bt.h", "stm32f4xx_hal_bt.h"]
            }
        }
    
    def generate_test_scenarios(self) -> List[TestScenario]:
        """Auto-generate diverse test scenarios covering the input space."""
        scenarios = []
        
        # Generate basic scenarios (MCU + Peripheral + Task)
        for mcu in self.microcontrollers:
            for peripheral in self.peripherals:
                for task in self.tasks:
                    # Skip incompatible combinations
                    if self._is_compatible(mcu, peripheral, task):
                        scenario = self._create_scenario(mcu, peripheral, task)
                        scenarios.append(scenario)
        
        # Add some edge cases and complex scenarios
        scenarios.extend(self._generate_edge_cases())
        scenarios.extend(self._generate_complex_scenarios())
        
        # Shuffle to avoid bias
        random.shuffle(scenarios)
        
        logger.info(f"Generated {len(scenarios)} test scenarios")
        return scenarios
    
    def _is_compatible(self, mcu: str, peripheral: str, task: str) -> bool:
        """Check if MCU + Peripheral + Task combination is reasonable."""
        # WiFi/Bluetooth not available on Arduino Uno
        if mcu == "Arduino Uno" and peripheral in ["WiFi", "Bluetooth"]:
            return False
        
        # Some tasks don't make sense for certain peripherals
        if peripheral == "LED" and task in ["Read input", "Sensor reading"]:
            return False
        
        if peripheral == "Button" and task in ["Transmit data", "Initialize connection"]:
            return False
        
        return True
    
    def _create_scenario(self, mcu: str, peripheral: str, task: str) -> TestScenario:
        """Create a test scenario with appropriate prompt and expectations."""
        mcu_config = self.mcu_configs[mcu]
        peripheral_config = self.peripheral_configs[peripheral]
        
        # Generate appropriate prompt
        prompt = self._generate_prompt(mcu, peripheral, task)
        
        # Combine expected keywords and libraries
        expected_keywords = list(set(mcu_config["keywords"] + peripheral_config["keywords"]))
        expected_libraries = list(set(mcu_config["libraries"] + peripheral_config["libraries"]))
        
        # Determine difficulty
        difficulty = self._assess_difficulty(mcu, peripheral, task)
        
        return TestScenario(
            mcu=mcu,
            peripheral=peripheral,
            task=task,
            prompt=prompt,
            expected_keywords=expected_keywords,
            expected_libraries=expected_libraries,
            difficulty=difficulty
        )
    
    def _generate_prompt(self, mcu: str, peripheral: str, task: str) -> str:
        """Generate a natural language prompt for the scenario."""
        prompts = {
            "Blink": f"How to blink an LED on {mcu}?",
            "Read input": f"How to read input from a {peripheral.lower()} on {mcu}?",
            "Transmit data": f"How to transmit data using {peripheral.lower()} on {mcu}?",
            "Initialize connection": f"How to initialize {peripheral.lower()} connection on {mcu}?",
            "Multi-peripheral setup": f"How to set up multiple peripherals ({peripheral.lower()}) on {mcu}?",
            "Sensor reading": f"How to read from {peripheral.lower()} sensor on {mcu}?",
            "Data logging": f"How to log data using {peripheral.lower()} on {mcu}?"
        }
        
        return prompts.get(task, f"How to use {peripheral.lower()} on {mcu} for {task.lower()}?")
    
    def _assess_difficulty(self, mcu: str, peripheral: str, task: str) -> str:
        """Assess the difficulty of a test scenario."""
        if task in ["Multi-peripheral setup", "Data logging"]:
            return "hard"
        elif peripheral in ["WiFi", "Bluetooth", "I2C Sensor"]:
            return "medium"
        else:
            return "easy"
    
    def _generate_edge_cases(self) -> List[TestScenario]:
        """Generate edge cases and error scenarios."""
        edge_cases = [
            TestScenario(
                mcu="ESP32",
                peripheral="LED",
                task="Blink",
                prompt="Blink LED on ESP32 pin 100 (invalid pin)",
                expected_keywords=[],
                expected_libraries=[],
                difficulty="easy"
            ),
            TestScenario(
                mcu="Arduino Uno",
                peripheral="I2C Sensor",
                task="Initialize connection",
                prompt="Use I2C on Arduino Uno with SDA=SCL (invalid config)",
                expected_keywords=[],
                expected_libraries=[],
                difficulty="medium"
            ),
            TestScenario(
                mcu="STM32",
                peripheral="UART",
                task="Multi-peripheral setup",
                prompt="Initialize UART and SPI together on STM32",
                expected_keywords=["HAL_UART_Init", "HAL_SPI_Init"],
                expected_libraries=["stm32f4xx_hal.h"],
                difficulty="hard"
            )
        ]
        return edge_cases
    
    def _generate_complex_scenarios(self) -> List[TestScenario]:
        """Generate complex, real-world scenarios."""
        complex_scenarios = [
            TestScenario(
                mcu="ESP32",
                peripheral="WiFi",
                task="Data logging",
                prompt="Create a complete IoT system with ESP32: WiFi connection, sensor reading, and data transmission",
                expected_keywords=["WiFi.begin", "WiFi.connect", "HTTPClient", "JSON"],
                expected_libraries=["WiFi.h", "HTTPClient.h", "ArduinoJson.h"],
                difficulty="hard"
            ),
            TestScenario(
                mcu="Raspberry Pi Pico",
                peripheral="I2C Sensor",
                task="Sensor reading",
                prompt="Read temperature from I2C sensor and display on LCD with Raspberry Pi Pico",
                expected_keywords=["i2c_write_blocking", "i2c_read_blocking", "gpio_put"],
                expected_libraries=["hardware/i2c.h", "hardware/gpio.h"],
                difficulty="hard"
            )
        ]
        return complex_scenarios
    
    async def run_evaluation(self) -> Dict:
        """Run the complete project evaluation."""
        logger.info("Starting comprehensive project evaluation...")
        
        # Generate test scenarios
        scenarios = self.generate_test_scenarios()
        
        # Convert to test cases
        test_cases = []
        for i, scenario in enumerate(scenarios):
            # Map MCU names to MicrocontrollerType enum
            mcu_type = self._map_mcu_to_enum(scenario.mcu)
            
            test_case = TestCase(
                name=f"Eval_{i:03d}_{scenario.mcu.replace(' ', '_')}_{scenario.peripheral}_{scenario.task.replace(' ', '_')}",
                prompt=scenario.prompt,
                microcontroller_type=mcu_type,
                expected_keywords=scenario.expected_keywords,
                expected_libraries=scenario.expected_libraries,
                description=f"{scenario.mcu} + {scenario.peripheral} + {scenario.task}",
                category="evaluation",
                difficulty=scenario.difficulty
            )
            test_cases.append(test_case)
        
        # Run all test cases
        results = []
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for test_case in test_cases:
                result = await self.runner.run_test_case(session, test_case)
                results.append(result)
        
        # Analyze results
        analysis = self._analyze_results(results, scenarios)
        
        return analysis
    
    def _map_mcu_to_enum(self, mcu: str) -> MicrocontrollerType:
        """Map MCU name to MicrocontrollerType enum."""
        mapping = {
            "Arduino Uno": MicrocontrollerType.ARDUINO_UNO,
            "ESP32": MicrocontrollerType.ESP32,
            "STM32": MicrocontrollerType.STM32,
            "Raspberry Pi Pico": MicrocontrollerType.ARDUINO_UNO  # Use Arduino as fallback
        }
        return mapping.get(mcu, MicrocontrollerType.ARDUINO_UNO)
    
    def _analyze_results(self, results: List, scenarios: List[TestScenario]) -> Dict:
        """Analyze test results and generate comprehensive report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == "pass")
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Calculate average score
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        
        # Analyze by MCU
        mcu_performance = self._analyze_by_mcu(results, scenarios)
        
        # Analyze by peripheral
        peripheral_performance = self._analyze_by_peripheral(results, scenarios)
        
        # Analyze by task
        task_performance = self._analyze_by_task(results, scenarios)
        
        # Find common failure patterns
        failure_patterns = self._find_failure_patterns(results, scenarios)
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(results, scenarios)
        weaknesses = self._identify_weaknesses(results, scenarios)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "average_score": avg_score,
            "mcu_performance": mcu_performance,
            "peripheral_performance": peripheral_performance,
            "task_performance": task_performance,
            "failure_patterns": failure_patterns,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "results": results,
            "scenarios": scenarios
        }
    
    def _analyze_by_mcu(self, results: List, scenarios: List[TestScenario]) -> Dict:
        """Analyze performance by microcontroller."""
        mcu_stats = {}
        
        for mcu in self.microcontrollers:
            mcu_results = [r for r, s in zip(results, scenarios) if s.mcu == mcu]
            if mcu_results:
                passed = sum(1 for r in mcu_results if r.status == "pass")
                avg_score = sum(r.score for r in mcu_results) / len(mcu_results)
                mcu_stats[mcu] = {
                    "total": len(mcu_results),
                    "passed": passed,
                    "success_rate": passed / len(mcu_results),
                    "avg_score": avg_score
                }
        
        return mcu_stats
    
    def _analyze_by_peripheral(self, results: List, scenarios: List[TestScenario]) -> Dict:
        """Analyze performance by peripheral."""
        peripheral_stats = {}
        
        for peripheral in self.peripherals:
            peripheral_results = [r for r, s in zip(results, scenarios) if s.peripheral == peripheral]
            if peripheral_results:
                passed = sum(1 for r in peripheral_results if r.status == "pass")
                avg_score = sum(r.score for r in peripheral_results) / len(peripheral_results)
                peripheral_stats[peripheral] = {
                    "total": len(peripheral_results),
                    "passed": passed,
                    "success_rate": passed / len(peripheral_results),
                    "avg_score": avg_score
                }
        
        return peripheral_stats
    
    def _analyze_by_task(self, results: List, scenarios: List[TestScenario]) -> Dict:
        """Analyze performance by task."""
        task_stats = {}
        
        for task in self.tasks:
            task_results = [r for r, s in zip(results, scenarios) if s.task == task]
            if task_results:
                passed = sum(1 for r in task_results if r.status == "pass")
                avg_score = sum(r.score for r in task_results) / len(task_results)
                task_stats[task] = {
                    "total": len(task_results),
                    "passed": passed,
                    "success_rate": passed / len(task_results),
                    "avg_score": avg_score
                }
        
        return task_stats
    
    def _find_failure_patterns(self, results: List, scenarios: List[TestScenario]) -> List[str]:
        """Find common patterns in failed tests."""
        failed_results = [(r, s) for r, s in zip(results, scenarios) if r.status != "pass"]
        
        patterns = []
        
        # Check for MCU-specific failures
        for mcu in self.microcontrollers:
            mcu_failures = [r for r, s in failed_results if s.mcu == mcu]
            if len(mcu_failures) > 3:  # More than 3 failures for this MCU
                patterns.append(f"Multiple failures with {mcu}")
        
        # Check for peripheral-specific failures
        for peripheral in self.peripherals:
            peripheral_failures = [r for r, s in failed_results if s.peripheral == peripheral]
            if len(peripheral_failures) > 3:
                patterns.append(f"Multiple failures with {peripheral}")
        
        # Check for low scores
        low_score_failures = [r for r, s in failed_results if r.score < 1.0]
        if len(low_score_failures) > 5:
            patterns.append("Many tests with low scores (< 1.0)")
        
        return patterns
    
    def _identify_strengths(self, results: List, scenarios: List[TestScenario]) -> List[str]:
        """Identify system strengths."""
        strengths = []
        
        # Find high-performing combinations
        high_performers = [(r, s) for r, s in zip(results, scenarios) if r.score >= 1.8]
        
        # Group by MCU
        for mcu in self.microcontrollers:
            mcu_high = [r for r, s in high_performers if s.mcu == mcu]
            if len(mcu_high) >= 3:
                strengths.append(f"Excellent performance with {mcu}")
        
        # Group by peripheral
        for peripheral in self.peripherals:
            peripheral_high = [r for r, s in high_performers if s.peripheral == peripheral]
            if len(peripheral_high) >= 3:
                strengths.append(f"Strong {peripheral} support")
        
        return strengths
    
    def _identify_weaknesses(self, results: List, scenarios: List[TestScenario]) -> List[str]:
        """Identify system weaknesses."""
        weaknesses = []
        
        # Find low-performing combinations
        low_performers = [(r, s) for r, s in zip(results, scenarios) if r.score < 1.0]
        
        # Group by MCU
        for mcu in self.microcontrollers:
            mcu_low = [r for r, s in low_performers if s.mcu == mcu]
            if len(mcu_low) >= 3:
                weaknesses.append(f"Poor performance with {mcu}")
        
        # Group by peripheral
        for peripheral in self.peripherals:
            peripheral_low = [r for r, s in low_performers if s.peripheral == peripheral]
            if len(peripheral_low) >= 3:
                weaknesses.append(f"Weak {peripheral} support")
        
        return weaknesses
    
    def generate_report(self, analysis: Dict) -> str:
        """Generate a comprehensive evaluation report."""
        report = f"""
# Project Evaluation Report - Microcontroller API Assistant
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Overall Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Total Tests** | {analysis['total_tests']} | - |
| **Success Rate** | {analysis['success_rate']:.1%} | {'✅' if analysis['success_rate'] >= 0.8 else '❌'} |
| **Average Score** | {analysis['average_score']:.2f}/2.0 | {'✅' if analysis['average_score'] >= 1.5 else '❌'} |

## 🎯 Performance by Microcontroller

| MCU | Tests | Success Rate | Avg Score | Status |
|-----|-------|--------------|-----------|--------|
"""
        
        for mcu, stats in analysis['mcu_performance'].items():
            status = "✅" if stats['success_rate'] >= 0.8 and stats['avg_score'] >= 1.5 else "❌"
            report += f"| {mcu} | {stats['total']} | {stats['success_rate']:.1%} | {stats['avg_score']:.2f} | {status} |\n"
        
        report += f"""
## 🔧 Performance by Peripheral

| Peripheral | Tests | Success Rate | Avg Score | Status |
|------------|-------|--------------|-----------|--------|
"""
        
        for peripheral, stats in analysis['peripheral_performance'].items():
            status = "✅" if stats['success_rate'] >= 0.8 and stats['avg_score'] >= 1.5 else "❌"
            report += f"| {peripheral} | {stats['total']} | {stats['success_rate']:.1%} | {stats['avg_score']:.2f} | {status} |\n"
        
        report += f"""
## 📋 Performance by Task

| Task | Tests | Success Rate | Avg Score | Status |
|------|-------|--------------|-----------|--------|
"""
        
        for task, stats in analysis['task_performance'].items():
            status = "✅" if stats['success_rate'] >= 0.8 and stats['avg_score'] >= 1.5 else "❌"
            report += f"| {task} | {stats['total']} | {stats['success_rate']:.1%} | {stats['avg_score']:.2f} | {status} |\n"
        
        report += f"""
## 💪 Strengths

"""
        
        if analysis['strengths']:
            for strength in analysis['strengths']:
                report += f"- {strength}\n"
        else:
            report += "- No significant strengths identified\n"
        
        report += f"""
## ⚠️ Weaknesses

"""
        
        if analysis['weaknesses']:
            for weakness in analysis['weaknesses']:
                report += f"- {weakness}\n"
        else:
            report += "- No significant weaknesses identified\n"
        
        report += f"""
## 🔍 Failure Patterns

"""
        
        if analysis['failure_patterns']:
            for pattern in analysis['failure_patterns']:
                report += f"- {pattern}\n"
        else:
            report += "- No clear failure patterns identified\n"
        
        report += f"""
## 🎯 Project Potential Assessment

### Generalization Capability
"""
        
        if analysis['success_rate'] >= 0.8:
            report += "**Excellent** - System generalizes well across different microcontrollers and peripherals.\n"
        elif analysis['success_rate'] >= 0.6:
            report += "**Good** - System shows decent generalization with some limitations.\n"
        else:
            report += "**Limited** - System struggles to generalize across different platforms.\n"
        
        report += f"""
### Production Readiness
"""
        
        if analysis['success_rate'] >= 0.9 and analysis['average_score'] >= 1.8:
            report += "**Production-Ready** - High success rate and quality scores indicate readiness for production use.\n"
        elif analysis['success_rate'] >= 0.7 and analysis['average_score'] >= 1.5:
            report += "**Research-Grade** - Good performance suitable for research and development.\n"
        else:
            report += "**Proof-of-Concept** - Basic functionality demonstrated, needs significant improvement.\n"
        
        report += f"""
### Recommended Improvements

1. **Expand Library Support**: Add more comprehensive library coverage for underperforming peripherals
2. **Error Handling**: Implement better error handling and edge case management
3. **Platform Optimization**: Focus on improving performance for weaker microcontrollers
4. **Documentation**: Enhance code generation with better comments and documentation
5. **Testing**: Add more edge cases and complex scenarios to improve robustness

---
*Evaluation completed with {analysis['total_tests']} test scenarios*
"""
        
        return report

async def main():
    """Run the complete project evaluation."""
    evaluator = ProjectEvaluation()
    
    try:
        # Run evaluation
        analysis = await evaluator.run_evaluation()
        
        # Generate report
        report = evaluator.generate_report(analysis)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path("evaluation_results") / f"project_evaluation_{timestamp}.md"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 PROJECT EVALUATION COMPLETE")
        print("="*60)
        print(f"📊 Total Tests: {analysis['total_tests']}")
        print(f"✅ Success Rate: {analysis['success_rate']:.1%}")
        print(f"📈 Average Score: {analysis['average_score']:.2f}/2.0")
        print(f"📄 Report saved to: {report_file}")
        print("="*60)
        
        # Print key findings
        print("\n🔍 KEY FINDINGS:")
        if analysis['strengths']:
            print("💪 Strengths:")
            for strength in analysis['strengths'][:3]:  # Top 3
                print(f"   - {strength}")
        
        if analysis['weaknesses']:
            print("⚠️ Weaknesses:")
            for weakness in analysis['weaknesses'][:3]:  # Top 3
                print(f"   - {weakness}")
        
        print(f"\n📋 Full report available at: {report_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
