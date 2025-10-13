#!/usr/bin/env python3
"""
Test script for the Microcontroller API Assistant.
This script provides automated testing for both Simple Query and Dynamic Code Generation endpoints.
"""

import requests
import json
import time
from typing import Dict, Any, List
import sys

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"

class APITester:
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
    
    def test_health_check(self):
        """Test the health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Health Check", True, f"Status: {data.get('status')}")
                return True
            else:
                self.log_test("Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Error: {str(e)}")
            return False
    
    def test_model_info(self):
        """Test the model info endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Model Info", True, f"Model: {data.get('model_name')}")
                return True
            else:
                self.log_test("Model Info", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Model Info", False, f"Error: {str(e)}")
            return False
    
    def test_peripherals(self):
        """Test the peripherals endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/peripherals")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Get Peripherals", True, f"Found {len(data)} peripherals: {', '.join(data)}")
                return data
            else:
                self.log_test("Get Peripherals", False, f"Status code: {response.status_code}")
                return []
        except Exception as e:
            self.log_test("Get Peripherals", False, f"Error: {str(e)}")
            return []
    
    def test_microcontrollers(self):
        """Test the microcontrollers endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/microcontrollers")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Get Microcontrollers", True, f"Found {len(data)} microcontrollers: {', '.join(data)}")
                return data
            else:
                self.log_test("Get Microcontrollers", False, f"Status code: {response.status_code}")
                return []
        except Exception as e:
            self.log_test("Get Microcontrollers", False, f"Error: {str(e)}")
            return []
    
    def test_languages(self):
        """Test the languages endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/languages")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Get Languages", True, f"Found {len(data)} languages: {', '.join(data)}")
                return data
            else:
                self.log_test("Get Languages", False, f"Status code: {response.status_code}")
                return []
        except Exception as e:
            self.log_test("Get Languages", False, f"Error: {str(e)}")
            return []
    
    def test_performance_metrics(self):
        """Test the performance metrics endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/performance")
            if response.status_code == 200:
                data = response.json()
                metrics = data.get('performance_metrics', {})
                self.log_test("Performance Metrics", True, 
                    f"Requests: {metrics.get('total_requests', 0)}, "
                    f"Avg Latency: {metrics.get('avg_latency', 0):.3f}s")
                return True
            else:
                self.log_test("Performance Metrics", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Performance Metrics", False, f"Error: {str(e)}")
            return False
    
    def test_operations(self, peripheral: str):
        """Test the operations endpoint for a specific peripheral."""
        try:
            response = self.session.get(f"{self.base_url}/operations/{peripheral}")
            if response.status_code == 200:
                data = response.json()
                operations = data.get('operations', [])
                self.log_test(f"Get Operations for {peripheral.upper()}", True, 
                    f"Found {len(operations)} operations: {', '.join(operations)}")
                return data
            else:
                self.log_test(f"Get Operations for {peripheral.upper()}", False, 
                    f"Status code: {response.status_code}")
                return None
        except Exception as e:
            self.log_test(f"Get Operations for {peripheral.upper()}", False, f"Error: {str(e)}")
            return None
    
    def test_simple_query(self, query: str):
        """Test the simple query endpoint."""
        try:
            payload = {"query": query}
            response = self.session.post(f"{self.base_url}/generate", json=payload)
            if response.status_code == 200:
                data = response.json()
                self.log_test(f"Simple Query: '{query[:30]}...'", True, 
                    f"Response length: {len(data.get('response', ''))}")
                return data
            else:
                self.log_test(f"Simple Query: '{query[:30]}...'", False, 
                    f"Status code: {response.status_code}")
                return None
        except Exception as e:
            self.log_test(f"Simple Query: '{query[:30]}...'", False, f"Error: {str(e)}")
            return None
    
    def test_code_generation(self, request_data: Dict[str, Any]):
        """Test the code generation endpoint."""
        try:
            response = self.session.post(f"{self.base_url}/generate-code", json=request_data)
            if response.status_code == 200:
                data = response.json()
                self.log_test(f"Code Generation: {request_data['peripheral']} {request_data['operation']}", True,
                    f"Code length: {len(data.get('code', ''))}")
                return data
            else:
                error_data = response.json() if response.content else {}
                self.log_test(f"Code Generation: {request_data['peripheral']} {request_data['operation']}", False,
                    f"Status code: {response.status_code}, Error: {error_data.get('detail', 'Unknown error')}")
                return None
        except Exception as e:
            self.log_test(f"Code Generation: {request_data['peripheral']} {request_data['operation']}", False, 
                f"Error: {str(e)}")
            return None
    
    def run_all_tests(self):
        """Run all available tests."""
        print("🚀 Starting Microcontroller API Assistant Tests")
        print("=" * 60)
        
        # Basic endpoint tests
        self.test_health_check()
        self.test_model_info()
        peripherals = self.test_peripherals()
        microcontrollers = self.test_microcontrollers()
        languages = self.test_languages()
        self.test_performance_metrics()
        
        # Test operations for each peripheral
        for peripheral in peripherals:
            self.test_operations(peripheral)
        
        # Simple query tests
        simple_queries = [
            "How to configure UART on Arduino Uno?",
            "How to initialize SPI on ESP32?",
            "How to read GPIO pin on STM32?",
            "How to configure I2C on Raspberry Pi Pico?",
        ]
        
        print("\n📝 Testing Simple Queries")
        print("-" * 30)
        for query in simple_queries:
            self.test_simple_query(query)
            time.sleep(0.5)  # Small delay between requests
        
        # Code generation tests
        print("\n🔧 Testing Code Generation")
        print("-" * 30)
        
        test_cases = [
            {
                "peripheral": "uart",
                "microcontroller": "arduino",
                "operation": "initialize",
                "parameters": {"uart_instance": "UART0"},
                "language": "c",
                "include_comments": True,
                "include_error_handling": True
            },
            {
                "peripheral": "gpio",
                "microcontroller": "esp32",
                "operation": "configure_pin",
                "parameters": {"pin_number": "13", "direction": "output"},
                "language": "c",
                "include_comments": True,
                "include_error_handling": True
            },
            {
                "peripheral": "spi",
                "microcontroller": "stm32",
                "operation": "send_data",
                "parameters": {"spi_instance": "SPI1", "data": "0xAA"},
                "language": "c",
                "include_comments": True,
                "include_error_handling": True
            },
            {
                "peripheral": "i2c",
                "microcontroller": "raspberry_pi_pico",
                "operation": "write_data",
                "parameters": {"i2c_instance": "I2C0", "device_address": "0x48", "data": "0x01"},
                "language": "c",
                "include_comments": True,
                "include_error_handling": True
            }
        ]
        
        for test_case in test_cases:
            self.test_code_generation(test_case)
            time.sleep(0.5)  # Small delay between requests
        
        # Summary
        print("\n📊 Test Summary")
        print("=" * 60)
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 All tests passed!")
            return True
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check the details above.")
            return False

def main():
    """Main function to run the tests."""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = API_BASE
    
    print(f"Testing API at: {base_url}")
    
    tester = APITester(base_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
