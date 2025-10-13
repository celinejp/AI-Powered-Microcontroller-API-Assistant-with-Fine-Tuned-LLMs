"""
Pytest configuration for the Microcontroller API Assistant tests.
"""

import pytest
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Add the training directory to the Python path
training_dir = Path(__file__).parent.parent / "training"
sys.path.insert(0, str(training_dir))


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_queries():
    """Sample queries for testing."""
    return [
        "How to initialize SPI on Arduino Uno?",
        "How to configure UART on ESP32?",
        "How to read accelerometer data?",
        "How to control GPIO pins?",
        "How to set up I2C communication?",
    ]


@pytest.fixture(scope="session")
def expected_keywords():
    """Expected keywords for different query types."""
    return {
        "spi": ["spi", "mosi", "miso", "sck", "ss"],
        "uart": ["uart", "serial", "tx", "rx", "baud"],
        "accelerometer": ["wire", "i2c", "accelerometer", "mpu", "gyro"],
        "gpio": ["gpio", "pin", "digital", "output", "input"],
        "i2c": ["wire", "i2c", "sda", "scl", "address"],
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "ERROR"  # Reduce log noise during tests
    
    yield
    
    # Clean up
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
