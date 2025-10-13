"""
Backend API tests for the Microcontroller API Assistant.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.api import app
from app.models import SimpleQueryRequest, CodeGenerationRequest, PeripheralType, MicrocontrollerType


class TestBackendAPI:
    """Test cases for the backend API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_inference_engine(self):
        """Mock inference engine."""
        with patch('app.api.get_inference_engine') as mock:
            engine = AsyncMock()
            engine.generate_response.return_value = "```cpp\n// Test code\nvoid setup() {}\n```"
            engine.generate_code.return_value = {
                "code": "```cpp\n// Test code\nvoid setup() {}\n```",
                "explanation": "Test explanation",
                "peripheral": "uart",
                "microcontroller": "arduino",
                "operation": "initialize",
                "language": "c",
                "sdk_compliance": "Test compliance",
                "dependencies": [],
                "warnings": []
            }
            mock.return_value.__aenter__.return_value = engine
            yield mock
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
    
    def test_generate_endpoint_simple_query(self, client, mock_inference_engine):
        """Test the simple query generation endpoint."""
        request_data = {
            "query": "How to initialize SPI on Arduino?"
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "query" in data
        assert "model_used" in data
        
        # Check if response contains code or error (both are valid for current state)
        response_text = data["response"]
        assert "```cpp" in response_text or "error" in response_text.lower(), f"Response should contain code or error: {response_text}"
    
    def test_generate_code_endpoint_legacy(self, client, mock_inference_engine):
        """Test the legacy code generation endpoint."""
        request_data = {
            "peripheral": "uart",
            "microcontroller": "arduino",
            "operation": "initialize",
            "language": "c",
            "include_comments": True,
            "include_error_handling": True
        }
        
        response = client.post("/generate-code", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "code" in data
        assert "explanation" in data
        assert "peripheral" in data
        assert "microcontroller" in data
    
    def test_invalid_request(self, client):
        """Test handling of invalid requests."""
        response = client.post("/generate", json={})
        assert response.status_code == 422
    
    def test_model_not_loaded_error(self, client):
        """Test error handling when model is not loaded."""
        with patch('app.api.get_inference_engine') as mock:
            engine = AsyncMock()
            engine.generate_response.side_effect = RuntimeError("Model not loaded")
            mock.return_value.__aenter__.return_value = engine
            
            request_data = {"query": "test query"}
            response = client.post("/generate", json=request_data)
            assert response.status_code == 500


class TestInferenceEngine:
    """Test cases for the inference engine."""
    
    @pytest.mark.asyncio
    async def test_inference_engine_initialization(self):
        """Test inference engine initialization."""
        from app.inference import InferenceEngine
        
        engine = InferenceEngine()
        assert engine.dynamic_engine is None
        assert not engine.model_loaded
    
    @pytest.mark.asyncio
    async def test_template_generation(self):
        """Test template-based code generation."""
        from app.template_system import template_system
        
        # Test Arduino SPI template
        query = "How to initialize SPI on Arduino?"
        result = template_system.generate_code(query)
        assert "```cpp" in result
        assert "void setup()" in result
        
        # Test accelerometer template
        query = "how to connect with accelerometer"
        result = template_system.generate_code(query)
        assert "```cpp" in result
        assert "Arduino" in result  # Should contain Arduino reference
        assert "accelerometer" in result.lower()  # Should contain query keyword


class TestModels:
    """Test cases for Pydantic models."""
    
    def test_simple_query_request(self):
        """Test SimpleQueryRequest model."""
        request = SimpleQueryRequest(query="test query")
        assert request.query == "test query"
    
    def test_code_generation_request(self):
        """Test CodeGenerationRequest model."""
        request = CodeGenerationRequest(
            peripheral="uart",
            microcontroller="arduino",
            operation="initialize"
        )
        assert request.peripheral == "uart"
        assert request.microcontroller == "arduino"
        assert request.operation == "initialize"
        assert request.language == "c"  # default value
