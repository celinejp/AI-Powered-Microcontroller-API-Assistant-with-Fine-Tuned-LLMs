# Microcontroller API Assistant Using Fine-Tuned LLMs

An AI-powered tool that generates accurate SDK-compliant API syntax and usage examples for microcontroller peripherals (UART, SPI, GPIO, I2C).

## 🎯 Project Overview

This project provides an intelligent assistant that helps developers generate correct API calls and usage examples for microcontroller peripherals. It uses fine-tuned language models to ensure accuracy and compliance with various SDK specifications.

### Features
- **Multi-Peripheral Support**: UART, SPI, GPIO, I2C
- **SDK Compliance**: Generates code that follows specific SDK guidelines
- **Real-time Inference**: Fast response times using vLLM optimization
- **Web Interface**: Simple React-based UI for easy interaction
- **Fine-tuned Models**: Custom models trained on microcontroller API datasets

## 🏗️ Architecture

```
Microcontroller-API-Assistant/
├── backend/           # FastAPI server with vLLM inference
├── frontend/          # React web interface
├── training/          # Model fine-tuning scripts
├── docs/             # Documentation and guides
└── docker/           # Containerization setup
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Poetry (for Python dependency management)
- Docker (optional, for containerized deployment)

### Backend Setup
```bash
cd backend
poetry install
poetry run python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Training Setup
```bash
cd training
poetry install
# Follow training/README.md for detailed instructions
```

## 📁 Project Structure

### Backend (`/backend`)
- **FastAPI Application**: REST API endpoints for code generation
- **vLLM Integration**: Optimized model inference server
- **Hugging Face Integration**: Model loading and management
- **Triton Optimization**: Kernel-level performance optimization

### Frontend (`/frontend`)
- **React Application**: Modern web interface
- **Peripheral Selection**: Dropdown-based UI for peripheral types
- **Code Display**: Syntax-highlighted code snippets
- **Real-time Generation**: Instant API code generation

### Training (`/training`)
- **Dataset Preparation**: Scripts for preparing training data
- **Fine-tuning Pipeline**: Hugging Face-based model training
- **Evaluation Tools**: Model performance assessment
- **Data Collection**: Utilities for gathering API examples

### Documentation (`/docs`)
- **API Documentation**: Backend API reference
- **Setup Guides**: Detailed installation instructions
- **Usage Examples**: How-to guides for different peripherals
- **Architecture Docs**: System design and implementation details

## 🔧 Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **vLLM**: High-performance LLM inference
- **Hugging Face Transformers**: Model loading and fine-tuning
- **Triton**: GPU kernel optimization
- **Poetry**: Dependency management

### Frontend
- **React**: Modern JavaScript framework
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Axios**: HTTP client for API calls

### Training
- **Hugging Face**: Model training and dataset management
- **PyTorch**: Deep learning framework
- **Datasets**: Data processing and management
- **Wandb**: Experiment tracking (optional)

## 📊 Supported Peripherals

- **UART**: Universal Asynchronous Receiver-Transmitter
- **SPI**: Serial Peripheral Interface
- **GPIO**: General Purpose Input/Output
- **I2C**: Inter-Integrated Circuit

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the transformer models and training infrastructure
- vLLM team for the high-performance inference engine
- Triton team for GPU optimization capabilities
- The open-source microcontroller community for API examples and documentation
