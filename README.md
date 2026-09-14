# Microcontroller API Assistant

An AI-assisted tool that generates example API code for microcontroller peripherals (UART, SPI, GPIO, I2C), built around a LoRA-fine-tuned language model served through a FastAPI backend and a React web interface.

## Status (as of 2026-09-13)

This project has a working, verified fine-tuning pipeline and a real backend/frontend serving loop — but is **not production-ready**. Specifics:

- ✅ **Fine-tuning is real.** A LoRA fine-tune of `microsoft/DialoGPT-medium` was run end-to-end on the bundled 224-example dataset (3 epochs, 33 steps). Eval loss fell monotonically (9.56 → 9.09 → 7.89) — genuine, measured learning.
- ✅ **The API genuinely serves the fine-tuned model.** `POST /generate-code` calls real inference first and only falls back to a template response on failure or empty output.
- ✅ **Inference latency optimizations are real and measured on CPU/Apple Silicon (MPS):** KV-caching + half precision cut generation latency by 66–79% versus a naive no-cache baseline (see [Verified results](#verified-results)).
- ⚠️ **Generated code is not yet coherent** at the current training budget. `DialoGPT-medium` has no code pretraining, and 33 steps on 179 examples isn't enough signal to produce reliable SDK-compliant snippets. More training and/or a stronger base model would be the next step — see [Known limitations](#known-limitations).
- ⚠️ **vLLM / Triton / FlashAttention 2** integrations exist in code but are CUDA-only; they are not installed or exercised in this environment (no NVIDIA GPU). 4-bit quantization (bitsandbytes) is likewise CUDA-only and correctly disabled off-GPU.
- ⚠️ **Only 3 of 7 advertised boards have full input validation** (Arduino, ESP32, STM32). Raspberry Pi Pico, Nordic nRF, TI MSP430, and Atmel AVR are defined in the API's type system but rejected by the request validator today.

## Features

- **Multi-peripheral support**: UART, SPI, GPIO, I2C
- **Fine-tuned inference**: real LoRA-adapted model generation, with a template-based fallback for reliability
- **Web interface**: React + TypeScript UI for peripheral/board/language selection and code display
- **Live performance dashboard**: request count, average/p50/p95 latency, throughput, token counts

## Architecture

```
Microcontroller-API-Assistant/
├── backend/           # FastAPI server + inference engine (HF Transformers, optional vLLM)
├── frontend/          # React + TypeScript web interface
├── training/          # LoRA fine-tuning pipeline and dataset
├── qa_system/         # Automated code-quality evaluation harness
└── tests/             # Backend/inference test suite
```

## Quick start

### Prerequisites
- Python 3.9–3.12 (3.13 is not yet supported by the Poetry environments)
- Node.js 18+
- Poetry
- Docker (optional, for containerized deployment)

### Backend
```bash
cd backend
poetry install
poetry run python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm start
```

### Fine-tuning (optional — a base checkpoint is not required to run the app; it falls back to templates)
```bash
cd training
poetry install
poetry run python finetune.py \
  --model microsoft/DialoGPT-medium \
  --dataset-path data/micro_api_dataset.jsonl \
  --output-dir real_checkpoints_run1 \
  --hub-repo-id local/not-used \
  --epochs 3 --max-steps -1 --save-steps 10 --eval-steps 10 \
  --no-wandb --no-push
```
Copy the resulting `real_checkpoints_run1/final_model/` into `backend/checkpoints/mcu-llm/<name>/` — the backend auto-detects the most recently modified subdirectory there on startup.

## Technology stack

| Layer | Tools | Notes |
|---|---|---|
| Backend | FastAPI, Hugging Face Transformers | Async API, real model inference |
| Fine-tuning | PEFT (LoRA), bitsandbytes (4-bit) | LoRA runs anywhere; 4-bit quantization is CUDA-only |
| Optional GPU serving | vLLM, Triton, flash-attn | CUDA-only, present in code, unused without an NVIDIA GPU |
| Frontend | React, TypeScript, Tailwind CSS, Axios | Typed UI and API client |
| Packaging | Poetry, Docker/docker-compose | |

## Supported peripherals

UART, SPI, GPIO, I2C.

## Supported microcontrollers

Declared in the API type system: Arduino Uno, ESP32, STM32, Raspberry Pi Pico, Nordic nRF, TI MSP430, Atmel AVR. **Request validation currently only covers Arduino, ESP32, and STM32** (see Known limitations).

## Verified results

Run on Apple Silicon (CPU/MPS), no CUDA GPU:

| Benchmark | Result |
|---|---|
| Fine-tuning eval loss (3 epochs, 33 steps) | 9.56 → 9.09 → 7.89 |
| Inference latency, naive vs. KV-cache | −66.1% |
| Inference latency, naive vs. KV-cache + fp16/MPS | −79.2% |
| Fused attention (`torch.nn.functional.scaled_dot_product_attention`) vs. naive attention | 1.6× (CPU), 1.58× (MPS) |

Reproduce the latency numbers with `backend/benchmark_latency.py`.

## Known limitations

- Generated code quality is early-stage — see [Status](#status-as-of-2026-09-13) above. This is a training-budget/base-model limitation, not a broken pipeline.
- `backend/app/validators.py` only has capability profiles for 3 of the 7 microcontrollers the API declares support for.
- 4-bit quantization and the vLLM/Triton/flash-attn serving path require an NVIDIA GPU and are untested in this environment.
- No user analytics, feedback collection, or production auth — this is a research/development-stage project.
