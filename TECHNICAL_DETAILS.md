# Technical Details: Microcontroller API Assistant

*Last verified 2026-09-13, on Apple Silicon (CPU/MPS), no CUDA GPU. Every number in this document was either measured directly against this codebase or is explicitly marked as unverified.*

## 1. LLM Fine-Tuning

### Base model
- **Model**: `microsoft/DialoGPT-medium` (~345M params, GPT-2 architecture, 2019, trained on Reddit conversation threads — no code pretraining)
- **Rationale**: small enough to fine-tune quickly on modest hardware; conversational pretraining gives it *some* general language ability to adapt from, though not a code-modeling head start

### Training dataset
- **Total examples**: 224, in `training/data/micro_api_dataset.jsonl` (instruction → code pairs)
- **Split**: 80/20 → 179 train / 45 validation (confirmed by an actual training run, not just computed from the file)
- **Coverage**: UART, SPI, GPIO, I2C across Arduino, ESP32, STM32, and Raspberry Pi Pico, sourced from each platform's official docs/SDK examples
- **Format**:
  ```json
  {"instruction": "Provide example code for how to initialize spi communication on raspberry pi pico?",
   "output": "Here's how to initialize spi communication on raspberry pi pico:\n\n```python\n...\n```\n\nThis code demonstrates:\n..."}
  ```

### LoRA configuration (`training/finetune.py`)
- Rank (r) = 16, α = 32, dropout = 0.1
- Target modules (DialoGPT/GPT-2): `c_attn`, `c_proj`, `c_fc`
- 4-bit quantization (bitsandbytes NF4): implemented via `BitsAndBytesConfig`, but **the code correctly disables it outside CUDA** (`RealModelLoader.get_quantization_config` returns `None` on `cpu`/`mps`). Confirmed directly: importing `bitsandbytes` and instantiating even a minimal `Linear4bit` layer on this Mac raises `ModuleNotFoundError: No module named 'scipy'` from inside bitsandbytes' own CUDA-oriented import chain — it is not a "slower" fallback, it is structurally CUDA-only.
- Training format: each example rendered as `"Instruction: {instruction}\nOutput: {output}\n"` and tokenized for standard causal-LM (next-token) loss.

### A completed training run (2026-09-13)

Six bugs had to be fixed before a run could complete or serve correctly — each is a real, independent defect found by executing the code, not by inspection:

1. **`training/finetune.py`** — `_save_final_model(model, tokenizer)` called `trainer.save_model(...)`, but `trainer` was never in scope (`NameError`). Would have crashed after a full training run, discarding it. Fixed the method signature and added `merge_and_unload()` so the saved checkpoint is a complete, standalone model rather than a bare LoRA adapter.
2. **`training/finetune.py`** — the dataset preprocessing step pre-set a `labels` field before tokenizer padding; `DataCollatorForLanguageModeling` (which is supposed to derive labels itself from the padded batch) then choked trying to tensor-ize a ragged, unpadded field. Fixed by removing the manual assignment.
3. **`training/pyproject.toml`** — `accelerate==0.25.0` predates an argument (`use_seedable_sampler`) that `transformers==4.40.2`'s `Trainer` now passes to `Accelerator.__init__`, raising a `TypeError` at trainer construction. Upgraded to `accelerate==0.30.1`.
4. **`backend/app/config.py`** — checkpoint auto-detection only checked for `pytorch_model.bin` or `adapter_model.safetensors`; modern `save_pretrained()` writes `model.safetensors`, so a correctly-trained checkpoint would have been invisible to the backend. Fixed in both detection code paths.
5. **`backend/app/inference.py`** — the live prompt-builder sent a long, multi-section structured prompt ("Peripheral: X\nMicrocontroller: Y\n...") that bears no resemblance to the `"Instruction: ...\nOutput: ..."` format the model was actually trained on. Even a well-trained checkpoint would look broken served this way. Rewrote `_build_dynamic_prompt` to match the training format exactly.
6. **`backend/app/api.py`** — `POST /generate-code`, the endpoint the main UI calls, previously returned hardcoded template strings regardless of any model output. Wired to call real inference first, template only as a failure/empty-output fallback.

With all six fixed, a real run completed: 3 epochs, 33 optimizer steps (effective batch size 16), ~35 minutes on CPU/MPS.

| Step | Train loss | Eval loss |
|---|---|---|
| 10 | 9.8424 | 9.5621 |
| 20 | 9.6589 | 9.0928 |
| 30 | 8.9915 | 7.8945 |

Eval loss fell monotonically and had not plateaued by step 30 — genuine, still-improving learning, not noise.

**Generation quality after this run**: with the prompt-format bug fixed, the checkpoint produces non-empty output, but it is not coherent code (e.g. `"1, pio 0"` for a GPIO configure-pin request). This is attributable to the small training budget (33 steps / 179 examples) and the base model's lack of code pretraining, not to any remaining plumbing defect — everything from checkpoint auto-detection through prompt formatting to API serving has been verified to work correctly end-to-end.

### Evaluation infrastructure that exists but was not run this session
- `qa_system/` contains a scoring harness (`qa_system/evaluator.py`, `scoring.py`) that auto-generates test scenarios across boards/peripherals/operations and scores generated code on syntax, keyword presence, and library usage. This is real, usable code, but it was not run against the fine-tuned checkpoint in this session — the "generation quality" assessment above is a direct read of sample outputs, not a `qa_system` score. Running it against the current checkpoint would give a more rigorous number and is a reasonable next step.

---

## 2. Inference latency: what's real, measured, on this hardware

The documented vLLM + Triton + FlashAttention 2 stack described below is real *code*, but it is **CUDA-only** and could not be exercised on this machine (no NVIDIA GPU; `torch.cuda.is_available()` is `False`; vLLM and Triton are not installed in this environment). No benchmark of that specific stack has been run. What follows is split accordingly.

### CUDA-only path (implemented, unverified in this environment)
- `backend/app/inference.py`'s `_initialize_vllm` configures `LLM(... tensor_parallel_size=..., gpu_memory_utilization=0.9, enable_prefix_caching=True, enforce_eager=False, dtype="float16")` when `ENABLE_VLLM=true` and vLLM is importable — falls back to Hugging Face Transformers otherwise (confirmed: it does fall back correctly on this machine).
- `backend/app/attention.py` implements a standalone FlashAttention 2 / Triton-kernel attention module with its own benchmark harness (`PerformanceBenchmark`). It is not wired into the DialoGPT generation path (which uses HF's/vLLM's own attention implementation) — it exists as a demonstrable optimization module, not as something exercised per-request today.
- No end-to-end latency benchmark of this specific stack (vLLM + Triton + flash-attn on CUDA) has been performed. Any prior "60–67% reduction" figure attributed to this stack should be treated as a design estimate, not a measurement, until run on CUDA hardware.

### CPU/MPS path (implemented and measured, `backend/benchmark_latency.py`)
Three configurations, 40 generated tokens, 5 timed runs each (1 warmup), greedy decoding:

| Configuration | Avg latency | Reduction vs. naive |
|---|---|---|
| Naive — CPU, fp32, `use_cache=False` (full attention recomputed every decode step) | 3216.5 ms | — |
| + KV-cache — CPU, fp32, `use_cache=True` | 1089.6 ms | 66.1% |
| + half precision on Apple Silicon (MPS), `use_cache=True` | 670.5 ms | 79.2% |

This demonstrates the same underlying *category* of optimization (avoid redundant computation via caching; use lower precision on available hardware) as the CUDA stack, using techniques that actually run here.

### A working local proxy for FlashAttention (`torch.nn.functional.scaled_dot_product_attention`)
PyTorch's built-in fused attention kernel — separate from the CUDA-only `flash-attn` package — was benchmarked against naive (unfused) attention on isolated Q/K/V tensors:

| Device | Naive | Fused SDPA | Speedup | Output match |
|---|---|---|---|---|
| CPU | 3.55 ms | 2.22 ms | 1.60× | max diff 9.8e-7 |
| MPS | 1.86 ms | 1.18 ms | 1.58× | bit-identical |

---

## 3. React + TypeScript application

### Features
- Peripheral (UART/SPI/GPIO/I2C), board, language, and operation selection, with dynamic parameter forms per operation
- Free-form natural-language query interface (`POST /generate`)
- Live performance dashboard: total requests, average/p50/p95 latency, throughput, total tokens (`GET /performance`)
- Syntax-highlighted code display with explanation, dependencies, and warnings

### API integration
- `GET /microcontrollers`, `GET /peripherals`, `GET /operations/{peripheral}`, `GET /languages`
- `POST /generate-code` — full-parameter generation. **Now genuinely calls the fine-tuned model** (fixed this session); template fallback only on failure or empty output.
- `POST /generate` — simple query generation, same real-inference-then-fallback pattern.
- `GET /performance`, `GET /model/info`

### Known gap: microcontroller validation coverage
`backend/app/validators.py`'s `microcontroller_capabilities` dictionary only has entries for Arduino, ESP32, and STM32. Raspberry Pi Pico, Nordic nRF, TI MSP430, and Atmel AVR are valid `MicrocontrollerType` enum members and appear in the README's supported-boards list, but any request naming them is rejected by the validator with "Unsupported microcontroller." Confirmed directly by requesting I2C/write_data on Raspberry Pi Pico. Not fixed this session — flagged as a separate, unrelated defect from the fine-tuning work.

### User feedback / analytics
Not implemented. This remains a development/research-stage project with no production user tracking, as in earlier versions of this document.

---

## Summary

| Area | Status |
|---|---|
| Fine-tuning pipeline | Real, runs correctly end-to-end, produces measurable learning |
| Fine-tuned model output quality | Early-stage — non-empty but not coherent code at current training budget |
| FastAPI backend serving the fine-tuned model | Real, verified this session (previously template-only) |
| Latency optimization (CPU/MPS) | Real, measured: 66–79% reduction |
| Latency optimization (CUDA: vLLM/Triton/flash-attn) | Real code, unverified — no CUDA GPU available to benchmark |
| 4-bit quantization | Real code, correctly CUDA-gated, confirmed non-functional off-GPU |
| React/TypeScript frontend | Real, fully functional |
| Board validation coverage | 3 of 7 declared boards implemented |
