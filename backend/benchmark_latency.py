"""
Local latency benchmark for the Microcontroller API Assistant's inference path.

Measures real wall-clock generation latency on whatever hardware is available
(no CUDA/vLLM/Triton/flash-attn required) across three configurations that map
to the same optimization *categories* used in the project's documented vLLM
stack:

  1. naive_no_kv_cache   - full attention recomputed at every decode step
                            (no caching at all - the "unoptimized" baseline)
  2. kv_cache             - standard KV-cache reuse during decoding
                            (same principle as vLLM's prefix/attention caching)
  3. kv_cache_fp16_mps    - KV-cache + half precision on Apple Silicon (MPS)
                            (same principle as the project's half-precision +
                            hardware-accelerated-kernel optimizations)

This does NOT reproduce the CUDA-only stack (vLLM PagedAttention, custom
Triton kernels, flash-attn) described in TECHNICAL_DETAILS.md - those require
an NVIDIA GPU. It gives a real, reproducible number for the optimization
categories that *are* available on this machine.
"""

import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/DialoGPT-medium"
PROMPT = "Generate GPIO configure_pin code for arduino in c:"
NEW_TOKENS = 40
WARMUP_RUNS = 1
TIMED_RUNS = 5


def load_model(device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()
    return tokenizer, model


def timed_generate(tokenizer, model, device, use_cache: bool, runs: int, warmup: int):
    inputs = tokenizer.encode(PROMPT, return_tensors="pt").to(device)
    latencies = []

    for i in range(warmup + runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                inputs,
                min_new_tokens=NEW_TOKENS,
                max_new_tokens=NEW_TOKENS,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            latencies.append(elapsed)

    return {
        "avg": statistics.mean(latencies),
        "p50": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "raw": latencies,
    }


def main():
    mps_available = torch.backends.mps.is_available()
    results = {}

    print(f"Prompt: {PROMPT!r}  |  new tokens per run: {NEW_TOKENS}  |  timed runs: {TIMED_RUNS} (+{WARMUP_RUNS} warmup)\n")

    # 1. Naive baseline: CPU, fp32, no KV cache at all.
    print("[1/3] naive_no_kv_cache  (cpu, float32, use_cache=False)...")
    tok, model = load_model("cpu", torch.float32)
    results["naive_no_kv_cache"] = timed_generate(tok, model, "cpu", use_cache=False, runs=TIMED_RUNS, warmup=WARMUP_RUNS)
    del model

    # 2. + KV cache (same device/precision, isolates the caching effect).
    print("[2/3] kv_cache            (cpu, float32, use_cache=True)...")
    tok, model = load_model("cpu", torch.float32)
    results["kv_cache"] = timed_generate(tok, model, "cpu", use_cache=True, runs=TIMED_RUNS, warmup=WARMUP_RUNS)
    del model

    # 3. + half precision on Apple Silicon GPU (MPS), still with KV cache.
    if mps_available:
        print("[3/3] kv_cache_fp16_mps   (mps, float16, use_cache=True)...")
        tok, model = load_model("mps", torch.float16)
        results["kv_cache_fp16_mps"] = timed_generate(tok, model, "mps", use_cache=True, runs=TIMED_RUNS, warmup=WARMUP_RUNS)
        del model
    else:
        print("[3/3] MPS not available on this machine - skipping.")

    print("\n=== Results (avg wall-clock latency for {} new tokens) ===".format(NEW_TOKENS))
    baseline = results["naive_no_kv_cache"]["avg"]
    for name, r in results.items():
        reduction = (1 - r["avg"] / baseline) * 100
        print(f"{name:22s} avg={r['avg']*1000:8.1f}ms  p50={r['p50']*1000:8.1f}ms  "
              f"min={r['min']*1000:8.1f}ms  max={r['max']*1000:8.1f}ms  "
              f"reduction_vs_naive={reduction:6.1f}%")

    if "kv_cache_fp16_mps" in results:
        final = results["kv_cache_fp16_mps"]["avg"]
    else:
        final = results["kv_cache"]["avg"]
    total_reduction = (1 - final / baseline) * 100
    print(f"\nTotal measured latency reduction, naive -> fully optimized (this machine): {total_reduction:.1f}%")


if __name__ == "__main__":
    main()
