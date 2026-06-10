#!/usr/bin/env python3
"""Single-request latency: baseline vs n-gram spec decode (B=1)."""
import time, json, statistics, os
os.environ["TRANSFORMERS_CACHE"] = "/storage/gxg8313/hf/hub"
from vllm import LLM, SamplingParams

MODEL = "/storage/gxg8313/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

PROMPT = "Explain the attention mechanism in transformers. The attention mechanism uses queries, keys, and values to compute weighted sums."
sampling = SamplingParams(max_tokens=256, temperature=0.0)

def bench_b1(llm, n_warmup=5, n_runs=20):
    for _ in range(n_warmup):
        llm.generate(PROMPT, sampling)
    runs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outs = llm.generate(PROMPT, sampling)
        elapsed = time.perf_counter() - t0
        total_tokens = len(outs[0].outputs[0].token_ids)
        runs.append(total_tokens / elapsed)
    return statistics.mean(runs), sorted(runs)

# Baseline
print("Loading baseline...")
base_llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.80, max_model_len=512)
base_avg, base_runs = bench_b1(base_llm)
print(f"Baseline B=1: {base_avg:.1f} tok/s  (p50={statistics.median(base_runs):.1f})")
del base_llm
import gc, torch; gc.collect(); torch.cuda.empty_cache()

# Spec decode (n-gram)
print("Loading spec decode (n-gram K=5)...")
spec_llm = LLM(
    model=MODEL, dtype="float16", gpu_memory_utilization=0.80, max_model_len=512,
    speculative_config={"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5},
)
spec_avg, spec_runs = bench_b1(spec_llm)
print(f"Spec B=1 (ngram K=5): {spec_avg:.1f} tok/s  (p50={statistics.median(spec_runs):.1f})")

speedup = spec_avg / base_avg
print(f"\nSpeedup: {speedup:.3f}x  ({base_avg:.1f} → {spec_avg:.1f} tok/s)")

results = {
    "baseline_b1_tok_s": round(base_avg, 1),
    "spec_b1_ngram_k5_tok_s": round(spec_avg, 1),
    "speedup": round(speedup, 3),
    "baseline_runs": [round(x,1) for x in base_runs],
    "spec_runs": [round(x,1) for x in spec_runs],
}
with open("single_req_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved single_req_results.json")
