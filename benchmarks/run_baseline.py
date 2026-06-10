#!/usr/bin/env python3
"""Baseline: Qwen2.5-7B-Instruct standard decode throughput."""
import time, json, statistics, os
os.environ["TRANSFORMERS_CACHE"] = "/storage/gxg8313/hf/hub"

from vllm import LLM, SamplingParams

MODEL = "/storage/gxg8313/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
PROMPTS = [
    "Explain the attention mechanism in transformers.",
    "What is CUDA and how does GPU parallelism work?",
    "Describe how PagedAttention improves KV cache utilization.",
    "What are the differences between RLHF and DPO?",
    "Explain speculative decoding and when it improves throughput.",
] * 10

sampling = SamplingParams(max_tokens=128, temperature=0.0)

llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.80, max_model_len=512)

# warmup
llm.generate(PROMPTS[:4], sampling)

runs = []
for _ in range(5):
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sampling)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    runs.append(total_tokens / elapsed)

avg = statistics.mean(runs)
print(f"Baseline: {avg:.1f} tok/s  runs={[round(x,1) for x in runs]}")

with open("baseline_results.json", "w") as f:
    json.dump({"avg_tok_s": round(avg, 1), "runs": [round(x,1) for x in runs]}, f)
