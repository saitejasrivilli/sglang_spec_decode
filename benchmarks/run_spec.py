#!/usr/bin/env python3
"""
Speculative decode: Qwen2.5-7B with n-gram prompt lookup (no draft model).
N-gram speculative decoding uses prompt n-gram matches as draft tokens.
K=5 draft tokens, n-gram window=5.
"""
import time, json, statistics, os
os.environ["TRANSFORMERS_CACHE"] = "/storage/gxg8313/hf/hub"

from vllm import LLM, SamplingParams

TARGET = "/storage/gxg8313/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# Prompts with repetitive patterns — good for n-gram spec decode
PROMPTS = [
    "Explain the attention mechanism in transformers. The attention mechanism in transformers uses queries, keys, and values.",
    "What is CUDA and how does GPU parallelism work? CUDA enables parallel computation on GPU cores.",
    "Describe how PagedAttention improves KV cache utilization in large language model serving.",
    "What are the differences between RLHF and DPO fine-tuning methods for language models?",
    "Explain speculative decoding and when it improves throughput. Speculative decoding uses a draft model.",
] * 10

sampling = SamplingParams(max_tokens=128, temperature=0.0)

llm = LLM(
    model=TARGET,
    dtype="float16",
    gpu_memory_utilization=0.80,
    max_model_len=512,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
    },
)

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
print(f"Spec (ngram K=5): {avg:.1f} tok/s  runs={[round(x,1) for x in runs]}")

with open("spec_results.json", "w") as f:
    json.dump({
        "method": "ngram",
        "K": 5,
        "avg_tok_s": round(avg, 1),
        "runs": [round(x, 1) for x in runs],
    }, f)
