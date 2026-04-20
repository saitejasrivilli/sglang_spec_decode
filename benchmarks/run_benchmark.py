"""
benchmarks/run_benchmark.py
────────────────────────────
Wall-clock latency + throughput benchmark.

Runs the SAME 200 ShareGPT prompts against:
  • SGLang with speculative decoding  (port from cluster.yaml)
  • vLLM baseline                      (port from cluster.yaml)

Writes results/benchmark_<timestamp>.json and prints a summary table.

All parameters come from configs/*.yaml — nothing hardcoded.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from benchmarks.sharegpt_loader import load_sharegpt_prompts
from sglang_patch.config_loader import (
    get_benchmark_config,
    get_server_port,
    load_cluster_config,
    load_experiment_config,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Request helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sglang_generate(
    url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout: float = 120.0,
) -> Dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{url}/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    latency = time.perf_counter() - t0
    data = resp.json()
    return {
        "latency_s": latency,
        "output_text": data.get("text", ""),
        "output_tokens": data.get("meta_info", {}).get("completion_tokens", 0),
        "ttft_s": data.get("meta_info", {}).get("ttft", None),
        "spec_acceptance_rate": data.get("meta_info", {}).get("spec_acceptance_rate", None),
    }


def _vllm_generate(
    url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout: float = 120.0,
) -> Dict:
    payload = {
        "model": load_experiment_config()["models"]["target"]["path"],
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{url}/v1/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    latency = time.perf_counter() - t0
    data = resp.json()
    choice = data.get("choices", [{}])[0]
    return {
        "latency_s": latency,
        "output_text": choice.get("text", ""),
        "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
        "ttft_s": None,  # vLLM baseline doesn't expose TTFT in this endpoint
        "spec_acceptance_rate": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Server readiness
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_server(url: str, timeout_s: int, health_path: str = "/health") -> bool:
    """Poll until server is ready or timeout expires."""
    deadline = time.time() + timeout_s
    interval = 2.0
    logger.info("Waiting for server at %s ...", url)
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}{health_path}", timeout=5)
            if r.status_code == 200:
                logger.info("Server ready: %s", url)
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(interval)
    logger.error("Server not ready after %ds: %s", timeout_s, url)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    url: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    generate_fn,
    max_workers: int = 1,
    label: str = "server",
) -> List[Dict]:
    """
    Send all prompts to a server and collect per-request metrics.

    Parameters
    ----------
    max_workers : 1 = serial (for latency measurement);
                  >1 = parallel (for throughput measurement)
    """
    results = []
    total = len(prompts)

    if max_workers == 1:
        for i, prompt in enumerate(prompts):
            try:
                r = generate_fn(
                    url=url,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                results.append(r)
            except Exception as exc:
                logger.warning("[%s] Request %d failed: %s", label, i, exc)
                results.append({"latency_s": None, "error": str(exc)})
            if (i + 1) % 20 == 0:
                logger.info("[%s] %d/%d done", label, i + 1, total)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    generate_fn,
                    url=url,
                    prompt=p,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ): idx
                for idx, p in enumerate(prompts)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    logger.warning("[%s] Request %d failed: %s", label, idx, exc)
                    results.append({"latency_s": None, "error": str(exc)})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metric aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(values: List[float], p: float) -> float:
    import statistics
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def aggregate(results: List[Dict], label: str) -> Dict:
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    ttfts = [r["ttft_s"] for r in results if r.get("ttft_s") is not None]
    output_tokens = [r.get("output_tokens", 0) for r in results if r.get("output_tokens")]
    acceptance_rates = [
        r["spec_acceptance_rate"]
        for r in results
        if r.get("spec_acceptance_rate") is not None
    ]

    total_tokens = sum(output_tokens)
    total_time = sum(latencies) if latencies else 1.0

    return {
        "label": label,
        "n_requests": len(results),
        "n_errors": sum(1 for r in results if r.get("error")),
        "p50_latency_s": _percentile(latencies, 50),
        "p95_latency_s": _percentile(latencies, 95),
        "p99_latency_s": _percentile(latencies, 99),
        "p50_ttft_s": _percentile(ttfts, 50) if ttfts else None,
        "p99_ttft_s": _percentile(ttfts, 99) if ttfts else None,
        "throughput_tok_per_s": total_tokens / total_time if total_time > 0 else 0,
        "avg_acceptance_rate": (
            sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else None
        ),
    }


def print_table(agg_list: List[Dict]) -> None:
    cols = [
        ("label", 20),
        ("p50_latency_s", 14),
        ("p99_latency_s", 14),
        ("p50_ttft_s", 12),
        ("throughput_tok_per_s", 20),
        ("avg_acceptance_rate", 20),
        ("n_errors", 10),
    ]
    header = "  ".join(f"{c[0]:<{c[1]}}" for c in cols)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for agg in agg_list:
        row = "  ".join(
            f"{str(agg.get(c[0], 'N/A')):<{c[1]}}" for c in cols
        )
        print(row)
    print("─" * len(header) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cfg_bench = get_benchmark_config()
    cfg_cluster = load_cluster_config()

    parser = argparse.ArgumentParser(description="SGLang vs vLLM benchmark")
    parser.add_argument(
        "--sglang-url",
        default=(
            f"http://localhost:{cfg_cluster['servers']['sglang']['port']}"
        ),
        help="SGLang server URL",
    )
    parser.add_argument(
        "--vllm-url",
        default=(
            f"http://localhost:{cfg_cluster['servers']['vllm']['port']}"
        ),
        help="vLLM server URL",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=cfg_bench["num_prompts"],
        help="Number of ShareGPT prompts",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=cfg_bench["max_new_tokens"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Only benchmark SGLang (skip vLLM)",
    )
    parser.add_argument(
        "--skip-sglang",
        action="store_true",
        help="Only benchmark vLLM (skip SGLang)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent requests (1 = serial latency test)",
    )
    args = parser.parse_args()

    # Load prompts
    prompts = load_sharegpt_prompts(n=args.num_prompts)
    logger.info("Loaded %d prompts", len(prompts))

    timeout = cfg_bench.get("server_ready_timeout_s", 300)
    all_agg = []
    raw_results = {}

    # ── SGLang benchmark ───────────────────────────────────────────────────
    if not args.skip_sglang:
        if not wait_for_server(args.sglang_url, timeout, health_path="/health"):
            logger.error("SGLang server not available; skipping")
        else:
            logger.info("Benchmarking SGLang (spec decode) ...")
            sglang_results = run_benchmark(
                url=args.sglang_url,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                generate_fn=_sglang_generate,
                max_workers=args.workers,
                label="sglang_spec",
            )
            agg = aggregate(sglang_results, "sglang_spec")
            all_agg.append(agg)
            raw_results["sglang_spec"] = sglang_results

    # ── vLLM benchmark ─────────────────────────────────────────────────────
    if not args.skip_vllm:
        if not wait_for_server(args.vllm_url, timeout, health_path="/health"):
            logger.error("vLLM server not available; skipping")
        else:
            logger.info("Benchmarking vLLM (baseline) ...")
            vllm_results = run_benchmark(
                url=args.vllm_url,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                generate_fn=_vllm_generate,
                max_workers=args.workers,
                label="vllm_baseline",
            )
            agg = aggregate(vllm_results, "vllm_baseline")
            all_agg.append(agg)
            raw_results["vllm_baseline"] = vllm_results

    # ── Speedup calculation ────────────────────────────────────────────────
    if len(all_agg) == 2:
        sglang_agg = next(a for a in all_agg if "sglang" in a["label"])
        vllm_agg = next(a for a in all_agg if "vllm" in a["label"])
        sglang_tps = sglang_agg["throughput_tok_per_s"]
        vllm_tps = vllm_agg["throughput_tok_per_s"]
        speedup = sglang_tps / vllm_tps if vllm_tps > 0 else float("nan")
        print(f"\n✓  Throughput speedup (SGLang spec / vLLM baseline): {speedup:.2f}x")

    print_table(all_agg)

    # ── Save results ───────────────────────────────────────────────────────
    out_dir = pathlib.Path(cfg_bench["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": ts,
                "config": {
                    "num_prompts": args.num_prompts,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
                "aggregated": all_agg,
                "raw": raw_results,
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
