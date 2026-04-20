"""
benchmarks/acceptance_vs_temp.py
──────────────────────────────────
Sweeps temperature across the values in experiment.yaml and measures
acceptance rate + wall-clock speedup at each.

Generates:
  results/acceptance_vs_temp_<ts>.json
  results/acceptance_vs_temp_<ts>.png

Run AFTER the SGLang server is up:
  python benchmarks/acceptance_vs_temp.py
"""
from __future__ import annotations

import json
import logging
import pathlib
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from benchmarks.sharegpt_loader import load_sharegpt_prompts
from benchmarks.run_benchmark import run_benchmark, aggregate, wait_for_server
from sglang_patch.config_loader import get_benchmark_config, load_cluster_config

logger = logging.getLogger(__name__)


def _sglang_generate_with_temp(
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
        "output_tokens": data.get("meta_info", {}).get("completion_tokens", 0),
        "ttft_s": data.get("meta_info", {}).get("ttft", None),
        "spec_acceptance_rate": data.get("meta_info", {}).get("spec_acceptance_rate", None),
    }


def run_sweep(
    sglang_url: str,
    prompts: List[str],
    temperatures: List[float],
    max_new_tokens: int,
    K: int,
    draft_overhead: float,
) -> List[Dict]:
    """
    For each temperature, run the benchmark and record acceptance rate + speedup.

    draft_overhead: ratio of draft model inference time to target model time.
    Used to compute theoretical speedup curve.  Measure empirically once and
    put it in experiment.yaml; typical value for TinyLlama/Llama-3-8B: 0.10–0.14.
    """
    results = []
    for temp in temperatures:
        logger.info("Running sweep at temperature=%.2f ...", temp)
        raw = run_benchmark(
            url=sglang_url,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            generate_fn=_sglang_generate_with_temp,
            max_workers=1,
            label=f"temp_{temp:.2f}",
        )
        agg = aggregate(raw, f"temp_{temp:.2f}")
        ar = agg.get("avg_acceptance_rate") or 0.0

        # Theoretical speedup: 1 + ar*K / (1 + K * draft_overhead)
        theoretical = 1.0 + ar * K / (1.0 + K * draft_overhead)

        results.append({
            "temperature": temp,
            "acceptance_rate": ar,
            "p50_latency_s": agg["p50_latency_s"],
            "throughput_tok_per_s": agg["throughput_tok_per_s"],
            "theoretical_speedup": theoretical,
        })
        logger.info(
            "  temp=%.2f  acceptance_rate=%.3f  theoretical_speedup=%.2fx",
            temp, ar, theoretical,
        )

    return results


def plot_sweep(sweep_results: List[Dict], out_path: pathlib.Path) -> None:
    """Generate acceptance rate + speedup chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart generation")
        return

    temps = [r["temperature"] for r in sweep_results]
    acc_rates = [r["acceptance_rate"] for r in sweep_results]
    theoretical = [r["theoretical_speedup"] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax1.set_facecolor("#0f1117")

    color_acc = "#4fc3f7"
    color_theory = "#aaa"

    ax1.plot(temps, acc_rates, "o-", color=color_acc, linewidth=2, markersize=7,
             label="acceptance rate")
    ax1.set_xlabel("temperature", color="white", fontsize=13)
    ax1.set_ylabel("acceptance rate", color=color_acc, fontsize=13)
    ax1.tick_params(axis="x", colors="white")
    ax1.tick_params(axis="y", colors=color_acc)
    ax1.set_ylim(0, 1.05)
    ax1.spines["bottom"].set_color("#444")
    ax1.spines["left"].set_color(color_acc)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.set_facecolor("#0f1117")
    ax2.plot(temps, theoretical, "s--", color=color_theory, linewidth=1.5,
             markersize=6, label="theoretical speedup", alpha=0.7)
    ax2.set_ylabel("theoretical speedup", color=color_theory, fontsize=13)
    ax2.tick_params(axis="y", colors=color_theory)
    ax2.spines["right"].set_color(color_theory)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", facecolor="#1e2130", edgecolor="#444",
               labelcolor="white", fontsize=11)

    plt.title("Spec Decode: Acceptance Rate & Speedup vs Temperature",
              color="white", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Chart saved to %s", out_path)


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    cfg_bench = get_benchmark_config()
    cfg_cluster = load_cluster_config()

    parser = argparse.ArgumentParser(description="Acceptance rate vs temperature sweep")
    parser.add_argument(
        "--sglang-url",
        default=f"http://localhost:{cfg_cluster['servers']['sglang']['port']}",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=min(50, cfg_bench["num_prompts"]),
        help="Prompts per temperature step (fewer = faster sweep)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=cfg_bench["max_new_tokens"],
    )
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="Number of draft tokens (reads from experiment.yaml if unset)",
    )
    parser.add_argument(
        "--draft-overhead",
        type=float,
        default=0.12,
        help=(
            "Ratio of draft model time to target model time. "
            "Measure once with --measure-overhead flag and store here."
        ),
    )
    args = parser.parse_args()

    from sglang_patch.config_loader import get_num_speculative_tokens
    K = args.K or get_num_speculative_tokens()

    if not wait_for_server(args.sglang_url, timeout_s=cfg_bench.get("server_ready_timeout_s", 300)):
        sys.exit(1)

    prompts = load_sharegpt_prompts(n=args.num_prompts)
    temperatures = cfg_bench.get("temperature_sweep", [0.0, 0.3, 0.6, 0.8, 1.0, 1.2])

    sweep = run_sweep(
        sglang_url=args.sglang_url,
        prompts=prompts,
        temperatures=temperatures,
        max_new_tokens=args.max_new_tokens,
        K=K,
        draft_overhead=args.draft_overhead,
    )

    out_dir = pathlib.Path(cfg_bench["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"acceptance_vs_temp_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({"K": K, "draft_overhead": args.draft_overhead, "sweep": sweep}, f, indent=2)
    logger.info("Sweep data saved to %s", json_path)

    png_path = out_dir / f"acceptance_vs_temp_{ts}.png"
    plot_sweep(sweep, png_path)


if __name__ == "__main__":
    main()
