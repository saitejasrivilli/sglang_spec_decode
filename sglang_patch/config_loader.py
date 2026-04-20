"""
sglang_patch/config_loader.py
─────────────────────────────
Central config loader.  Every module imports from here so there is exactly
one place where YAML is parsed and validated.  No hardcoded constants anywhere.
"""
from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from typing import Any, Dict, Optional

import yaml


# ── Locate config directory ──────────────────────────────────────────────────
# Walks up from this file to find the project root (contains configs/).
def _find_project_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError(
        "Cannot locate 'configs/' directory. "
        "Make sure you run from inside the sglang_spec_decode project."
    )


PROJECT_ROOT = _find_project_root()
CLUSTER_CFG_PATH = PROJECT_ROOT / "configs" / "cluster.yaml"
EXPERIMENT_CFG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


# ── Loaders ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_cluster_config() -> Dict[str, Any]:
    with open(CLUSTER_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    _validate_cluster(cfg)
    return cfg


@lru_cache(maxsize=1)
def load_experiment_config() -> Dict[str, Any]:
    with open(EXPERIMENT_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    _validate_experiment(cfg)
    return cfg


# ── Convenience accessors (type-safe) ────────────────────────────────────────

def get_gpu_ids() -> list[int]:
    return load_cluster_config()["cluster"]["gpu_ids"]


def get_tensor_parallel_size(server: str = "sglang") -> int:
    return load_cluster_config()["servers"][server]["tensor_parallel_size"]


def get_server_port(server: str = "sglang") -> int:
    return load_cluster_config()["servers"][server]["port"]


def get_target_model_path() -> str:
    path = load_experiment_config()["models"]["target"]["path"]
    return _resolve_model_path(path)


def get_draft_model_path() -> str:
    path = load_experiment_config()["models"]["draft"]["path"]
    return _resolve_model_path(path)


def get_num_speculative_tokens() -> int:
    return load_experiment_config()["speculative_decoding"]["num_speculative_tokens"]


def get_acceptance_threshold() -> float:
    return load_experiment_config()["speculative_decoding"]["acceptance_threshold"]


def get_benchmark_config() -> Dict[str, Any]:
    return load_experiment_config()["benchmark"]


def get_hf_cache_dir() -> Optional[str]:
    val = load_cluster_config()["cluster"].get("hf_cache_dir")
    if val is not None:
        return str(val)
    return os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_model_path(path: str) -> str:
    """
    If path is a local directory that exists, return it as-is.
    Otherwise treat it as a HF repo ID (download handled by transformers).
    """
    p = pathlib.Path(path)
    if p.exists() and p.is_dir():
        return str(p.resolve())
    return path  # HF repo ID — let HF hub handle caching


def _validate_cluster(cfg: Dict[str, Any]) -> None:
    assert "cluster" in cfg, "cluster.yaml must have a 'cluster' key"
    assert "gpu_ids" in cfg["cluster"], "cluster.yaml: cluster.gpu_ids is required"
    assert "servers" in cfg, "cluster.yaml must have a 'servers' key"
    assert "sglang" in cfg["servers"]
    assert "vllm" in cfg["servers"]


def _validate_experiment(cfg: Dict[str, Any]) -> None:
    assert "models" in cfg
    assert "target" in cfg["models"]
    assert "draft" in cfg["models"]
    assert "speculative_decoding" in cfg
    sd = cfg["speculative_decoding"]
    assert "num_speculative_tokens" in sd
    assert isinstance(sd["num_speculative_tokens"], int) and sd["num_speculative_tokens"] >= 1
