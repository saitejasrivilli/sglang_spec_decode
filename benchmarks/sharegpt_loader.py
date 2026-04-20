"""
benchmarks/sharegpt_loader.py
──────────────────────────────
Downloads the ShareGPT dataset and extracts the first `n` human turns
for use as benchmark prompts.  Caches locally to avoid repeated downloads.

Uses real ShareGPT data (no synthetic prompts).
"""
from __future__ import annotations

import json
import logging
import pathlib
import urllib.request
from typing import List, Optional

logger = logging.getLogger(__name__)

# ShareGPT dataset — widely used for LLM benchmarks
_SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
    "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)

_CACHE_DIR = pathlib.Path(__file__).parent / ".cache"
_CACHE_FILE = _CACHE_DIR / "sharegpt_v3.json"


def load_sharegpt_prompts(
    n: int = 200,
    min_length: int = 10,
    max_length: int = 1024,
    cache_path: Optional[pathlib.Path] = None,
    force_download: bool = False,
) -> List[str]:
    """
    Return the first `n` human-turn prompts from ShareGPT.

    Parameters
    ----------
    n            : number of prompts to return
    min_length   : discard prompts shorter than this (chars)
    max_length   : truncate prompts longer than this (chars)
    cache_path   : override cache file location
    force_download: re-download even if cache exists

    Returns
    -------
    List[str] of prompts, length == n (or fewer if dataset is smaller)
    """
    cache = pathlib.Path(cache_path) if cache_path else _CACHE_FILE
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not cache.exists() or force_download:
        logger.info("Downloading ShareGPT dataset to %s ...", cache)
        _download(cache)

    logger.info("Loading ShareGPT from %s", cache)
    with open(cache, encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[str] = []
    for conversation in data:
        if len(prompts) >= n:
            break
        convs = conversation.get("conversations", [])
        for turn in convs:
            if turn.get("from", "").lower() in ("human", "user"):
                text = turn.get("value", "").strip()
                if len(text) < min_length:
                    continue
                text = text[:max_length]
                prompts.append(text)
                break  # one prompt per conversation

    if len(prompts) < n:
        logger.warning(
            "Requested %d prompts but only %d available after filtering",
            n, len(prompts),
        )

    logger.info("Loaded %d prompts (requested %d)", len(prompts), n)
    return prompts[:n]


def _download(dest: pathlib.Path) -> None:
    """Download the ShareGPT JSON with a simple progress indicator."""
    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  Downloading... {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(_SHAREGPT_URL, dest, reporthook=_progress)
        print()  # newline after progress
        logger.info("Download complete: %s", dest)
    except Exception as exc:
        logger.error("Failed to download ShareGPT: %s", exc)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prompts = load_sharegpt_prompts(n=5)
    for i, p in enumerate(prompts):
        print(f"\n[{i}] {p[:120]}...")
