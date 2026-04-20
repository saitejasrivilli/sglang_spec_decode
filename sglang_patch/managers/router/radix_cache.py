"""
sglang_patch/managers/router/radix_cache.py
────────────────────────────────────────────
Drop-in replacement / extension for sglang/srt/managers/router/radix_cache.py.

Adds three new methods to RadixCache:
  • insert_provisional(seq_id, tokens, kv_ptrs)
  • commit_provisional(seq_id, accepted_len)
  • evict_provisional(seq_id)

These three methods are the ONLY correct way to handle speculative-decoding
KV cache entries in SGLang.  The provisional layer sits outside the radix trie
so that draft tokens:
  1. Are never shared as prefixes with other requests.
  2. Can be atomically committed or discarded after target verification.
  3. Hold their physical KV blocks during the draft phase without leaking.

Critical correctness invariant
──────────────────────────────
After a rejection at position k < K:
  - commit_provisional(seq_id, k)  inserts tokens[0:k] into the real trie.
  - The physical GPU blocks for tokens[k:K] are freed immediately.
  - If this is NOT done the target model will read stale KV state and produce
    silently wrong outputs — no crash, just wrong tokens.

Thread safety
─────────────
All public methods on _ProvisionalNode are called from the model_runner's
decode loop, which is single-threaded per GPU worker.  No locking needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Try to import the real SGLang RadixCache ─────────────────────────────────
try:
    from sglang.srt.managers.router.radix_cache import (
        RadixCache as _SGLangRadixCache,
        RadixCacheState,
    )
    _SGLANG_AVAILABLE = True
except ImportError:
    _SGLANG_AVAILABLE = False

    class _SGLangRadixCache:  # type: ignore[no-redef]
        """Minimal stub used when SGLang is not installed (for unit tests)."""

        def __init__(self):
            self._trie: Dict[int, list] = {}
            self._block_pool: set = set()

        def match_prefix(self, tokens: List[int]):
            return [], []

        def insert(self, tokens: List[int], kv_ptrs: List[int]) -> None:
            for tok, ptr in zip(tokens, kv_ptrs):
                self._trie[tok] = self._trie.get(tok, []) + [ptr]

        def free_blocks(self, kv_ptrs: List[int]) -> None:
            self._block_pool.update(kv_ptrs)

        def get_node(self, seq_id: int):
            return None


# ── Provisional node ──────────────────────────────────────────────────────────

@dataclass
class _ProvisionalNode:
    """
    Holds the draft tokens + their KV block pointers for one sequence.
    Lives outside the radix trie until commit or eviction.
    """
    seq_id: int
    tokens: List[int] = field(default_factory=list)
    kv_ptrs: List[int] = field(default_factory=list)

    # Snapshot of the sequence length BEFORE the draft phase started.
    # Needed so commit can anchor the trie insertion correctly.
    base_len: int = 0

    def __len__(self) -> int:
        return len(self.tokens)

    def is_empty(self) -> bool:
        return len(self.tokens) == 0

    def slice_accepted(self, accepted_len: int):
        """Return (accepted_tokens, accepted_ptrs, rejected_ptrs)."""
        accepted_tokens = self.tokens[:accepted_len]
        accepted_ptrs = self.kv_ptrs[:accepted_len]
        rejected_ptrs = self.kv_ptrs[accepted_len:]
        return accepted_tokens, accepted_ptrs, rejected_ptrs


# ── Main RadixCache with provisional extension ────────────────────────────────

class RadixCache(_SGLangRadixCache):
    """
    SGLang RadixCache augmented with speculative-decoding provisional layer.

    All original RadixCache functionality is preserved.  The three new methods
    (insert_provisional / commit_provisional / evict_provisional) add the
    provisional layer on top without touching the radix trie internals.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Map from seq_id → _ProvisionalNode
        self._provisional: Dict[int, _ProvisionalNode] = {}

    # ── Public speculative-decoding API ───────────────────────────────────────

    def insert_provisional(
        self,
        seq_id: int,
        tokens: List[int],
        kv_ptrs: List[int],
        base_len: int,
    ) -> None:
        """
        Store draft tokens + their KV block pointers as provisional.

        Parameters
        ----------
        seq_id   : unique sequence identifier (same as used by the scheduler)
        tokens   : list of K draft token ids, in order
        kv_ptrs  : list of K physical KV block pointers (one per token)
        base_len : number of tokens already committed to the trie for this seq
                   (= len before the draft phase started)

        Raises
        ------
        RuntimeError if there is already a provisional node for seq_id that
        has not been committed or evicted — indicates a logic error in the
        calling code (missing evict after a rejection).
        """
        if seq_id in self._provisional and not self._provisional[seq_id].is_empty():
            raise RuntimeError(
                f"[RadixCache] insert_provisional called for seq_id={seq_id} "
                "but an uncommitted provisional node already exists. "
                "Call evict_provisional or commit_provisional first."
            )

        if len(tokens) != len(kv_ptrs):
            raise ValueError(
                f"tokens ({len(tokens)}) and kv_ptrs ({len(kv_ptrs)}) must have the same length"
            )

        node = _ProvisionalNode(
            seq_id=seq_id,
            tokens=list(tokens),
            kv_ptrs=list(kv_ptrs),
            base_len=base_len,
        )
        self._provisional[seq_id] = node
        logger.debug(
            "insert_provisional seq_id=%d  K=%d  base_len=%d",
            seq_id, len(tokens), base_len,
        )

    def commit_provisional(self, seq_id: int, accepted_len: int) -> List[int]:
        """
        Commit the first `accepted_len` draft tokens into the real radix trie.
        Free the KV blocks for any rejected tail tokens.

        Parameters
        ----------
        accepted_len : number of tokens to accept (0 ≤ accepted_len ≤ K)

        Returns
        -------
        List of KV block pointers that were freed (for diagnostics / testing).
        """
        node = self._provisional.get(seq_id)
        if node is None or node.is_empty():
            logger.warning(
                "commit_provisional called for seq_id=%d but no provisional node found",
                seq_id,
            )
            return []

        if accepted_len < 0 or accepted_len > len(node):
            raise ValueError(
                f"accepted_len={accepted_len} out of range [0, {len(node)}]"
            )

        accepted_tokens, accepted_ptrs, rejected_ptrs = node.slice_accepted(accepted_len)

        # ── Insert accepted tokens into the real trie ─────────────────────
        if accepted_tokens:
            try:
                self.insert(accepted_tokens, accepted_ptrs)
            except Exception as exc:
                # Don't let a trie error leak KV blocks — free everything.
                logger.error(
                    "commit_provisional: trie insert failed for seq_id=%d: %s. "
                    "Freeing all blocks to prevent leak.",
                    seq_id, exc,
                )
                self.free_blocks(accepted_ptrs + rejected_ptrs)
                self._provisional.pop(seq_id, None)
                raise

        # ── Free rejected tail blocks ─────────────────────────────────────
        if rejected_ptrs:
            self.free_blocks(rejected_ptrs)
            logger.debug(
                "commit_provisional seq_id=%d  accepted=%d  freed=%d blocks",
                seq_id, accepted_len, len(rejected_ptrs),
            )

        self._provisional.pop(seq_id, None)
        return rejected_ptrs

    def evict_provisional(self, seq_id: int) -> List[int]:
        """
        Discard ALL provisional tokens for seq_id and free their KV blocks.

        Call this when:
          • The sequence finishes (EOS) during the draft phase.
          • A hard rejection at position 0 (nothing to commit).
          • The sequence is preempted by the scheduler.

        Returns
        -------
        List of freed KV block pointers (for diagnostics / testing).
        """
        node = self._provisional.pop(seq_id, None)
        if node is None or node.is_empty():
            return []

        self.free_blocks(node.kv_ptrs)
        logger.debug(
            "evict_provisional seq_id=%d  freed=%d blocks",
            seq_id, len(node.kv_ptrs),
        )
        return node.kv_ptrs

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def num_provisional_sequences(self) -> int:
        """Number of sequences currently holding provisional KV blocks."""
        return sum(1 for n in self._provisional.values() if not n.is_empty())

    def provisional_block_count(self) -> int:
        """Total number of provisional KV blocks in flight."""
        return sum(len(n.kv_ptrs) for n in self._provisional.values())

    def clear_all_provisional(self) -> None:
        """
        Emergency: free all provisional blocks (e.g., on server shutdown).
        """
        for node in list(self._provisional.values()):
            if not node.is_empty():
                self.free_blocks(node.kv_ptrs)
        self._provisional.clear()
        logger.info("clear_all_provisional: all provisional blocks freed")
