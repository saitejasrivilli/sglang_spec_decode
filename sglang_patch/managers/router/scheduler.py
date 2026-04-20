"""
sglang_patch/managers/router/scheduler.py
──────────────────────────────────────────
Spec-decode-aware scheduler extension for SGLang.

Key addition: when a sequence is preempted (swapped out) during a draft
phase, we must evict its provisional KV blocks BEFORE the block manager
reclaims them for another sequence.  Missing this step causes a use-after-free
of KV memory that silently corrupts outputs.

All other scheduling logic is delegated to the parent SGLang Scheduler.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from sglang.srt.managers.router.scheduler import Scheduler as _SGLangScheduler
    _SGLANG_AVAILABLE = True
except ImportError:
    _SGLANG_AVAILABLE = False

    class _SGLangScheduler:  # type: ignore[no-redef]
        """Minimal stub for unit testing."""
        def __init__(self, *args, **kwargs):
            self.radix_cache = None

        def preempt(self, seq_ids: List[int]) -> None:
            pass

        def schedule(self):
            return []


class Scheduler(_SGLangScheduler):
    """
    SGLang Scheduler with provisional KV eviction on preemption.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # radix_cache is set by the parent or by ModelRunner after init
        # We just ensure preempt() calls evict_provisional.

    def preempt(self, seq_ids: List[int]) -> None:
        """
        Called by the engine when sequences must be swapped out.

        We evict provisional KV entries BEFORE delegating to the parent so
        the block manager never sees those blocks as "in use" after preemption.
        """
        if hasattr(self, "radix_cache") and self.radix_cache is not None:
            for seq_id in seq_ids:
                freed = self.radix_cache.evict_provisional(seq_id)
                if freed:
                    logger.debug(
                        "Preempt: evicted %d provisional blocks for seq_id=%d",
                        len(freed), seq_id,
                    )

        super().preempt(seq_ids)  # type: ignore[misc]

    def on_sequence_finish(self, seq_id: int) -> None:
        """
        Called when a sequence reaches EOS.  Clean up any leftover provisional
        blocks (e.g., if EOS was sampled during a draft step).
        """
        if hasattr(self, "radix_cache") and self.radix_cache is not None:
            self.radix_cache.evict_provisional(seq_id)

        if hasattr(super(), "on_sequence_finish"):
            super().on_sequence_finish(seq_id)  # type: ignore[misc]
