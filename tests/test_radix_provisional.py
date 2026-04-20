"""
tests/test_radix_provisional.py
──────────────────────────────────
Unit tests for the RadixCache provisional KV management.

These tests run WITHOUT SGLang installed.  They verify the correctness of:
  • insert_provisional
  • commit_provisional (partial and full)
  • evict_provisional
  • Error handling (double-insert, out-of-range accepted_len)

Run:
    python -m pytest tests/test_radix_provisional.py -v
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from sglang_patch.managers.router.radix_cache import RadixCache, _ProvisionalNode


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _RecordingCache(RadixCache):
    """RadixCache subclass that records insert/free calls for assertions."""

    def __init__(self):
        # Skip super().__init__ — we only need the provisional layer
        self._provisional = {}
        self._inserted: list = []   # (tokens, ptrs) tuples
        self._freed: list = []      # flat list of freed ptr ints

    def insert(self, tokens, kv_ptrs):
        self._inserted.append((list(tokens), list(kv_ptrs)))

    def free_blocks(self, kv_ptrs):
        self._freed.extend(kv_ptrs)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInsertProvisional:
    def test_basic_insert(self):
        cache = _RecordingCache()
        cache.insert_provisional(
            seq_id=1,
            tokens=[10, 20, 30, 40],
            kv_ptrs=[100, 101, 102, 103],
            base_len=5,
        )
        assert 1 in cache._provisional
        node = cache._provisional[1]
        assert node.tokens == [10, 20, 30, 40]
        assert node.kv_ptrs == [100, 101, 102, 103]
        assert node.base_len == 5

    def test_length_mismatch_raises(self):
        cache = _RecordingCache()
        with pytest.raises(ValueError, match="same length"):
            cache.insert_provisional(
                seq_id=2,
                tokens=[1, 2, 3],
                kv_ptrs=[10, 20],   # wrong length
                base_len=0,
            )

    def test_double_insert_without_commit_raises(self):
        cache = _RecordingCache()
        cache.insert_provisional(seq_id=3, tokens=[1], kv_ptrs=[10], base_len=0)
        with pytest.raises(RuntimeError, match="uncommitted"):
            cache.insert_provisional(seq_id=3, tokens=[2], kv_ptrs=[11], base_len=0)

    def test_insert_after_commit_succeeds(self):
        cache = _RecordingCache()
        cache.insert_provisional(seq_id=4, tokens=[1, 2], kv_ptrs=[10, 11], base_len=0)
        cache.commit_provisional(seq_id=4, accepted_len=2)
        # Should not raise
        cache.insert_provisional(seq_id=4, tokens=[3, 4], kv_ptrs=[12, 13], base_len=2)


class TestCommitProvisional:
    def test_full_accept(self):
        """All K tokens accepted → all KV ptrs committed, nothing freed."""
        cache = _RecordingCache()
        cache.insert_provisional(
            seq_id=1, tokens=[10, 20, 30, 40], kv_ptrs=[100, 101, 102, 103], base_len=5
        )
        freed = cache.commit_provisional(seq_id=1, accepted_len=4)

        assert freed == []
        assert cache._inserted == [([10, 20, 30, 40], [100, 101, 102, 103])]
        assert cache._freed == []
        assert 1 not in cache._provisional

    def test_partial_accept(self):
        """Accept 2 of 4 → commit 2, free 2."""
        cache = _RecordingCache()
        cache.insert_provisional(
            seq_id=1, tokens=[10, 20, 30, 40], kv_ptrs=[100, 101, 102, 103], base_len=5
        )
        freed = cache.commit_provisional(seq_id=1, accepted_len=2)

        assert freed == [102, 103]
        assert cache._inserted == [([10, 20], [100, 101])]
        assert set(cache._freed) == {102, 103}
        assert 1 not in cache._provisional

    def test_zero_accept(self):
        """Accept 0 tokens → nothing committed, all KV ptrs freed."""
        cache = _RecordingCache()
        cache.insert_provisional(
            seq_id=1, tokens=[10, 20, 30], kv_ptrs=[100, 101, 102], base_len=0
        )
        freed = cache.commit_provisional(seq_id=1, accepted_len=0)

        assert freed == [100, 101, 102]
        assert cache._inserted == []
        assert set(cache._freed) == {100, 101, 102}

    def test_out_of_range_raises(self):
        cache = _RecordingCache()
        cache.insert_provisional(seq_id=1, tokens=[1, 2], kv_ptrs=[10, 11], base_len=0)
        with pytest.raises(ValueError, match="out of range"):
            cache.commit_provisional(seq_id=1, accepted_len=5)

    def test_missing_node_returns_empty(self):
        cache = _RecordingCache()
        freed = cache.commit_provisional(seq_id=999, accepted_len=0)
        assert freed == []


class TestEvictProvisional:
    def test_evict_frees_all_blocks(self):
        cache = _RecordingCache()
        cache.insert_provisional(
            seq_id=1, tokens=[10, 20, 30, 40], kv_ptrs=[100, 101, 102, 103], base_len=0
        )
        freed = cache.evict_provisional(seq_id=1)

        assert freed == [100, 101, 102, 103]
        assert set(cache._freed) == {100, 101, 102, 103}
        assert 1 not in cache._provisional

    def test_evict_missing_seq_returns_empty(self):
        cache = _RecordingCache()
        freed = cache.evict_provisional(seq_id=42)
        assert freed == []

    def test_evict_after_commit_returns_empty(self):
        cache = _RecordingCache()
        cache.insert_provisional(seq_id=1, tokens=[1], kv_ptrs=[10], base_len=0)
        cache.commit_provisional(seq_id=1, accepted_len=1)
        freed = cache.evict_provisional(seq_id=1)
        assert freed == []


class TestDiagnostics:
    def test_provisional_counts(self):
        cache = _RecordingCache()
        cache.insert_provisional(seq_id=1, tokens=[1, 2, 3], kv_ptrs=[10, 11, 12], base_len=0)
        cache.insert_provisional(seq_id=2, tokens=[4, 5], kv_ptrs=[13, 14], base_len=0)

        assert cache.num_provisional_sequences() == 2
        assert cache.provisional_block_count() == 5

        cache.commit_provisional(seq_id=1, accepted_len=3)
        assert cache.num_provisional_sequences() == 1
        assert cache.provisional_block_count() == 2

    def test_clear_all_provisional(self):
        cache = _RecordingCache()
        cache.insert_provisional(seq_id=1, tokens=[1, 2], kv_ptrs=[10, 11], base_len=0)
        cache.insert_provisional(seq_id=2, tokens=[3, 4], kv_ptrs=[12, 13], base_len=0)

        cache.clear_all_provisional()

        assert cache.num_provisional_sequences() == 0
        assert set(cache._freed) == {10, 11, 12, 13}


class TestCriticalInvariant:
    """
    The critical correctness invariant:

    When spec decoding rejects at position k, commit_provisional(accepted_len=k)
    must free the KV blocks for positions k..K-1.  These tests verify
    that no block is lost or double-freed.
    """

    def test_rejection_at_position_1_of_4(self):
        """Reject at k=1: accept tokens[0], free tokens[1,2,3]."""
        cache = _RecordingCache()
        K = 4
        tokens = [10, 20, 30, 40]
        ptrs = [100, 101, 102, 103]

        cache.insert_provisional(seq_id=1, tokens=tokens, kv_ptrs=ptrs, base_len=0)
        freed = cache.commit_provisional(seq_id=1, accepted_len=1)

        assert freed == [101, 102, 103], "Must free exactly the rejected tail"
        assert cache._inserted == [([10], [100])], "Must commit only accepted tokens"
        assert set(cache._freed) == {101, 102, 103}

    def test_no_blocks_leaked_across_multiple_sequences(self):
        """Every allocated block must end up either committed or freed."""
        cache = _RecordingCache()
        seqs = {
            1: ([10, 11, 12], [100, 101, 102]),
            2: ([20, 21, 22, 23], [200, 201, 202, 203]),
            3: ([30], [300]),
        }
        for seq_id, (toks, ptrs) in seqs.items():
            cache.insert_provisional(seq_id=seq_id, tokens=toks, kv_ptrs=ptrs, base_len=0)

        # Partial accept seq 1: accept 2 of 3
        cache.commit_provisional(seq_id=1, accepted_len=2)
        # Full accept seq 2
        cache.commit_provisional(seq_id=2, accepted_len=4)
        # Evict seq 3 (preempted)
        cache.evict_provisional(seq_id=3)

        committed_ptrs = set()
        for toks, ptrs in cache._inserted:
            committed_ptrs.update(ptrs)
        freed_ptrs = set(cache._freed)

        all_ptrs = {100, 101, 102, 200, 201, 202, 203, 300}
        assert committed_ptrs | freed_ptrs == all_ptrs, "No blocks leaked"
        assert committed_ptrs & freed_ptrs == set(), "No block double-counted"
