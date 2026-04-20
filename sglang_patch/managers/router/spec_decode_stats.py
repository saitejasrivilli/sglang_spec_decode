"""
sglang_patch/managers/router/spec_decode_stats.py
──────────────────────────────────────────────────
Thread-safe running statistics for speculative decoding.
Imported by model_runner.py to track acceptance rate, tokens/step, etc.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _StepRecord:
    """One batch's worth of spec-decode statistics."""
    wall_time_s: float
    K: int
    n_accepted: int        # tokens accepted this step (0 ≤ n_accepted ≤ K+1)
    batch_size: int
    temperature: float


class SpecDecodeStats:
    """
    Running statistics for speculative decoding.

    Thread-safe (one Lock).  All heavy math is done lazily in properties.
    """

    def __init__(self, K: int, log_interval: int = 100):
        """
        Parameters
        ----------
        K            : number of draft tokens per step (from experiment.yaml)
        log_interval : emit a log line every N steps (0 = disabled)
        """
        self._K = K
        self._log_interval = log_interval
        self._lock = Lock()

        # Counters
        self._total_draft_tokens: int = 0
        self._total_accepted_tokens: int = 0
        self._total_steps: int = 0
        self._total_wall_time_s: float = 0.0

        # Per-temperature accumulation for the sweep analysis
        self._by_temperature: Dict[float, Dict[str, int]] = {}

        # Recent history for rolling window (last 1000 steps)
        self._history: List[_StepRecord] = []
        self._history_maxlen = 1000

        self._t_start = time.perf_counter()

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        K: int,
        n_accepted: int,
        batch_size: int = 1,
        wall_time_s: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        """
        Record the results of one speculative decode step.

        Parameters
        ----------
        K          : draft tokens proposed this step
        n_accepted : draft tokens accepted (0 ≤ n_accepted ≤ K)
                     NOTE: the +1 bonus token is counted separately below
        batch_size : number of sequences in the batch
        wall_time_s: wall-clock time for this step
        temperature: sampling temperature (used for sweep analysis)
        """
        with self._lock:
            self._total_draft_tokens += K * batch_size
            self._total_accepted_tokens += n_accepted
            self._total_steps += 1
            self._total_wall_time_s += wall_time_s

            # Per-temperature tracking
            t = round(temperature, 2)
            if t not in self._by_temperature:
                self._by_temperature[t] = {"draft": 0, "accepted": 0, "steps": 0}
            self._by_temperature[t]["draft"] += K * batch_size
            self._by_temperature[t]["accepted"] += n_accepted
            self._by_temperature[t]["steps"] += 1

            # Rolling history
            rec = _StepRecord(
                wall_time_s=wall_time_s,
                K=K,
                n_accepted=n_accepted,
                batch_size=batch_size,
                temperature=temperature,
            )
            self._history.append(rec)
            if len(self._history) > self._history_maxlen:
                self._history.pop(0)

            # Periodic logging
            if self._log_interval > 0 and self._total_steps % self._log_interval == 0:
                self._log_summary()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate since instantiation."""
        with self._lock:
            if self._total_draft_tokens == 0:
                return 0.0
            return self._total_accepted_tokens / self._total_draft_tokens

    @property
    def avg_tokens_per_step(self) -> float:
        """
        Expected accepted tokens per decode step.
        Theoretical: 1 + acceptance_rate * K
        """
        return 1.0 + self.acceptance_rate * self._K

    @property
    def rolling_acceptance_rate(self, window: int = 200) -> float:
        """Acceptance rate over the last `window` steps."""
        with self._lock:
            recent = self._history[-window:]
            if not recent:
                return 0.0
            draft = sum(r.K * r.batch_size for r in recent)
            accepted = sum(r.n_accepted for r in recent)
            return accepted / draft if draft > 0 else 0.0

    @property
    def total_steps(self) -> int:
        with self._lock:
            return self._total_steps

    @property
    def total_wall_time_s(self) -> float:
        with self._lock:
            return self._total_wall_time_s

    def acceptance_rate_at_temperature(self, temperature: float) -> Optional[float]:
        """Acceptance rate for a specific temperature (for sweep charts)."""
        t = round(temperature, 2)
        with self._lock:
            data = self._by_temperature.get(t)
            if data is None or data["draft"] == 0:
                return None
            return data["accepted"] / data["draft"]

    def temperature_sweep_data(self) -> Dict[float, float]:
        """Returns {temperature: acceptance_rate} for all observed temperatures."""
        with self._lock:
            result = {}
            for t, data in self._by_temperature.items():
                if data["draft"] > 0:
                    result[t] = data["accepted"] / data["draft"]
            return dict(sorted(result.items()))

    # ── Serialization (for API response) ─────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "total_steps": self._total_steps,
                "total_draft_tokens": self._total_draft_tokens,
                "total_accepted_tokens": self._total_accepted_tokens,
                "acceptance_rate": self.acceptance_rate,
                "avg_tokens_per_step": self.avg_tokens_per_step,
                "total_wall_time_s": self._total_wall_time_s,
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _log_summary(self) -> None:
        """Called inside the lock."""
        ar = (self._total_accepted_tokens / self._total_draft_tokens
              if self._total_draft_tokens > 0 else 0.0)
        avg_tps = 1.0 + ar * self._K
        elapsed = time.perf_counter() - self._t_start
        logger.info(
            "[SpecDecode] step=%d  acceptance_rate=%.3f  avg_tokens/step=%.2f  "
            "elapsed=%.1fs",
            self._total_steps, ar, avg_tps, elapsed,
        )
