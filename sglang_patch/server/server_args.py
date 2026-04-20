"""
sglang_patch/server/server_args.py
───────────────────────────────────
Drop this file over sglang/srt/server_args.py  (or import and extend).

Adds speculative-decoding fields to ServerArgs without hardcoding any
model name, path, or numeric constant — all values come from experiment.yaml
when not provided explicitly on the command line.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Optional

# ── Import base SGLang ServerArgs ─────────────────────────────────────────────
# We monkey-patch rather than rewrite so we stay compatible with future
# SGLang releases.  Only the new fields + __post_init__ are added.
try:
    from sglang.srt.server_args import ServerArgs as _BaseServerArgs
except ImportError:
    # Allow the file to be imported for testing without SGLang installed
    import dataclasses as _dc

    @_dc.dataclass
    class _BaseServerArgs:  # type: ignore[no-redef]
        model_path: str = ""
        tokenizer_path: Optional[str] = None
        dtype: str = "bfloat16"
        tensor_parallel_size: int = 1
        port: int = 30000
        host: str = "0.0.0.0"
        max_num_seqs: int = 256
        disable_cuda_graph: bool = False

from sglang_patch.config_loader import (
    get_draft_model_path,
    get_num_speculative_tokens,
    get_acceptance_threshold,
)


@dataclasses.dataclass
class ServerArgs(_BaseServerArgs):
    """
    Extended ServerArgs with speculative-decoding support.

    CLI usage:
        python -m sglang.launch_server \\
            --model-path  <target>  \\
            --draft-model-path <draft>  \\
            --num-speculative-tokens 4

    If --draft-model-path is omitted, speculative decoding is disabled.
    If --num-speculative-tokens is omitted, value is read from experiment.yaml.
    """

    # ── New fields ────────────────────────────────────────────────────────────
    draft_model_path: Optional[str] = None
    """HF repo ID or local path to the draft (small) model."""

    num_speculative_tokens: int = dataclasses.field(default_factory=get_num_speculative_tokens)
    """K: number of draft tokens to propose each step."""

    spec_decode_acceptance_threshold: float = dataclasses.field(
        default_factory=get_acceptance_threshold
    )
    """
    Minimum acceptance probability.  0.0 = standard lossless sampling.
    Setting > 0 makes decoding lossy but can improve acceptance rate.
    """

    draft_dtype: Optional[str] = None
    """dtype override for draft model.  Defaults to same as target model."""

    spec_decode_log_interval: int = 100
    """Log acceptance-rate stats every N decode steps (0 = disabled)."""

    # ── Post-init resolution ──────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # Call parent post_init if it exists
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore[misc]

        # Resolve draft_dtype fallback
        if self.draft_dtype is None:
            self.draft_dtype = self.dtype

        # Validate K
        if self.num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {self.num_speculative_tokens}"
            )

        # Validate threshold
        if not (0.0 <= self.spec_decode_acceptance_threshold < 1.0):
            raise ValueError(
                "spec_decode_acceptance_threshold must be in [0, 1), "
                f"got {self.spec_decode_acceptance_threshold}"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def use_spec_decode(self) -> bool:
        """True iff the draft model path was provided."""
        return self.draft_model_path is not None

    @classmethod
    def add_cli_args(cls, parser) -> None:  # type: ignore[override]
        """Register speculative-decoding flags with argparse."""
        try:
            super().add_cli_args(parser)  # type: ignore[misc]
        except AttributeError:
            pass  # base class may not have this

        group = parser.add_argument_group("Speculative Decoding")
        group.add_argument(
            "--draft-model-path",
            type=str,
            default=None,
            help=(
                "Path or HF repo ID for the draft (small) model. "
                "Omit to disable speculative decoding."
            ),
        )
        group.add_argument(
            "--num-speculative-tokens",
            type=int,
            default=get_num_speculative_tokens(),
            help="Number of draft tokens to propose per step (default: from experiment.yaml).",
        )
        group.add_argument(
            "--spec-decode-acceptance-threshold",
            type=float,
            default=get_acceptance_threshold(),
            help="Minimum acceptance probability (0 = lossless).",
        )
        group.add_argument(
            "--draft-dtype",
            type=str,
            default=None,
            choices=["bfloat16", "float16", "float32"],
            help="dtype for draft model weights (default: same as target).",
        )
        group.add_argument(
            "--spec-decode-log-interval",
            type=int,
            default=100,
            help="Log acceptance-rate stats every N steps (0 = disabled).",
        )
