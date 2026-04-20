#!/usr/bin/env bash
# scripts/launch_sglang.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launch SGLang with speculative decoding enabled.
# All parameters are read from configs/cluster.yaml and configs/experiment.yaml.
# No model names, GPU indices, or ports are hardcoded here.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Parse config values via Python (single source of truth) ─────────────────
eval "$(python3 - <<'PYEOF'
import sys
sys.path.insert(0, ''"$PROJECT_ROOT"'')
from sglang_patch.config_loader import (
    get_target_model_path,
    get_draft_model_path,
    get_num_speculative_tokens,
    get_server_port,
    get_tensor_parallel_size,
    load_cluster_config,
    load_experiment_config,
)

cluster = load_cluster_config()
exp = load_experiment_config()

print(f'TARGET_MODEL="{get_target_model_path()}"')
print(f'DRAFT_MODEL="{get_draft_model_path()}"')
print(f'K={get_num_speculative_tokens()}')
print(f'PORT={get_server_port("sglang")}')
print(f'TP={get_tensor_parallel_size("sglang")}')
print(f'DTYPE="{exp["models"]["target"]["dtype"]}"')
print(f'LOG_DIR="{exp["logging"]["log_dir"]}"')
PYEOF
)"

mkdir -p "$PROJECT_ROOT/$LOG_DIR"
LOG_FILE="$PROJECT_ROOT/$LOG_DIR/sglang_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================"
echo "  Launching SGLang with speculative decoding"
echo "  target   : $TARGET_MODEL"
echo "  draft    : $DRAFT_MODEL"
echo "  K        : $K"
echo "  tp       : $TP"
echo "  port     : $PORT"
echo "  dtype    : $DTYPE"
echo "  log      : $LOG_FILE"
echo "======================================================"

# Apply patches (copy our extended modules over the SGLang source)
python3 "$SCRIPT_DIR/apply_patches.py"

# Launch
python3 -m sglang.launch_server \
    --model-path          "$TARGET_MODEL" \
    --draft-model-path    "$DRAFT_MODEL" \
    --num-speculative-tokens "$K" \
    --dtype               "$DTYPE" \
    --tp                  "$TP" \
    --port                "$PORT" \
    --host                "0.0.0.0" \
    2>&1 | tee "$LOG_FILE"
