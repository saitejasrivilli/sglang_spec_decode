#!/usr/bin/env bash
# scripts/launch_vllm.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launch vLLM baseline server (no speculative decoding).
# All parameters come from configs/*.yaml — nothing hardcoded.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

eval "$(python3 - <<'PYEOF'
import sys
sys.path.insert(0, ''"$PROJECT_ROOT"'')
from sglang_patch.config_loader import (
    get_target_model_path,
    get_server_port,
    get_tensor_parallel_size,
    load_experiment_config,
)
exp = load_experiment_config()
print(f'TARGET_MODEL="{get_target_model_path()}"')
print(f'PORT={get_server_port("vllm")}')
print(f'TP={get_tensor_parallel_size("vllm")}')
print(f'DTYPE="{exp["models"]["target"]["dtype"]}"')
print(f'LOG_DIR="{exp["logging"]["log_dir"]}"')
PYEOF
)"

mkdir -p "$PROJECT_ROOT/$LOG_DIR"
LOG_FILE="$PROJECT_ROOT/$LOG_DIR/vllm_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================"
echo "  Launching vLLM baseline"
echo "  model    : $TARGET_MODEL"
echo "  tp       : $TP"
echo "  port     : $PORT"
echo "  dtype    : $DTYPE"
echo "  log      : $LOG_FILE"
echo "======================================================"

python3 -m vllm.entrypoints.openai.api_server \
    --model                "$TARGET_MODEL" \
    --dtype                "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --port                 "$PORT" \
    --host                 "0.0.0.0" \
    2>&1 | tee "$LOG_FILE"
