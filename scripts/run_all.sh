#!/usr/bin/env bash
# scripts/run_all.sh
# ─────────────────────────────────────────────────────────────────────────────
# Full end-to-end pipeline:
#   1. Apply SGLang patches
#   2. Run unit tests
#   3. Start SGLang (spec decode) + vLLM (baseline) in background
#   4. Wait for both servers
#   5. Run wall-clock benchmark
#   6. Run acceptance rate vs temperature sweep
#   7. Print final summary
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ── Read ports from config (Python is the single source of truth) ──────────
eval "$(python3 -c "
import sys; sys.path.insert(0, '.')
from sglang_patch.config_loader import get_server_port, get_benchmark_config
cfg = get_benchmark_config()
print(f'SGLANG_PORT={get_server_port(\"sglang\")}')
print(f'VLLM_PORT={get_server_port(\"vllm\")}')
print(f'TIMEOUT={cfg.get(\"server_ready_timeout_s\", 300)}')
print(f'OUT_DIR=\"{cfg[\"output_dir\"]}\"')
")"

SGLANG_URL="http://localhost:$SGLANG_PORT"
VLLM_URL="http://localhost:$VLLM_PORT"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   SGLang Speculative Decode — Full Pipeline                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Unit tests ───────────────────────────────────────────────────────
echo "▶ Step 1: Running unit tests"
python3 -m pytest tests/ -v --tb=short
echo "  ✓ All tests passed"
echo ""

# ── Step 2: Apply patches ────────────────────────────────────────────────────
echo "▶ Step 2: Applying SGLang patches"
python3 scripts/apply_patches.py
echo ""

# ── Step 3: Start servers in background ──────────────────────────────────────
echo "▶ Step 3: Starting servers"
mkdir -p logs

bash scripts/launch_sglang.sh > logs/sglang_stdout.log 2>&1 &
SGLANG_PID=$!
echo "  SGLang PID=$SGLANG_PID  (log: logs/sglang_stdout.log)"

bash scripts/launch_vllm.sh > logs/vllm_stdout.log 2>&1 &
VLLM_PID=$!
echo "  vLLM   PID=$VLLM_PID   (log: logs/vllm_stdout.log)"

# Cleanup on exit
trap "echo 'Stopping servers...'; kill $SGLANG_PID $VLLM_PID 2>/dev/null || true" EXIT
echo ""

# ── Step 4: Wait for servers ──────────────────────────────────────────────────
echo "▶ Step 4: Waiting for servers (timeout=${TIMEOUT}s)"
python3 - <<PYEOF
import sys, time, requests
urls = [("SGLang", "$SGLANG_URL"), ("vLLM", "$VLLM_URL")]
deadline = time.time() + $TIMEOUT
for label, url in urls:
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                print(f"  ✓ {label} ready")
                break
        except:
            time.sleep(3)
    else:
        print(f"  ✗ {label} not ready after ${TIMEOUT}s")
        sys.exit(1)
PYEOF
echo ""

# ── Step 5: Wall-clock benchmark ─────────────────────────────────────────────
echo "▶ Step 5: Running wall-clock benchmark"
python3 benchmarks/run_benchmark.py \
    --sglang-url "$SGLANG_URL" \
    --vllm-url   "$VLLM_URL"
echo ""

# ── Step 6: Acceptance rate vs temperature sweep ──────────────────────────────
echo "▶ Step 6: Running acceptance rate sweep"
python3 benchmarks/acceptance_vs_temp.py \
    --sglang-url "$SGLANG_URL"
echo ""

# ── Step 7: Summary ───────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Done.  Results saved to: $OUT_DIR"
echo "╚══════════════════════════════════════════════════════════════╝"
ls -lh "$OUT_DIR"/ 2>/dev/null || echo "  (output dir not yet created)"
