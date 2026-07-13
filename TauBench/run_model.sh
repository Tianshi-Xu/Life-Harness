#!/usr/bin/env bash
set -euo pipefail
# Usage: ./run_model.sh <model_name>
# Runs 6 experiments: 3 domains x (baseline + harness), trials=3, concurrency=10

cd "$(dirname "$0")"
MODEL="$1"
AGENT="openai/${MODEL}"
TEACHER="openai/deepseek-v4-pro"
TAG=$(echo "$MODEL" | tr '.' '_')

echo "=== [$MODEL] start ==="

# ── airline baseline ──
echo "=== [$MODEL] airline baseline ==="
uv run python scripts/eval_harness.py \
  --domain airline --split test --trials 3 --concurrency 10 \
  --agent-llm "$AGENT" --user-llm "$TEACHER" \
  --output "airline/${TAG}-baseline"

# ── airline harness ──
echo "=== [$MODEL] airline harness ==="
uv run python scripts/eval_harness.py \
  --domain airline --split test --trials 3 --concurrency 10 \
  --agent-llm "$AGENT" --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 --h5-top-k 1 \
  --output "airline/${TAG}-harness"

# ── retail baseline ──
echo "=== [$MODEL] retail baseline ==="
uv run python scripts/eval_harness.py \
  --domain retail --split test --trials 3 --concurrency 10 --nl \
  --agent-llm "$AGENT" --user-llm "$TEACHER" \
  --output "retail/${TAG}-baseline"

# ── retail harness ──
echo "=== [$MODEL] retail harness ==="
uv run python scripts/eval_harness.py \
  --domain retail --split test --trials 3 --concurrency 10 --nl \
  --agent-llm "$AGENT" --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 \
  --output "retail/${TAG}-harness"

# ── telecom baseline ──
echo "=== [$MODEL] telecom baseline ==="
uv run python scripts/eval_harness.py \
  --domain telecom --split train --trials 3 --concurrency 10 \
  --agent-llm "$AGENT" --user-llm "$TEACHER" \
  --output "telecom/${TAG}-baseline"

# ── telecom harness ──
echo "=== [$MODEL] telecom harness ==="
uv run python scripts/eval_harness.py \
  --domain telecom --split train --trials 3 --concurrency 10 \
  --agent-llm "$AGENT" --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 --h5-top-k 1 \
  --output "telecom/${TAG}-harness"

echo "=== [$MODEL] all done ==="
