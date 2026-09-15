#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# TauBench 运行命令清单
# 手动执行,自行控制并发
# 先: cd TauBench
# Teacher (user simulator) 默认用 deepseek-flash
# ============================================================

cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
if [[ -n "${TAU_TEACHER_ENV_FILE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$TAU_TEACHER_ENV_FILE"
fi
TEACHER="${TAU_TEACHER_MODEL:-openai/deepseek-flash}"

# ── claude-opus-4-8 ──────────────────────────────────────────

# airline
uv run python scripts/eval_harness.py \
  --domain airline --split test --trials 3 \
  --agent-llm openai/claude-opus-4-8 --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 --h5-top-k 1 \
  --concurrency 10 \
  --output airline/claude-opus-4-8-harness

# retail
uv run python scripts/eval_harness.py \
  --domain retail --split test --trials 3 --nl \
  --agent-llm openai/claude-opus-4-8 --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 \
  --concurrency 10 \
  --output retail/claude-opus-4-8-harness

# telecom
uv run python scripts/eval_harness.py \
  --domain telecom --split train --trials 3 \
  --agent-llm openai/claude-opus-4-8 --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 --h5-top-k 1 \
  --concurrency 10 \
  --output telecom/claude-opus-4-8-harness

# ── gemini-3.1-pro-preview ───────────────────────────────────

# airline
uv run python scripts/eval_harness.py \
  --domain airline --split test --trials 3 \
  --agent-llm openai/gemini-3.1-pro-preview --user-llm "$TEACHER" \
  --h2 --h3 --h4 --h5 --h5-top-k 1 \
  --concurrency 10 \
  --output airline/gemini-3.1-pro-preview

# retail
uv run python scripts/eval_harness.py \
  --domain retail --split test --trials 3 --nl \
  --agent-llm openai/gemini-3.1-pro-preview --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 \
  --concurrency 10 \
  --output retail/gemini-3.1-pro-preview-harness

# telecom
uv run python scripts/eval_harness.py \
  --domain telecom --split train --trials 3 \
  --agent-llm openai/gemini-3.1-pro-preview --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 --h5-top-k 1 \
  --concurrency 10 \
  --output telecom/gemini-3.1-pro-preview-harness

# ── gpt-5.5 ──────────────────────────────────────────────────

# airline
uv run python scripts/eval_harness.py \
  --domain airline --split test --trials 3 \
  --agent-llm openai/gpt-5.5 --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 --h5-top-k 1 \
  --concurrency 10 \
  --output airline/gpt-5.5-harness

# retail
uv run python scripts/eval_harness.py \
  --domain retail --split test --trials 3 --nl \
  --agent-llm openai/gpt-5.5 --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 \
  --concurrency 10 \
  --output retail/gpt-5.5-harness

# telecom
uv run python scripts/eval_harness.py \
  --domain telecom --split train --trials 3 \
  --agent-llm openai/gpt-5.5 --user-llm "$TEACHER" \
  --enabled --h2 --h3 --h4 --h5 --h5-top-k 1 \
  --concurrency 10 \
  --output telecom/gpt-5.5-harness
