#!/usr/bin/env bash

set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPWORLD_ROOT="${APPWORLD_ROOT:-$(cd "$HARNESS_ROOT/../.." && pwd)/appworld}"
POLICY_DIRECTORY="${POLICY_DIRECTORY:-$HARNESS_ROOT/policies/v005}"
TASK_IDS_FILE="${TASK_IDS_FILE:?Set TASK_IDS_FILE to a newline-delimited task list}"
DATASET="${DATASET:-train}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME for this subset run}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-simplified_function_calling_agent/local/qwen3-4b/train}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
BASELINE="${BASELINE:-}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PREDICTOR_TEMPERATURE="${PREDICTOR_TEMPERATURE:-$TEMPERATURE}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
MODEL_SERVER_URL="${MODEL_SERVER_URL:-http://127.0.0.1:8000}"
NO_API_KEY="${NO_API_KEY:-EMPTY}"
OPENAI_API_KEY="${OPENAI_API_KEY:-$NO_API_KEY}"
RUN_LOG_DIRECTORY="${RUN_LOG_DIRECTORY:-$HARNESS_ROOT/artifacts/run_logs/$(basename "$EXPERIMENT_NAME")}"

if [[ ! -f "$TASK_IDS_FILE" ]]; then
    echo "Task ID list does not exist: $TASK_IDS_FILE" >&2
    exit 1
fi

mapfile -t task_ids < <(rg -v '^\s*(#|$)' "$TASK_IDS_FILE")
if (( ${#task_ids[@]} == 0 )); then
    echo "Task ID list is empty: $TASK_IDS_FILE" >&2
    exit 1
fi

mkdir -p "$RUN_LOG_DIRECTORY"
export APPWORLD_ROOT MODEL_SERVER_URL NO_API_KEY OPENAI_API_KEY MODEL_NAME TEMPERATURE PREDICTOR_TEMPERATURE DATASET
export PYTHONPATH="$HARNESS_ROOT/src:$APPWORLD_ROOT/experiments${PYTHONPATH:+:$PYTHONPATH}"

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM

for ((index = 0; index < NUM_PROCESSES; index++)); do
    log_file="$RUN_LOG_DIRECTORY/process_${index}.log"
    (
        "$APPWORLD_ROOT/.venv/bin/python" -m life_harness_appworld.runner \
            --appworld-root "$APPWORLD_ROOT" \
            --policy-directory "$POLICY_DIRECTORY" \
            --task-ids-file "$TASK_IDS_FILE" \
            --base-experiment "$BASE_EXPERIMENT" \
            --model-name "$MODEL_NAME" \
            --temperature "$TEMPERATURE" \
            --api-predictor-temperature "$PREDICTOR_TEMPERATURE" \
            --experiment-name "$EXPERIMENT_NAME" \
            --num-processes "$NUM_PROCESSES" \
            --process-index "$index" \
            ${BASELINE:+--baseline}
    ) >"$log_file" 2>&1 &
    pids+=("$!")
done

failures=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "process $index failed; inspect $RUN_LOG_DIRECTORY/process_${index}.log" >&2
        failures=$((failures + 1))
    fi
done

if ((failures > 0)); then
    echo "$failures of $NUM_PROCESSES processes failed" >&2
    exit 1
fi

if [[ "${SKIP_EVALUATE:-0}" != "1" ]]; then
    for task_id in "${task_ids[@]}"; do
        "$APPWORLD_ROOT/.venv/bin/appworld" evaluate \
            "$EXPERIMENT_NAME" --task-id "$task_id" --root "$APPWORLD_ROOT" || true
    done
fi
