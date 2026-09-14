#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
task=${1:?Usage: worker.sh alfworld|dbbench [profile]}
profile=${2:-$task-std}
case "$task" in
  dbbench) python_bin=.native/core/bin/python; port=15022 ;;
  alfworld) python_bin=.native/alfworld-venv/bin/python; port=15021 ;;
  *) echo 'Only verified native transports are exposed here.' >&2; exit 2 ;;
esac
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ALFWORLD_DATA="$PWD/data/alfworld"
exec "$python_bin" -m agentrl.worker -c ".native/configs/$task.yaml" \
  --controller http://127.0.0.1:15020/api --self "http://127.0.0.1:$port/api" --host 127.0.0.1 --port "$port" "$profile"
