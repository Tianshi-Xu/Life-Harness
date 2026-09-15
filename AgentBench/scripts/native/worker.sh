#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
task=${1:?Usage: worker.sh alfworld|dbbench|webshop [profile]}
profile=${2:-$task-std}
case "$task" in
  dbbench)
    python_bin=.native/core/bin/python
    port=15022
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    ;;
  alfworld)
    python_bin=.native/alfworld-venv/bin/python
    port=15021
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    ;;
  webshop)
    python_bin=.native/webshop-venv/bin/python
    port=15023
    if [[ -n "${WEBSHOP_JAVA_HOME:-}" ]]; then
      export JAVA_HOME="$WEBSHOP_JAVA_HOME"
    elif [[ -x "$PWD/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64/bin/java" ]]; then
      export JAVA_HOME="$PWD/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64"
    elif [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/java" ]]; then
      export JAVA_HOME
    elif command -v java >/dev/null 2>&1; then
      export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
    else
      echo 'WebShop requires Java 11. Set WEBSHOP_JAVA_HOME or install a JDK.' >&2
      exit 2
    fi
    export PATH="$JAVA_HOME/bin:$PATH"
    export _JAVA_OPTIONS="${_JAVA_OPTIONS:--Xmx4g}"
    export PYTHONPATH="$PWD/.native/webshop-src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
    ;;
  *) echo 'Only verified native transports are exposed here.' >&2; exit 2 ;;
esac
export ALFWORLD_DATA="$PWD/data/alfworld"
exec "$python_bin" -m agentrl.worker -c ".native/configs/$task.yaml" \
  --controller http://127.0.0.1:15020/api --self "http://127.0.0.1:$port/api" --host 127.0.0.1 --port "$port" "$profile"
