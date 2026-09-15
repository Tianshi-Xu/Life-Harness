#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

source_repo="$PWD/data/webshop_repo"
runtime="$PWD/.native/webshop-src"
data_dir="$source_repo/data"
download_dir="$PWD/.native/downloads"
data_base="${WEBSHOP_DATA_BASE:-https://hf-mirror.com/datasets/sparklabutah/timewarp-env-data/resolve/main/webshop}"
spacy_url="${WEBSHOP_SPACY_WHEEL_URL:-https://hf-mirror.com/spacy/en_core_web_lg/resolve/main/en_core_web_lg-any-py3-none-any.whl}"
spacy_wheel="$download_dir/en_core_web_lg-3.7.1-py3-none-any.whl"
spacy_sha256=ab70aeb6172cde82508f7739f35ebc9918a3d07debeed637403c8f794ba3d3dc

if [[ ! -d "$source_repo/.git" && ! -f "$source_repo/.git" ]]; then
  git submodule update --init -- data/webshop_repo
fi
expected_commit=$(git ls-files -s data/webshop_repo | awk '{print $2}')
actual_commit=$(git -C "$source_repo" rev-parse HEAD)
if [[ -z "$expected_commit" || "$actual_commit" != "$expected_commit" ]]; then
  echo "WebShop source must be at the pinned gitlink: expected=$expected_commit actual=$actual_commit" >&2
  exit 2
fi

mkdir -p "$data_dir" "$download_dir"
for name in items_shuffle.json items_ins_v2.json items_human_ins.json; do
  target="$data_dir/$name"
  if [[ ! -f "$target" ]]; then
    echo "Downloading $name"
    curl --fail --location --retry 8 --retry-all-errors --continue-at - \
      --output "$target" "$data_base/$name"
  fi
done

marker="$runtime/.life-harness-source"
if [[ ! -f "$marker" ]]; then
  if [[ -e "$runtime" && -n "$(find "$runtime" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Existing unmarked WebShop runtime found at $runtime; move it aside before rebuilding." >&2
    exit 2
  fi
  mkdir -p "$runtime"
  git -C "$source_repo" archive HEAD | tar -x -C "$runtime"
  patch --directory="$runtime" --strip=1 < src/server/tasks/webshop/webshop.patch
  printf '%s\n' "$actual_commit" > "$marker"
fi
if [[ "$(cat "$marker")" != "$actual_commit" ]]; then
  echo "WebShop runtime source does not match the pinned submodule commit." >&2
  exit 2
fi
if [[ ! -L "$runtime/data" ]]; then
  if [[ -e "$runtime/data" ]]; then
    echo "$runtime/data exists but is not the expected data symlink." >&2
    exit 2
  fi
  ln -s ../../data/webshop_repo/data "$runtime/data"
fi

bootstrap_python="${WEBSHOP_PYTHON:-$PWD/.native/core/bin/python}"
if [[ ! -x "$bootstrap_python" ]]; then
  bootstrap_python=$(command -v python3)
fi
if [[ ! -x "$PWD/.native/webshop-venv/bin/python" ]]; then
  "$bootstrap_python" -m venv --system-site-packages .native/webshop-venv
fi
webshop_python="$PWD/.native/webshop-venv/bin/python"
"$webshop_python" -m pip install -r scripts/native/requirements-webshop.txt
if ! "$webshop_python" -c 'import torch' >/dev/null 2>&1; then
  echo 'The text environment imports torch. Set WEBSHOP_PYTHON to a Python with torch installed.' >&2
  exit 2
fi
if ! "$webshop_python" -c 'import spacy; spacy.load("en_core_web_lg")' >/dev/null 2>&1; then
  if [[ ! -f "$spacy_wheel" ]]; then
    curl --fail --location --retry 8 --retry-all-errors \
      --output "$spacy_wheel" "$spacy_url"
  fi
  printf '%s  %s\n' "$spacy_sha256" "$spacy_wheel" | sha256sum --check --status
  "$webshop_python" -m pip install "$spacy_wheel"
fi

if [[ -n "${WEBSHOP_JAVA_HOME:-}" ]]; then
  java_home="$WEBSHOP_JAVA_HOME"
elif [[ -x "$PWD/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64/bin/java" ]]; then
  java_home="$PWD/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64"
elif [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/java" ]]; then
  java_home="$JAVA_HOME"
elif command -v java >/dev/null 2>&1; then
  java_home=$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")
else
  echo 'WebShop requires Java 11. Set WEBSHOP_JAVA_HOME or install a JDK.' >&2
  exit 2
fi
java_version=$("$java_home/bin/java" -version 2>&1 | head -1)
if [[ "$java_version" != *'"11.'* ]]; then
  echo "WebShop/Pyserini requires Java 11; found: $java_version" >&2
  exit 2
fi

export JAVA_HOME="$java_home"
export PATH="$JAVA_HOME/bin:$PATH"
export _JAVA_OPTIONS="${_JAVA_OPTIONS:--Xmx4g}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="$runtime:$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$webshop_python" scripts/native/build_webshop_100k.py
.native/core/bin/python scripts/native/prepare_configs.py
"$webshop_python" scripts/native/smoke_webshop.py
