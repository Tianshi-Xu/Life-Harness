"""Report whether this checkout has the local assets needed for new evaluations.

The committed results can be audited without these runtime assets. This check
does not contact model endpoints or inspect credential contents.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AGENTBENCH = ROOT / "AgentBench"
TAUBENCH = ROOT / "TauBench"

TAU_PYTHON = TAUBENCH / ".venv/bin/python"
CORE_PYTHON = AGENTBENCH / ".native/core/bin/python"
ALFWORLD_PYTHON = AGENTBENCH / ".native/alfworld-venv/bin/python"
WEBSHOP_PYTHON = AGENTBENCH / ".native/webshop-venv/bin/python"


def has_packages(python: Path, packages: tuple[str, ...]) -> bool:
    if not python.is_file():
        return False
    code = "import importlib.util,sys;sys.exit(not all(importlib.util.find_spec(p) for p in sys.argv[1:]))"
    try:
        result = subprocess.run(
            [str(python), "-c", code, *packages],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def can_import(python: Path, package: str) -> bool:
    if not python.is_file():
        return False
    try:
        result = subprocess.run(
            [str(python), "-c", f"import {package}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def has_java11() -> bool:
    candidates = [AGENTBENCH / ".native/sysroot/usr/lib/jvm/java-11-openjdk-amd64/bin/java"]
    if os.environ.get("WEBSHOP_JAVA_HOME"):
        candidates.append(Path(os.environ["WEBSHOP_JAVA_HOME"]) / "bin/java")
    if os.environ.get("JAVA_HOME"):
        candidates.append(Path(os.environ["JAVA_HOME"]) / "bin/java")
    if shutil.which("java"):
        candidates.append(Path(shutil.which("java")))
    for java in candidates:
        if not java.is_file():
            continue
        try:
            result = subprocess.run(
                [str(java), "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and '"11.' in result.stderr:
            return True
    return False


def runtime_checks() -> dict[str, list[bool]]:
    tau = has_packages(TAU_PYTHON, ("tau2", "litellm", "numpy")) and can_import(TAU_PYTHON, "tau2")
    core = has_packages(CORE_PYTHON, ("agentrl", "mysql", "yaml"))
    alfworld = has_packages(ALFWORLD_PYTHON, ("alfworld", "textworld", "numpy"))
    webshop = has_packages(WEBSHOP_PYTHON, ("gym", "pyserini", "spacy"))
    return {
        "airline": [tau],
        "retail": [tau],
        "telecom": [tau],
        "alfworld": [alfworld],
        "dbbench": [core, (AGENTBENCH / ".native/sysroot/usr/sbin/mysqld").is_file()],
        "webshop": [webshop, (AGENTBENCH / ".native/webshop-src/.life-harness-source").is_file(), has_java11()],
    }


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    runtimes = runtime_checks()
    print("Domain     Frozen data       Runtime checks  Local file status")
    for domain in ("airline", "retail", "telecom", "alfworld", "dbbench", "webshop"):
        if domain in ("airline", "retail", "telecom"):
            data = [TAUBENCH / f"data/tau2/domains/{domain}/tasks.json"]
        elif domain == "webshop":
            data = [AGENTBENCH / name for name in manifest["data_hashes"] if name.startswith(".native/webshop")]
        else:
            data = [AGENTBENCH / name for name in manifest["data_hashes"] if name.startswith(f"data/{domain}/")]
        present_data = sum(path.is_file() for path in data)
        runtime = runtimes[domain]
        present_runtime = sum(runtime)
        ready_files = present_data == len(data) and all(runtime)
        print(f"{domain:10} {present_data:>4}/{len(data):<4} "
              f"{present_runtime:>4}/{len(runtime):<4}       "
              f"{'ready for service check' if ready_files else 'incomplete'}")
    for name in ("alfworld_repo", "webshop_repo"):
        path = AGENTBENCH / "data" / name
        print(f"{name}: {'initialized' if (path / '.git').exists() else 'not initialized'}")
    print("Service health, model availability, and credentials require separate checks.")


if __name__ == "__main__":
    main()
