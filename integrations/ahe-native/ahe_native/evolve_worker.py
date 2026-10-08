"""Execute AHE's ORIGINAL Evolve Agent invocation inside an isolated filesystem."""
import argparse
import importlib.util
import json
import sys
import subprocess
import uuid
from pathlib import Path

from .connection import bind_connections
from .records import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    sys.path.insert(0, "/upstream")
    spec = importlib.util.spec_from_file_location("ahe_original_evolve", "/upstream/evolve.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    bind_connections(module)
    if request.get("mode") == "debugger":
        original_run = subprocess.run
        def observe_cli(command, *positional, **kwargs):
            result = original_run(command, *positional, **kwargs)
            if isinstance(command, list) and len(command) > 1 and Path(command[0]).name == "adb" and command[1] == "ask":
                write_json(Path(request["iteration_dir"]) / "debugger/subprocess_diagnostics" / (uuid.uuid4().hex + ".json"),
                    {"returncode": result.returncode, "stdout": result.stdout,
                     "stderr": result.stderr, "question": command[command.index("-q") + 1]})
            return result
        subprocess.run = observe_cli  # diagnostic observation; original result is unchanged
        result = module.run_parallel_adb_ask(request["config"], Path(request["job_dir"]),
            request["task_results"], Path(request["iteration_dir"]), request["iteration"],
            timeout_tasks=set(request["timeout_tasks"]), k=request["k"])
    else:
        result = module.run_evolve_agent(request["config"], Path(request["experiment"]),
            request["iteration"], request["query"], Path(request["job_dir"]),
            Path(request["iteration_dir"]))
    write_json(args.out, {"result": result})


if __name__ == "__main__":
    main()
