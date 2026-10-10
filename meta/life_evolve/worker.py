"""Shared six-domain offline replay; reuse frozen safety/action validation."""
import builtins
import copy
import importlib
import json
from pathlib import Path
import resource
import sys
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meta import harness_hooks as runtime
from meta.agentbench_hooks_v31.contract import SAFE_IMPORTS
from meta.life_optimizer_v25.worker import check_action
from meta.environment_state import StateReader


def main(payload):
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024**2, 384 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for module in SAFE_IMPORTS:
        importlib.import_module(module)
    codes = {r["name"]: compile(r["source"], r["name"], "exec") for r in payload["artifacts"]}

    def audit(event, args):
        if (event == "open" or event.startswith(("os.", "socket.", "subprocess.", "ctypes.", "shutil."))
                or event in {"sys.addaudithook", "sys.settrace", "sys.setprofile", "resource.setrlimit"}):
            raise PermissionError("offline replay denies file/network/process access")

    sys.addaudithook(audit)
    counter = [0]

    def load(path, target):
        module = ModuleType("life_shared_replay_" + str(counter[0]))
        counter[0] += 1
        sys.modules[module.__name__] = module
        module.__dict__["__builtins__"] = {**vars(builtins), "print": lambda *a, **k: None}
        exec(codes[path], module.__dict__)
        return module

    runtime.legacy.load_artifact = load  # Isolated process, not historical source.
    pairs = [(r["name"], r["layer"]) for r in payload["artifacts"]]

    def fresh(context, state=None):
        harness = runtime.CompositeHarness(pairs, payload["domain"])
        harness.bind_task_context(copy.deepcopy(context))
        harness.bind_environment_state(state)
        return harness

    replayed, results = 0, []
    for case in payload["cases"]:
        try:
            states = case.get("environment_states", [])
            reader = StateReader(states)
            current_index = case.get("initial_environment_state_index")
            current_state = reader.get(current_index)
            harness = fresh(case["context"], current_state)
            tools = harness.h3(copy.deepcopy(case["tools"]))
            for ordinal, call in enumerate(case["calls"]):
                index = call.get("environment_state_index")
                if index != current_index:
                    current_state, current_index = reader.get(index), index
                harness.bind_environment_state(current_state)
                messages = [copy.deepcopy(case["message_pool"][i]) for i in call["message_indices"]]
                for message in messages:
                    harness.h3_message(message)
                # Recorded public budget wins. Fallback matches existing host
                # budgets; replay does not evaluate environment success.
                limit = case.get("turn_limit", {"alfworld": 50, "dbbench": 15, "webshop": 20, "os": 16}.get(payload["domain"], 200))
                remaining = call.get("remaining", max(0, limit - ordinal))
                if ordinal == 0:
                    harness.h5(messages)
                else:
                    harness.h4(messages, remaining)
                harness.h3_messages(messages)
                raw = copy.deepcopy(call["assistant"])
                index = call.get("action_environment_state_index")
                if index != current_index:
                    current_state, current_index = reader.get(index), index
                harness.bind_environment_state(current_state)
                repaired = harness.h2(raw, messages)
                check_action(raw, repaired, tools)
                harness.h2_no_tool_message(repaired, remaining)
                harness.drain_h2_messages()
                replayed += 1
            results.append({"case_id": case["case_id"], "passed": True, "calls": len(case["calls"])})
        except Exception as exc:
            results.append({"case_id": case["case_id"], "passed": False, "error": f"{type(exc).__name__}: {exc}"[:1000]})
    return {"passed": all(r["passed"] for r in results), "calls_replayed": replayed,
            "cases": results, "scope": "frozen-input hook replay, not environment rollout or score"}


if __name__ == "__main__":
    try:
        result = main(json.load(sys.stdin))
    except Exception as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:1000]}
    print(json.dumps(result, ensure_ascii=False))
