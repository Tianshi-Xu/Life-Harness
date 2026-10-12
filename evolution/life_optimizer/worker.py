"""Disposable, resource-limited hook replay. Never evaluates task success."""
import builtins
import copy
import importlib
import json
from pathlib import Path
import resource
import sys
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolution.agentbench_hooks import runtime
from evolution.agentbench_hooks.contract import SAFE_IMPORTS


def subset(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and subset(v, actual[k]) for k, v in expected.items())
    return expected == actual


def check_action(raw, value, tools):
    if not isinstance(value, dict) or value.get("role") != "assistant":
        raise ValueError("H2 must return an assistant dictionary")
    control = value.get("_harness_control")
    if control:
        if not isinstance(control, dict) or control.get("action") != "suppress_tool":
            raise ValueError("invalid H2 veto")
        return
    known = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    for call in value.get("tool_calls") or []:
        # Frozen malformed solver responses are allowed to remain unchanged;
        # preserving one is not evidence that it would execute successfully.
        if any(call == old for old in raw.get("tool_calls") or []):
            continue
        if not isinstance(call.get("id"), str) or call.get("type", "function") != "function":
            raise ValueError("invalid tool-call envelope")
        function = call["function"]
        if function["name"] not in known or not isinstance(function["arguments"], str):
            raise ValueError("unknown tool or non-string arguments")
        arguments = json.loads(function["arguments"])
        schema = known[function["name"]]
        check_schema(arguments, schema)


def check_schema(value, schema):
    expected = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool, "null": type(None)}
    if expected == "number":
        valid = type(value) in (int, float)
    else:
        valid = expected not in types or type(value) is types[expected]
    if not valid or ("enum" in schema and value not in schema["enum"]):
        raise ValueError("repaired argument type/enum mismatch")
    if isinstance(value, dict):
        if not set(schema.get("required", [])) <= set(value):
            raise ValueError("repaired arguments miss required fields")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError("repaired arguments introduce fields")
        for key in set(value) & set(properties):
            check_schema(value[key], properties[key])
    if isinstance(value, list) and "items" in schema:
        for item in value:
            check_schema(item, schema["items"])


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
    module_count = [0]

    def load(path, target):
        # Match the real runtime: every new episode reloads each artifact,
        # not just Harness instances with reused module/class globals.
        module = ModuleType("life_replay_" + str(module_count[0]))
        module_count[0] += 1
        sys.modules[module.__name__] = module
        module.__dict__["__builtins__"] = {**vars(builtins), "print": lambda *a, **k: None}
        exec(codes[path], module.__dict__)
        return module

    runtime.load_artifact = load
    pairs = [(r["name"], r["layer"]) for r in payload["artifacts"]]

    def fresh(context):
        harness = runtime.CompositeHarness(pairs, payload["domain"])
        harness.bind_task_context(copy.deepcopy(context))
        return harness

    replayed, results = 0, []
    for case in payload["cases"]:
        try:
            harness = fresh(case["context"])
            calls = case["calls"]
            tools = harness.h3(copy.deepcopy(case["tools"]))
            for ordinal, call in enumerate(calls):
                messages = [copy.deepcopy(case["message_pool"][i]) for i in call["message_indices"]]
                for message in messages:
                    harness.h3_message(message)
                remaining = max(0, {"alfworld": 50, "dbbench": 15, "webshop": 20}[payload["domain"]] - ordinal)
                if ordinal == 0:
                    harness.h5(messages)
                else:
                    harness.h4(messages, remaining)
                raw = copy.deepcopy(call["assistant"])
                repaired = harness.h2(raw, messages)
                check_action(raw, repaired, tools)
                harness.h2_no_tool_message(repaired, remaining)
                harness.drain_h2_messages()
                replayed += 1
            results.append({"case_id": case["case_id"], "passed": True, "calls": len(calls)})
        except Exception as exc:
            results.append({"case_id": case["case_id"], "passed": False, "error": f"{type(exc).__name__}: {exc}"[:1000]})
    fixtures = []
    for test in payload["tests"]:
        try:
            harness = fresh(test.get("context"))
            tools = test.get("tools", [])
            for step in test["steps"]:
                args = copy.deepcopy(step["args"])
                actual = getattr(harness, step["hook"])(*args)
                if step["hook"] == "h2":
                    check_action(step["args"][0], actual, tools)
                if "expected" in step and actual != step["expected"]:
                    raise ValueError("hook output does not equal expected")
                if "expected_subset" in step and not subset(step["expected_subset"], actual):
                    raise ValueError("hook output does not match expected_subset")
                if "expected_state" in step and not subset(step["expected_state"], harness.shared_state):
                    raise ValueError("shared state does not match expected_state")
            fixtures.append({"test_id": test["test_id"], "passed": True})
        except Exception as exc:
            fixtures.append({"test_id": test["test_id"], "passed": False, "error": f"{type(exc).__name__}: {exc}"[:1000]})
    return {"passed": all(r["passed"] for r in results + fixtures), "calls_replayed": replayed,
            "cases": results, "tests": fixtures, "scope": "frozen-input hook replay, not environment rollout or score"}


if __name__ == "__main__":
    try:
        result = main(json.load(sys.stdin))
    except Exception as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:1000]}
    print(json.dumps(result, ensure_ascii=False))
