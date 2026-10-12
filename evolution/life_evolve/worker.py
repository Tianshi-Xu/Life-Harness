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
from evolution import harness_hooks as runtime
from evolution.agentbench_hooks_v31.contract import SAFE_IMPORTS
from evolution.life_optimizer_v25.worker import check_action
from evolution.environment_state import StateReader


def main(payload):
    if "public_context_fields" in payload:
        runtime.register_domain(payload["domain"], payload["public_context_fields"])
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024**2, 384 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for module in SAFE_IMPORTS:
        importlib.import_module(module)
    codes = {r["name"]: compile(r["source"], r.get("artifact", r["name"]), "exec") for r in payload["artifacts"]}
    locations = {r.get("artifact", r["name"]): r["name"] for r in payload["artifacts"]}

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

    replayed, results, previews = 0, [], []
    for case in payload["cases"]:
        stage, ordinal = "bind_context_and_environment_state", None
        try:
            states = case.get("environment_states", [])
            reader = StateReader(states)
            current_index = case.get("initial_environment_state_index")
            current_state = reader.get(current_index)
            harness = fresh(case["context"], current_state)
            stage = "h3"
            tools = harness.h3(copy.deepcopy(case["tools"]))
            for ordinal, call in enumerate(case["calls"]):
                stage = "bind_environment_state"
                index = call.get("environment_state_index")
                if index != current_index:
                    current_state, current_index = reader.get(index), index
                harness.bind_environment_state(current_state)
                messages = [copy.deepcopy(case["message_pool"][i]) for i in call["message_indices"]]
                stage = "h3_message"
                for message in messages:
                    harness.h3_message(message)
                # Recorded public budget wins. Fallback matches existing host
                # budgets; replay does not evaluate environment success.
                limit = case.get("turn_limit", {"alfworld": 50, "dbbench": 15, "webshop": 20, "os": 16}.get(payload["domain"], 200))
                remaining = call.get("remaining", max(0, limit - ordinal))
                if ordinal == 0:
                    stage = "h5"
                    guidance = harness.h5(messages)
                else:
                    stage = "h4"
                    guidance = harness.h4(messages, remaining)
                stage = "h3_messages"
                model_view = harness.h3_messages(messages)
                raw = copy.deepcopy(call["assistant"])
                stage = "bind_action_environment_state"
                index = call.get("action_environment_state_index")
                if index != current_index:
                    current_state, current_index = reader.get(index), index
                harness.bind_environment_state(current_state)
                stage = "h2"
                repaired = harness.h2(raw, messages)
                stage = "h2_action_schema"
                check_action(raw, repaired, tools)
                stage = "h2_no_tool_message"
                nudge = harness.h2_no_tool_message(repaired, remaining)
                stage = "drain_h2_messages"
                side_messages = harness.drain_h2_messages()
                replayed += 1
                if ordinal == payload.get('preview_call_index'):
                    previews.append({'case_id': case['case_id'], 'call_index': ordinal, 'remaining': remaining,
                        'guidance_hook': 'h5' if ordinal == 0 else 'h4', 'guidance': guidance,
                        'model_view_changes': [{'message_index': index, 'before_content': before.get('content'),
                            'after_content': after.get('content')} for index, (before, after)
                            in enumerate(zip(messages, model_view)) if before != after],
                        'tool_description_changes': [{'name': before['function']['name'],
                            'before': before['function'].get('description'), 'after': after['function'].get('description')}
                            for before, after in zip(case['tools'], tools)
                            if before['function'].get('description') != after['function'].get('description')],
                        'action_before': raw, 'action_after': repaired, 'no_tool_nudge': nudge,
                        'side_messages': side_messages,
                        'shared_state_summary': {key: {'type': type(value).__name__,
                            'value': value[:200] if isinstance(value, str) else value,
                            **({'length': len(value), 'truncated': len(value) > 200} if isinstance(value, str) else {})}
                            if value is None or type(value) in (str, bool, int) else {'type': type(value).__name__}
                            for key, value in harness.shared_state.items() if isinstance(key, str)}})
                    break
            results.append({"case_id": case["case_id"], "passed": True,
                            "calls": ordinal + 1 if payload.get('preview_call_index') is not None else len(case["calls"])})
        except Exception as exc:
            detail = {"case_id": case["case_id"], "passed": False, "call_index": ordinal,
                      "hook": stage, "error": f"{type(exc).__name__}: {exc}"[:2000]}
            detail['related_artifacts'] = [r.get('artifact', r['name']) for r in payload['artifacts']
                                          if stage.startswith(r['layer']) or stage.startswith('bind_')]
            trace = exc.__traceback__
            while trace is not None:
                filename = trace.tb_frame.f_code.co_filename
                if filename in locations:
                    detail.update(artifact=filename, mechanism_id=locations[filename], line=trace.tb_lineno)
                # Contract failures occur just after a hook returns. Locate the
                # responsible implementation without dumping frame locals/data.
                instance = trace.tb_frame.f_locals.get("instance")
                method = type(instance).__dict__.get(stage) if instance is not None else None
                code = getattr(method, "__code__", None)
                if code is not None and code.co_filename in locations and "artifact" not in detail:
                    detail.update(artifact=code.co_filename, mechanism_id=locations[code.co_filename],
                                  line=code.co_firstlineno)
                trace = trace.tb_next
            results.append(detail)
    report = {"passed": all(r["passed"] for r in results), "calls_replayed": replayed,
              "cases": results, "scope": "frozen-input hook replay, not environment rollout or score"}
    if 'preview_call_index' in payload:
        report['previews'] = previews
    return report


if __name__ == "__main__":
    try:
        result = main(json.load(sys.stdin))
    except Exception as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:1000]}
    print(json.dumps(result, ensure_ascii=False))
