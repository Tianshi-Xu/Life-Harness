"""Run shared H2–H5 hooks around a benchmark's original episode callbacks."""
import copy
import hashlib
import json
from pathlib import Path

from evolution import harness_hooks as hooks
from evolution.life_optimizer_v25.worker import check_action
from .adapter import EpisodeSetup, ModelResponse, StepResult


def case_id(task_id):
    return "case_" + hashlib.sha256(task_id.encode()).hexdigest()


def artifacts(manifest):
    root, records = hooks.manifest_records(manifest)
    pairs = [(root / row["artifact_path"], row["target_layer"]) for row in records]
    for path, layer in pairs:
        hooks.validate_source(path, layer)
    return pairs, {"manifest": hooks.digest(Path(manifest)),
                   **{str(path): hooks.digest(path) for path, _ in pairs}}


def json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def messages(values):
    if not isinstance(values, list) or any(not isinstance(m, dict) or m.get("role") not in
        {"system", "user", "assistant", "tool"} for m in values):
        raise ValueError("adapter must expose public chat messages")
    return json_copy(values)


def run_episode(adapter, task_id, split, pairs, ledger, scope):
    row = {"index": case_id(task_id), "native_task_id": task_id, "success": 0,
           "status": "infrastructure_error", "infrastructure_error": False,
           "turns": 0, "history": [], "calls": [], "harness_trace": []}
    episode = None
    try:
        episode = adapter.open_episode(task_id, split)
        bind_recorder = getattr(episode, "bind_request_recorder", None)
        if callable(bind_recorder):
            bind_recorder(lambda role, request: ledger.call(role, request, **scope, task_id=task_id))
        setup = episode.setup
        if not isinstance(setup, EpisodeSetup) or type(setup.max_turns) is not int or setup.max_turns < 1:
            raise ValueError("episode requires EpisodeSetup with a positive max_turns")
        row["turn_limit"] = setup.max_turns
        context = json_copy(setup.public_context)
        harness = hooks.CompositeHarness(pairs, adapter.name)
        harness.bind_task_context(context)
        row["public_task_context"] = context
        history = row["history"]

        def inject(items):
            for original in messages(items):
                conditioned = harness.h3_message(original)
                if original != conditioned:
                    row["harness_trace"].append({"layer": "h3", "turn": row["turns"],
                                                 "before": original, "after": conditioned})
                history.append(conditioned)

        inject(setup.messages)
        native_tools = json_copy(setup.tools)
        done = False
        for turn in range(1, setup.max_turns + 1):
            row["turns"] = turn
            remaining = setup.max_turns - turn + 1
            tools = harness.h3(native_tools)
            if tools != native_tools:
                row["harness_trace"].append({"layer": "h3", "turn": turn, "before": native_tools, "after": tools})
            layer = "h5" if turn == 1 else "h4"
            hints = harness.h5(history) if turn == 1 else harness.h4(history, remaining)
            view = copy.deepcopy(history)
            if hints:
                prefix = harness.h5_message_prefix if turn == 1 else harness.h4_message_prefix
                hint = {"role": "user", "content": prefix + "\n".join(hints)}
                if turn > 1 and harness.h4_persistent:
                    inject([hint])
                    view = copy.deepcopy(history)
                else:
                    view.append(hint)
                row["harness_trace"].append({"layer": layer, "turn": turn, "hints": hints})
            view = harness.h3_messages(view)
            call = {"turn": turn, "remaining": remaining, "request": {"messages": view, "tools": tools}}
            row["calls"].append(call)
            request_id = ledger.start("solver", **scope, task_id=task_id, turn=turn)
            try:
                response = episode.complete(copy.deepcopy(view), copy.deepcopy(tools))
                if not isinstance(response, ModelResponse):
                    raise TypeError("complete must return ModelResponse")
            except Exception as exc:
                ledger.finish(request_id, None, error_type=type(exc).__name__)
                raise
            ledger.finish(request_id, response.usage, usage_source=response.usage_source)
            raw = messages([response.message])[0]
            if raw["role"] != "assistant":
                raise ValueError("solver response must have role assistant")
            call["response"] = {"choices": [{"message": raw}], "usage": response.usage}
            repaired = harness.h2(raw, history)
            check_action(raw, repaired, tools)
            if raw != repaired:
                row["harness_trace"].append({"layer": "h2", "turn": turn, "before": raw, "after": copy.deepcopy(repaired)})
            control = repaired.pop("_harness_control", None)
            inject([raw if control or harness.h2_preserve_raw_history else repaired])
            if control:
                inject(control["messages"])
                continue
            if not repaired.get("tool_calls"):
                nudge = harness.h2_no_tool_message(repaired, remaining)
                if nudge:
                    inject([{"role": "user", "content": nudge}])
                    continue
            inject(harness.drain_h2_messages())
            result = episode.execute(copy.deepcopy(repaired))
            if not isinstance(result, StepResult) or type(result.done) is not bool:
                raise TypeError("execute must return StepResult")
            inject(result.messages)
            done = result.done
            if result.done:
                break
        success = episode.grade()
        if type(success) not in (bool, int) or success not in (0, 1):
            raise ValueError("native grade must be binary success")
        row.update(success=int(success), status="completed" if done else "turn_limit")
    except Exception as exc:
        row.update(success=0, status="infrastructure_error", infrastructure_error=True,
                   error_type=type(exc).__name__)
    finally:
        if episode is not None:
            try:
                episode.close()
            except Exception as exc:
                row.update(success=0, status="infrastructure_error", infrastructure_error=True,
                           error_type=type(exc).__name__)
    return row
