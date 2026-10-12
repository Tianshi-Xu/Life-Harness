"""No-model four-hook contract replay, independent of training pool size."""
import copy
import json


def _public_context(domain, index, evaluator):
    if domain != "dbbench":
        return None
    path = evaluator.AB / "data/dbbench/db_out_new.jsonl"
    entry = json.loads(path.read_text().splitlines()[index])
    return {
        key: copy.deepcopy(entry[key])
        for key in ("description", "type", "table", "evidence", "add_description")
        if key in entry
    }


def preflight(candidate, domain, run_root, evaluator):
    manifest = evaluator.MANIFEST
    mod = evaluator.validate_plugin(candidate)
    for idx in manifest[domain]["indices"]:
        row = json.loads(
            (run_root / domain / "evals/baseline" / f"episode_{idx}.json").read_text()
        )
        harness = mod.Harness()
        bind = getattr(harness, "bind_task_context", None)
        if callable(bind):
            bind(_public_context(domain, idx, evaluator))

        calls = row["calls"]
        tools = copy.deepcopy(calls[0]["request"]["tools"])
        before = copy.deepcopy(tools)
        after = harness.h3(tools)
        for item in before:
            item["function"].pop("description", None)
        clean = copy.deepcopy(after)
        for item in clean:
            item["function"].pop("description", None)
        assert before == clean, "Changed tool names or parameter schema"

        for turn, call in enumerate(calls):
            messages = copy.deepcopy(call["request"]["messages"])
            hints = (
                harness.h5(messages)
                if turn == 0
                else harness.h4(
                    messages, (50 if domain == "alfworld" else 15) - turn
                )
            )
            word_counts = [
                len(item.split()) if isinstance(item, str) else str(type(item))
                for item in hints
            ]
            assert (
                isinstance(hints, list)
                and len(hints) <= 1
                and all(isinstance(item, str) and len(item.split()) <= 120 for item in hints)
            ), (
                f"Hint budget/schema exceeded at sample {idx}, turn {turn + 1}: "
                f"word counts {word_counts}"
            )
            raw = {
                key: value
                for key, value in call["response"]["choices"][0]["message"].items()
                if key in ("role", "content", "tool_calls") and value is not None
            }
            repaired = harness.h2(copy.deepcopy(raw), messages)
            assert isinstance(repaired, dict) and repaired.get("role") == "assistant"
            for tool_call in repaired.get("tool_calls", []) or []:
                assert isinstance(tool_call["id"], str)
                assert isinstance(tool_call["function"]["arguments"], str)
                known_name = tool_call["function"]["name"] in [
                    tool["function"]["name"] for tool in after
                ]
                preserved_name = any(
                    old.get("id") == tool_call["id"]
                    and old.get("function", {}).get("name")
                    == tool_call["function"]["name"]
                    for old in raw.get("tool_calls", []) or []
                )
                assert known_name or preserved_name, "Introduced an unknown tool name"
    return {
        "baseline_episodes_replayed": len(manifest[domain]["indices"]),
        "syntax_and_ast": "pass",
        "h3_schema": "preserved",
        "hint_budget": "pass",
        "public_context_bound": domain == "dbbench",
    }
