"""Offline regression replay for deterministic harness triggers.

This does not re-execute or modify AppWorld. It replays proposed actions and
recorded observations to estimate repairs, blocks, H4 triggers, and H5 coverage.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from life_harness_appworld.analysis.trajectory_analysis import (
    instruction_from_lm_calls,
    message_tool_calls,
    offered_tools,
    read_jsonl,
)
from life_harness_appworld.layers.h2_action_gate import H2ActionGate
from life_harness_appworld.layers.h4_trajectory_monitor import H4TrajectoryMonitor
from life_harness_appworld.layers.h5_skill_guidance import H5SkillGuidance
from life_harness_appworld.runtime_types import ToolCall
from appworld import update_root
from appworld.task import Task


def parse_call(raw_call: dict[str, Any]) -> ToolCall:
    function = raw_call.get("function", {})
    raw_arguments = function.get("arguments", "{}")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(
        call_id=str(raw_call.get("call_id", raw_call.get("id", "replay"))),
        name=str(function.get("name", "")),
        arguments=arguments,
    )


def h4_replay(calls: list[dict[str, Any]], config: dict[str, Any]) -> Counter[str]:
    monitor = H4TrajectoryMonitor(config)
    triggers: Counter[str] = Counter()
    for index, lm_call in enumerate(calls):
        raw_calls = message_tool_calls(lm_call)
        if not lm_call.get("input", {}).get("tools"):
            continue
        parsed = [parse_call(raw) for raw in raw_calls]
        outputs: list[str] = []
        if index + 1 < len(calls):
            next_messages = calls[index + 1].get("input", {}).get("messages") or []
            by_id = {
                str(message.get("tool_call_id")): str(message.get("content", ""))
                for message in next_messages
                if message.get("role") == "tool"
            }
            outputs = [by_id.get(call.call_id, "") for call in parsed]
        events = monitor.observe(
            step=index + 1,
            max_steps=50,
            calls=parsed,
            outputs=outputs,
            no_tool=not raw_calls,
        )
        triggers.update(trigger for trigger, _ in events)
    return triggers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--policy-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--appworld-root", type=Path, required=True)
    args = parser.parse_args()
    os.environ["APPWORLD_ROOT"] = str(args.appworld_root.resolve())
    update_root(str(args.appworld_root.resolve()))
    evaluation = json.loads(
        (args.baseline_output / "evaluations" / "train.json").read_text(encoding="utf-8")
    )["individual"]
    h2 = H2ActionGate(
        json.loads((args.policy_directory / "h2.json").read_text(encoding="utf-8"))
    )
    h4_config = json.loads(
        (args.policy_directory / "h4.json").read_text(encoding="utf-8")
    )
    h5 = H5SkillGuidance(args.policy_directory / "h5_skills.json", top_k=1)
    summary: dict[str, Counter[str]] = {
        "successful": Counter(),
        "failed": Counter(),
    }
    detail: list[dict[str, Any]] = []
    for task_dir in sorted((args.baseline_output / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        success = bool(evaluation.get(task_dir.name, {}).get("success", False))
        group = "successful" if success else "failed"
        calls = read_jsonl(task_dir / "logs" / "lm_calls.jsonl")
        task = Task.load(task_dir.name, load_ground_truth=False)
        environment_tools = {
            tool["function"]["name"]: tool["function"]
            for tool in task.api_docs.function_calling()
        }
        h2_repairs: Counter[str] = Counter()
        h2_blocks: Counter[str] = Counter()
        for lm_call in calls:
            offered = offered_tools(lm_call)
            if not offered:
                continue
            for raw_call in message_tool_calls(lm_call):
                function = raw_call.get("function", {})
                decision = h2.realize(
                    call_id=str(raw_call.get("call_id", raw_call.get("id", "replay"))),
                    name=str(function.get("name", "")),
                    raw_arguments=function.get("arguments", "{}"),
                    tools=environment_tools,
                )
                h2_repairs.update(decision.repairs)
                if decision.blocked:
                    h2_blocks.update([decision.feedback.split(".", 1)[0]])
        h4_triggers = h4_replay(calls, h4_config)
        skills = h5.retrieve(instruction_from_lm_calls(calls))
        summary[group]["tasks"] += 1
        summary[group]["tasks_with_h2_repairs"] += bool(h2_repairs)
        summary[group]["h2_repairs"] += sum(h2_repairs.values())
        summary[group]["tasks_with_h2_blocks"] += bool(h2_blocks)
        summary[group]["h2_blocks"] += sum(h2_blocks.values())
        summary[group]["tasks_with_h4_triggers"] += bool(h4_triggers)
        summary[group]["h4_triggers"] += sum(h4_triggers.values())
        summary[group]["tasks_with_h5_skills"] += bool(skills)
        detail.append(
            {
                "task_id": task_dir.name,
                "success": success,
                "h2_repairs": dict(h2_repairs),
                "h2_blocks": dict(h2_blocks),
                "h4_triggers": dict(h4_triggers),
                "h5_skills": [skill.id for skill in skills],
            }
        )
    output = {
        "summary": {key: dict(value) for key, value in summary.items()},
        "tasks": detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
