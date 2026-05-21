"""Analyze tau2 results.json and export trajectory diagnostics.

Usage:
    uv run python scripts/analyze_results.py \
        --results data/simulations/<run>/results.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reusable diagnostics for a tau2 results.json file."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results.json",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for analysis files. Default: "
            "<results_dir>/analysis_<results_stem>"
        ),
    )
    parser.add_argument(
        "--max-msg-chars",
        type=int,
        default=500,
        help="Maximum chars per message snippet in trajectory export.",
    )
    return parser.parse_args()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    calls = msg.get("tool_calls")
    return calls if isinstance(calls, list) else []


def _build_task_report(
    sim: dict[str, Any],
    task_meta: dict[str, Any] | None,
    max_msg_chars: int,
) -> tuple[str, dict[str, Any]]:
    task_id = str(sim.get("task_id"))
    reward_info = sim.get("reward_info") or {}
    messages = sim.get("messages") or []

    tool_call_names: list[str] = []
    tool_errors: list[dict[str, Any]] = []
    trajectory_lines: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        turn_idx = msg.get("turn_idx")
        if role == "assistant":
            for tc in _extract_tool_calls(msg):
                name = tc.get("name", "")
                if name:
                    tool_call_names.append(name)
                arguments = _safe_text(tc.get("arguments"))
                trajectory_lines.append(
                    f"- turn {turn_idx} assistant -> tool `{name}` args: "
                    f"`{_trim(arguments, max_msg_chars)}`"
                )
        elif role == "tool":
            content = _safe_text(msg.get("content"))
            is_error = bool(msg.get("error"))
            if is_error:
                tool_errors.append(
                    {
                        "turn_idx": turn_idx,
                        "content": content,
                    }
                )
            trajectory_lines.append(
                f"- turn {turn_idx} tool (error={is_error}) "
                f"`{_trim(content, max_msg_chars)}`"
            )
        elif role in {"user", "assistant"}:
            content = _safe_text(msg.get("content"))
            trajectory_lines.append(
                f"- turn {turn_idx} {role}: `{_trim(content, max_msg_chars)}`"
            )

    action_checks = reward_info.get("action_checks") or []
    failed_actions = [
        a
        for a in action_checks
        if not bool(a.get("action_match"))
    ]
    failed_nl = [
        item
        for item in (reward_info.get("nl_assertions") or [])
        if not bool(item.get("met"))
    ]
    failed_comm = [
        item
        for item in (reward_info.get("communicate_checks") or [])
        if not bool(item.get("met"))
    ]

    report_lines: list[str] = []
    report_lines.append(f"# Task {task_id} Analysis")
    report_lines.append("")
    report_lines.append(f"- termination_reason: `{sim.get('termination_reason')}`")
    report_lines.append(f"- reward: `{reward_info.get('reward')}`")
    report_lines.append(
        f"- reward_breakdown: `{_safe_text(reward_info.get('reward_breakdown'))}`"
    )
    report_lines.append(f"- total_messages: `{len(messages)}`")
    report_lines.append("")

    if task_meta:
        report_lines.append("## Task Intent")
        report_lines.append("")
        scenario = ((task_meta.get("user_scenario") or {}).get("instructions") or {})
        report_lines.append(f"- reason_for_call: `{_safe_text(scenario.get('reason_for_call'))}`")
        report_lines.append(
            f"- task_instructions: `{_trim(_safe_text(scenario.get('task_instructions')), 1200)}`"
        )
        report_lines.append("")

    report_lines.append("## Evaluation Failures")
    report_lines.append("")
    if not failed_actions and not failed_nl and not failed_comm:
        report_lines.append("- No failed checks.")
    else:
        for a in failed_actions:
            action = a.get("action") or {}
            report_lines.append(
                f"- failed action: `{action.get('name')}` "
                f"args=`{_safe_text(action.get('arguments'))}`"
            )
        for item in failed_nl:
            report_lines.append(
                f"- failed nl_assertion: `{_safe_text(item.get('nl_assertion'))}`; "
                f"reason: `{_trim(_safe_text(item.get('justification')), 500)}`"
            )
        for item in failed_comm:
            report_lines.append(
                f"- failed communicate_check: `{_safe_text(item.get('info'))}`; "
                f"reason: `{_trim(_safe_text(item.get('justification')), 500)}`"
            )
    report_lines.append("")

    report_lines.append("## Tool Usage")
    report_lines.append("")
    call_counter = Counter(tool_call_names)
    if call_counter:
        for name, count in sorted(call_counter.items(), key=lambda x: (-x[1], x[0])):
            report_lines.append(f"- `{name}`: {count}")
    else:
        report_lines.append("- No tool calls")
    report_lines.append("")

    report_lines.append("## Tool Errors")
    report_lines.append("")
    if tool_errors:
        for err in tool_errors:
            report_lines.append(
                f"- turn {err['turn_idx']}: `{_trim(err['content'], 900)}`"
            )
    else:
        report_lines.append("- No tool errors")
    report_lines.append("")

    report_lines.append("## Trajectory (Condensed)")
    report_lines.append("")
    report_lines.extend(trajectory_lines)
    report_lines.append("")

    report_text = "\n".join(report_lines)
    stats = {
        "task_id": task_id,
        "reward": reward_info.get("reward"),
        "failed_action_count": len(failed_actions),
        "failed_nl_count": len(failed_nl),
        "failed_communicate_count": len(failed_comm),
        "tool_error_count": len(tool_errors),
        "tool_calls": dict(call_counter),
    }
    return report_text, stats


def main() -> int:
    args = parse_args()
    results_path = Path(args.results)
    data = json.loads(results_path.read_text(encoding="utf-8"))

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else results_path.parent / f"analysis_{results_path.stem}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = data.get("tasks") or []
    task_meta_by_id = {str(t.get("id")): t for t in tasks}
    simulations = data.get("simulations") or []

    index_lines: list[str] = []
    index_lines.append("# Results Analysis")
    index_lines.append("")
    index_lines.append(f"- source: `{results_path}`")
    index_lines.append(f"- simulations: `{len(simulations)}`")
    index_lines.append("")
    index_lines.append("## Per-task reports")
    index_lines.append("")

    stats_list: list[dict[str, Any]] = []
    for sim in simulations:
        task_id = str(sim.get("task_id"))
        report_text, stats = _build_task_report(
            sim=sim,
            task_meta=task_meta_by_id.get(task_id),
            max_msg_chars=args.max_msg_chars,
        )
        stats_list.append(stats)
        report_name = f"task_{task_id}.md"
        (out_dir / report_name).write_text(report_text, encoding="utf-8")
        index_lines.append(f"- task `{task_id}`: `{report_name}`")

    (out_dir / "summary.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "source": str(results_path),
                "num_simulations": len(simulations),
                "tasks": stats_list,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Analysis written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
