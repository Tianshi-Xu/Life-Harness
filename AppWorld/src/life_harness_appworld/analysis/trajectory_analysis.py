"""Analyze AppWorld trajectories without reading solution or private oracle data.

The analyzer consumes only runtime logs plus the already-produced success boolean
from the aggregate evaluation JSON.  It intentionally does not open task ground
truth, evaluation requirements, solution code, private data, or final databases.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ERROR_PATTERNS: dict[str, re.Pattern[str]] = {
    "traceback": re.compile(r"execution failed|traceback|exception", re.I),
    "validation": re.compile(r"validationerror|field required|missing required|required field", re.I),
    "unexpected_argument": re.compile(r"unexpected keyword|extra inputs? are not permitted", re.I),
    "authentication": re.compile(r"unauthorized|not authenticated|invalid (?:access )?token|login required", re.I),
    "not_found": re.compile(r"not found|does not exist|no .* found", re.I),
    "invalid_value": re.compile(r"invalid (?:value|argument|parameter)|must be (?:one of|a valid)", re.I),
    "state_conflict": re.compile(r"already (?:exists|followed|liked|completed|cancelled|deleted)", re.I),
}

EXHAUSTIVE_WORDS = re.compile(
    r"\b(all|every|each|most|highest|lowest|least|entire|complete list|how many|total)\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def message_tool_calls(call: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        calls = call["output"]["choices"][0]["message"].get("tool_calls") or []
    except (KeyError, IndexError, TypeError):
        return []
    return [item for item in calls if isinstance(item, dict)]


def offered_tools(call: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tools = call.get("input", {}).get("tools") or []
    output: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        if isinstance(name, str):
            output[name] = function
    return output


def instruction_from_lm_calls(calls: list[dict[str, Any]]) -> str:
    for call in calls:
        messages = call.get("input", {}).get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and "# Real Task Instruction" in content:
                match = re.search(
                    r"# Real Task Instruction\s*\n(.+?)(?:\n\nDisclaimer:|\n\n#|$)",
                    content,
                    re.S,
                )
                if match:
                    return match.group(1).strip()
    return ""


@dataclass
class TaskStats:
    task_id: str
    success: bool
    instruction: str = ""
    lm_calls: int = 0
    action_count: int = 0
    environment_error_count: int = 0
    environment_error_types: dict[str, int] = field(default_factory=dict)
    environment_error_fingerprints: dict[str, int] = field(default_factory=dict)
    no_tool_turns: int = 0
    malformed_json_arguments: int = 0
    non_object_arguments: int = 0
    unknown_tool_calls: int = 0
    missing_required_fields: int = 0
    unknown_argument_fields: int = 0
    string_integer_fields: int = 0
    consecutive_duplicate_actions: int = 0
    abab_loops: int = 0
    max_action_repetition: int = 0
    completion_calls: int = 0
    fail_submissions: int = 0
    missing_completion: bool = False
    max_step: int = 0
    budget_exhausted: bool = False
    paginated_tool_calls: int = 0
    single_page_exhaustive_candidates: int = 0
    action_names: list[str] = field(default_factory=list)
    schema_issues: list[str] = field(default_factory=list)


def error_fingerprint(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    candidates = [
        line
        for line in lines
        if re.search(r"(?:Error|Exception):|not found|does not exist|unauthorized|invalid", line, re.I)
        and not line.startswith("File ")
    ]
    line = candidates[-1] if candidates else (lines[-1] if lines else "unknown error")
    line = re.sub(r"eyJ[A-Za-z0-9_.-]+", "<TOKEN>", line)
    line = re.sub(r"[\w.+-]+@[\w.-]+", "<EMAIL>", line)
    line = re.sub(r"\b\d+\b", "<N>", line)
    line = re.sub(r"\s+", " ", line)
    return line[:300]


def analyze_task(task_dir: Path, success: bool) -> TaskStats:
    task_id = task_dir.name
    lm_calls = read_jsonl(task_dir / "logs" / "lm_calls.jsonl")
    logger_rows = read_jsonl(task_dir / "logs" / "logger.jsonl")
    api_calls = read_jsonl(task_dir / "logs" / "api_calls.jsonl")
    instruction = instruction_from_lm_calls(lm_calls)
    stats = TaskStats(
        task_id=task_id,
        success=success,
        instruction=instruction,
        lm_calls=len(lm_calls),
    )

    signatures: list[str] = []
    page_groups: dict[str, set[int]] = defaultdict(set)
    page_group_calls: Counter[str] = Counter()

    for call in lm_calls:
        offered = offered_tools(call)
        tools_present = bool(offered)
        tool_calls = message_tool_calls(call)
        if tools_present and not tool_calls:
            stats.no_tool_turns += 1
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            name = function.get("name")
            if not isinstance(name, str):
                stats.unknown_tool_calls += 1
                continue
            stats.action_names.append(name)
            raw_arguments = function.get("arguments", "{}")
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError:
                stats.malformed_json_arguments += 1
                arguments = {}
            if not isinstance(arguments, dict):
                stats.non_object_arguments += 1
                arguments = {}

            schema = offered.get(name)
            if schema is None:
                stats.unknown_tool_calls += 1
            else:
                parameters = schema.get("parameters", {})
                properties = parameters.get("properties", {})
                required = set(parameters.get("required", []))
                missing = required - set(arguments)
                unknown = set(arguments) - set(properties)
                stats.missing_required_fields += len(missing)
                stats.unknown_argument_fields += len(unknown)
                stats.schema_issues.extend(f"{name}:missing:{field}" for field in sorted(missing))
                stats.schema_issues.extend(f"{name}:unknown:{field}" for field in sorted(unknown))
                for field_name, field_value in arguments.items():
                    if (
                        properties.get(field_name, {}).get("type") == "integer"
                        and isinstance(field_value, str)
                        and re.fullmatch(r"-?\d+", field_value)
                    ):
                        stats.string_integer_fields += 1

                if "page_index" in properties:
                    stats.paginated_tool_calls += 1
                    group_arguments = {
                        key: value
                        for key, value in arguments.items()
                        if key not in {"page_index", "page_limit", "access_token"}
                    }
                    group_key = name + "|" + canonical(group_arguments)
                    page_index = arguments.get("page_index", 0)
                    if isinstance(page_index, int):
                        page_groups[group_key].add(page_index)
                    page_group_calls[group_key] += 1

            signature = name + "|" + canonical(arguments)
            signatures.append(signature)

    stats.action_count = len(signatures)
    stats.max_action_repetition = max(Counter(signatures).values(), default=0)
    stats.consecutive_duplicate_actions = sum(
        left == right for left, right in zip(signatures, signatures[1:], strict=False)
    )
    stats.abab_loops = sum(
        signatures[index] == signatures[index + 2]
        and signatures[index + 1] == signatures[index + 3]
        and signatures[index] != signatures[index + 1]
        for index in range(max(0, len(signatures) - 3))
    )

    if EXHAUSTIVE_WORDS.search(instruction):
        stats.single_page_exhaustive_candidates = sum(
            len(pages) == 1 and calls == 1
            for group, pages in page_groups.items()
            for calls in [page_group_calls[group]]
        )

    error_counts: Counter[str] = Counter()
    error_fingerprints: Counter[str] = Counter()
    last_action_name = "unknown_tool"
    for row in logger_rows:
        step = row.get("step_number")
        if isinstance(step, int):
            stats.max_step = max(stats.max_step, step)
        if row.get("role") == "agent":
            names = re.findall(r"\b([a-z][a-z0-9_]*__[a-z][a-z0-9_]*)\s*\(", str(row.get("content", "")))
            if names:
                last_action_name = names[-1]
            continue
        if row.get("role") != "environment":
            continue
        content = str(row.get("content", ""))
        matched = False
        for error_type, pattern in ERROR_PATTERNS.items():
            if pattern.search(content):
                error_counts[error_type] += 1
                matched = True
        if matched:
            stats.environment_error_count += 1
            error_fingerprints[f"{last_action_name} | {error_fingerprint(content)}"] += 1
    stats.environment_error_types = dict(error_counts)
    stats.environment_error_fingerprints = dict(error_fingerprints)

    for call in api_calls:
        if call.get("url") != "/supervisor/message":
            continue
        stats.completion_calls += 1
        if call.get("data", {}).get("status") == "fail":
            stats.fail_submissions += 1
    stats.missing_completion = stats.completion_calls == 0
    stats.budget_exhausted = stats.max_step >= 49 and stats.missing_completion
    return stats


def aggregate(task_stats: list[TaskStats]) -> dict[str, Any]:
    scalar_fields = [
        "lm_calls",
        "action_count",
        "environment_error_count",
        "no_tool_turns",
        "malformed_json_arguments",
        "non_object_arguments",
        "unknown_tool_calls",
        "missing_required_fields",
        "unknown_argument_fields",
        "string_integer_fields",
        "consecutive_duplicate_actions",
        "abab_loops",
        "fail_submissions",
        "paginated_tool_calls",
        "single_page_exhaustive_candidates",
    ]
    boolean_fields = ["missing_completion", "budget_exhausted"]

    def summarize(group: list[TaskStats]) -> dict[str, Any]:
        return {
            "tasks": len(group),
            **{field: sum(getattr(item, field) for item in group) for field in scalar_fields},
            **{field: sum(bool(getattr(item, field)) for item in group) for field in boolean_fields},
            "tasks_with_environment_errors": sum(item.environment_error_count > 0 for item in group),
            "tasks_with_duplicate_actions": sum(
                item.consecutive_duplicate_actions > 0 for item in group
            ),
            "tasks_with_abab_loops": sum(item.abab_loops > 0 for item in group),
            "tasks_with_fail_submission": sum(item.fail_submissions > 0 for item in group),
        }

    errors = Counter()
    error_fingerprints = Counter()
    schema_issues = Counter()
    actions = Counter()
    for item in task_stats:
        errors.update(item.environment_error_types)
        error_fingerprints.update(item.environment_error_fingerprints)
        schema_issues.update(item.schema_issues)
        actions.update(item.action_names)
    failed = [item for item in task_stats if not item.success]
    successful = [item for item in task_stats if item.success]
    return {
        "all": summarize(task_stats),
        "failed": summarize(failed),
        "successful": summarize(successful),
        "environment_error_types": dict(errors.most_common()),
        "environment_error_fingerprints": dict(error_fingerprints.most_common(80)),
        "schema_issues": dict(schema_issues.most_common()),
        "most_common_actions": dict(actions.most_common(30)),
        "tasks": [asdict(item) for item in task_stats],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AppWorld Qwen3-4B Baseline Trajectory Analysis",
        "",
        "This report uses runtime trajectories and final success booleans only. It does not read solutions, private oracle data, or evaluation requirement text.",
        "",
        "## Aggregate",
        "",
        "| Metric | All | Failed | Successful |",
        "|---|---:|---:|---:|",
    ]
    keys = [
        "tasks",
        "lm_calls",
        "action_count",
        "tasks_with_environment_errors",
        "environment_error_count",
        "no_tool_turns",
        "missing_required_fields",
        "unknown_argument_fields",
        "string_integer_fields",
        "tasks_with_duplicate_actions",
        "consecutive_duplicate_actions",
        "tasks_with_abab_loops",
        "tasks_with_fail_submission",
        "missing_completion",
        "budget_exhausted",
        "single_page_exhaustive_candidates",
    ]
    for key in keys:
        lines.append(
            f"| {key} | {summary['all'][key]} | {summary['failed'][key]} | {summary['successful'][key]} |"
        )
    lines += ["", "## Environment error types", ""]
    for key, value in summary["environment_error_types"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Environment error fingerprints", ""]
    for key, value in summary["environment_error_fingerprints"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Schema issues", ""]
    for key, value in summary["schema_issues"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Most common actions", ""]
    for key, value in summary["most_common_actions"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    evaluation_path = args.baseline_output / "evaluations" / "train.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    individual = evaluation.get("individual", {})
    task_stats = [
        analyze_task(task_dir, bool(individual.get(task_dir.name, {}).get("success", False)))
        for task_dir in sorted((args.baseline_output / "tasks").iterdir())
        if task_dir.is_dir()
    ]
    summary = aggregate(task_stats)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "trajectory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_directory / "trajectory_summary.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    print(markdown_report(summary))


if __name__ == "__main__":
    main()
