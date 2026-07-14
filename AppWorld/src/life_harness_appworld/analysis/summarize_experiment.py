from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    triggers: Counter[str] = Counter()
    layers: Counter[str] = Counter()
    tasks_with_trigger: Counter[str] = Counter()
    task_rows: list[dict[str, Any]] = []
    max_prompt_tokens = 0
    breaking_tasks: list[str] = []

    for task_directory in sorted((args.experiment_output / "tasks").glob("*")):
        if not task_directory.is_dir():
            continue
        task_triggers: Counter[str] = Counter()
        for event in read_jsonl(task_directory / "logs" / "harness_events.jsonl"):
            trigger = str(event.get("trigger", "unknown"))
            layer = str(event.get("layer", "unknown"))
            triggers[trigger] += 1
            layers[layer] += 1
            task_triggers[trigger] += 1
        for trigger in task_triggers:
            tasks_with_trigger[trigger] += 1

        prompt_tokens = 0
        breaking = False
        for call in read_jsonl(task_directory / "logs" / "lm_calls.jsonl"):
            usage = call.get("output", {}).get("usage", {})
            prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens", 0) or 0))
            output_text = json.dumps(call.get("output", {}), ensure_ascii=False).lower()
            breaking = breaking or "maximum context length" in output_text
        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
        if breaking:
            breaking_tasks.append(task_directory.name)
        task_rows.append(
            {
                "task_id": task_directory.name,
                "finished": (task_directory / "misc" / "finished").exists(),
                "max_prompt_tokens": prompt_tokens,
                "context_error": breaking,
                "triggers": dict(task_triggers),
            }
        )

    evaluation_path = args.experiment_output / "evaluations" / "train.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.exists()
        else None
    )
    summary = {
        "tasks_present": len(task_rows),
        "tasks_finished": sum(row["finished"] for row in task_rows),
        "max_prompt_tokens": max_prompt_tokens,
        "context_error_tasks": breaking_tasks,
        "layer_events": dict(layers),
        "trigger_events": dict(triggers),
        "tasks_with_trigger": dict(tasks_with_trigger),
        "evaluation": evaluation,
        "tasks": task_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
