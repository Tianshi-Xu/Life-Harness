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


def summarize(experiment_output: Path, dataset: str = "train") -> dict[str, Any]:
    triggers: Counter[str] = Counter()
    layers: Counter[str] = Counter()
    tasks_with_trigger: Counter[str] = Counter()
    task_rows: list[dict[str, Any]] = []
    max_prompt_tokens = 0
    breaking_tasks: list[str] = []
    tokens = {"calls": 0, "calls_missing_usage": 0, "known_input_tokens": 0,
              "known_output_tokens": 0, "known_total_tokens": 0,
              "total_tokens_derived_calls": 0, "tasks_missing_call_logs": 0}

    for task_directory in sorted((experiment_output / "tasks").glob("*")):
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
        calls = read_jsonl(task_directory / "logs" / "lm_calls.jsonl")
        if not calls:
            tokens["tasks_missing_call_logs"] += 1
        for call in calls:
            output = call.get("output")
            output = output if isinstance(output, dict) else {}
            usage = output.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            tokens["calls"] += 1
            values = [usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")]
            known = [type(value) is int and value >= 0 for value in values]
            if not known[2] and known[0] and known[1]:
                values[2] = values[0] + values[1]
                known[2] = True
                tokens["total_tokens_derived_calls"] += 1
            for field, value, valid in zip(
                ("known_input_tokens", "known_output_tokens", "known_total_tokens"), values, known
            ):
                if valid:
                    tokens[field] += value
            if not all(known):
                tokens["calls_missing_usage"] += 1
            if known[0]:
                prompt_tokens = max(prompt_tokens, values[0])
            output_text = json.dumps(output, ensure_ascii=False).lower()
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

    evaluation_path = experiment_output / "evaluations" / f"{dataset}.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.exists()
        else None
    )
    return {
        "dataset": dataset,
        "tasks_present": len(task_rows),
        "tasks_finished": sum(row["finished"] for row in task_rows),
        "max_prompt_tokens": max_prompt_tokens,
        "context_error_tasks": breaking_tasks,
        "layer_events": dict(layers),
        "trigger_events": dict(triggers),
        "tasks_with_trigger": dict(tasks_with_trigger),
        "evaluation": evaluation,
        "model_token_usage": {**tokens, "accounting_complete": tokens["calls"] > 0
            and tokens["calls_missing_usage"] == tokens["tasks_missing_call_logs"] == 0,
            "scope": "recorded inference calls; known totals exclude missing usage; raw lm_calls.jsonl retained"},
        "tasks": task_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("train", "dev", "test_normal", "test_challenge"), default="train")
    args = parser.parse_args()
    summary = summarize(args.experiment_output, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
