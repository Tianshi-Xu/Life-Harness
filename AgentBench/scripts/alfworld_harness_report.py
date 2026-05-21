#!/usr/bin/env python3
"""
Summarize ALFWorld harness traces from a runs.jsonl file.

Usage:
  python scripts/alfworld_harness_report.py \
      --runs outputs/<ts>/<model>/alfworld-env_train/runs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _extract_output(row: Dict[str, Any]) -> Dict[str, Any]:
    out = row.get("output", {})
    return out if isinstance(out, dict) else {}


def summarize(runs_path: Path) -> Dict[str, Any]:
    rows = _load_rows(runs_path)
    total = len(rows)
    pass_count = 0
    task_limit = 0
    invalid_action = 0
    validation_failed = 0
    h2_blocked = 0
    h2_canonicalized = 0
    h4_soft = 0
    h4_hard = 0
    h5_injections = 0
    h3_hits = 0

    for row in rows:
        out = _extract_output(row)
        status = out.get("status", "")
        result = out.get("result", {})
        reward = result.get("reward", 0)
        if reward == 1:
            pass_count += 1
        if status == "task limit reached":
            task_limit += 1
        if status == "agent invalid action":
            invalid_action += 1
        if status == "agent validation failed":
            validation_failed += 1

        trace = result.get("harness_trace", {}) if isinstance(result, dict) else {}
        h2_events = trace.get("h2", []) if isinstance(trace, dict) else []
        h3_events = trace.get("h3", []) if isinstance(trace, dict) else []
        h4_events = trace.get("h4", []) if isinstance(trace, dict) else []
        h5_events = trace.get("h5", []) if isinstance(trace, dict) else []

        for e in h2_events:
            if e.get("blocked"):
                h2_blocked += 1
            if e.get("canonicalized"):
                h2_canonicalized += 1
        for e in h4_events:
            level = e.get("intervention_level")
            if level == "soft":
                h4_soft += 1
            elif level == "hard":
                h4_hard += 1
        h3_hits += len(h3_events)
        h5_injections += len(h5_events)

    success_rate = pass_count / total if total else 0
    return {
        "total": total,
        "pass": pass_count,
        "success_rate": success_rate,
        "task_limit_reached": task_limit,
        "agent_invalid_action": invalid_action,
        "agent_validation_failed": validation_failed,
        "h2": {"blocked": h2_blocked, "canonicalized": h2_canonicalized},
        "h3": {"hint_hits": h3_hits},
        "h4": {"soft_interventions": h4_soft, "hard_interventions": h4_hard},
        "h5": {"skill_injections": h5_injections},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="Path to runs.jsonl")
    args = parser.parse_args()
    summary = summarize(Path(args.runs))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
