"""Automated harness failure analysis pipeline.

This script automates the "run → analyse failures → generate improvement report"
cycle.  It can either:
  (a) run a fresh evaluation, then analyse the results, or
  (b) skip the evaluation and analyse an existing results.json.

The report includes:
  • Pass/fail summary
  • Failure-category breakdown (via LLM triage)
  • Per-task failure narrative
  • Harness improvement recommendations (H2 rules + H3 hints)

Usage
-----
# Fresh run on 3 airline train tasks (cheap test)
uv run python scripts/harness_failure_report.py \\
    --domain airline --split train --harness --h3 --num-tasks 3

# Full airline train run
uv run python scripts/harness_failure_report.py \\
    --domain airline --split train --harness --h3

# Analyse existing results (skip evaluation)
uv run python scripts/harness_failure_report.py \\
    --run-dir data/simulations/eval_train_harness_h3_20260407_201921

# Retail domain
uv run python scripts/harness_failure_report.py \\
    --domain retail --split train --harness --h3 --num-tasks 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Failure categories (used as labels in LLM prompt + aggregation)
# ---------------------------------------------------------------------------

CATEGORIES = {
    "POLICY_HALLUCINATION": "Agent invented a policy rule that does not exist",
    "PREMATURE_TRANSFER": "Agent transferred to human when the request was in-scope",
    "WRONG_TOOL_ARGS": "Agent called the right tool but with incorrect arguments or amounts",
    "WRONG_TOOL_CHOICE": "Agent called the wrong tool or skipped a required tool",
    "NO_HARNESS_RECOVERY": "Harness blocked a call; agent failed to self-correct and gave up",
    "INCOMPLETE_TASK": "Agent completed part of the task but left required steps undone",
    "WRONG_DB_STATE": "Agent acted without checking current DB state (wrong assumption)",
    "USER_INSTRUCTION_IGNORED": "Agent ignored or misunderstood a specific user instruction",
    "CONFIRM_MISSING": "Agent performed a DB write without obtaining explicit user confirmation",
    "OTHER": "Failure does not fit the categories above",
}

CATEGORY_LIST = list(CATEGORIES.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _build_conversation_snippet(sim: dict, max_chars_per_msg: int = 600) -> str:
    """Return a condensed conversation transcript for the LLM analyst."""
    lines: list[str] = []
    for msg in sim.get("messages") or []:
        role = msg.get("role", "")
        turn = msg.get("turn_idx", "?")
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = _trim(_safe_text(tc.get("arguments")), max_chars_per_msg)
                    lines.append(f"[{turn}] AGENT calls {name}({args})")
            else:
                content = _trim(_safe_text(msg.get("content")), max_chars_per_msg)
                if content:
                    lines.append(f"[{turn}] AGENT: {content}")
        elif role == "tool":
            content = _trim(_safe_text(msg.get("content")), max_chars_per_msg)
            is_error = bool(msg.get("error"))
            prefix = "TOOL ERROR" if is_error else "TOOL"
            lines.append(f"[{turn}] {prefix}: {content}")
        elif role == "user":
            content = _trim(_safe_text(msg.get("content")), max_chars_per_msg)
            if content:
                lines.append(f"[{turn}] USER: {content}")
    return "\n".join(lines)


def _failed_action_summary(sim: dict) -> str:
    ri = sim.get("reward_info") or {}
    action_checks = ri.get("action_checks") or []
    failed = [a for a in action_checks if not a.get("action_match")]
    if not failed:
        return "No failed action checks."
    parts = []
    for a in failed:
        action = a.get("action") or {}
        parts.append(
            f"  - MISSING: {action.get('name')}({_safe_text(action.get('arguments'))})"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM triage
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM = """\
You are an expert evaluator analysing customer service agent simulation failures.
Your job is to read a conversation transcript, identify the ROOT CAUSE of the failure,
and suggest harness improvements that could prevent the same mistake.

Failure categories and their codes:
{category_block}

Harness types:
  H2 = pre-execution validation rule (inspects DB state BEFORE tool runs, blocks with error message)
  H3 = policy hint embedded in the tool description (always visible in every LLM API request)

You MUST respond in valid JSON only — no prose outside the JSON object.
""".format(
    category_block="\n".join(
        f"  {code}: {desc}" for code, desc in CATEGORIES.items()
    )
)

_TRIAGE_USER_TEMPLATE = """\
## Task {task_id} — reward {reward}

### User's goal
{user_goal}

### Expected actions that were NOT taken (evaluation failures)
{failed_actions}

### Conversation transcript
{transcript}

---
Respond with JSON:
{{
  "category": "<one of {category_codes}>",
  "root_cause": "<1-2 sentence explanation of WHY the agent failed>",
  "harness_type": "<H2|H3|BOTH|NONE>",
  "harness_suggestion": "<specific rule or hint that would prevent this failure, or 'none' if no harness can help>",
  "confidence": "<HIGH|MEDIUM|LOW>"
}}
"""


def _call_llm(prompt_system: str, prompt_user: str, llm: str, api_base: str) -> str:
    """Call the LLM and return the raw text response."""
    import litellm

    response = litellm.completion(
        model=llm,
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
        api_base=api_base if api_base else None,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def _triage_failure(
    sim: dict,
    task_meta: dict | None,
    llm: str,
    api_base: str,
) -> dict:
    """Return triage result dict for a single failed simulation."""
    task_id = str(sim.get("task_id"))
    reward = (sim.get("reward_info") or {}).get("reward", 0)

    # Build user goal from task meta
    if task_meta:
        scenario = (task_meta.get("user_scenario") or {}).get("instructions") or {}
        user_goal = scenario.get("reason_for_call") or scenario.get("task_instructions") or "(unknown)"
    else:
        # Fall back to first user message
        user_msgs = [
            m.get("content", "")
            for m in (sim.get("messages") or [])
            if m.get("role") == "user"
        ]
        user_goal = user_msgs[0] if user_msgs else "(unknown)"
    user_goal = _trim(_safe_text(user_goal), 600)

    transcript = _build_conversation_snippet(sim)
    failed_actions = _failed_action_summary(sim)

    prompt = _TRIAGE_USER_TEMPLATE.format(
        task_id=task_id,
        reward=reward,
        user_goal=user_goal,
        failed_actions=failed_actions,
        transcript=_trim(transcript, 4000),
        category_codes="|".join(CATEGORY_LIST),
    )

    try:
        raw = _call_llm(_TRIAGE_SYSTEM, prompt, llm, api_base)
        # Extract JSON from response (handle markdown code fences)
        raw_strip = raw.strip()
        if raw_strip.startswith("```"):
            lines = raw_strip.split("\n")
            raw_strip = "\n".join(lines[1:-1])
        result = json.loads(raw_strip)
        result["task_id"] = task_id
        result["reward"] = reward
    except Exception as e:
        result = {
            "task_id": task_id,
            "reward": reward,
            "category": "OTHER",
            "root_cause": f"(LLM triage failed: {e})",
            "harness_type": "NONE",
            "harness_suggestion": "none",
            "confidence": "LOW",
        }
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(
    results_path: Path,
    domain: str,
    triages: list[dict],
    all_task_rewards: list[tuple[str, float]],
    harness_flags: str,
) -> str:
    total = len(all_task_rewards)
    passed = sum(1 for _, r in all_task_rewards if r >= 1.0)
    failed = total - passed
    pass_rate = passed / total if total else 0.0

    lines: list[str] = []

    lines += [
        f"# Harness Failure Analysis Report",
        f"",
        f"- **Domain**: {domain}",
        f"- **Harness config**: {harness_flags}",
        f"- **Results source**: `{results_path}`",
        f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"---",
        f"",
        f"## 1. Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tasks | {total} |",
        f"| Passed | {passed} ({pass_rate:.0%}) |",
        f"| Failed | {failed} ({1-pass_rate:.0%}) |",
        f"| Tasks with LLM triage | {len(triages)} |",
        f"",
    ]

    if not triages:
        lines.append("No failures to analyse — all tasks passed! 🎉")
        return "\n".join(lines)

    # --- Category breakdown ---
    cat_counter: Counter = Counter(t["category"] for t in triages)
    harness_type_counter: Counter = Counter(t["harness_type"] for t in triages)

    lines += [
        f"## 2. Failure Category Breakdown",
        f"",
        f"| Category | Count | % of failures | Description |",
        f"|----------|------:|--------------|-------------|",
    ]
    for cat, cnt in cat_counter.most_common():
        pct = cnt / len(triages) * 100
        desc = CATEGORIES.get(cat, "")
        lines.append(f"| `{cat}` | {cnt} | {pct:.0f}% | {desc} |")
    lines.append("")

    lines += [
        f"## 3. Harness Actionability",
        f"",
        f"| Harness type | Count |",
        f"|-------------|------:|",
    ]
    for ht, cnt in harness_type_counter.most_common():
        lines.append(f"| {ht} | {cnt} |")
    lines.append("")

    harness_fixable = sum(
        1 for t in triages if t["harness_type"] in ("H2", "H3", "BOTH")
    )
    lines += [
        f"> **{harness_fixable} / {len(triages)} failures** ({harness_fixable/len(triages):.0%}) "
        f"could potentially be prevented by new H2 rules or H3 hints.",
        f"",
    ]

    # --- Per-category harness suggestions ---
    lines += [
        f"## 4. Harness Improvement Recommendations",
        f"",
    ]
    suggestions_by_cat: dict[str, list[str]] = defaultdict(list)
    for t in triages:
        if t["harness_type"] != "NONE" and t["harness_suggestion"] not in ("none", "", None):
            suggestions_by_cat[t["category"]].append(
                f"  - **Task {t['task_id']}** ({t['harness_type']}, conf={t['confidence']}): "
                f"{t['harness_suggestion']}"
            )

    for cat in CATEGORY_LIST:
        if cat not in suggestions_by_cat:
            continue
        lines.append(f"### {cat}")
        lines.append("")
        lines.append(f"*{CATEGORIES[cat]}*")
        lines.append("")
        lines.extend(suggestions_by_cat[cat])
        lines.append("")

    if not any(suggestions_by_cat.values()):
        lines.append("No actionable harness improvements identified in these failures.")
        lines.append("")

    # --- Per-task details ---
    lines += [
        f"## 5. Per-Task Failure Details",
        f"",
        f"| Task | Reward | Category | Confidence | Root Cause |",
        f"|------|--------|----------|------------|-----------|",
    ]
    for t in sorted(triages, key=lambda x: str(x["task_id"])):
        root = _trim(t["root_cause"], 120)
        lines.append(
            f"| {t['task_id']} | {t['reward']} | `{t['category']}` | "
            f"{t['confidence']} | {root} |"
        )
    lines.append("")

    # --- Full triage data (expandable) ---
    lines += [
        f"## 6. Full Triage Data",
        f"",
        f"<details>",
        f"<summary>Click to expand raw triage JSON</summary>",
        f"",
        f"```json",
        json.dumps(triages, ensure_ascii=False, indent=2),
        f"```",
        f"",
        f"</details>",
        f"",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation runner (thin wrapper around eval_harness.py logic)
# ---------------------------------------------------------------------------


def _run_evaluation(args: argparse.Namespace) -> Path:
    """Run evaluation and return the save_dir path."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_tasks
    from tau2.utils.utils import DATA_DIR

    if args.domain == "airline":
        from tau2.domains.airline.environment import get_tasks
    elif args.domain == "retail":
        from tau2.domains.retail.environment import get_tasks
    else:
        from tau2.domains.telecom.environment import get_tasks

    parts = [args.domain, args.split]
    if args.harness:
        parts.append("harness")
    if args.h3:
        parts.append("h3")
    label = "_".join(parts)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"eval_{label}_{ts}"
    save_dir = DATA_DIR / "simulations" / run_name
    save_path = save_dir / "results.json"

    config = TextRunConfig(
        domain=args.domain,
        llm_agent=args.agent_llm,
        llm_args_agent={"api_base": args.agent_api_base, "api_key": "EMPTY"},
        llm_user=args.user_llm,
        num_trials=1,
        task_split_name=args.split,
        max_concurrency=args.concurrency,
        harness_enabled=args.harness,
        harness_h3=args.h3,
        harness_h5=False,
        harness_h5_rag=False,
        harness_h5_rag_tool=False,
    )

    tasks = get_tasks(args.split)
    if args.task_ids:
        id_set = {str(i) for i in args.task_ids}
        tasks = [t for t in tasks if t.id in id_set]
    elif args.num_tasks:
        tasks = tasks[: args.num_tasks]

    print(f"\n{'='*60}")
    print(f"Running evaluation: {label}")
    print(f"  Tasks       : {len(tasks)}")
    print(f"  Agent LLM   : {args.agent_llm}")
    print(f"  H2 harness  : {args.harness}")
    print(f"  H3 hints    : {args.h3}")
    print(f"  Save dir    : {save_dir}")
    print(f"{'='*60}\n")

    run_tasks(
        config,
        tasks,
        save_path=save_path,
        save_dir=save_dir,
        evaluation_type=EvaluationType.ALL,
    )
    return save_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Automated harness failure analysis: run eval → triage failures "
            "with LLM → generate improvement report."
        )
    )
    # Evaluation options
    p.add_argument("--domain", default="airline", choices=["airline", "retail", "telecom"])
    p.add_argument("--split", default="train", choices=["train", "test", "base"])
    p.add_argument("--harness", action="store_true", help="Enable H2 harness")
    p.add_argument("--h3", action="store_true", help="Enable H3 tool hints")
    p.add_argument("--num-tasks", type=int, default=None, help="Limit number of tasks")
    p.add_argument(
        "--task-ids", nargs="+", type=int, default=None,
        help="Run specific task IDs only",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--agent-llm", default="openai/Qwen/Qwen3-4B-Instruct",
        help="Agent LLM identifier",
    )
    p.add_argument(
        "--agent-api-base", default="http://localhost:30000/v1",
        help="Agent LLM API base URL",
    )
    p.add_argument(
        "--user-llm", default="openai/deepseek-v3-2-251201",
        help="User LLM identifier",
    )

    # Analysis options
    p.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Path to an existing simulation run directory containing results.json. "
            "If provided, skips evaluation and uses this data directly."
        ),
    )
    p.add_argument(
        "--analyst-llm",
        default="openai/deepseek-v3-2-251201",
        help="LLM used for failure triage analysis (default: deepseek-v3)",
    )
    p.add_argument(
        "--analyst-api-base",
        default=None,
        help="API base for analyst LLM (default: None = use provider default)",
    )
    p.add_argument(
        "--max-triage",
        type=int,
        default=None,
        help="Max number of failed tasks to triage with LLM (cost control). Default: all.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save the report (default: <run_dir>/failure_report/)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Step 1: Get results
    if args.run_dir:
        run_dir = Path(args.run_dir)
        print(f"Using existing run directory: {run_dir}")
    else:
        run_dir = _run_evaluation(args)

    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"ERROR: results.json not found in {run_dir}", file=sys.stderr)
        return 1

    # Step 2: Load results
    print(f"\nLoading results from {results_path} ...")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    simulations = data.get("simulations") or []
    tasks_data = data.get("tasks") or []
    task_meta_by_id: dict[str, dict] = {str(t.get("id")): t for t in tasks_data}

    all_rewards = [
        (str(s.get("task_id")), float((s.get("reward_info") or {}).get("reward", 0)))
        for s in simulations
    ]
    failed_sims = [
        s for s in simulations
        if float((s.get("reward_info") or {}).get("reward", 0)) < 1.0
    ]

    total = len(all_rewards)
    n_failed = len(failed_sims)
    print(f"  Total simulations : {total}")
    print(f"  Failed            : {n_failed}")

    # Determine harness config label
    harness_flags_parts = []
    if args.harness or args.run_dir:
        harness_flags_parts.append("H2" if args.harness else "unknown")
    if args.h3 or args.run_dir:
        harness_flags_parts.append("H3" if args.h3 else "unknown")
    harness_flags = "+".join(harness_flags_parts) if harness_flags_parts else "baseline"
    if args.run_dir:
        # Try to infer from directory name
        dir_name = run_dir.name
        inferred = []
        if "harness" in dir_name:
            inferred.append("H2")
        if "h3" in dir_name:
            inferred.append("H3")
        harness_flags = "+".join(inferred) if inferred else "unknown"

    # Step 3: Triage failures with LLM
    triage_targets = failed_sims
    if args.max_triage is not None:
        triage_targets = triage_targets[: args.max_triage]

    triages: list[dict] = []
    if triage_targets:
        print(f"\nTriaging {len(triage_targets)} failure(s) with LLM ({args.analyst_llm}) ...")
        for i, sim in enumerate(triage_targets, 1):
            task_id = str(sim.get("task_id"))
            print(f"  [{i}/{len(triage_targets)}] Analysing task {task_id} ...", end=" ", flush=True)
            result = _triage_failure(
                sim=sim,
                task_meta=task_meta_by_id.get(task_id),
                llm=args.analyst_llm,
                api_base=args.analyst_api_base or "",
            )
            triages.append(result)
            print(f"→ {result['category']} ({result['confidence']})")
    else:
        print("No failures to triage.")

    # Step 4: Generate report
    domain = args.domain if not args.run_dir else run_dir.name.split("_")[1] if "_" in run_dir.name else "unknown"
    report_md = _generate_report(
        results_path=results_path,
        domain=domain,
        triages=triages,
        all_task_rewards=all_rewards,
        harness_flags=harness_flags,
    )

    # Step 5: Save
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "failure_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    triage_path = out_dir / "triage.json"
    triage_path.write_text(
        json.dumps(triages, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print brief summary to console
    cat_counter: Counter = Counter(t["category"] for t in triages)
    print(f"\n{'='*60}")
    print(f"Failure Analysis Report")
    print(f"{'='*60}")
    print(f"  Pass rate : {sum(1 for _,r in all_rewards if r>=1)}/{total} "
          f"({sum(1 for _,r in all_rewards if r>=1)/total:.0%})")
    if triages:
        print(f"  Failure categories:")
        for cat, cnt in cat_counter.most_common():
            print(f"    {cat:35s}: {cnt}")
        harness_fixable = sum(1 for t in triages if t["harness_type"] in ("H2", "H3", "BOTH"))
        print(f"  Harness-fixable failures: {harness_fixable}/{len(triages)}")
    print(f"  Report saved → {report_path}")
    print(f"  Triage JSON → {triage_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
