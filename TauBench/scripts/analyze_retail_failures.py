#!/usr/bin/env python3
"""
分析 retail train harness_h3 评测结果，提取失败任务的关键信息。

用法：
  uv run python scripts/analyze_retail_failures.py \
      data/simulations/eval_retail_train_harness_h3_20260408_095839
"""

import json
import sys
from pathlib import Path


def fmt_tool_call(tc: dict) -> str:
    # Support both flat format {name, arguments} and nested {function: {name, arguments}}
    if "function" in tc:
        fn = tc["function"]
        name = fn.get("name", "?")
        raw_args = fn.get("arguments", "{}")
    else:
        name = tc.get("name", "?")
        raw_args = tc.get("arguments", {})

    try:
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
        else:
            args = raw_args
        args_str = json.dumps(args, ensure_ascii=False)
    except Exception:
        args_str = str(raw_args)
    return f"{name}({args_str[:300]})"


def extract_conversation(messages: list) -> list[dict]:
    """把 messages 压成精简对话行。"""
    turns = []
    for msg in messages:
        role = msg["role"]
        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []

        if role == "tool":
            turns.append({"role": "tool_result", "text": content[:400]})
        elif tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    turns.append({"role": "tool_call", "text": fmt_tool_call(tc)[:400]})
        elif content:
            turns.append({"role": role, "text": content[:400]})
    return turns


def summarize_reward_info(ri: dict) -> dict:
    reward = ri.get("reward", 0)
    db_match = ri.get("db_check", {}).get("db_match", None)
    db_reward = ri.get("db_check", {}).get("db_reward", None)

    action_checks = ri.get("action_checks", [])
    matched = sum(1 for a in action_checks if a.get("action_match"))
    total_actions = len(action_checks)

    failed_actions = []
    for ac in action_checks:
        if not ac.get("action_match"):
            act = ac.get("action", {})
            failed_actions.append({
                "name": act.get("name"),
                "args": act.get("arguments"),
                "actual_reward": ac.get("action_reward"),
            })

    env_assertions = ri.get("env_assertions", [])
    failed_env_assertions = [a for a in env_assertions if not a.get("passed", True)]

    # NL assertions (newer format)
    nl_assertions = ri.get("nl_assertions", [])
    failed_nl = [a for a in nl_assertions if not a.get("met", True)]

    communicate_checks = ri.get("communicate_checks") or []
    failed_communicate = [c for c in communicate_checks if not c.get("met", True)]

    reward_breakdown = ri.get("reward_breakdown", {})

    return {
        "reward": reward,
        "db_match": db_match,
        "db_reward": db_reward,
        "action_match_rate": f"{matched}/{total_actions}",
        "failed_gold_actions": failed_actions,
        "failed_assertions": [a.get("assertion") for a in failed_env_assertions],
        "failed_nl_assertions": [
            {"text": a.get("nl_assertion", ""), "justification": a.get("justification", "")[:200]}
            for a in failed_nl
        ],
        "failed_communicate": [
            {"info": c.get("info"), "justification": c.get("justification", "")[:200]}
            for c in failed_communicate
        ],
        "reward_breakdown": reward_breakdown,
    }


def detect_harness_hits(messages: list) -> list[str]:
    """找到所有 harness 拦截（tool_result 含 error 标志）。"""
    hits = []
    for msg in messages:
        if msg["role"] == "tool":
            content = msg.get("content") or ""
            # harness error messages 通常以特定词汇开头
            if any(kw in content.lower() for kw in [
                "error:", "cannot", "only pending", "only delivered",
                "must be", "not allowed", "invalid", "rejected",
                "harness", "rule", "policy violation",
            ]):
                hits.append(content[:300])
    return hits


def find_tool_call_sequence(messages: list) -> list[str]:
    """提取 agent 调用工具的顺序。"""
    seq = []
    for msg in messages:
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                if "function" in tc:
                    name = tc["function"].get("name", "?")
                else:
                    name = tc.get("name", "?")
                seq.append(name)
    return seq


def analyze_run(run_dir: str) -> None:
    run_path = Path(run_dir)
    results_file = run_path / "results.json"
    summary_file = run_path / "harness_summary.json"

    with open(results_file) as f:
        data = json.load(f)

    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print("=" * 70)
        print("HARNESS SUMMARY")
        print("=" * 70)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print()

    tasks_meta = {str(t["id"]): t for t in data.get("tasks", [])}
    simulations = data["simulations"]

    passed = [s for s in simulations if s["reward_info"]["reward"] >= 1.0]
    failed = [s for s in simulations if s["reward_info"]["reward"] < 1.0]

    print(f"总计: {len(simulations)} 任务 | 通过: {len(passed)} | 失败: {len(failed)}")
    print(f"通过率: {len(passed)/len(simulations)*100:.1f}%\n")

    # ── 提取每个失败任务的精简摘要 ──────────────────────────────────────
    extracted = []
    for sim in failed:
        task_id = str(sim["task_id"])
        task = tasks_meta.get(task_id, {})
        scenario = task.get("user_scenario", {})
        instructions_raw = scenario.get("instructions", "")
        if isinstance(instructions_raw, dict):
            instructions = (
                instructions_raw.get("reason_for_call", "")
                + " | "
                + instructions_raw.get("task_instructions", "")
            )
        else:
            instructions = str(instructions_raw)

        ri = summarize_reward_info(sim["reward_info"])
        conv = extract_conversation(sim["messages"])
        harness_hits = detect_harness_hits(sim["messages"])
        tool_seq = find_tool_call_sequence(sim["messages"])

        extracted.append({
            "task_id": task_id,
            "instructions": instructions[:600],
            "reward": ri["reward"],
            "db_match": ri["db_match"],
            "db_reward": ri["db_reward"],
            "action_match_rate": ri["action_match_rate"],
            "failed_gold_actions": ri["failed_gold_actions"],
            "failed_assertions": ri["failed_assertions"],
            "failed_nl_assertions": ri["failed_nl_assertions"],
            "failed_communicate": ri["failed_communicate"],
            "reward_breakdown": ri["reward_breakdown"],
            "harness_hits": harness_hits,
            "tool_call_sequence": tool_seq,
            "conversation": conv,
        })

    # ── 打印逐任务摘要 ──────────────────────────────────────────────────
    print("=" * 70)
    print("失败任务逐条摘要")
    print("=" * 70)
    for e in extracted:
        print(f"\n{'─'*60}")
        print(f"Task {e['task_id']}  reward={e['reward']}  db_match={e['db_match']}  actions={e['action_match_rate']}")
        print(f"Instructions: {e['instructions']}")

        if e["failed_gold_actions"]:
            print(f"\n[未匹配 gold actions]")
            for fa in e["failed_gold_actions"]:
                print(f"  - {fa['name']}  args={json.dumps(fa['args'], ensure_ascii=False)[:200]}")

        if e.get("reward_breakdown"):
            print(f"  Breakdown: {e['reward_breakdown']}")

        if e["failed_assertions"]:
            print(f"\n[失败 env_assertions]")
            for a in e["failed_assertions"]:
                print(f"  - {a}")

        if e.get("failed_nl_assertions"):
            print(f"\n[失败 NL assertions]")
            for a in e["failed_nl_assertions"]:
                print(f"  - {a['text'][:150]}")
                print(f"    reason: {a['justification'][:150]}")

        if e.get("failed_communicate"):
            print(f"\n[未传达的信息]")
            for c in e["failed_communicate"]:
                print(f"  - '{c['info']}': {c['justification'][:150]}")

        if e["harness_hits"]:
            print(f"\n[Harness 拦截]")
            for h in e["harness_hits"]:
                print(f"  !! {h[:200]}")

        print(f"\n[Tool 调用序列] {' -> '.join(e['tool_call_sequence'])}")

        print(f"\n[对话摘要]")
        for turn in e["conversation"]:
            role = turn["role"]
            text = turn["text"]
            prefix = {
                "assistant": "A",
                "user": "U",
                "tool_call": "→",
                "tool_result": "←",
            }.get(role, role[0].upper())
            print(f"  {prefix}: {text[:300]}")

    # ── 保存精简 JSON ────────────────────────────────────────────────────
    out_file = run_path / "failures_extracted.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)
    print(f"\n\n已保存精简数据到: {out_file}")

    # ── 统计工具调用序列中的失败模式 ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("失败任务工具调用分布（按工具名）")
    print("=" * 70)
    from collections import Counter
    tool_counter: Counter = Counter()
    for e in extracted:
        for t in e["tool_call_sequence"]:
            tool_counter[t] += 1
    for tool, cnt in tool_counter.most_common():
        print(f"  {tool}: {cnt}")

    # ── 按 DB 失败 vs 行为失败 分类 ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("失败类型初步分类")
    print("=" * 70)
    db_fail = [e for e in extracted if not e["db_match"]]
    nl_fail = [e for e in extracted if e.get("failed_nl_assertions") or e.get("failed_communicate")]
    comm_fail = [e for e in extracted if e.get("failed_communicate")]
    db_only_fail = [e for e in extracted if not e["db_match"] and not nl_fail]
    harness_hit_failed = [e for e in extracted if e["harness_hits"]]

    print(f"  DB 状态不符: {len(db_fail)} 任务")
    print(f"  NL Assertion 失败: {len(nl_fail)} 任务")
    print(f"  未传达关键信息: {len(comm_fail)} 任务")
    print(f"  含 Harness 拦截: {len(harness_hit_failed)} 任务")

    # ── 失败的 gold action 工具分布 ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("未匹配 gold action 按工具名分布")
    print("=" * 70)
    gold_fail_counter: Counter = Counter()
    for e in extracted:
        for fa in e["failed_gold_actions"]:
            gold_fail_counter[fa["name"]] += 1
    for tool, cnt in gold_fail_counter.most_common():
        print(f"  {tool}: {cnt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        default = "data/simulations/eval_retail_train_harness_h3_20260408_095839"
        print(f"Usage: python {sys.argv[0]} <run_dir>  (defaulting to {default})")
        run_dir = default
    else:
        run_dir = sys.argv[1]
    analyze_run(run_dir)
