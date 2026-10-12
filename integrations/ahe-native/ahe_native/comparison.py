"""Host-only per-optimization-step comparison; never sent to evolving agents."""
import hashlib
import json
from pathlib import Path

from .backend import HERE, LIFE
from .records import write_json
from .training import AGENTBENCH

SIX = LIFE / "evolution/experiments/life_agentbench_six_resume_mr_20261001/round_comparison.json"
LIFE_V3 = LIFE / "evolution/experiments/life_agentbench_v3_from_six_db_web_20261002/training_results.json"


def reference_rows():
    rows = json.loads(SIX.read_text())["train_rounds"] if SIX.is_file() else []
    result = {domain: {} for domain in AGENTBENCH}
    for row in rows:
        domain, step = row["domain"], row["round"]
        result[domain][step] = {name: {
            "batch": row[name + "_batch"], "incumbent": row[name + "_incumbent"],
            "candidate": row[name + "_candidate"], "accepted": row[name + "_accepted"],
            "effective": row[name + "_candidate"] if row[name + "_accepted"] else row[name + "_incumbent"],
            "source": str(SIX), "version": "historical six-round"}
            for name in ("meta", "life")}
    for row in (json.loads(LIFE_V3.read_text())["new_rounds"] if LIFE_V3.is_file() else []):
        result[row["domain"]].setdefault(row["round"], {})["life"] = {
            "batch": row["train_batch"], "incumbent": row["incumbent_score"],
            "candidate": row["candidate_score"], "accepted": row["accepted"],
            "effective": row["candidate_score"] if row["accepted"] else row["incumbent_score"],
            "source": str(LIFE_V3), "version": "v3 continuation from sealed round six"}
    for domain in AGENTBENCH:
        if 1 not in result[domain]:
            continue
        result[domain][0] = {name: {"effective": result[domain][1][name]["incumbent"],
            "batch": result[domain][1][name]["batch"], "source": str(SIX), "version": "initial baseline"}
            for name in ("meta", "life")}
    return result


def reference_cell(value):
    if not value:
        return "未测"
    if "candidate" not in value:
        return f"b{value['batch']}：{value['effective']:.0%} 基线"
    decision = "✓" if value["accepted"] else "×"
    suffix = " [v3]" if value["version"].startswith("v3") else ""
    return f"b{value['batch']}：{value['incumbent']:.0%}→{value['candidate']:.0%} {decision}；生效 {value['effective']:.0%}{suffix}"


def publish_comparison(batch_dir):
    batch_dir = Path(batch_dir)
    protocol = json.loads((batch_dir / "protocol.json").read_text())
    status_path = batch_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {"domains": {}}
    references = reference_rows()
    last_step = protocol["iterations_per_domain"]
    profile = protocol.get("capability_profile", "historical-unrestricted")
    rows = []
    for domain in AGENTBENCH:
        measured = {value["iteration"] - 1: {**value, "batch": value["iteration"] - 1,
            "provenance": "original pre-evolve evaluation", "accuracy": value["successes"] / value["tasks"]}
            for value in status.get("domains", {}).get(domain, {}).get("evaluations", [])}
        final = batch_dir / f"{domain}_final_candidate_train.json"
        if final.is_file():
            value = json.loads(final.read_text())
            measured[value["optimization_step"]] = {**value, "accuracy": value["successes"] / value["tasks"],
                "provenance": "separate final candidate train evaluation; no selection change"}
        for step in range(protocol["iterations_per_domain"] + 1):
            reference = references[domain].get(step, {})
            rows.append({"domain": domain, "optimization_step": step,
                "ahe": measured.get(step), "meta": reference.get("meta"), "life": reference.get("life")})
    report = {"schema_version": 1, "batch": protocol["name"], "updated_at": status.get("updated_at"),
        "split": "train", "rows": rows,
        "capability_profile": profile,
        "unrestricted_ahe_reproduction": protocol.get("unrestricted_ahe_reproduction", True),
        "debugger_tasks_per_iteration": protocol.get("debugger_tasks_per_iteration"),
        "references": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (SIX, LIFE_V3) if path.is_file()},
        "notes": ["All three methods use different training batches; not a controlled same-task improvement curve.",
            "Optimization-step zero is baseline. AHE step N is measured by original iteration N+1.",
            "Final AHE step is measured separately after the unchanged original loop; not used for best-ever selection.",
            "Meta has six measured rounds; ALFWorld Life has six. Unmeasured later rounds remain null.",
            "Life steps 7-9 for DBBench/WebShop use the existing v3 continuation, not a fresh from-scratch run.",
            "Infrastructure errors count as zero in the full denominator. Test scores are not mixed with train scores.",
            "This report stays outside the optimizer filesystem and is never used in candidate selection."]}
    write_json(batch_dir / "round_comparison.json", report)
    lines = ["# AHE / Meta / Life 逐轮训练对比", "",
        f"AHE 能力配置：`{profile}`；限制版不等同于无限制原版复现。" if profile != "historical-unrestricted" else "AHE 为历史未加本次能力限制的接入版本。",
        "按已完成的优化次数对齐，0 为初始基线。各方法批次不同，不能当成同题连续提升曲线。",
        f"AHE 原版 iteration 1 测基线、iteration N+1 测第 N 次优化产物；第 {last_step} 次产物在原版循环结束后另测同批训练题，不参与选择。",
        "Meta 只有 6 轮；ALFWorld Life 只有 6 轮。DBBench/WebShop Life 第 7–9 轮为此前 v3 续跑，明确标记版本。",
        "✓ 接受，× 拒绝；Meta/Life 同时列 incumbent→candidate 和生效成绩。AHE 不套用 Life 的接受/拒绝规则。", "",
        "| 环境 | 优化次数 | AHE train | Meta train | Life train |", "| --- | ---: | --- | --- | --- |"]
    for row in rows:
        value = row["ahe"]
        ahe = "未测"
        if value:
            ahe = f"b{value['batch']:02d}：{value['successes']}/{value['tasks']} = {value['accuracy']:.0%}；infra {value['infrastructure_errors']}"
        lines.append(f"| {row['domain']} | {row['optimization_step']} | {ahe} | {reference_cell(row['meta'])} | {reference_cell(row['life'])} |")
    lines.extend(["", "这里只比较训练成绩，不使用测试集。对比报告在 optimizer 工作区之外，不反馈给 AHE。", ""])
    temporary = batch_dir / "COMPARISON.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(batch_dir / "COMPARISON.md")
    return report
