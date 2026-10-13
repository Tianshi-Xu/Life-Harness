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
        return "Not measured"
    if "candidate" not in value:
        return f"b{value['batch']}: {value['effective']:.0%} baseline"
    decision = "✓" if value["accepted"] else "×"
    suffix = " [v3]" if value["version"].startswith("v3") else ""
    return f"b{value['batch']}: {value['incumbent']:.0%}→{value['candidate']:.0%} {decision}; effective {value['effective']:.0%}{suffix}"


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
    lines = ["# AHE / Meta-Harness / Life-Harness training comparison by round", "",
        f"AHE capability profile: `{profile}`; restricted AHE is not an unrestricted upstream reproduction." if profile != "historical-unrestricted" else "AHE uses the historical adapter without the current capability restrictions.",
        "Align results by completed optimization steps; step 0 is the initial baseline. Methods use different batches, so this is not a same-task improvement curve.",
        f"Upstream AHE iteration 1 evaluates the baseline; iteration N+1 evaluates the result of optimization step N. The step {last_step} candidate is evaluated separately on the same TRAIN batch after the upstream loop ends, without affecting selection.",
        "Meta-Harness has 6 rounds; Life-Harness on ALFWorld has 6 rounds. Life-Harness steps 7–9 on DBBench/WebShop are the earlier v3 continuation and are labeled accordingly.",
        "✓ accepted, × rejected; Meta-Harness/Life-Harness show incumbent→candidate and effective scores. AHE retains its own selection rules.", "",
        "| Environment | Optimization steps | AHE train | Meta-Harness train | Life-Harness train |", "| --- | ---: | --- | --- | --- |"]
    for row in rows:
        value = row["ahe"]
        ahe = "Not measured"
        if value:
            ahe = f"b{value['batch']:02d}: {value['successes']}/{value['tasks']} = {value['accuracy']:.0%}; infra {value['infrastructure_errors']}"
        lines.append(f"| {row['domain']} | {row['optimization_step']} | {ahe} | {reference_cell(row['meta'])} | {reference_cell(row['life'])} |")
    lines.extend(["", "This report compares TRAIN scores only, without using TEST results. It stays outside the optimizer workspace and is not supplied to AHE.", ""])
    temporary = batch_dir / "COMPARISON.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(batch_dir / "COMPARISON.md")
    return report
