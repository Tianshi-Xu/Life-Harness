"""Expose public training evidence on demand, require a review, run one proposer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from evolution import evidence
from evolution.agentbench_hooks.contract import LAYERS, validate_source
from evolution.agentbench_hooks.runtime import CONTEXT_FIELDS, object_digest
from . import PROFILE
from .bundle import IDENTIFIER, filter_operations
from .context import ContextLimit, ProposalFailure, OUTPUT_LIMIT

ROOT = Path(__file__).resolve().parents[2]
MAX_TURNS = 32
OUTPUT_FILES = {"bundle.json", "diagnosis.json", "checks.json"}
READ_LOG = "evidence_reads.json"
HOOK_ARITIES = {"h2": 2, "h3": 1, "h4": 2, "h5": 1, "h3_message": 1,
                "h2_no_tool_message": 2, "drain_h2_messages": 0}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((ROOT / "evolution/agentbench_hooks").glob("*.py"))
    paths += [ROOT / "evolution/evidence.py", Path(__file__).with_name("PROMPT.md")]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def public_message(raw):
    if not isinstance(raw, dict) or raw.get("role") not in {"system", "user", "assistant", "tool"}:
        return None
    if str(raw.get("content", "")).startswith("[HARNESS_TRACE_V1]"):
        return None
    result = {k: copy.deepcopy(raw[k]) for k in ("role", "content", "tool_call_id", "name") if k in raw}
    if raw.get("tool_calls"):
        result["tool_calls"] = [{"id": c.get("id"), "type": c.get("type", "function"),
            "function": {k: c.get("function", {}).get(k) for k in ("name", "arguments")}}
            for c in raw["tool_calls"]]
    return result


def compact_case(row, context):
    pool, positions = [], {}

    def indices(messages):
        result = []
        for raw in messages:
            message = public_message(raw)
            if message is None:
                continue
            key = json.dumps(message, ensure_ascii=False, sort_keys=True)
            if key not in positions:
                positions[key] = len(pool)
                pool.append(message)
            result.append(positions[key])
        return result

    history = indices(row["history"])
    calls = []
    tools = None
    for call in row["calls"]:
        request = call.get("request", {})
        choices = call.get("response", {}).get("choices") or []
        if not choices or not request.get("tools"):
            continue  # Infra/incomplete calls are not behavioral replay cases.
        native_tools = [{"type": t.get("type", "function"), "function": {
            k: copy.deepcopy(t["function"][k]) for k in ("name", "description", "parameters")
            if k in t["function"]}} for t in request["tools"]]
        if tools is None:
            tools = native_tools
        if native_tools != tools:
            raise ValueError("tool descriptions/schemas changed inside recorded episode")
        assistant = public_message(choices[0]["message"])
        if assistant is None or assistant["role"] != "assistant":
            raise ValueError("recorded response is not an assistant message")
        calls.append({"turn": call.get("turn"), "message_indices": indices(request["messages"]),
                      "assistant": assistant})
    return {"case_id": str(row["index"]), "success": int(row["success"]), "status": row["status"],
            "infrastructure_error": bool(row.get("infrastructure_error")),
            "signals": evidence.classify(evidence.public_episode(row)), "context": context,
            "message_pool": pool, "history_indices": history, "calls": calls, "tools": tools or []}


def public_context(domain, row, db_rows):
    if domain == "dbbench":
        entry = db_rows[int(row["index"])]
        return {k: copy.deepcopy(entry[k]) for k in CONTEXT_FIELDS[domain] if k in entry}
    if domain == "webshop":
        instruction = next((m.get("content", "") for m in row["history"] if m.get("role") == "user"), "")
        return {"instruction": instruction}
    return None


def prepare(seed_run, eval_dir, destination, *, seed=0, db_source=None):
    """Read an incumbent TRAIN evaluation; never edit the seed or run an LLM."""
    seed_run, eval_dir, destination = map(lambda p: Path(p).resolve(), (seed_run, eval_dir, destination))
    if destination.exists() or any(p == destination or p in destination.parents for p in (seed_run, eval_dir)):
        raise ValueError("use a new destination outside historical seed/evaluation directories")
    state = json.loads((seed_run / "state.json").read_text())
    summary = json.loads((eval_dir / "summary.json").read_text())
    domain, records = state["domain"], state["active_snapshot"]
    if domain not in CONTEXT_FIELDS or summary.get("split") != "train" or summary.get("domain") != domain:
        raise ValueError("matching domain and explicit train split required; tests are forbidden")
    if object_digest(records) != state["active_snapshot_sha256"]:
        raise ValueError("active snapshot hash mismatch")
    content_hashes = sorted(v for k, v in summary.get("chain_hashes", {}).items() if k != "manifest")
    if content_hashes != sorted(r["sha256"] for r in records):
        raise ValueError("evidence is not from the exact incumbent content")
    outcomes = summary.get("per_task")
    if (not isinstance(outcomes, dict) or not outcomes or summary.get("n") != len(outcomes)
            or any(type(v) is not int or v not in (0, 1) for v in outcomes.values())
            or summary.get("successes") != sum(outcomes.values())):
        raise ValueError("complete binary evaluation summary required")
    active_sources = []
    for record in records:
        if not IDENTIFIER.fullmatch(record["mechanism_id"]):
            raise ValueError("invalid incumbent mechanism id")
        path = (seed_run / record["artifact_path"]).resolve()
        if seed_run not in path.parents or digest(path) != record["sha256"]:
            raise ValueError("incumbent artifact path/hash mismatch")
        validate_source(path, record["target_layer"])
        active_sources.append((record, path))
    rows, episodes, infra = {}, {}, 0
    for task, success in outcomes.items():
        if not re_task_id(task):
            raise ValueError("unsafe task id")
        row = json.loads((eval_dir / f"episode_{task}.json").read_text())
        if str(row["index"]) != task or type(row["success"]) is not int or row["success"] != success:
            raise ValueError("episode/outcome mismatch")
        if row.get("infrastructure_error"):
            infra += 1
            if success != 0:
                raise ValueError("infra failure must remain zero in the full denominator")
        rows[task], episodes[task] = row, evidence.public_episode(row)
    if infra != summary.get("infra_errors", 0):
        raise ValueError("infra count mismatch")
    db_rows = None
    if domain == "dbbench":
        path = Path(db_source) if db_source else ROOT / "AgentBench/data/dbbench/db_out_new.jsonl"
        # Only public context fields of confirmed TRAIN IDs enter the workspace.
        db_rows = [json.loads(line) for line in path.read_text().splitlines()]
    # Prioritize metadata, but do not filter or cap the available cases.
    choices = evidence.representatives(episodes, len(episodes), seed)
    destination.mkdir(parents=True)
    copied = []
    for record, source in active_sources:
        filename = f"active/{record['mechanism_id']}_r{record['revision']}.py"
        target = destination / filename
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != record["sha256"]:
            raise ValueError("incumbent changed during workspace preparation")
        copied.append({**record, "visible_path": filename})
    write_json(destination / "inventory.json", {"domain": domain, "snapshot_sha256": state["active_snapshot_sha256"],
                                              "records": copied})
    history_fields = {"round", "train_batch", "incumbent_score", "candidate_score", "accepted",
                      "bundle_id", "bundle_hypothesis", "active_snapshot_sha256"}
    operation_fields = {"operation_id", "mechanism_id", "op", "target_layer", "hypothesis"}
    history = []
    for row in state.get("rounds", []):
        history.append({**{k: row[k] for k in history_fields if k in row},
                        "operations": [{k: op[k] for k in operation_fields if k in op} for op in row.get("operations", [])]})
    write_json(destination / "history.json", history)
    case_dir = destination / "evidence"
    case_dir.mkdir()
    index, used = [], 0
    for task in choices:
        case = compact_case(rows[task], public_context(domain, rows[task], db_rows))
        filename = f"evidence/case_{task}.json"
        text = json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n"
        (destination / filename).write_text(text)
        used += len(text)
        index.append({k: case[k] for k in ("case_id", "success", "status", "infrastructure_error", "signals")})
        index[-1].update(path=filename, turns=rows[task]["turns"], calls=len(case["calls"]), chars=len(text), bytes=len(text.encode("utf-8")))
    write_json(destination / "evidence.json", index)
    write_json(destination / "summary.json", {
        "split": "train", "domain": domain, "n": len(outcomes), "successes": sum(outcomes.values()),
        "infra_errors": infra, "available_cases": len(index), "evidence_access": "on_demand",
        "evidence_chars": used, "case_limit": None, "evidence_char_limit": None, "seed": seed,
        "signals": dict(Counter(g for e in episodes.values() if not e["success"] for g in evidence.classify(e))),
        "note": "Signals are observations, not root causes. Infra failures remain in the full score denominator."})
    view = destination / "source_view"
    view.mkdir()
    for name in ("INTERFACE.md", "contract.py", "runtime.py"):
        shutil.copy2(ROOT / "evolution/agentbench_hooks" / name, view / name)
    shutil.copytree(ROOT / "evolution/agentbench_hooks/templates", view / "templates", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__).with_name("PROMPT.md"), destination / "OPTIMIZER.md")
    hashes = {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*") if p.is_file()}
    write_json(destination / "receipt.json", {"profile": PROFILE, "source_hashes": source_hashes(),
        "inputs": hashes, "seed_state_sha256": digest(seed_run / "state.json"), "evaluation_sha256": digest(eval_dir / "summary.json"),
        "prior_test_exposure": bool(state.get("prior_test_exposure", True)),
        "budget": {"max_proposer_turns": MAX_TURNS},
        "evidence": {"access": "on_demand", "available_cases": len(index), "case_limit": None, "char_limit": None},
        "acceptance": "offline checks cannot accept: strict improvement on the same complete training batch required"})
    return destination


def re_task_id(value):
    return isinstance(value, str) and bool(value) and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value)


def verify_inputs(workspace):
    workspace = Path(workspace).resolve()
    receipt = json.loads((workspace / "receipt.json").read_text())
    if receipt["profile"] != PROFILE or receipt["source_hashes"] != source_hashes():
        raise ValueError("profile/source changed: prepare a new workspace")
    for relative, expected in receipt["inputs"].items():
        if digest(safe_path(workspace, relative)) != expected:
            raise ValueError("frozen input changed")
    return receipt


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("workspace-relative paths only")
    result = (Path(root) / path).resolve()
    if Path(root).resolve() not in result.parents:
        raise ValueError("path escapes workspace")
    return result


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def load_cases(workspace, case_ids=None):
    """Host-side public case loading; proposer only opens requested case files."""
    workspace = Path(workspace)
    index = json.loads((workspace / "evidence.json").read_text())
    paths = {row["case_id"]: row["path"] for row in index}
    ids = list(paths) if case_ids is None else case_ids
    if any(not isinstance(case, str) or case not in paths for case in ids):
        raise ValueError("unknown training case id")
    return [json.loads(safe_path(workspace, paths[case]).read_text()) for case in ids]


def cited_cases(workspace, review):
    if not isinstance(review, dict) or not isinstance(review.get("cases"), list) or not review["cases"]:
        raise ValueError("diagnose at least one cited case; all available cases need not be analyzed")
    ids = []
    for case in review["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError("invalid cited case")
        ids.append(case["case_id"])
    return load_cases(workspace, ids)


def check_case_diagnoses(review, cases, declared_ids=None):
    selected = {c["case_id"]: c for c in cases}
    if (not isinstance(review, dict) or set(review) != {"schema_version", "cases", "rules", "shared_state_contract"}
            or review["schema_version"] != 1 or not isinstance(review["cases"], list)):
        raise ValueError("explicit diagnosis schema-v1 required before code")
    diagnosed, supported = set(), set()
    for row in review["cases"]:
        fields = {"case_id", "first_error_message_index", "observation", "root_cause", "confidence",
                  "public_trigger", "operation_ids", "regression_risk"}
        if not isinstance(row, dict) or set(row) != fields or row["case_id"] not in selected or row["case_id"] in diagnosed:
            raise ValueError("diagnose each cited case exactly once, with exact case fields")
        case = selected[row["case_id"]]
        index = row["first_error_message_index"]
        if index is not None and (type(index) is not int or not 0 <= index < len(case["history_indices"])):
            raise ValueError("first error must point to an observed public history message, or null if uncertain")
        if case["success"] and index is not None:
            raise ValueError("successful control must not be labelled an observed failure")
        if row["confidence"] not in {"low", "medium", "high"} or any(not nonempty(row[k]) for k in
                ("observation", "root_cause", "public_trigger", "regression_risk")):
            raise ValueError("explicit diagnosis, public trigger, uncertainty and regression risk required")
        refs = row["operation_ids"]
        if not isinstance(refs, list) or any(not isinstance(r, str) or (declared_ids is not None and r not in declared_ids) for r in refs):
            raise ValueError("diagnosis references unknown operation")
        diagnosed.add(row["case_id"])
        supported.update(refs)
    if diagnosed != set(selected):
        raise ValueError("every cited case needs diagnosis; uncited cases are optional")
    return supported


def validate_review(review, checks, cases, bundle, declared, active_records=()):
    if not isinstance(review, dict) or set(review) != {"schema_version", "cases", "rules", "shared_state_contract"} or review["schema_version"] != 1:
        raise ValueError("diagnosis.json requires cases, rules, shared_state_contract, schema_version=1")
    if not isinstance(checks, dict) or set(checks) != {"schema_version", "tests"} or checks["schema_version"] != 1:
        raise ValueError("checks.json requires schema_version=1 and tests")
    declared_ids = {o.get("operation_id") for o in declared["operations"] if isinstance(o, dict) and isinstance(o.get("operation_id"), str)}
    valid = {o["operation_id"]: o for o in bundle["operations"]}
    supported = check_case_diagnoses(review, cases, declared_ids)
    if supported & set(valid) != set(valid):
        raise ValueError("every surviving operation needs cited-case support")
    tests = checks["tests"]
    if not isinstance(tests, list) or not 1 <= len(tests) <= 64 or len(json.dumps(checks)) > 256000:
        raise ValueError("1..64 bounded declarative tests required")
    test_index, total_steps = {}, 0
    for test in tests:
        required = {"test_id", "rule_id", "kind", "steps"}
        if (not isinstance(test, dict) or not required <= set(test) or set(test) - required - {"context"}
                or not isinstance(test["test_id"], str) or not IDENTIFIER.fullmatch(test["test_id"])
                or test["test_id"] in test_index or test["kind"] not in {"trigger", "counterexample", "boundary", "cross_layer"}):
            raise ValueError("invalid or duplicate declarative test")
        steps = test["steps"]
        if not isinstance(steps, list) or not 1 <= len(steps) <= 40:
            raise ValueError("test needs 1..40 hook steps")
        total_steps += len(steps)
        asserted = False
        for step in steps:
            if (not isinstance(step, dict) or not {"hook", "args"} <= set(step)
                    or set(step) - {"hook", "args", "expected", "expected_subset", "expected_state"}
                    or step["hook"] not in HOOK_ARITIES or not isinstance(step["args"], list)
                    or len(step["args"]) != HOOK_ARITIES[step["hook"]]):
                raise ValueError("invalid hook/arguments; fixture code is forbidden")
            asserted |= any(k in step for k in ("expected", "expected_subset", "expected_state"))
        if not asserted:
            raise ValueError("fixture must assert behavior, not merely call hooks")
        test_index[test["test_id"]] = test
    if total_steps > 256:
        raise ValueError("offline step budget exceeded")
    covered, rule_ids, enabled, discarded = set(), set(), [], []
    for rule in review["rules"]:
        required = {"rule_id", "invariant", "layer_contracts", "operation_ids", "test_ids"}
        if not isinstance(rule, dict) or set(rule) != required or not nonempty(rule["rule_id"]) or rule["rule_id"] in rule_ids:
            raise ValueError("invalid or duplicate rule")
        rule_ids.add(rule["rule_id"])
        if not nonempty(rule["invariant"]) or not isinstance(rule["layer_contracts"], dict) or set(rule["layer_contracts"]) != LAYERS:
            raise ValueError("rule needs an invariant and explicit responsibilities for all four layers")
        if any(not nonempty(v) for v in rule["layer_contracts"].values()):
            raise ValueError("unchanged layers need a reason, not an omitted contract")
        refs = rule["operation_ids"]
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in declared_ids for r in refs):
            raise ValueError("rule references unknown operations")
        survivors = set(refs) & set(valid)
        if not survivors:
            discarded.append(rule["rule_id"])
            continue
        ids = rule["test_ids"]
        if not isinstance(ids, list) or any(not isinstance(t, str) or t not in test_index or test_index[t]["rule_id"] != rule["rule_id"] for t in ids):
            raise ValueError("rule references unknown/unrelated tests")
        kinds = {test_index[t]["kind"] for t in ids}
        if not {"trigger", "counterexample"} <= kinds:
            raise ValueError("each surviving rule requires trigger and non-trigger counterexample tests")
        if len({valid[r]["target_layer"] for r in survivors}) > 1 and "cross_layer" not in kinds:
            raise ValueError("cross-layer changes require a cross-layer behavior test")
        covered.update(survivors)
        enabled.extend(test_index[t] for t in ids)
    if covered != set(valid) or any(t["rule_id"] not in rule_ids for t in tests):
        raise ValueError("every surviving operation needs a tested rule; no orphan tests")
    state_contract = review["shared_state_contract"]
    if not isinstance(state_contract, list):
        raise ValueError("shared_state_contract must be a list (empty if unused)")
    mechanisms = {r["mechanism_id"] for r in active_records}
    for op in bundle["operations"]:
        if op["op"] == "delete":
            mechanisms.discard(op["mechanism_id"])
        else:
            mechanisms.add(op["mechanism_id"])
    state_keys = set()
    for item in state_contract:
        if (not isinstance(item, dict) or set(item) != {"key", "writers", "readers", "value_contract", "reset"}
                or item["reset"] != "per_episode" or not nonempty(item["key"]) or not nonempty(item["value_contract"])
                or not isinstance(item["writers"], list) or not isinstance(item["readers"], list)):
            raise ValueError("shared-state ownership/value/reset contract required")
        if item["key"] in state_keys or any(not isinstance(m, str) or m not in mechanisms for m in item["writers"] + item["readers"]):
            raise ValueError("duplicate state key or unknown writer/reader mechanism")
        state_keys.add(item["key"])
    return list({t["test_id"]: t for t in enabled}.values()), discarded


def check_candidate(workspace, candidate):
    """A preflight gate only: never commits artifacts or computes a success rate."""
    try:
        verify_inputs(workspace)
        workspace, candidate = Path(workspace), Path(candidate)
        inventory = json.loads((workspace / "inventory.json").read_text())
        declared = json.loads((candidate / "bundle.json").read_text())
        bundle, rejected = filter_operations(declared, inventory["domain"], inventory["records"],
                                              inventory["snapshot_sha256"], candidate)
        review = json.loads((candidate / "diagnosis.json").read_text())
        cases = cited_cases(workspace, review)
        checks = json.loads((candidate / "checks.json").read_text())
        tests, discarded_rules = validate_review(review, checks, cases, bundle, declared, inventory["records"])
        surviving_ids = {op["operation_id"] for op in bundle["operations"]}
        normalized_review = copy.deepcopy(review)
        for row in normalized_review["cases"]:
            row["operation_ids"] = [op for op in row["operation_ids"] if op in surviving_ids]
        normalized_review["rules"] = [rule for rule in normalized_review["rules"] if rule["rule_id"] not in discarded_rules]
        for rule in normalized_review["rules"]:
            rule["operation_ids"] = [op for op in rule["operation_ids"] if op in surviving_ids]
        normalized_checks = {"schema_version": 1, "tests": copy.deepcopy(tests)}
        tests = copy.deepcopy(tests)
        artifacts = [{"name": r["mechanism_id"], "layer": r["target_layer"],
                      "source": safe_path(workspace, r["visible_path"]).read_text()} for r in inventory["records"]]
        for op in bundle["operations"]:
            position = next((i for i, r in enumerate(artifacts) if r["name"] == op["mechanism_id"]), None)
            if op["op"] == "delete":
                artifacts.pop(position)
            else:
                item = {"name": op["mechanism_id"], "layer": op["target_layer"],
                        "source": safe_path(candidate, op["artifact"]).read_text()}
                if position is None:
                    artifacts.append(item)
                else:
                    artifacts[position] = item
        for test in tests:
            context = test.get("context")
            if context is not None and (not isinstance(context, dict) or set(context) - CONTEXT_FIELDS[inventory["domain"]]):
                raise ValueError("fixture context must contain public fields only")
            test["tools"] = next((case["tools"] for case in cases if case["tools"]), [])
        payload = {"domain": inventory["domain"], "artifacts": artifacts, "cases": cases, "tests": tests}
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(Path(__file__).with_name("worker.py"))],
            input=json.dumps(payload), capture_output=True, text=True, timeout=10, env={})
        if result.returncode:
            raise ValueError(f"replay worker exited {result.returncode} (resource limit or execution error)")
        report = json.loads(result.stdout)
        return {**report, "profile": PROFILE, "bundle": bundle, "rejected_operations": rejected,
                "diagnosis": normalized_review, "checks": normalized_checks,
                "discarded_rules": discarded_rules, "acceptance": "not evaluated; requires same-batch strict improvement"}
    except (ValueError, KeyError, TypeError, OSError, SyntaxError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000], "profile": PROFILE,
                "acceptance": "not evaluated"}


TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a bounded byte range of an allowlisted workspace file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_byte": {"type": "integer", "minimum": 0},
        "max_bytes": {"type": "integer", "minimum": 1, "maximum": 16000}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write bundle.json, diagnosis.json, checks.json or one artifacts/*.py file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "check_candidate", "description": "Validate diagnosis, cross-layer contracts, counterexamples and frozen-input hook replay. No LLM or environment call.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def tool_call(workspace, name, args):
    receipt = verify_inputs(workspace)
    if name == "check_candidate":
        report = check_candidate(workspace, workspace)
        return {k: v for k, v in report.items() if k not in {"bundle", "diagnosis", "checks"}}
    path = safe_path(workspace, args["path"])
    relative = str(path.relative_to(Path(workspace).resolve()))
    output = relative in OUTPUT_FILES or (path.parent == Path(workspace).resolve() / "artifacts" and path.suffix == ".py")
    if name == "read_file":
        if relative not in receipt["inputs"] and relative not in {"receipt.json", READ_LOG} and not output:
            raise ValueError("read outside input/output allowlist")
        start, maximum = args.get("start_byte", 0), args.get("max_bytes", 12000)
        if type(start) is not int or start < 0 or type(maximum) is not int or not 1 <= maximum <= 16000:
            raise ValueError("invalid byte range")
        with path.open("rb") as stream:
            stream.seek(start)
            data = stream.read(maximum)
        if relative.startswith("evidence/") and data:
            ledger_path = Path(workspace) / READ_LOG
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"case_ids": [], "requests": []}
            case_id = path.stem.removeprefix("case_")
            if case_id not in ledger["case_ids"]:
                ledger["case_ids"].append(case_id)
            ledger["requests"].append({"path": relative, "start_byte": start, "bytes": len(data)})
            write_json(ledger_path, ledger)
        return {"content": data.decode("utf-8", errors="replace"), "next_byte": start + len(data), "eof": len(data) < maximum}
    if name != "write_file" or not output or not isinstance(args["content"], str) or len(args["content"]) > 256000:
        raise ValueError("only bounded candidate outputs may be written; no shell, fixture code or frozen-input edits")
    if (relative == "bundle.json" or path.suffix == ".py") and not (Path(workspace) / "diagnosis.json").is_file():
        raise ValueError("write the explicit case diagnosis and rule contracts before code")
    if relative == "bundle.json" or path.suffix == ".py":
        review = json.loads((Path(workspace) / "diagnosis.json").read_text())
        check_case_diagnoses(review, cited_cases(workspace, review))
    path.parent.mkdir(exist_ok=True)
    path.write_text(args["content"])
    return {"path": relative, "bytes": path.stat().st_size}


def model_router_completion(messages, tools):
    key = os.environ.get("MR_API_KEY")
    if not key:
        raise ValueError("MR_API_KEY must be supplied in the process environment; never written to the workspace")
    host = os.environ.get("MR_HOST", "https://api.example.com").rstrip("/")
    request = urllib.request.Request(host + "/protocol/openai/v1/chat/completions", method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps({"model": "deepseek-v4-flash", "messages": messages, "tools": tools,
                         "tool_choice": "auto", "max_tokens": OUTPUT_LIMIT, "stream": False}).encode())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=900) as response:
        return json.load(response)


@contextmanager
def agent_workspace(workspace, checkpoint_dir, input_hash, max_turns):
    """Optional persistent public-only session; never holds a credential."""
    if checkpoint_dir is None:
        with tempfile.TemporaryDirectory(prefix="life-optimizer-") as temporary:
            sandbox = Path(temporary) / "workspace"
            shutil.copytree(workspace, sandbox)
            yield sandbox, None, None
        return
    root = Path(checkpoint_dir).resolve()
    if root == workspace or workspace in root.parents:
        raise ValueError("checkpoint must be separate from frozen inputs")
    checkpoint = root / "agent_checkpoint.json"
    sandbox = root / "workspace"
    saved = None
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["input_sha256"] != input_hash or saved["max_turns"] != max_turns:
            raise ValueError("agent checkpoint input/budget mismatch")
        if saved.get("inflight"):
            raise ValueError("interrupted API request has unknown outcome; retain checkpoint, do not blindly retry")
        if saved.get("error") == "output_truncated":
            raise ProposalFailure("output_truncated", "truncated response retained; do not retry this proposal")
        if saved.get("error") == "context_limit":
            raise ProposalFailure("context_limit", "context limit recorded; do not retry this proposal")
        actual = {str(p.relative_to(sandbox)): digest(p) for p in sandbox.rglob("*") if p.is_file()}
        if actual != saved["workspace_hashes"]:
            raise ValueError("agent checkpoint workspace changed")
        verify_inputs(sandbox)
    else:
        root.mkdir(parents=True, exist_ok=True)
        if sandbox.exists():
            raise ValueError("unbound checkpoint workspace already exists")
        shutil.copytree(workspace, sandbox)
    yield sandbox, checkpoint, saved


def save_agent(checkpoint, sandbox, **value):
    if checkpoint is None:
        return
    value["workspace_hashes"] = {str(p.relative_to(sandbox)): digest(p)
                               for p in sandbox.rglob("*") if p.is_file()}
    temporary = checkpoint.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(checkpoint)


def propose(workspace, destination, *, completion=model_router_completion, max_turns=MAX_TURNS,
            checkpoint_dir=None, recovery_outputs=None):
    """One agent, shared diagnostic/evolution context; cached validated output only."""
    workspace, destination = Path(workspace).resolve(), Path(destination).resolve()
    receipt = verify_inputs(workspace)
    if type(max_turns) is not int or not 1 <= max_turns <= MAX_TURNS:
        raise ValueError("proposer turn budget may be reduced, not expanded")
    input_hash = object_digest(receipt)
    if destination.exists():
        cache = json.loads((destination / "validation.json").read_text())
        if not cache.get("passed") or cache["input_sha256"] != input_hash:
            raise ValueError("cached proposal input mismatch")
        outputs = {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*")
                   if p.is_file() and p.name != "validation.json"}
        if outputs != cache["outputs"]:
            raise ValueError("cached proposal output changed")
        return destination / "bundle.json"
    if workspace == destination or workspace in destination.parents:
        raise ValueError("candidate output must be separate from the frozen workspace")
    with agent_workspace(workspace, checkpoint_dir, input_hash, max_turns) as (sandbox, checkpoint, saved):
        files = sorted(receipt["inputs"])
        messages = [{"role": "system", "content": (sandbox / "OPTIMIZER.md").read_text()},
            {"role": "user", "content": "Optimize one Life bundle. Workspace files: " + json.dumps(files) +
             ". Read inventory.json, summary.json, evidence.json (metadata index) and the full interface. Open individual evidence/case_ID.json traces as needed; no case-count or total-character quota. Diagnose only cases you cite. Create the review BEFORE code; use check_candidate and fix failures. Stop after validation passes."}]
        usage, validation, requests = {}, None, 0
        if saved:
            messages, usage, requests = saved["messages"], saved["usage"], saved["requests"]
        elif recovery_outputs is not None:
            # Caller supplies an audited, same-round archive, not old evolved policies.
            recovery = Path(recovery_outputs).resolve()
            for name in OUTPUT_FILES | {READ_LOG}:
                source = recovery / name
                if source.is_file():
                    shutil.copy2(source, sandbox / name)
            if (recovery / "artifacts").is_dir():
                shutil.copytree(recovery / "artifacts", sandbox / "artifacts", dirs_exist_ok=True)
            verify_inputs(sandbox)
            messages.append({"role": "user", "content":
                "Recover this interrupted same-round proposal from the existing diagnosis/output files if present. "
                "The interruption was output-token truncation, not a performance rejection. Reuse valid diagnosis; "
                "create the remaining outputs and call check_candidate. All 50 case files remain available on demand. "
                f"Only {max_turns} API requests remain in the original round budget; do not restart exhaustive analysis."})
        for _ in range(requests, max_turns):
            requests += 1
            save_agent(checkpoint, sandbox, input_sha256=input_hash, max_turns=max_turns,
                       messages=messages, usage=usage, requests=requests, inflight=True)
            try:
                response = completion(messages, TOOLS)
            except ContextLimit as exc:
                save_agent(checkpoint, sandbox, input_sha256=input_hash, max_turns=max_turns,
                           messages=messages, usage=usage, requests=requests, inflight=False,
                           error="context_limit", detail=str(exc))
                raise ProposalFailure("context_limit", str(exc)) from exc
            for key, value in response.get("usage", {}).items():
                if type(value) is int:
                    usage[key] = usage.get(key, 0) + value
            message = response["choices"][0]["message"]
            assistant = {k: message[k] for k in ("role", "content", "tool_calls", "reasoning_content") if k in message}
            messages.append(assistant)
            calls = message.get("tool_calls") or []
            if response["choices"][0].get("finish_reason") == "length":
                save_agent(checkpoint, sandbox, input_sha256=input_hash, max_turns=max_turns,
                           messages=messages, usage=usage, requests=requests, inflight=False,
                           error="output_truncated")
                raise ProposalFailure("output_truncated", "Flash output truncated: checkpoint retained; no blind preflight-feedback retry")
            for call in calls:
                try:
                    result = tool_call(sandbox, call["function"]["name"], json.loads(call["function"]["arguments"]))
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
            if not calls:
                validation = check_candidate(sandbox, sandbox)
                if validation["passed"]:
                    save_agent(checkpoint, sandbox, input_sha256=input_hash, max_turns=max_turns,
                               messages=messages, usage=usage, requests=requests, inflight=False)
                    break
                messages.append({"role": "user", "content": "Preflight failed; fix candidate outputs within the remaining turn budget: " + json.dumps(validation)})
            remaining = max_turns - requests
            if remaining in (8, 4, 2):
                messages.append({"role": "user", "content": f"{remaining} API requests remain. Prioritize finishing the diagnosis, artifacts, bundle and checks, then validate. Read additional cases only as needed; no case-count quota."})
            save_agent(checkpoint, sandbox, input_sha256=input_hash, max_turns=max_turns,
                       messages=messages, usage=usage, requests=requests, inflight=False)
        else:
            validation = check_candidate(sandbox, sandbox)
        if not validation["passed"]:
            raise ProposalFailure("no_valid_candidate", "no valid candidate within proposer budget: " + json.dumps(validation))
        verify_inputs(workspace)
        verify_inputs(sandbox)
        destination.mkdir(parents=True)
        write_json(destination / "diagnosis.json", validation["diagnosis"])
        write_json(destination / "checks.json", validation["checks"])
        write_json(destination / "bundle.json", validation["bundle"])
        for operation in validation["bundle"]["operations"]:
            if "artifact" in operation:
                target = destination / operation["artifact"]
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(sandbox / operation["artifact"], target)
        outputs = {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*") if p.is_file()}
        read_log = sandbox / READ_LOG
        access = json.loads(read_log.read_text()) if read_log.exists() else {"case_ids": [], "requests": []}
        write_json(destination / "validation.json", {**{k: v for k, v in validation.items() if k not in {"bundle", "diagnosis", "checks"}},
            "input_sha256": input_hash, "outputs": outputs, "usage": usage, "proposer_requests": requests,
            "evidence_access": access, "max_proposer_turns": max_turns, "model": "deepseek-v4-flash"})
    return destination / "bundle.json"


def strict_improvement(candidate, incumbent, *, expected_task_ids):
    """Integration gate: matching TRAIN tasks, complete denominator, strictly better."""
    expected = {str(task) for task in expected_task_ids}
    if len(expected) != 50:
        raise ValueError("predeclared complete 50-task batch required, not a screen subset")
    def values(summary):
        scores = summary.get("per_task")
        if (summary.get("split") != "train" or not isinstance(scores, dict) or not scores
                or summary.get("n") != len(scores) or any(type(v) is not int or v not in (0, 1) for v in scores.values())
                or type(summary.get("successes")) is not int or summary.get("successes") != sum(scores.values()) or set(scores) != expected):
            raise ValueError("complete binary train summary required")
        accuracy = float(summary.get("accuracy", sum(scores.values()) / len(scores)))
        if not math.isfinite(accuracy) or not math.isclose(accuracy, sum(scores.values()) / len(scores), abs_tol=1e-12):
            raise ValueError("accuracy does not match full-batch outcomes")
        return scores
    a, b = values(candidate), values(incumbent)
    if candidate.get("domain") not in CONTEXT_FIELDS or candidate.get("domain") != incumbent.get("domain") or set(a) != set(b):
        raise ValueError("candidate/incumbent domain or task keys differ")
    return sum(a.values()) > sum(b.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="offline only; make all public TRAIN cases available on demand")
    prepare_parser.add_argument("--seed-run", type=Path, required=True)
    prepare_parser.add_argument("--eval-dir", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--db-source", type=Path)
    propose_parser = commands.add_parser("propose", help="explicitly invokes the paid Flash proposer")
    propose_parser.add_argument("--workspace", type=Path, required=True)
    propose_parser.add_argument("--output", type=Path, required=True)
    check_parser = commands.add_parser("check", help="offline only; never accepts or evaluates a candidate")
    check_parser.add_argument("--workspace", type=Path, required=True)
    check_parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.seed_run, args.eval_dir, args.output, seed=args.seed, db_source=args.db_source))
    elif args.command == "propose":
        print(propose(args.workspace, args.output))
    else:
        report = check_candidate(args.workspace, args.candidate)
        print(json.dumps({k: v for k, v in report.items() if k not in {"bundle", "diagnosis", "checks"}}, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
