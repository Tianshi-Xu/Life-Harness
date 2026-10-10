"""Expose public training evidence on demand and validate executable H2–H5 code."""
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

from meta import evidence
from meta.harness_hooks import CONTEXT_FIELDS, object_digest, validate_source
from meta.environment_state import resolve as resolve_environment_state
from . import PROFILE
from meta.life_optimizer_v25.bundle import IDENTIFIER, filter_operations as legacy_filter_operations
from .context import ContextLimit, ProposalFailure, OUTPUT_LIMIT, verify_usage

ROOT = Path(__file__).resolve().parents[2]
MAX_REPAIRS = 3
WORKER_PATH = Path(__file__).with_name("worker.py")
OUTPUT_FILES = {"bundle.json", "diagnosis.json", "checks.json"}
READ_LOG = "evidence_reads.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def filter_operations(document, domain, records, snapshot, workspace):
    # Inputs are already bound to a verified, immutable host receipt. The model
    # need not copy a digest or invent a syntactically valid bookkeeping ID.
    if isinstance(document, dict):
        document = {**document, "domain": domain, "base_snapshot_sha256": snapshot}
        identifier = document.get("bundle_id")
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
            document["bundle_id"] = "bundle_" + object_digest(document)[:16]
    bundle, rejected = legacy_filter_operations(document, domain, records, snapshot, workspace)
    valid = []
    for op in bundle["operations"]:
        try:
            if op["op"] != "delete":
                validate_source(Path(workspace) / op["artifact"], op["target_layer"])
            valid.append(op)
        except (ValueError, SyntaxError, OSError, TypeError) as exc:
            rejected.append({"index": document["operations"].index(op), "operation_id": op["operation_id"],
                             "error": f"{type(exc).__name__}: {exc}"})
    if not valid:
        raise ValueError("all operations rejected")
    return {**bundle, "operations": valid}, rejected


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((ROOT / "meta/agentbench_hooks_v31").glob("*.py"))
    # Sealed validators, transport accounting and replay worker are reused unchanged.
    paths += list((ROOT / "meta/life_optimizer_v25").glob("*.py"))
    paths += [ROOT / "meta/evidence.py", Path(__file__).with_name("PROMPT.md"),
              Path(__file__).with_name("SCHEMAS.md"), ROOT / "meta/harness_hooks.py",
              ROOT / "meta/environment_state.py",
              ROOT / "meta/HARNESS_INTERFACE.md", ROOT / "meta/tau_hooks.py",
              ROOT / "AgentBench/src/server/harness/session.py",
              ROOT / "TauBench/src/tau2/agent/llm_agent.py",
              ROOT / "TauBench/src/tau2/orchestrator/orchestrator.py"]
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
        if "remaining" in call:
            if type(call["remaining"]) is not int or call["remaining"] < 0:
                raise ValueError("invalid recorded public turn budget")
            calls[-1]["remaining"] = call["remaining"]
        for field in ("environment_state_index", "action_environment_state_index"):
            if field in call:
                calls[-1][field] = call[field]
    result = {"case_id": str(row["index"]), "success": int(row["success"]), "status": row["status"],
            "infrastructure_error": bool(row.get("infrastructure_error")),
            "signals": evidence.classify(evidence.public_episode(row)), "context": context,
            "message_pool": pool, "history_indices": history, "calls": calls, "tools": tools or []}
    if "turn_limit" in row:
        if type(row["turn_limit"]) is not int or row["turn_limit"] <= 0:
            raise ValueError("invalid recorded public turn limit")
        result["turn_limit"] = row["turn_limit"]
    if "environment_states" in row:
        records = row["environment_states"]
        if not isinstance(records, list):
            raise ValueError("invalid recorded environment-state history")
        indices = [row.get("initial_environment_state_index")]
        indices += [call.get(field) for call in calls for field in
                    ("environment_state_index", "action_environment_state_index")]
        if any(index is not None and (type(index) is not int or not 0 <= index < len(records)) for index in indices):
            raise ValueError("invalid recorded environment-state reference")
        result["environment_states"] = copy.deepcopy(records)
        result["initial_environment_state_index"] = indices[0]
    elif any(field in call for call in calls for field in
             ("environment_state_index", "action_environment_state_index")):
        raise ValueError("environment-state reference has no recorded history")
    return result


def public_context(domain, row, db_rows):
    if domain in {"airline", "retail", "telecom"}:
        policy = next((m.get("content", "") for m in row["history"] if m.get("role") == "system"), "")
        return {"policy": policy}
    if domain == "dbbench":
        entry = db_rows[int(row["index"])]
        return {k: copy.deepcopy(entry[k]) for k in CONTEXT_FIELDS[domain] if k in entry}
    if domain == "webshop":
        instruction = next((m.get("content", "") for m in row["history"] if m.get("role") == "user"), "")
        return {"instruction": instruction}
    if domain == "os":
        # The host records only the description already shown to the solver.
        context = row.get("public_task_context")
        if not isinstance(context, dict) or set(context) != {"description"} or not isinstance(context["description"], str):
            raise ValueError("OS evidence requires the exact public task description")
        return copy.deepcopy(context)
    return None


def pattern_overview(index):
    """Group public observations without generating diagnoses or ranking policies."""
    groups = {}
    for row in index:
        for signal in sorted(set(row["signals"]) or {"unclassified"}):
            group = groups.setdefault(signal, {"signal": signal, "failed_case_ids": [],
                "successful_case_ids": [], "infra_case_ids": []})
            category = ("infra_case_ids" if row["infrastructure_error"] else
                        "successful_case_ids" if row["success"] else "failed_case_ids")
            group[category].append(row["case_id"])
    return {"note": "Overlapping symptom groups, not root causes or matched controls. "
            "Inspect traces to select comparable successes; infrastructure cases are separate.",
            "groups": [groups[key] for key in sorted(groups)],
            "successful_case_ids": [r["case_id"] for r in index if r["success"]],
            "behavioral_failure_case_ids": [r["case_id"] for r in index
                                            if not r["success"] and not r["infrastructure_error"]],
            "infra_case_ids": [r["case_id"] for r in index if r["infrastructure_error"]]}


def strategy_history(history, records):
    """Index observed bundle outcomes by mechanism; never infer operation-level gains."""
    mechanisms = {}
    for record in records:
        mechanisms[record["mechanism_id"]] = {
            "mechanism_id": record["mechanism_id"], "active": True,
            "target_layer": record["target_layer"], "revision": record["revision"], "attempts": []}
    fields = {"round", "train_batch", "incumbent_score", "candidate_score",
              "incumbent_successes", "candidate_successes", "accepted", "bundle_id"}
    for row in history:
        for op in row["operations"]:
            key = op.get("mechanism_id")
            if not isinstance(key, str):
                continue
            mechanism = mechanisms.setdefault(key, {"mechanism_id": key, "active": False,
                "target_layer": op.get("target_layer"), "attempts": []})
            mechanism["attempts"].append({**{k: row[k] for k in fields if k in row},
                **{k: op[k] for k in ("operation_id", "op", "hypothesis") if k in op}})
    return {"note": "Scores belong to whole bundles on each named batch, not individual "
            "mechanisms. Different batches are not a continuous curve. Rejected edits are "
            "not active policies; absence of history does not imply a new mechanism.",
            "mechanisms": [mechanisms[key] for key in sorted(mechanisms)]}


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
                      "incumbent_successes", "candidate_successes",
                      "bundle_id", "bundle_hypothesis", "active_snapshot_sha256"}
    operation_fields = {"operation_id", "mechanism_id", "op", "target_layer", "hypothesis"}
    history = []
    for row in state.get("rounds", []):
        history.append({**{k: row[k] for k in history_fields if k in row},
                        "operations": [{k: op[k] for k in operation_fields if k in op} for op in row.get("operations", [])]})
    write_json(destination / "history.json", history)
    write_json(destination / "strategy_history.json", strategy_history(history, copied))
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
    write_json(destination / "pattern_overview.json", pattern_overview(index))
    write_json(destination / "summary.json", {
        "split": "train", "domain": domain, "n": len(outcomes), "successes": sum(outcomes.values()),
        "infra_errors": infra, "available_cases": len(index), "evidence_access": "on_demand",
        "evidence_chars": used, "case_limit": None, "evidence_char_limit": None, "seed": seed,
        "signals": dict(Counter(g for e in episodes.values() if not e["success"] for g in evidence.classify(e))),
        "note": "Signals are observations, not root causes. Infra failures remain in the full score denominator."})
    view = destination / "source_view"
    view.mkdir()
    for name in ("INTERFACE.md", "contract.py", "runtime.py"):
        shutil.copy2(ROOT / "meta/agentbench_hooks_v31" / name, view / name)
    shutil.copy2(ROOT / "meta/HARNESS_INTERFACE.md", view / "INTERFACE.md")
    shutil.copy2(ROOT / "meta/harness_hooks.py", view / "harness_hooks.py")
    shutil.copy2(ROOT / "meta/environment_state.py", view / "environment_state.py")
    shutil.copy2(ROOT / "meta/tau_hooks.py", view / "tau_hooks.py")
    shutil.copy2(ROOT / "AgentBench/src/server/harness/session.py", view / "session.py")
    shutil.copytree(ROOT / "meta/agentbench_hooks_v31/templates", view / "templates", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__).with_name("PROMPT.md"), destination / "OPTIMIZER.md")
    shutil.copy2(Path(__file__).with_name("SCHEMAS.md"), destination / "SCHEMAS.md")
    hashes = {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*") if p.is_file()}
    write_json(destination / "receipt.json", {"profile": PROFILE, "source_hashes": source_hashes(),
        "inputs": hashes, "seed_state_sha256": digest(seed_run / "state.json"), "evaluation_sha256": digest(eval_dir / "summary.json"),
        "prior_test_exposure": bool(state.get("prior_test_exposure", True)),
        "budget": {"max_proposer_turns": None, "max_repairs": MAX_REPAIRS},
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


def load_cases(workspace, case_ids=None):
    """Host-side public case loading; proposer only opens requested case files."""
    workspace = Path(workspace)
    index = json.loads((workspace / "evidence.json").read_text())
    paths = {row["case_id"]: row["path"] for row in index}
    ids = list(paths) if case_ids is None else case_ids
    if any(not isinstance(case, str) or case not in paths for case in ids):
        raise ValueError("unknown training case id")
    return [json.loads(safe_path(workspace, paths[case]).read_text()) for case in ids]


def check_candidate(workspace, candidate):
    """A preflight gate only: never commits artifacts or computes a success rate."""
    try:
        verify_inputs(workspace)
        workspace, candidate = Path(workspace), Path(candidate)
        inventory = json.loads((workspace / "inventory.json").read_text())
        declared = json.loads((candidate / "bundle.json").read_text())
        bundle, rejected = filter_operations(declared, inventory["domain"], inventory["records"],
                                              inventory["snapshot_sha256"], candidate)
        # Diagnosis and hand-authored tests are optional notes, not code gates.
        # Replay selection is host-owned and cannot be narrowed by those notes.
        cases = load_cases(workspace)
        artifacts = [{"name": r["mechanism_id"], "layer": r["target_layer"],
                      "source": safe_path(workspace, r["visible_path"]).read_text()} for r in inventory["records"]]
        for op in bundle["operations"]:
            position = next((i for i, r in enumerate(artifacts) if r["name"] == op["mechanism_id"]), None)
            if op["op"] == "delete":
                artifacts.pop(position)
            else:
                validate_source(safe_path(candidate, op["artifact"]), op["target_layer"])
                item = {"name": op["mechanism_id"], "layer": op["target_layer"],
                        "source": safe_path(candidate, op["artifact"]).read_text()}
                if position is None:
                    artifacts.append(item)
                else:
                    artifacts[position] = item
        # Business databases are megabytes, unlike message-only replay inputs.
        # Keep the existing sandbox bounds, but isolate one state-bearing case
        # per process so 50 full initial databases cannot exhaust its memory.
        batches = [[case] for case in cases] if any(case.get("environment_states") for case in cases) else [cases]
        reports = []
        for batch in batches:
            payload = {"domain": inventory["domain"], "artifacts": artifacts,
                       "cases": batch}
            result = subprocess.run([sys.executable, "-I", "-S", "-B", str(WORKER_PATH)],
                input=json.dumps(payload), capture_output=True, text=True, timeout=10, env={})
            if result.returncode:
                raise ValueError(f"replay worker exited {result.returncode} (resource limit or execution error)")
            reports.append(json.loads(result.stdout))
        report = {**reports[0], "passed": all(r["passed"] for r in reports),
                  "calls_replayed": sum(r.get("calls_replayed", 0) for r in reports),
                  "cases": [case for r in reports for case in r.get("cases", [])]}
        if any("error" in r for r in reports):
            report["error"] = "; ".join(r["error"] for r in reports if "error" in r)[:1000]
        return {**report, "profile": PROFILE, "bundle": bundle, "rejected_operations": rejected,
                "acceptance": "not evaluated; requires same-batch strict improvement"}
    except (ValueError, KeyError, TypeError, OSError, SyntaxError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000], "profile": PROFILE,
                "acceptance": "not evaluated"}


TOOLS = [
    {"type": "function", "function": {"name": "read_case", "description": "Read TRAIN history, one solver call or approved business DB facts on demand. No total read quota; DB is not included unless explicitly requested.",
        "parameters": {"type": "object", "properties": {"case_id": {"type": "string"},
        "start_message": {"type": "integer", "minimum": 0}, "message_count": {"type": "integer", "minimum": 1},
        "call_index": {"type": "integer", "minimum": 0},
        "environment_state": {"type": "string", "enum": ["model", "action"]},
        "state_path": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]}}},
        "required": ["case_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a bounded byte range of an allowlisted workspace file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_byte": {"type": "integer", "minimum": 0},
        "max_bytes": {"type": "integer", "minimum": 1, "maximum": 16000}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write bundle.json, diagnosis.json, checks.json or one artifacts/*.py file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "check_candidate", "description": "Check executable H2–H5 safety, interfaces and frozen-input hook replay. Optional diagnosis/checks notes do not affect the result. No LLM or environment call.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def record_read(workspace, case_id, request):
    ledger_path = Path(workspace) / READ_LOG
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"case_ids": [], "requests": []}
    if case_id not in ledger["case_ids"]:
        ledger["case_ids"].append(case_id)
    ledger["requests"].append(request)
    write_json(ledger_path, ledger)


def read_case(workspace, args):
    if not isinstance(args, dict) or set(args) - {"case_id", "start_message", "message_count", "call_index", "environment_state", "state_path"}:
        raise ValueError("invalid read_case arguments")
    if ("environment_state" in args and (args["environment_state"] not in {"model", "action"} or "call_index" not in args)
            or "state_path" in args and "environment_state" not in args):
        raise ValueError("business-state reads require a call index and model/action boundary")
    case = load_cases(workspace, [args["case_id"]])[0]
    result = {k: copy.deepcopy(case[k]) for k in
              ("case_id", "success", "status", "infrastructure_error", "signals", "context", "tools")}
    result.update(history_length=len(case["history_indices"]), call_count=len(case["calls"]))
    if "call_index" in args:
        position = args["call_index"]
        if (set(args) & {"start_message", "message_count"} or type(position) is not int
                or not 0 <= position < len(case["calls"])):
            raise ValueError("invalid call index or mixed call/history request")
        call = case["calls"][position]
        result["call"] = {"call_index": position, "turn": call["turn"],
            "messages": [copy.deepcopy(case["message_pool"][i]) for i in call["message_indices"]],
            "assistant": copy.deepcopy(call["assistant"])}
        selection = {"call_index": position}
        for field in ("environment_state_index", "action_environment_state_index"):
            if field in call:
                result["call"][field] = call[field]
        if "environment_state" in args:
            phase = args["environment_state"]
            field = "environment_state_index" if phase == "model" else "action_environment_state_index"
            state = resolve_environment_state(case.get("environment_states", []), call.get(field))
            path = args.get("state_path", [])
            if not isinstance(path, list) or any(type(key) not in (str, int) or type(key) is int and key < 0 for key in path):
                raise ValueError("state_path must contain JSON object keys or nonnegative list indices")
            for key in path:
                if state is None:
                    raise ValueError("recorded environment state/path is unavailable")
                state = state[key]
            result["environment_state"] = state
            selection.update(environment_state=phase, state_path=path)
    else:
        start, count = args.get("start_message", 0), args.get("message_count", 12)
        if (type(start) is not int or type(count) is not int
                or not 0 <= start <= len(case["history_indices"]) or count < 1):
            raise ValueError("invalid history message range")
        end = min(len(case["history_indices"]), start + count)
        result["history"] = [{"message_index": i, "message": copy.deepcopy(
            case["message_pool"][case["history_indices"][i]])} for i in range(start, end)]
        result.update(next_message=end, eof=end == len(case["history_indices"]))
        selection = {"start_message": start, "message_count": end - start}
    index = json.loads((Path(workspace) / "evidence.json").read_text())
    path = next(r["path"] for r in index if r["case_id"] == case["case_id"])
    record_read(workspace, case["case_id"], {"tool": "read_case", "path": path, **selection,
                "bytes": len(json.dumps(result, ensure_ascii=False).encode("utf-8"))})
    return result


def tool_call(workspace, name, args):
    receipt = verify_inputs(workspace)
    if name == "read_case":
        return read_case(workspace, args)
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
            case_id = path.stem.removeprefix("case_")
            record_read(workspace, case_id, {"path": relative, "start_byte": start, "bytes": len(data)})
        return {"content": data.decode("utf-8", errors="replace"), "next_byte": start + len(data), "eof": len(data) < maximum}
    if name != "write_file" or not output or not isinstance(args["content"], str) or len(args["content"]) > 256000:
        raise ValueError("only bounded candidate outputs may be written; no shell, fixture code or frozen-input edits")
    path.parent.mkdir(exist_ok=True)
    path.write_text(args["content"])
    return {"path": relative, "bytes": path.stat().st_size}


def model_router_completion(messages, tools):
    key = os.environ.get("MR_API_KEY")
    if not key:
        raise ValueError("MR_API_KEY must be supplied in the process environment; never written to the workspace")
    host = os.environ.get("MR_HOST", "https://routify-pub.alibaba-inc.com").rstrip("/")
    request = urllib.request.Request(host + "/protocol/openai/v1/chat/completions", method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps({"model": "deepseek-v4-flash", "messages": messages, "tools": tools,
                         "tool_choice": "auto", "max_tokens": OUTPUT_LIMIT, "stream": False}).encode())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=900) as response:
        result = json.load(response)
    verify_usage(result)
    return result


@contextmanager
def agent_workspace(workspace, checkpoint_dir, input_hash, max_repairs):
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
        if saved["input_sha256"] != input_hash or saved.get("max_repairs") != max_repairs:
            raise ValueError("agent checkpoint input/budget mismatch")
        if saved.get("error") == "repair_limit":
            raise ProposalFailure("repair_limit", "repair opportunities exhausted; do not restart the same proposal")
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


def valid_snapshot(validation, sandbox, input_hash, requests):
    files = {'bundle.json': json.dumps(validation['bundle'], ensure_ascii=False, sort_keys=True)}
    for name in OUTPUT_FILES - {'bundle.json'}:
        path = safe_path(sandbox, name)
        if path.is_file():
            files[name] = path.read_text()
    for operation in validation['bundle']['operations']:
        if 'artifact' in operation:
            files[operation['artifact']] = safe_path(sandbox, operation['artifact']).read_text()
    body = {'input_sha256':input_hash, 'files':files, 'validated_at_request':requests,
            'operation_count':len(validation['bundle']['operations'])}
    return {**body, 'snapshot_sha256':object_digest(body)}


def verify_valid_snapshot(snapshot, input_hash):
    body = {k:v for k,v in snapshot.items() if k != 'snapshot_sha256'}
    if snapshot.get('input_sha256') != input_hash or snapshot.get('snapshot_sha256') != object_digest(body):
        raise ValueError('validated candidate snapshot changed or belongs to different inputs')


def restore_valid_snapshot(snapshot, sandbox, checkpoint, input_hash):
    verify_valid_snapshot(snapshot, input_hash)
    base = checkpoint.parent if checkpoint is not None else sandbox.parent
    destination = base / ('validated_' + snapshot['snapshot_sha256'])
    if destination.exists():
        actual = {str(p.relative_to(destination)):p.read_text() for p in destination.rglob('*') if p.is_file()}
        if actual != snapshot['files']:
            raise ValueError('restored validated candidate changed')
    else:
        destination.mkdir()
        for name, content in snapshot['files'].items():
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    validation = check_candidate(sandbox, destination)
    if not validation['passed']:
        raise ValueError('previously validated snapshot no longer passes: ' + json.dumps(validation))
    return destination, validation


def propose(workspace, destination, *, completion=model_router_completion,
            max_repairs=MAX_REPAIRS, checkpoint_dir=None, recovery_outputs=None):
    """One agent; unlimited interaction turns, at most three preflight repairs."""
    workspace, destination = Path(workspace).resolve(), Path(destination).resolve()
    receipt = verify_inputs(workspace)
    if type(max_repairs) is not int or not 0 <= max_repairs <= MAX_REPAIRS:
        raise ValueError("max_repairs must be an integer between 0 and 3")
    input_hash = object_digest(receipt)
    if destination.exists():
        cache = json.loads((destination / "validation.json").read_text())
        if (not cache.get("passed") or cache["input_sha256"] != input_hash
                or cache.get("max_repairs") != max_repairs):
            raise ValueError("cached proposal input/repair-policy mismatch")
        outputs = {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*")
                   if p.is_file() and p.name != "validation.json"}
        if outputs != cache["outputs"]:
            raise ValueError("cached proposal output changed")
        return destination / "bundle.json"
    if workspace == destination or workspace in destination.parents:
        raise ValueError("candidate output must be separate from the frozen workspace")
    with agent_workspace(workspace, checkpoint_dir, input_hash, max_repairs) as (sandbox, checkpoint, saved):
        files = sorted(receipt["inputs"])
        messages = [{"role": "system", "content": (sandbox / "OPTIMIZER.md").read_text()},
            {"role": "user", "content": "Optimize one Life bundle. Workspace files: " + json.dumps(files) +
             ". Read inventory.json, summary.json, pattern_overview.json, strategy_history.json, history.json and the full interface. Inspect failures and successful controls with read_case or read_file, as needed without a total evidence quota. Choose an evidence-backed mechanism, replacing/deleting ineffective policies when justified. Read SCHEMAS.md, write bundle.json and executable artifacts, then use check_candidate and fix genuine code failures. Diagnosis and checks files are optional notes, not requirements. Stop after validation passes."}]
        usage, validation, requests = {}, None, 0
        repairs_granted, failed_preflights = 0, 0
        recovered_qualified = False
        selected, fallback_reason = sandbox, None
        snapshot_file = checkpoint.parent / "validated_candidate.json" if checkpoint is not None else None
        snapshot = json.loads(snapshot_file.read_text()) if snapshot_file is not None and snapshot_file.exists() else None
        if snapshot is not None:
            verify_valid_snapshot(snapshot, input_hash)

        def remember(report):
            nonlocal snapshot
            if report["passed"]:
                snapshot = valid_snapshot(report, sandbox, input_hash, requests)
                if snapshot_file is not None:
                    temporary = snapshot_file.with_suffix(".tmp")
                    write_json(temporary, snapshot)
                    temporary.replace(snapshot_file)

        def persist(**extra):
            save_agent(checkpoint, sandbox, input_sha256=input_hash, max_repairs=max_repairs,
                       messages=messages, usage=usage, requests=requests,
                       repairs_granted=repairs_granted, failed_preflights=failed_preflights,
                       **extra)

        def repair_feedback(report):
            nonlocal repairs_granted, failed_preflights
            failed_preflights += 1
            if repairs_granted >= max_repairs:
                return False
            repairs_granted += 1
            messages.append({"role": "user", "content":
                f"Preflight failed. Repair opportunity {repairs_granted}/{max_repairs}. "
                "Fix the reported candidate defects and recheck. This is NOT an API-request limit: "
                "reading, reasoning and editing within this repair have no interaction-count cap. "
                + json.dumps({k: v for k, v in report.items() if k not in {"bundle", "diagnosis", "checks"}})})
            return True

        if saved:
            messages, usage, requests = saved["messages"], saved["usage"], saved["requests"]
            repairs_granted, failed_preflights = saved["repairs_granted"], saved["failed_preflights"]
        elif recovery_outputs is not None:
            # Caller supplies an audited, same-round archive, not old evolved policies.
            recovery = Path(recovery_outputs).resolve()
            for name in OUTPUT_FILES | {READ_LOG}:
                original = recovery / name
                if original.is_file():
                    shutil.copy2(original, sandbox / name)
            if (recovery / "artifacts").is_dir():
                shutil.copytree(recovery / "artifacts", sandbox / "artifacts", dirs_exist_ok=True)
            verify_inputs(sandbox)
            messages.append({"role": "user", "content":
                "Recover this interrupted same-round proposal from the existing diagnosis/output files if present. "
                "The preceding attempt was interrupted or failed preflight, not rejected on performance. Preserve useful code and notes; "
                "create the remaining outputs and call check_candidate. All 50 case files remain available on demand. "
                "There is no API-request-count cap; at most three preflight repair opportunities are allowed."})
            validation = check_candidate(sandbox, sandbox)
            remember(validation)
            recovered_qualified = bool(validation["passed"] and not validation.get("rejected_operations"))
            if recovered_qualified:
                persist(inflight=False, recovered_qualified=True)
            elif not validation["passed"] and not repair_feedback(validation):
                persist(inflight=False, error="repair_limit")
                raise ProposalFailure("repair_limit", "recovered outputs need repair but no repair opportunity remains")

        while not recovered_qualified:
            requests += 1
            persist(inflight=True)
            try:
                response = completion(messages, TOOLS)
            except ContextLimit as exc:
                persist(inflight=False, error="context_limit", detail=str(exc))
                if snapshot is not None:
                    selected, validation = restore_valid_snapshot(snapshot, sandbox, checkpoint, input_hash)
                    fallback_reason = "context_limit_after_valid_candidate"
                    persist(inflight=False, error="context_limit", detail=str(exc), fallback_reason=fallback_reason)
                    break
                raise ProposalFailure("context_limit", str(exc)) from exc
            for key, value in response.get("usage", {}).items():
                if type(value) is int:
                    usage[key] = usage.get(key, 0) + value
            message = response["choices"][0]["message"]
            assistant = {k: message[k] for k in ("role", "content", "tool_calls", "reasoning_content") if k in message}
            messages.append(assistant)
            calls = message.get("tool_calls") or []
            if response["choices"][0].get("finish_reason") == "length":
                persist(inflight=False, error="output_truncated")
                if snapshot is not None:
                    selected, validation = restore_valid_snapshot(snapshot, sandbox, checkpoint, input_hash)
                    fallback_reason = "output_truncated_after_valid_candidate"
                    persist(inflight=False, fallback_reason=fallback_reason)
                    break
                raise ProposalFailure("output_truncated", "Flash output truncated: checkpoint retained; no blind preflight-feedback retry")
            failed_report = None
            for call in calls:
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if call["function"]["name"] == "check_candidate":
                        validation = check_candidate(sandbox, sandbox)
                        remember(validation)
                        failed_report = None if validation["passed"] else validation
                        result = {k: v for k, v in validation.items() if k not in {"bundle", "diagnosis", "checks"}}
                    else:
                        result = tool_call(sandbox, call["function"]["name"], arguments)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
                    if call.get("function", {}).get("name") == "check_candidate":
                        validation = failed_report = result
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
            if not calls:
                validation = check_candidate(sandbox, sandbox)
                remember(validation)
                if validation["passed"]:
                    persist(inflight=False)
                    break
                if snapshot is not None:
                    selected, validation = restore_valid_snapshot(snapshot, sandbox, checkpoint, input_hash)
                    fallback_reason = "later_edits_invalid_after_valid_candidate"
                    persist(inflight=False, fallback_reason=fallback_reason)
                    break
                failed_report = validation
            if failed_report is not None and not repair_feedback(failed_report):
                if snapshot is not None:
                    selected, validation = restore_valid_snapshot(snapshot, sandbox, checkpoint, input_hash)
                    fallback_reason = "repair_limit_after_valid_candidate"
                    persist(inflight=False, fallback_reason=fallback_reason)
                    break
                persist(inflight=False, error="repair_limit")
                raise ProposalFailure("repair_limit",
                    f"No valid candidate after {max_repairs} preflight repairs; incumbent retained. "
                    + json.dumps(failed_report))
            persist(inflight=False)
        if not validation["passed"]:
            raise ProposalFailure("no_valid_candidate", "no valid candidate: " + json.dumps(validation))
        verify_inputs(workspace)
        verify_inputs(sandbox)
        destination.mkdir(parents=True)
        for name in OUTPUT_FILES - {"bundle.json"}:
            original = safe_path(selected, name)
            if original.is_file():
                shutil.copy2(original, destination / name)
        write_json(destination / "bundle.json", validation["bundle"])
        for operation in validation["bundle"]["operations"]:
            if "artifact" in operation:
                target = destination / operation["artifact"]
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(selected / operation["artifact"], target)
        outputs = {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*") if p.is_file()}
        read_log = sandbox / READ_LOG
        access = json.loads(read_log.read_text()) if read_log.exists() else {"case_ids": [], "requests": []}
        write_json(destination / "validation.json", {**{k: v for k, v in validation.items() if k not in {"bundle", "diagnosis", "checks"}},
            "input_sha256": input_hash, "outputs": outputs, "usage": usage, "proposer_requests": requests,
            "evidence_access": access, "max_proposer_turns": None, "max_repairs": max_repairs,
            "repairs_granted": repairs_granted, "failed_preflights": failed_preflights,
            "model": "deepseek-v4-flash", "fallback_reason": fallback_reason,
            "recovered_qualified": recovered_qualified,
            "validated_snapshot_sha256": snapshot.get("snapshot_sha256") if snapshot is not None else None})
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
    propose_parser.add_argument("--checkpoint", type=Path,
                                help="public-only session/usage archive; defaults beside the candidate output")
    check_parser = commands.add_parser("check", help="offline only; never accepts or evaluates a candidate")
    check_parser.add_argument("--workspace", type=Path, required=True)
    check_parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.seed_run, args.eval_dir, args.output, seed=args.seed, db_source=args.db_source))
    elif args.command == "propose":
        checkpoint = args.checkpoint or args.output.parent / (args.output.name + "_session")
        print(propose(args.workspace, args.output, checkpoint_dir=checkpoint))
    else:
        report = check_candidate(args.workspace, args.candidate)
        print(json.dumps({k: v for k, v in report.items() if k not in {"bundle", "diagnosis", "checks"}}, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
