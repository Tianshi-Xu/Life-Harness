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
import signal
import subprocess
import sys
import tempfile
import urllib.request

from evolution import evidence
from evolution.harness_hooks import CONTEXT_FIELDS, object_digest, validate_source, register_domain
from evolution.environment_state import resolve as resolve_environment_state
from . import PROFILE
from evolution.life_optimizer_v25.bundle import IDENTIFIER, RejectedOperations, filter_operations as legacy_filter_operations
from .context import ContextLimit, ProposalFailure, OUTPUT_LIMIT, verify_usage, flash_counter

ROOT = Path(__file__).resolve().parents[2]
MAX_REPAIRS = 3
MAX_TOTAL_TOKENS = 10_000_000
WORKER_PATH = Path(__file__).with_name("worker.py")
OUTPUT_FILES = {"bundle.json", "diagnosis.json", "checks.json"}
READ_LOG = "evidence_reads.json"
CONTEXT_ARCHIVE = 'context_history'
CONTEXT_CHECKPOINT_TOKENS = 65536  # Refresh trigger, never a Flash context/turn limit.


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
                             "mechanism_id": op["mechanism_id"], "artifact": op.get("artifact"),
                             "error": f"{type(exc).__name__}: {exc}"})
    if not valid:
        raise RejectedOperations(rejected)
    return {**bundle, "operations": valid}, rejected


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((ROOT / "evolution/agentbench_hooks_v31").glob("*.py"))
    # Bind all shared validators, transport accounting and replay code to this workspace.
    paths += list((ROOT / "evolution/life_optimizer_v25").glob("*.py"))
    paths += [ROOT / "evolution/evidence.py", Path(__file__).with_name("PROMPT.md"),
              Path(__file__).with_name("SCHEMAS.md"), ROOT / "evolution/harness_hooks.py",
              ROOT / "evolution/environment_state.py",
              ROOT / "evolution/HARNESS_INTERFACE.md", ROOT / "evolution/tau_hooks.py",
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

    def tool_contracts(values):
        return [{**tool, 'function': {key: value for key, value in tool['function'].items()
                                     if key != 'description'}} for tool in values]

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
        if tool_contracts(native_tools) != tool_contracts(tools):
            raise ValueError(f"native tool names/parameters changed inside recorded episode: case={row.get('index')}, "
                             f"turn={call.get('turn')}; H3 description-only changes are allowed, native contracts must match")
        assistant = public_message(choices[0]["message"])
        if assistant is None or assistant["role"] != "assistant":
            raise ValueError(f"recorded response is not an assistant message: case={row.get('index')}, "
                             f"turn={call.get('turn')}; require a public role='assistant' response")
        calls.append({"turn": call.get("turn"), "message_indices": indices(request["messages"]),
                      "assistant": assistant})
        if native_tools != tools:
            calls[-1]['tool_descriptions'] = {tool['function']['name']: tool['function'].get('description')
                                              for tool in native_tools}
        if "remaining" in call:
            if type(call["remaining"]) is not int or call["remaining"] < 0:
                raise ValueError(f"invalid recorded public turn budget: case={row.get('index')}, "
                                 f"turn={call.get('turn')}, remaining={call['remaining']!r}; expected nonnegative integer")
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
            raise ValueError(f"invalid recorded public turn limit: case={row.get('index')}, "
                             f"turn_limit={row['turn_limit']!r}; expected positive integer")
        result["turn_limit"] = row["turn_limit"]
    if "environment_states" in row:
        records = row["environment_states"]
        if not isinstance(records, list):
            raise ValueError(f"invalid recorded environment-state history: case={row.get('index')}; "
                             "environment_states must be a list of hash-bound records")
        indices = [row.get("initial_environment_state_index")]
        indices += [call.get(field) for call in calls for field in
                    ("environment_state_index", "action_environment_state_index")]
        if any(index is not None and (type(index) is not int or not 0 <= index < len(records)) for index in indices):
            raise ValueError(f"invalid recorded environment-state reference: case={row.get('index')}; "
                             f"expected None or integer index in [0,{len(records)})")
        result["environment_states"] = copy.deepcopy(records)
        result["initial_environment_state_index"] = indices[0]
    elif any(field in call for call in calls for field in
             ("environment_state_index", "action_environment_state_index")):
        raise ValueError(f"environment-state reference has no recorded history: case={row.get('index')}; "
                         "export the native public snapshots or omit unavailable state references")
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
    if domain == "alfworld":
        return None
    context = row.get("public_task_context")
    if context is not None and (not isinstance(context, dict) or set(context) - CONTEXT_FIELDS[domain]):
        raise ValueError("custom TRAIN context contains undeclared fields")
    return copy.deepcopy(context)


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
    fields = {"round", "train_batch", "incumbent_score", "candidate_score", "history_source",
              "incumbent_successes", "candidate_successes", "accepted", "bundle_id"}
    for row in history:
        for op in row["operations"]:
            key = op.get("mechanism_id")
            if not isinstance(key, str):
                continue
            mechanism = mechanisms.setdefault(key, {"mechanism_id": key, "active": False,
                "target_layer": op.get("target_layer"), "attempts": []})
            mechanism["attempts"].append({**{k: row[k] for k in fields if k in row},
                **{k: op[k] for k in ("operation_id", "op", "hypothesis", "historical_code_path",
                                      "historical_code_sha256", "historical_code_unavailable") if k in op}})
    return {"note": "Scores belong to whole bundles on each named batch, not individual "
            "mechanisms. Different batches are not a continuous curve. Rejected edits are "
            "not active policies; absence of history does not imply a new mechanism.",
            "mechanisms": [mechanisms[key] for key in sorted(mechanisms)]}


def archive_history_code(seed_run, destination, row, operation, namespace=None):
    """Optional, hash-bound past code for on-demand inspection, never execution."""
    relative, expected = operation.get('artifact'), operation.get('artifact_sha256')
    number, identifier = row.get('round'), operation.get('operation_id')
    if (not isinstance(relative, str) or not isinstance(expected, str) or type(number) is not int
            or not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier)):
        return {}
    try:
        source = safe_path(seed_run, f'proposals/round_{number:02d}/{relative}')
        if not source.is_file() or digest(source) != expected:
            return {'historical_code_unavailable': 'not archived locally or recorded hash mismatch'}
        folder = 'history_code' if namespace is None else f'history_code/{namespace}'
        filename = f'{folder}/r{number:02d}_{identifier}.py'
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != expected:
            raise RuntimeError('historical code changed during copy; retain the original history and audit before preparation')
        return {'historical_code_path': filename, 'historical_code_sha256': expected}
    except (ValueError, OSError):
        return {'historical_code_unavailable': 'historical path is unavailable or outside the seed workspace'}


def paired_history_outcomes(seed_run, row):
    """Optional task-level TRAIN feedback from the native round receipt layout."""
    number, batch = row.get('round'), row.get('train_batch')
    if type(number) is not int or type(batch) is not int:
        return {}
    try:
        paths = [safe_path(seed_run, f'evals/{name}/life_train/summary.json') for name in
                 (f'initial_batch_{batch:02d}', f'candidate_{number:02d}')]
        if not all(path.is_file() for path in paths):
            return {}
        before, after = [json.loads(path.read_text()) for path in paths]
        a, b = before.get('per_task'), after.get('per_task')
        if (before.get('split') != 'train' or after.get('split') != 'train'
                or not isinstance(a, dict) or not isinstance(b, dict) or not a or set(a) != set(b)
                or before.get('n') != len(a) or after.get('n') != len(b)
                or any(type(value) is not int or value not in (0, 1) for value in [*a.values(), *b.values()])
                or before.get('successes') != sum(a.values()) or after.get('successes') != sum(b.values())
                or row.get('incumbent_successes') != sum(a.values()) or row.get('candidate_successes') != sum(b.values())):
            return {}
        return {'paired_gain_case_ids': sorted(key for key in a if not a[key] and b[key]),
                'paired_regression_case_ids': sorted(key for key in a if a[key] and not b[key]),
                'paired_train_n': len(a)}
    except (ValueError, OSError, KeyError, TypeError):
        return {}


def history_evidence(source, destination, row, domain, db_rows, ordinal, variants=('candidate', 'incumbent')):
    """Pin optional public TRAIN files for lazy reads, without copying large DB logs."""
    if not paired_history_outcomes(source, row):
        return {}
    result = {}
    for variant, name in (('candidate', f"candidate_{row['round']:02d}"),
                          ('incumbent', f"initial_batch_{row['train_batch']:02d}")):
        if variant not in variants:
            continue
        try:
            directory = safe_path(source, f'evals/{name}/life_train')
            summary_path = directory / 'summary.json'
            summary = json.loads(summary_path.read_text())
            if summary.get('domain') != domain or any(not re_task_id(task) for task in summary['per_task']):
                raise ValueError('matching domain and safe TRAIN case IDs required')
            episodes = {}
            for task, success in summary['per_task'].items():
                episodes[task] = {'sha256': digest(safe_path(directory, f'episode_{task}.json')), 'success': success}
                if domain == 'dbbench':
                    episodes[task]['context'] = public_context(domain, {'index': task}, db_rows)
            suffix = '' if variant == 'candidate' else '_incumbent'
            relative = f'history_evidence/attempt_{ordinal:02d}{suffix}.json'
            write_json(destination / relative, {'split': 'train', 'domain': domain,
                'source': str(directory), 'summary_sha256': digest(summary_path), 'episodes': episodes})
            result.update({f'{variant}_evidence': relative, f'{variant}_evidence_cases': len(episodes)})
        except (ValueError, OSError, KeyError, TypeError, IndexError):
            result[f'{variant}_evidence_unavailable'] = 'matching complete public TRAIN episodes are not archived locally'
    return result


def load_history_case(workspace, history_index, case_id, variant='candidate'):
    history = json.loads((Path(workspace) / 'history.json').read_text())
    if type(history_index) is not int or not 0 <= history_index < len(history):
        raise ValueError(f'history_index={history_index!r}; expected integer in [0,{len(history)}) from history.json')
    attempt = history[history_index]
    relative = attempt.get(f'{variant}_evidence')
    if relative is None:
        raise ValueError(f'history[{history_index}] has no archived public {variant} TRAIN trace; '
                         f'use available {variant}_evidence entries; never substitute TEST or another view')
    index = json.loads(safe_path(workspace, relative).read_text())
    if index.get('split') != 'train' or not isinstance(case_id, str) or case_id not in index['episodes']:
        raise ValueError(f'unknown historical TRAIN case_id={case_id!r}; use cases from {relative}')
    directory = Path(index['source'])
    if digest(directory / 'summary.json') != index['summary_sha256']:
        raise ValueError('historical TRAIN summary changed; restore the archived evidence, not its hash')
    entry = index['episodes'][case_id]
    data = safe_path(directory, f'episode_{case_id}.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError(f'historical TRAIN episode changed: history_index={history_index}, case_id={case_id}; '
                         'restore the recorded file, never rewrite the evidence index or replay paid work')
    row = json.loads(data)
    if (str(row['index']) != case_id or type(row['success']) is not int or row['success'] != entry['success']
            or row.get('infrastructure_error') and row['success']):
        raise ValueError('historical episode/outcome mismatch; require matching index, binary TRAIN result and infra=0-success')
    context = entry.get('context') if index['domain'] == 'dbbench' else public_context(index['domain'], row, None)
    return compact_case(row, context), relative, entry['sha256']


def prepare(seed_run, eval_dir, destination, *, seed=0, db_source=None, additional_history=()):
    """Read an incumbent TRAIN evaluation; never edit the seed or run an LLM."""
    seed_run, eval_dir, destination = map(lambda p: Path(p).resolve(), (seed_run, eval_dir, destination))
    if destination.exists() or any(p == destination or p in destination.parents for p in (seed_run, eval_dir)):
        raise ValueError(f"use a new destination outside historical seed/evaluation directories: {destination}; "
                         "never overwrite existing paid workspaces or historical evidence")
    state = json.loads((seed_run / "state.json").read_text())
    summary = json.loads((eval_dir / "summary.json").read_text())
    domain, records = state["domain"], state["active_snapshot"]
    if "public_context_fields" in state:
        register_domain(domain, state["public_context_fields"])
    history_sources = [(seed_run, state, None)]
    for source in additional_history:
        source = Path(source).resolve()
        past = json.loads((source / 'state.json').read_text())
        if past.get('domain') != domain:
            raise ValueError(f"additional TRAIN history domain mismatch: {source}; expected {domain!r}")
        history_sources.append((source, past, 'past_' + object_digest(past)[:12]))
    if domain not in CONTEXT_FIELDS or summary.get("split") != "train" or summary.get("domain") != domain:
        raise ValueError(f"matching domain and explicit train split required; tests are forbidden: "
                         f"seed domain={domain!r}, summary domain={summary.get('domain')!r}, split={summary.get('split')!r}")
    if object_digest(records) != state["active_snapshot_sha256"]:
        raise ValueError("active snapshot hash mismatch: state.active_snapshot_sha256 must match "
                         "the complete active_snapshot; audit/restore the original state, not the hash field")
    content_hashes = sorted(v for k, v in summary.get("chain_hashes", {}).items() if k != "manifest")
    if content_hashes != sorted(r["sha256"] for r in records):
        raise ValueError("evidence is not from the exact incumbent content: summary.chain_hashes must contain "
                         "every active artifact's matching sha256; use that snapshot's verified TRAIN evaluation")
    outcomes = summary.get("per_task")
    if (not isinstance(outcomes, dict) or not outcomes or summary.get("n") != len(outcomes)
            or any(type(v) is not int or v not in (0, 1) for v in outcomes.values())
            or summary.get("successes") != sum(outcomes.values())):
        raise ValueError("complete binary evaluation summary required: per_task must be a nonempty "
                         "task_id→integer 0/1 map, n=len(per_task), successes=sum(per_task); do not drop failures")
    active_sources = []
    for record in records:
        if not IDENTIFIER.fullmatch(record["mechanism_id"]):
            raise ValueError(f"invalid incumbent mechanism id: {record['mechanism_id']!r}; expected {IDENTIFIER.pattern!r}")
        path = (seed_run / record["artifact_path"]).resolve()
        if seed_run not in path.parents or digest(path) != record["sha256"]:
            raise ValueError(f"incumbent artifact path/hash mismatch: mechanism={record['mechanism_id']!r}, "
                             f"artifact={record['artifact_path']!r}; require an in-seed file with its recorded sha256")
        validate_source(path, record["target_layer"])
        active_sources.append((record, path))
    rows, episodes, infra = {}, {}, 0
    for task, success in outcomes.items():
        if not re_task_id(task):
            raise ValueError(f"unsafe task id: {task!r}; use nonempty string IDs containing only letters/digits/_-; "
                             "not paths or parent traversal")
        row = json.loads((eval_dir / f"episode_{task}.json").read_text())
        if str(row["index"]) != task or type(row["success"]) is not int or row["success"] != success:
            raise ValueError(f"episode/outcome mismatch: task={task!r}, expected index={task!r}, success={success}; "
                             "episode index and integer success must agree with the complete summary")
        if row.get("infrastructure_error"):
            infra += 1
            if success != 0:
                raise ValueError(f"infra failure must remain zero in the full denominator: task={task!r}, "
                                 f"recorded success={success}; retain the episode as 0, never remove it")
        rows[task], episodes[task] = row, evidence.public_episode(row)
    if infra != summary.get("infra_errors", 0):
        raise ValueError(f"infra count mismatch: summary infra_errors={summary.get('infra_errors', 0)}, "
                         f"counted episodes={infra}; reconcile the original complete evaluation")
    db_rows = None
    if domain == "dbbench":
        path = Path(db_source) if db_source else ROOT / "AgentBench/data/dbbench/db_out_new.jsonl"
        # Only public context fields of confirmed TRAIN IDs enter the workspace.
        db_rows = [json.loads(line) for line in path.read_text().splitlines()]
    # Prioritize metadata, but do not filter or cap the available cases.
    choices = evidence.representatives(episodes, len(episodes), seed)
    destination.mkdir(parents=True)
    write_json(destination / "domain_schema.json", {"name": domain,
        "public_context_fields": sorted(CONTEXT_FIELDS[domain])})
    copied = []
    for record, source in active_sources:
        filename = f"active/{record['mechanism_id']}_r{record['revision']}.py"
        target = destination / filename
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != record["sha256"]:
            raise ValueError(f"incumbent changed during workspace preparation: {record['artifact_path']!r}; "
                             "stop concurrent artifact edits and restore verified content before preparation")
        copied.append({**record, "visible_path": filename})
    write_json(destination / "inventory.json", {"domain": domain, "snapshot_sha256": state["active_snapshot_sha256"],
                                              "records": copied})
    history_fields = {"round", "train_batch", "incumbent_score", "candidate_score", "accepted",
                      "incumbent_successes", "candidate_successes",
                      "bundle_id", "bundle_hypothesis", "active_snapshot_sha256"}
    operation_fields = {"operation_id", "mechanism_id", "op", "target_layer", "hypothesis"}
    history, seen = [], {}
    for source, past, namespace in history_sources:
        for row in past.get("rounds", []):
            public = {**{k: row[k] for k in history_fields if k in row},
                      "operations": [{k: op[k] for k in operation_fields if k in op}
                                     for op in row.get("operations", [])]}
            identity = object_digest({**public, 'code_hashes': [op.get('artifact_sha256') for op in row.get('operations', [])]})
            if identity in seen:
                # Copied ancestor rows may lack their original code/episode files.
                # Enrich from a registered source, without activating or duplicating them.
                number = seen[identity]
                existing = history[number]
                supplement = {**paired_history_outcomes(source, row),
                              **history_evidence(source, destination, row, domain, db_rows, number,
                                  variants=[v for v in ('candidate', 'incumbent') if f'{v}_evidence' not in existing])}
                existing.update({key: value for key, value in supplement.items()
                                 if key not in existing})
                for variant in ('candidate', 'incumbent'):
                    if f'{variant}_evidence' in existing:
                        existing.pop(f'{variant}_evidence_unavailable', None)
                for visible, operation in zip(existing['operations'], row.get('operations', [])):
                    if 'historical_code_path' not in visible:
                        visible.update(archive_history_code(source, destination, row, operation, namespace))
                    if 'historical_code_path' in visible:
                        visible.pop('historical_code_unavailable', None)
                continue
            seen[identity] = len(history)
            history.append({**public, "history_source": source.parent.name,
                            **paired_history_outcomes(source, row),
                            **history_evidence(source, destination, row, domain, db_rows, len(history)),
                            "operations": [{**visible, **archive_history_code(source, destination, row, op, namespace)}
                                           for visible, op in zip(public['operations'], row.get('operations', []))]})
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
        index[-1].update(path=filename, turns=rows[task]["turns"], calls=len(case["calls"]),
                        history_length=len(case['history_indices']), chars=len(text), bytes=len(text.encode("utf-8")))
    write_json(destination / "evidence.json", index)
    write_json(destination / "pattern_overview.json", pattern_overview(index))
    write_json(destination / "summary.json", {
        "split": "train", "domain": domain, "n": len(outcomes), "successes": sum(outcomes.values()),
        "infra_errors": infra, "available_cases": len(index), "evidence_access": "on_demand",
        "evidence_chars": used, "case_limit": None, "evidence_char_limit": None, "seed": seed,
        "signals": dict(Counter(g for e in episodes.values() if not e["success"] for g in evidence.classify(e))),
        "latest_history_index": next((i for i, entry in reversed(list(enumerate(history)))
            if entry['history_source'] == seed_run.parent.name), None),
        "note": "Signals are observations, not root causes. Infra failures remain in the full score denominator."})
    view = destination / "source_view"
    view.mkdir()
    for name in ("INTERFACE.md", "contract.py", "runtime.py"):
        shutil.copy2(ROOT / "evolution/agentbench_hooks_v31" / name, view / name)
    shutil.copy2(ROOT / "evolution/HARNESS_INTERFACE.md", view / "INTERFACE.md")
    shutil.copy2(ROOT / "evolution/harness_hooks.py", view / "harness_hooks.py")
    shutil.copy2(ROOT / "evolution/environment_state.py", view / "environment_state.py")
    shutil.copy2(ROOT / "evolution/tau_hooks.py", view / "tau_hooks.py")
    shutil.copy2(ROOT / "AgentBench/src/server/harness/session.py", view / "session.py")
    shutil.copytree(ROOT / "evolution/agentbench_hooks_v31/templates", view / "templates", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__).with_name("PROMPT.md"), destination / "OPTIMIZER.md")
    shutil.copy2(Path(__file__).with_name("SCHEMAS.md"), destination / "SCHEMAS.md")
    hashes = {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*") if p.is_file()}
    write_json(destination / "receipt.json", {"profile": PROFILE, "source_hashes": source_hashes(),
        "inputs": hashes, "seed_state_sha256": digest(seed_run / "state.json"), "evaluation_sha256": digest(eval_dir / "summary.json"),
        "prior_test_exposure": bool(state.get("prior_test_exposure", True)),
        "budget": {"max_proposer_turns": None, "max_repairs": MAX_REPAIRS,
                   "max_total_tokens": MAX_TOTAL_TOKENS},
        "evidence": {"access": "on_demand", "available_cases": len(index), "case_limit": None, "char_limit": None},
        "acceptance": "offline checks cannot accept: strict improvement on the same complete training batch required"})
    return destination


def re_task_id(value):
    return isinstance(value, str) and bool(value) and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value)


def verify_inputs(workspace):
    workspace = Path(workspace).resolve()
    receipt = json.loads((workspace / "receipt.json").read_text())
    if receipt["profile"] != PROFILE or receipt["source_hashes"] != source_hashes():
        current = source_hashes()
        changed = sorted(name for name in set(current) | set(receipt["source_hashes"])
                         if current.get(name) != receipt["source_hashes"].get(name))
        raise ValueError(f"profile/source changed: prepare a new workspace; changed files={changed}, "
                         f"expected profile={PROFILE!r}, recorded={receipt['profile']!r}; "
                         "do not edit frozen receipts or regenerate a paid proposal blindly")
    for relative, expected in receipt["inputs"].items():
        if digest(safe_path(workspace, relative)) != expected:
            raise ValueError(f"frozen input changed: {relative}; expected sha256={expected}; "
                             "restore the verified input, not its receipt hash")
    schema_path = workspace / "domain_schema.json"
    if schema_path.exists():
        if "domain_schema.json" not in receipt["inputs"]:
            raise ValueError("domain schema must be bound to the frozen input receipt")
        schema = json.loads(schema_path.read_text())
        register_domain(schema["name"], schema["public_context_fields"])
    return receipt


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"workspace-relative paths only: received {relative!r}; "
                         "use a nonempty relative path without '..'")
    result = (Path(root) / path).resolve()
    if Path(root).resolve() not in result.parents:
        raise ValueError(f"path escapes workspace: {relative!r}; symlink/absolute escapes are not allowed")
    return result


def load_cases(workspace, case_ids=None):
    """Host-side public case loading; proposer only opens requested case files."""
    workspace = Path(workspace)
    index = json.loads((workspace / "evidence.json").read_text())
    paths = {row["case_id"]: row["path"] for row in index}
    ids = list(paths) if case_ids is None else case_ids
    if any(not isinstance(case, str) or case not in paths for case in ids):
        raise ValueError(f"unknown training case id: {[case for case in ids if not isinstance(case, str) or case not in paths]!r}; "
                         "use string case_id values from the current TRAIN evidence.json or pattern_overview.json. "
                         "Batches rotate: for a historical case use read_case(history_index=..., "
                         "history_variant='candidate' or 'incumbent'), not a current evidence path")
    return [json.loads(safe_path(workspace, paths[case]).read_text()) for case in ids]


def incumbent_chain(workspace):
    """Load the source-bound active chain for staging or read-only diagnostics."""
    workspace = Path(workspace)
    inventory = json.loads((workspace / "inventory.json").read_text())
    artifacts = [{"name": r["mechanism_id"], "layer": r["target_layer"], "artifact": r["visible_path"],
                  "source": safe_path(workspace, r["visible_path"]).read_text()} for r in inventory["records"]]
    return inventory, artifacts


def candidate_chain(workspace, candidate):
    """One staging implementation for full preflight and optional diagnostics."""
    workspace, candidate = Path(workspace), Path(candidate)
    inventory, artifacts = incumbent_chain(workspace)
    declared = json.loads((candidate / "bundle.json").read_text())
    # Version bookkeeping is known from frozen inputs, not an evolved policy.
    # Explicit stale/wrong values remain errors; never replace supplied values.
    if isinstance(declared, dict) and isinstance(declared.get('operations'), list):
        for op in declared['operations']:
            if not isinstance(op, dict) or op.get('op') not in ('modify', 'delete'):
                continue
            record = next((r for r in inventory['records'] if r['mechanism_id'] == op.get('mechanism_id')), None)
            if record is not None:
                op.setdefault('expected_revision', record['revision'])
                op.setdefault('expected_sha256', record['sha256'])
    bundle, rejected = filter_operations(declared, inventory["domain"], inventory["records"],
                                        inventory["snapshot_sha256"], candidate)
    for op in bundle["operations"]:
        position = next((i for i, r in enumerate(artifacts) if r["name"] == op["mechanism_id"]), None)
        if op["op"] == "delete":
            artifacts.pop(position)
        else:
            validate_source(safe_path(candidate, op["artifact"]), op["target_layer"])
            item = {"name": op["mechanism_id"], "layer": op["target_layer"], "artifact": op["artifact"],
                    "source": safe_path(candidate, op["artifact"]).read_text()}
            if position is None:
                artifacts.append(item)
            else:
                artifacts[position] = item
    return inventory['domain'], bundle, rejected, artifacts


def replay(payload):
    result = subprocess.run([sys.executable, "-I", "-S", "-B", str(WORKER_PATH)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=10, env={})
    if result.returncode:
        try:
            reason = signal.Signals(-result.returncode).name if result.returncode < 0 else f'exit status {result.returncode}'
        except ValueError:
            reason = 'unrecognized termination signal'
        raise ValueError(f"replay worker exited {result.returncode} (resource limit or execution error); "
                         f"termination={reason}; "
                         f"case_ids={[case['case_id'] for case in payload['cases']]}; "
                         f"sandbox bounds: CPU=3s, RAM=384MiB, wall=10s; stderr={result.stderr[-2000:]!r}")
    return json.loads(result.stdout)


def check_candidate(workspace, candidate):
    """A preflight gate only: never commits artifacts or computes a success rate."""
    stage = 'frozen_inputs'
    try:
        verify_inputs(workspace)
        stage = 'bundle_and_source'
        domain, bundle, rejected, artifacts = candidate_chain(workspace, candidate)
        # Diagnosis and hand-authored tests are optional notes, not code gates.
        # Replay selection is host-owned and cannot be narrowed by those notes.
        cases = load_cases(workspace)
        stage = 'hook_replay'
        # Business databases are megabytes, unlike message-only replay inputs.
        # Keep the existing sandbox bounds, but isolate one state-bearing case
        # per process so 50 full initial databases cannot exhaust its memory.
        batches = [[case] for case in cases] if any(case.get("environment_states") for case in cases) else [cases]
        reports = []
        for batch in batches:
            payload = {"domain": domain, "public_context_fields": sorted(CONTEXT_FIELDS[domain]), "artifacts": artifacts,
                       "cases": batch}
            reports.append(replay(payload))
        report = {**reports[0], "passed": all(r["passed"] for r in reports),
                  "calls_replayed": sum(r.get("calls_replayed", 0) for r in reports),
                  "cases": [case for r in reports for case in r.get("cases", [])]}
        if any("error" in r for r in reports):
            report["error"] = "; ".join(r["error"] for r in reports if "error" in r)[:1000]
        return {**report, "profile": PROFILE, "bundle": bundle, "rejected_operations": rejected,
                "acceptance": "not evaluated; requires same-batch strict improvement"}
    except (ValueError, KeyError, TypeError, OSError, SyntaxError, subprocess.TimeoutExpired) as exc:
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000], "profile": PROFILE, "stage": stage,
                  "acceptance": "not evaluated"}
        if isinstance(exc, RejectedOperations):
            report["rejected_operations"] = exc.rejected_operations
        return report


def preview_candidate(workspace, case_id, call_index, *, chain=None):
    """Inspect one TRAIN prefix; never qualifies, consumes repairs or calls an LLM."""
    verify_inputs(workspace)
    if chain is not None and (not isinstance(chain, str) or chain not in ('incumbent', 'candidate')):
        raise ValueError("preview chain must be 'incumbent' or 'candidate'; omit it to preview the "
                         "candidate if bundle.json exists, otherwise the current incumbent")
    bundle_exists = safe_path(workspace, 'bundle.json').exists()
    chain = ('candidate' if bundle_exists else 'incumbent') if chain is None else chain
    if chain == 'incumbent':
        inventory, artifacts = incumbent_chain(workspace)
        domain, rejected = inventory['domain'], []
    else:
        if not bundle_exists:
            raise ValueError("candidate preview needs bundle.json and its artifacts; write them first or "
                             "use chain='incumbent' to inspect the current policy before coding")
        domain, _, rejected, artifacts = candidate_chain(workspace, workspace)
    case = load_cases(workspace, [case_id])[0]
    if type(call_index) is not int or not 0 <= call_index < len(case['calls']):
        raise ValueError(f"preview call_index={call_index!r}; expected integer in [0,{len(case['calls'])})")
    report = replay({'domain': domain, 'public_context_fields': sorted(CONTEXT_FIELDS[domain]),
        'artifacts': artifacts, 'cases': [case], 'preview_call_index': call_index})
    return {**report, 'rejected_operations': rejected, 'qualified': False, 'chain': chain,
            'scope': 'one frozen TRAIN prefix with the selected chain; includes incumbent model inputs, not a new rollout',
            'acceptance': 'not evaluated; run full check_candidate before finishing'}


def candidate_feedback(report):
    """Group repeated replay errors without dropping any failure occurrence."""
    result = {k: v for k, v in report.items() if k not in {'bundle', 'diagnosis', 'checks', 'cases'}}
    if 'cases' in report:
        cases, groups = report['cases'], {}
        result.update(replayed_cases=len(cases), passed_cases=sum(bool(case['passed']) for case in cases))
        for case in cases:
            if case['passed']:
                continue
            detail = {k: v for k, v in case.items() if k not in {'case_id', 'call_index', 'passed'}}
            group = groups.setdefault(json.dumps(detail, sort_keys=True), {**detail, 'occurrences': []})
            group['occurrences'].append({'case_id': case['case_id'], 'call_index': case.get('call_index')})
        result['replay_failures'] = list(groups.values())
    if report.get('passed'):
        result['next_step'] = ('Qualified for host native TRAIN evaluation, not a measured success score. '
            'Hand off the candidate unless a concrete observed defect remains. Rechecking the same frozen prefix '
            'cannot measure accuracy; no interaction-count cap, but all requests share the token budget.')
    return result


TOOLS = [
    {"type": "function", "function": {"name": "read_case", "description": "Read current incumbent TRAIN evidence, or a past attempt using history_index and optional history_variant=candidate|incumbent. Page history or call messages; setup omitted unless include_setup=true, DB only when requested. No total read quota.",
        "parameters": {"type": "object", "properties": {"case_id": {"type": "string"},
        "history_index": {"type": "integer", "minimum": 0}, "include_setup": {"type": "boolean", "default": False},
        "history_variant": {"type": "string", "enum": ["candidate", "incumbent"], "default": "candidate"},
        "start_message": {"type": "integer", "minimum": 0}, "message_count": {"type": "integer", "minimum": 1},
        "call_index": {"type": "integer", "minimum": 0},
        "environment_state": {"type": "string", "enum": ["model", "action"]},
        "state_path": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]}}},
        "required": ["case_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a bounded byte range of an allowlisted workspace file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_byte": {"type": "integer", "minimum": 0},
        "max_bytes": {"type": "integer", "minimum": 1, "maximum": 16000}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_file", "description": "Find literal text in an allowlisted file; return UTF-8 byte offsets and short context. Page using next_byte, then read exact ranges. No total search/read quota.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string", "minLength": 1},
        "start_byte": {"type": "integer", "minimum": 0}, "max_matches": {"type": "integer", "minimum": 1, "maximum": 20}},
        "required": ["path", "text"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write bundle.json, diagnosis.json, checks.json or one artifacts/*.py file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "copy_file", "description": "Copy one frozen active/history Python source into artifacts/*.py unchanged, then use replace_file for targeted edits. Avoid reprinting unchanged code.",
        "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
        "required": ["source", "destination"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "replace_file", "description": "Replace one exact, unique text span in a candidate output; preserves the rest of the file. Read/search first. Frozen inputs cannot be edited.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string", "minLength": 1},
        "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "check_candidate", "description": "Check executable H2–H5 safety, interfaces and frozen-input hook replay. Optional diagnosis/checks notes do not affect the result. No LLM or environment call.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "preview_candidate", "description": "Inspect H4/H5 guidance, H3 model-view edits and H2 actions at one current TRAIN call. Choose chain=incumbent before coding or to compare a control; default previews the candidate if bundle.json exists, otherwise incumbent. Isolated prefix only, never qualification or a new rollout; no LLM/environment call.",
        "parameters": {"type": "object", "properties": {"case_id": {"type": "string"}, "call_index": {"type": "integer", "minimum": 0},
        "chain": {"type": "string", "enum": ["incumbent", "candidate"]}},
        "required": ["case_id", "call_index"], "additionalProperties": False}}},
]

CHECKPOINT_TOOL = {"type": "function", "function": {"name": "checkpoint_context",
    "description": "Preserve your working memory before continuing in a fresh context. Record findings and exact evidence/code references, hypothesis, rejected alternatives, regression risks, current candidate, unresolved defects and next steps. Do not invent findings or reimplement code here. Full original history stays readable.",
    "parameters": {"type": "object", "properties": {"working_memory": {"type": "string", "minLength": 1}},
                   "required": ["working_memory"], "additionalProperties": False}}}


def record_read(workspace, case_id, request):
    ledger_path = Path(workspace) / READ_LOG
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"case_ids": [], "requests": []}
    if case_id not in ledger["case_ids"]:
        ledger["case_ids"].append(case_id)
    ledger["requests"].append(request)
    write_json(ledger_path, ledger)


def read_case(workspace, args):
    if not isinstance(args, dict) or set(args) - {"case_id", "history_index", "history_variant", "include_setup", "start_message", "message_count", "call_index", "environment_state", "state_path"}:
        raise ValueError("invalid read_case arguments: expected object with case_id and optional "
                         "history_index, history_variant, include_setup, start_message/message_count, call_index, environment_state, state_path; "
                         f"received keys={list(args) if isinstance(args, dict) else type(args).__name__}")
    if ("environment_state" in args and (args["environment_state"] not in {"model", "action"} or "call_index" not in args)
            or "state_path" in args and "environment_state" not in args):
        raise ValueError("business-state reads require a call index and model/action boundary: "
                         "supply call_index=<recorded call number>, environment_state='model' or 'action'; "
                         "state_path is optional and requires environment_state")
    if type(args.get('include_setup', False)) is not bool:
        raise ValueError('include_setup must be boolean; true requests public context and tool definitions')
    variant = args.get('history_variant', 'candidate')
    if not isinstance(variant, str) or variant not in ('candidate', 'incumbent') or 'history_variant' in args and 'history_index' not in args:
        raise ValueError("history_variant requires history_index and must be 'candidate' or 'incumbent'; omit both for current evidence")
    origin = {'kind': 'incumbent'}
    if 'history_index' in args:
        case, path, source_sha = load_history_case(workspace, args['history_index'], args['case_id'], variant)
        origin = {'kind': 'historical_' + variant, 'history_index': args['history_index'], 'source_sha256': source_sha}
    else:
        case = load_cases(workspace, [args["case_id"]])[0]
        index = json.loads((Path(workspace) / 'evidence.json').read_text())
        path = next(r['path'] for r in index if r['case_id'] == case['case_id'])
    result = {k: copy.deepcopy(case[k]) for k in
              ("case_id", "success", "status", "infrastructure_error", "signals")}
    if args.get('include_setup', False):
        result.update({k: copy.deepcopy(case[k]) for k in ('context', 'tools')})
    result.update(origin=origin, history_length=len(case["history_indices"]), call_count=len(case["calls"]))
    if "call_index" in args:
        position = args["call_index"]
        if type(position) is not int or not 0 <= position < len(case["calls"]):
            raise ValueError(f"invalid call index: call_index={position!r}; expected integer in [0,{len(case['calls'])})")
        call = case["calls"][position]
        if args.get('include_setup', False):
            for tool in result['tools']:
                name = tool['function']['name']
                if name in call.get('tool_descriptions', {}):
                    description = call['tool_descriptions'][name]
                    if description is None:
                        tool['function'].pop('description', None)
                    else:
                        tool['function']['description'] = description
        start, count = args.get('start_message', 0), args.get('message_count', len(call['message_indices']))
        if (type(start) is not int or type(count) is not int
                or not 0 <= start <= len(call['message_indices']) or count < 1):
            raise ValueError(f"invalid recorded-call message range: start_message={start!r}, message_count={count!r}; "
                             f"expected start in [0,{len(call['message_indices'])}] and positive count")
        end = min(len(call['message_indices']), start + count)
        result["call"] = {"call_index": position, "turn": call["turn"],
            "messages": [copy.deepcopy(case["message_pool"][i]) for i in call["message_indices"][start:end]],
            "message_start": start, "message_count": end - start, "next_message": end,
            "eof": end == len(call['message_indices']), "full_message_count": len(call['message_indices']),
            "assistant": copy.deepcopy(call["assistant"])}
        selection = {"call_index": position, 'start_message': start, 'message_count': end - start}
        for field in ("environment_state_index", "action_environment_state_index"):
            if field in call:
                result["call"][field] = call[field]
        if "environment_state" in args:
            phase = args["environment_state"]
            field = "environment_state_index" if phase == "model" else "action_environment_state_index"
            state = resolve_environment_state(case.get("environment_states", []), call.get(field))
            state_path = args.get("state_path", [])
            if not isinstance(state_path, list) or any(type(key) not in (str, int) or type(key) is int and key < 0 for key in state_path):
                raise ValueError("state_path must contain JSON object keys or nonnegative list indices: "
                                 "use an array of strings/integers, never booleans")
            for depth, key in enumerate(state_path):
                if state is None:
                    raise ValueError(f"recorded environment state/path is unavailable: "
                                     f"case={case['case_id']}, call={position}, boundary={phase}, path={state_path[:depth+1]!r}; "
                                     "old traces may have no DB snapshot; use recorded history/tool observations")
                try:
                    state = state[key]
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError(f"state_path not found at {state_path[:depth+1]!r}: "
                                     f"container type={type(state).__name__}; read the parent path to inspect available keys/indices") from exc
            result["environment_state"] = state
            selection.update(environment_state=phase, state_path=state_path)
    else:
        start, count = args.get("start_message", 0), args.get("message_count", 12)
        if (type(start) is not int or type(count) is not int
                or not 0 <= start <= len(case["history_indices"]) or count < 1):
            raise ValueError(f"invalid history message range: start_message={start!r}, message_count={count!r}; "
                             f"expected integer start in [0,{len(case['history_indices'])}] and positive integer count")
        end = min(len(case["history_indices"]), start + count)
        result["history"] = [{"message_index": i, "message": copy.deepcopy(
            case["message_pool"][case["history_indices"][i]])} for i in range(start, end)]
        result.update(next_message=end, eof=end == len(case["history_indices"]))
        selection = {"start_message": start, "message_count": end - start}
    record_read(workspace, case["case_id"], {"tool": "read_case", "path": path, **origin, **selection,
                "bytes": len(json.dumps(result, ensure_ascii=False).encode("utf-8"))})
    return result


def tool_call(workspace, name, args):
    receipt = verify_inputs(workspace)
    if name == 'copy_file':
        source = safe_path(workspace, args['source'])
        destination = safe_path(workspace, args['destination'])
        relative = str(source.relative_to(Path(workspace).resolve()))
        if (relative not in receipt['inputs'] or source.suffix != '.py'
                or not relative.startswith(('active/', 'history_code/'))
                or destination.parent != Path(workspace).resolve() / 'artifacts'
                or destination.suffix != '.py'):
            raise ValueError('copy_file requires a frozen active/history Python source and an artifacts/*.py destination')
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        return {'path': str(destination.relative_to(Path(workspace).resolve())),
                'bytes': destination.stat().st_size, 'sha256': digest(destination)}
    if name == "read_case":
        return read_case(workspace, args)
    if name == "check_candidate":
        report = check_candidate(workspace, workspace)
        return candidate_feedback(report)
    if name == "preview_candidate":
        return preview_candidate(workspace, args['case_id'], args['call_index'], chain=args.get('chain'))
    path = safe_path(workspace, args["path"])
    relative = str(path.relative_to(Path(workspace).resolve()))
    output = relative in OUTPUT_FILES or (path.parent == Path(workspace).resolve() / "artifacts" and path.suffix == ".py")
    if name in {"read_file", "search_file"}:
        archive = (path.parent == Path(workspace).resolve() / CONTEXT_ARCHIVE and path.suffix == '.json'
                   and len(path.stem) == 64 and all(c in '0123456789abcdef' for c in path.stem))
        if archive and digest(path) != path.stem:
            raise ValueError('context history archive hash mismatch; retain original evidence, do not replace it')
        if relative not in receipt["inputs"] and relative not in {"receipt.json", READ_LOG} and not output and not archive:
            hint = ("; evidence/ contains this rotated batch only: use read_case(history_index=..., "
                    "history_variant='candidate' or 'incumbent') for historical cases from history.json"
                    if relative.startswith('evidence/') else '')
            raise ValueError(f"read outside input/output allowlist: {relative!r}; "
                             "read receipt.inputs, receipt.json, evidence_reads.json or candidate outputs only" + hint)
    if name == "search_file":
        needle, start, maximum = args.get("text"), args.get("start_byte", 0), args.get("max_matches", 20)
        if (not isinstance(needle, str) or not needle or type(start) is not int or start < 0
                or type(maximum) is not int or not 1 <= maximum <= 20):
            raise ValueError("search_file requires nonempty literal text, integer start_byte>=0 and "
                             "1<=max_matches<=20 per page; continue with next_byte, without a total quota")
        data, needle = path.read_bytes(), needle.encode('utf-8')
        cursor, matches = start, []
        while len(matches) < maximum:
            position = data.find(needle, cursor)
            if position < 0:
                break
            cursor = position + len(needle)
            context_start = max(0, position - 120)
            matches.append({"start_byte": position, "end_byte": cursor, "context_start_byte": context_start,
                            "context": data[context_start:cursor + 240].decode('utf-8', errors='replace')})
        result = {"path": relative, "matches": matches, "next_byte": cursor,
                  "eof": data.find(needle, cursor) < 0}
        if relative.startswith('evidence/'):
            record_read(workspace, path.stem.removeprefix('case_'), {"tool": name, "path": relative,
                "start_byte": start, "text": args['text'], "bytes": len(json.dumps(result, ensure_ascii=False).encode('utf-8'))})
        return result
    if name == "read_file":
        start, maximum = args.get("start_byte", 0), args.get("max_bytes", 12000)
        if type(start) is not int or start < 0 or type(maximum) is not int or not 1 <= maximum <= 16000:
            raise ValueError(f"invalid byte range: start_byte={start!r}, max_bytes={maximum!r}; "
                             "expected integer start_byte>=0 and 1<=max_bytes<=16000; "
                             "continue from next_byte for larger files, without a total read quota")
        with path.open("rb") as stream:
            stream.seek(start)
            data = stream.read(maximum)
        if relative.startswith("evidence/") and data:
            case_id = path.stem.removeprefix("case_")
            record_read(workspace, case_id, {"path": relative, "start_byte": start, "bytes": len(data)})
        return {"content": data.decode("utf-8", errors="replace"), "next_byte": start + len(data), "eof": len(data) < maximum}
    if name == "replace_file":
        if not output:
            raise ValueError(f"replace_file cannot edit frozen input {relative!r}; only candidate outputs are writable")
        old, new = args.get('old_text'), args.get('new_text')
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("replace_file requires nonempty string old_text and string new_text")
        content = path.read_text()
        count = content.count(old)
        if count != 1:
            raise ValueError(f"replace_file expected one exact match, found {count}; "
                             "read/search the current file and supply a unique span with surrounding context; no edit was made")
        args, name = {**args, 'content': content.replace(old, new, 1)}, 'write_file'
    if name != "write_file" or not output or not isinstance(args["content"], str) or len(args["content"]) > 256000:
        raise ValueError(f"only bounded candidate outputs may be written; no shell, fixture code or frozen-input edits: "
                         f"tool={name!r}, path={relative!r}; use write_file for bundle.json, optional "
                         "diagnosis.json/checks.json or artifacts/name.py with string content<=256000 characters")
    path.parent.mkdir(exist_ok=True)
    path.write_text(args["content"])
    return {"path": relative, "bytes": path.stat().st_size}


def needs_context_checkpoint(messages):
    return flash_counter().count(messages, TOOLS) >= CONTEXT_CHECKPOINT_TOKENS


def archive_context(workspace, messages):
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    relative = f'{CONTEXT_ARCHIVE}/{hashlib.sha256(raw.encode()).hexdigest()}.json'
    path = safe_path(workspace, relative)
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_text(raw)
    if digest(path) != path.stem:
        raise ValueError('context history archive changed; retain the original conversation')
    return relative


def checkpoint_memory(response):
    message = response['choices'][0]['message']
    calls = message.get('tool_calls') or []
    if not calls:
        memory = message.get('content')
    elif len(calls) == 1 and calls[0]['function']['name'] == 'checkpoint_context':
        arguments = json.loads(calls[0]['function']['arguments'])
        if not isinstance(arguments, dict):
            raise ValueError('context checkpoint arguments must contain working_memory')
        memory = arguments.get('working_memory')
    else:
        raise ValueError('context checkpoint requires working memory, not candidate edits or other tool calls')
    if not isinstance(memory, str) or not memory.strip():
        raise ValueError('context checkpoint working memory is empty; keep original context')
    return memory


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
        result = json.load(response)
    verify_usage(result)
    return result


@contextmanager
def agent_workspace(workspace, checkpoint_dir, input_hash, max_repairs, token_budget):
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
        if (saved["input_sha256"] != input_hash or saved.get("max_repairs") != max_repairs
                or saved.get("max_total_tokens") != token_budget):
            raise ValueError(f"agent checkpoint input/budget mismatch: recorded max_repairs={saved.get('max_repairs')!r}, "
                             f"requested={max_repairs}; total token budget recorded={saved.get('max_total_tokens')!r}, "
                             f"requested={token_budget}; input receipt must be identical; retain checkpoint and audit before resuming")
        if saved.get("error") == "token_budget":
            raise ProposalFailure("token_budget", "token budget exhausted; retain checkpoint, do not reset paid usage")
        if saved.get('error') == 'context_checkpoint':
            raise ProposalFailure('context_checkpoint', 'failed working-memory checkpoint retained; audit before any retry')
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
            changed = sorted(name for name in set(actual) | set(saved["workspace_hashes"])
                             if actual.get(name) != saved["workspace_hashes"].get(name))
            raise ValueError(f"agent checkpoint workspace changed: files={changed}; "
                             "retain paid outputs and audit modifications, do not reset the checkpoint")
        verify_inputs(sandbox)
    else:
        root.mkdir(parents=True, exist_ok=True)
        if sandbox.exists():
            raise ValueError(f"unbound checkpoint workspace already exists: {sandbox}; retain it and locate/audit "
                             "its original checkpoint instead of overwriting paid outputs")
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
        raise ValueError('validated candidate snapshot changed or belongs to different inputs: require matching '
                         'input_sha256 and snapshot_sha256 over the saved body; retain the original validated snapshot')


def restore_valid_snapshot(snapshot, sandbox, checkpoint, input_hash):
    verify_valid_snapshot(snapshot, input_hash)
    base = checkpoint.parent if checkpoint is not None else sandbox.parent
    destination = base / ('validated_' + snapshot['snapshot_sha256'])
    if destination.exists():
        actual = {str(p.relative_to(destination)):p.read_text() for p in destination.rglob('*') if p.is_file()}
        if actual != snapshot['files']:
            raise ValueError(f'restored validated candidate changed: {destination}; files must match the saved '
                             'validated snapshot exactly, not later edits')
    else:
        destination.mkdir()
        for name, content in snapshot['files'].items():
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    validation = check_candidate(sandbox, destination)
    if not validation['passed']:
        raise ValueError('previously validated snapshot no longer passes: ' + json.dumps(candidate_feedback(validation)))
    return destination, validation


def startup_feedback(workspace):
    """Make measured history discoverable, not an inferred diagnosis or policy."""
    summary = json.loads((Path(workspace) / 'summary.json').read_text())
    index = summary.get('latest_history_index')
    history = json.loads((Path(workspace) / 'history.json').read_text())
    latest = None if index is None else {key: value for key, value in history[index].items() if key in {
        'round', 'history_source', 'accepted', 'incumbent_successes', 'candidate_successes', 'paired_train_n',
        'bundle_hypothesis', 'paired_gain_case_ids', 'paired_regression_case_ids', 'candidate_evidence', 'incumbent_evidence'}}
    contrasts = []
    for case_id in (latest or {}).get('paired_regression_case_ids', []):
        pair = {'case_id': case_id, 'history_index': index}
        for variant in ('incumbent', 'candidate'):
            try:
                case, _, _ = load_history_case(workspace, index, case_id, variant)
                tail = []
                for position in range(max(0, len(case['history_indices']) - 4), len(case['history_indices'])):
                    message = case['message_pool'][case['history_indices'][position]]
                    tail.append({'message_index': position, 'role': message['role'],
                        'content_excerpt': str(message.get('content') or '')[:500],
                        'tool_calls': [{'name': call['function']['name'],
                            'arguments_excerpt': str(call['function'].get('arguments') or '')[:500]}
                            for call in message.get('tool_calls', [])]})
                pair[variant] = {'success': case['success'], 'tail_excerpts': tail}
            except (ValueError, OSError, KeyError, TypeError) as exc:
                pair[variant] = {'unavailable': f'{type(exc).__name__}: {exc}'}
        contrasts.append(pair)
    return {'current_TRAIN': {key: summary[key] for key in ('n', 'successes', 'infra_errors')},
        'latest_primary_attempt_index': index, 'latest_measured_TRAIN_attempt': latest,
        'latest_regression_contrasts': contrasts,
        'note': 'History is grouped by source, not chronology. These are whole-bundle outcomes on the named past batch, '
                'not the current rotated batch or operation-level causal effects. Compare past candidate and incumbent '
                'with read_case(history_index=..., history_variant=...). Tail excerpts are public observations, not '
                'root-cause diagnoses or complete traces; full reads remain unlimited. New current evidence uses neither argument.'}


def request_token_reserve(messages, tools):
    """Conservative byte upper bound with full output reservation, not a context cap."""
    values = copy.deepcopy(messages)
    if not values or values[0].get('role') != 'system':
        values.insert(0, {'role': 'system', 'content': ''})
    values[0]['tools'] = copy.deepcopy(tools)
    prompt = flash_counter().encode_messages(values, thinking_mode='thinking', drop_thinking=False)
    # Byte-level BPE cannot emit more tokens than encoded bytes. Reserve a
    # transport allowance too; actual charges always use provider usage.
    return len(prompt.encode('utf-8')) + OUTPUT_LIMIT + 4096


def propose(workspace, destination, *, completion=model_router_completion,
            max_repairs=MAX_REPAIRS, checkpoint_dir=None, recovery_outputs=None, token_budget=None):
    """One agent; total token budget, unlimited turn count, up to three repairs."""
    workspace, destination = Path(workspace).resolve(), Path(destination).resolve()
    receipt = verify_inputs(workspace)
    ceiling = receipt['budget'].get('max_total_tokens', MAX_TOTAL_TOKENS)
    token_budget = ceiling if token_budget is None else token_budget
    if type(token_budget) is not int or not 0 < token_budget <= min(ceiling, MAX_TOTAL_TOKENS):
        raise ValueError("token_budget must be a positive integer within the frozen 10000000-token ceiling")
    if type(max_repairs) is not int or not 0 <= max_repairs <= MAX_REPAIRS:
        raise ValueError("max_repairs must be an integer between 0 and 3")
    input_hash = object_digest(receipt)
    if destination.exists():
        cache = json.loads((destination / "validation.json").read_text())
        if (not cache.get("passed") or cache["input_sha256"] != input_hash
                or cache.get("max_repairs") != max_repairs
                or cache.get('max_total_tokens', ceiling) != token_budget):
            raise ValueError("cached proposal input/repair-policy mismatch: requires passed=True, "
                             "the same input receipt and max_repairs; do not overwrite or blindly replay a paid proposal")
        outputs = {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*")
                   if p.is_file() and p.name != "validation.json"}
        if outputs != cache["outputs"]:
            changed = sorted(name for name in set(outputs) | set(cache["outputs"])
                             if outputs.get(name) != cache["outputs"].get(name))
            raise ValueError(f"cached proposal output changed: files={changed}; retain and audit the original artifacts")
        return destination / "bundle.json"
    if workspace == destination or workspace in destination.parents:
        raise ValueError("candidate output must be separate from the frozen workspace")
    with agent_workspace(workspace, checkpoint_dir, input_hash, max_repairs, token_budget) as (sandbox, checkpoint, saved):
        files = sorted(receipt["inputs"])
        messages = [{"role": "system", "content": (sandbox / "OPTIMIZER.md").read_text()},
            {"role": "user", "content": "Optimize one Life bundle. Workspace files: " + json.dumps(files) +
             ". Measured TRAIN feedback: " + json.dumps(startup_feedback(sandbox), ensure_ascii=False) +
             ". Inspect the latest regressions and gains in the actual past candidate AND incumbent before broadening their mechanism; this is evidence guidance, not a preflight gate. Read current failures and successful controls on demand. evidence.json includes history_length and calls for targeted tail/call reads; full evidence remains unlimited. Orient with inventory, summary and the interface; consult pattern_overview, strategy_history and history for relevant hypotheses; use SCHEMAS.md for files. Long contexts receive a same-agent working-memory checkpoint, not a token truncation. Full archived context remains available via read_file/search_file on context_history/. Cite source paths/case/call indices in memory; reread original evidence when uncertain. Replace/delete ineffective policies when justified. Write bundle.json and artifacts; check_candidate and repair genuine defects. Optional notes are not requirements. Hand off a coherent qualified candidate for real evaluation rather than repeatedly polishing a frozen prefix. No case-read or request-count quota; input + output token budget: " + str(token_budget) + "."}]
        usage, validation, requests = {}, None, 0
        context_checkpoints = []
        context_checkpoint_failures = []
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
                       max_total_tokens=token_budget,
                       messages=messages, usage=usage, requests=requests, validation=validation,
                       context_checkpoints=context_checkpoints,
                       context_checkpoint_failures=context_checkpoint_failures,
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
                + json.dumps(candidate_feedback(report))})
            return True

        if saved:
            messages, usage, requests = saved["messages"], saved["usage"], saved["requests"]
            validation = saved.get('validation')
            context_checkpoints = saved.get('context_checkpoints', [])
            context_checkpoint_failures = saved.get('context_checkpoint_failures', [])
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
            last_refresh = max([0] + [c['request'] for c in context_checkpoints]
                               + [c['request'] for c in context_checkpoint_failures])
            compacting = (needs_context_checkpoint(messages)
                          and (not (context_checkpoints or context_checkpoint_failures) or requests > last_refresh))
            request_tools = [CHECKPOINT_TOOL] if compacting else TOOLS
            if compacting:
                messages.append({'role': 'user', 'content':
                    'Checkpoint your working memory now, then continue the SAME proposal. Preserve factual findings '
                    'with case/history/call or file references, hypothesis and alternatives, regressions/non-triggers, '
                    'current code and validation status, unresolved errors, and next action. Be concise (aim below 3000 '
                    'tokens) without dropping necessary facts; use source references instead of copying policies/code. '
                    'No candidate edits in this response. Full old context will be archived and remains readable. '
                    'The token threshold triggers memory management, not a context limit or request-count quota.'})
            spent = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
            if spent >= token_budget or request_token_reserve(messages, request_tools) > token_budget - spent:
                persist(inflight=False, error='token_budget')
                if snapshot is not None:
                    selected, validation = restore_valid_snapshot(snapshot, sandbox, checkpoint, input_hash)
                    fallback_reason = 'token_budget_after_valid_candidate'
                    persist(inflight=False, error='token_budget', fallback_reason=fallback_reason)
                    break
                raise ProposalFailure('token_budget', 'Not enough remaining input + output budget for another request; '
                                      'no qualified candidate. Paid usage and incumbent retained.')
            requests += 1
            persist(inflight=True)
            try:
                response = completion(messages, request_tools)
            except ContextLimit as exc:
                persist(inflight=False, error="context_limit", detail=str(exc))
                if snapshot is not None:
                    selected, validation = restore_valid_snapshot(snapshot, sandbox, checkpoint, input_hash)
                    fallback_reason = "context_limit_after_valid_candidate"
                    persist(inflight=False, error="context_limit", detail=str(exc), fallback_reason=fallback_reason)
                    break
                raise ProposalFailure("context_limit", str(exc)) from exc
            verify_usage(response)
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
            if compacting:
                try:
                    memory = checkpoint_memory(response)
                except (ValueError, KeyError, TypeError) as exc:
                    # A failed optional summary is not a failed proposal. Keep
                    # all messages and spend; do not execute unadvertised tools.
                    context_checkpoint_failures.append({'request': requests,
                        'error': str(exc), 'usage': copy.deepcopy(response['usage'])})
                    for call in calls:
                        messages.append({'role': 'tool', 'tool_call_id': call['id'],
                            'content': json.dumps({'executed': False,
                                'error': 'Working memory was not supplied. Context retained; reissue this action in the normal tool turn.'})})
                    messages.append({'role': 'user', 'content':
                        'Memory refresh was deferred; your full context, code, evidence and budgets are unchanged. '
                        'Continue the SAME proposal with normal tools. Do not restart analysis or routinely reread '
                        'unchanged inventory/code. Copy existing code with copy_file, then use replace_file for exact edits.'})
                    persist(inflight=False)
                    continue
                for call in calls:
                    messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                     'content': json.dumps({'saved': True, 'scope': 'working memory, not qualification'})})
                archive = archive_context(sandbox, messages)
                context_checkpoints.append({'archive': archive, 'request': requests,
                    'memory': memory, 'usage': copy.deepcopy(response['usage']),
                    'usage_so_far': copy.deepcopy(usage)})
                host_state = {'repairs_granted': repairs_granted, 'failed_preflights': failed_preflights,
                    'remaining_total_tokens': token_budget - usage['prompt_tokens'] - usage['completion_tokens'],
                    'last_validation': candidate_feedback(validation) if validation is not None else None,
                    'qualified_candidate_available': snapshot is not None,
                    'candidate_path': 'bundle.json', 'prior_context_archives': [r['archive'] for r in context_checkpoints]}
                messages = [messages[0], messages[1], {'role': 'user', 'content':
                    'Continue the same proposal from YOUR working memory:\n' + memory +
                    '\nVerified host bookkeeping (not a diagnosis): ' + json.dumps(host_state, ensure_ascii=False) +
                    '\nAll source files, TRAIN evidence and full prior contexts remain readable. Do not assume '
                    'the summary proves a fact: consult cited original evidence when needed. Token/repair budgets '
                    'and candidate files are unchanged; this is not a new proposal.'}]
                persist(inflight=False)
                continue
            failed_report = None
            for call in calls:
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if call["function"]["name"] == "check_candidate":
                        validation = check_candidate(sandbox, sandbox)
                        remember(validation)
                        failed_report = None if validation["passed"] else validation
                        result = candidate_feedback(validation)
                    else:
                        result = tool_call(sandbox, call["function"]["name"], arguments)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000],
                              "tool": call.get("function", {}).get("name"),
                              "expected_arguments": next((tool["function"]["parameters"] for tool in TOOLS
                                  if tool["function"]["name"] == call.get("function", {}).get("name")), None)}
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
                    + json.dumps(candidate_feedback(failed_report)))
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
            "max_total_tokens": token_budget,
            "context_checkpoints": context_checkpoints,
            "context_checkpoint_failures": context_checkpoint_failures,
            "repairs_granted": repairs_granted, "failed_preflights": failed_preflights,
            "model": "deepseek-v4-flash", "fallback_reason": fallback_reason,
            "recovered_qualified": recovered_qualified,
            "validated_snapshot_sha256": snapshot.get("snapshot_sha256") if snapshot is not None else None})
    return destination / "bundle.json"


def strict_improvement(candidate, incumbent, *, expected_task_ids):
    """Integration gate: matching TRAIN tasks, complete denominator, strictly better."""
    expected = {str(task) for task in expected_task_ids}
    if len(expected) != 50:
        raise ValueError(f"predeclared complete 50-task batch required, not a screen subset: received "
                         f"{len(expected)} unique IDs; use the frozen full TRAIN batch")
    def values(summary):
        scores = summary.get("per_task")
        if (summary.get("split") != "train" or not isinstance(scores, dict) or not scores
                or summary.get("n") != len(scores) or any(type(v) is not int or v not in (0, 1) for v in scores.values())
                or type(summary.get("successes")) is not int or summary.get("successes") != sum(scores.values()) or set(scores) != expected):
            raise ValueError("complete binary train summary required: split='train', matching full task IDs, "
                             "integer 0/1 outcomes, n=task count and successes=sum(outcomes); no subset or dropped failures")
        accuracy = float(summary.get("accuracy", sum(scores.values()) / len(scores)))
        if not math.isfinite(accuracy) or not math.isclose(accuracy, sum(scores.values()) / len(scores), abs_tol=1e-12):
            raise ValueError(f"accuracy does not match full-batch outcomes: received {accuracy!r}, "
                             f"expected {sum(scores.values()) / len(scores)} from all {len(scores)} tasks")
        return scores
    a, b = values(candidate), values(incumbent)
    if candidate.get("domain") not in CONTEXT_FIELDS or candidate.get("domain") != incumbent.get("domain") or set(a) != set(b):
        raise ValueError(f"candidate/incumbent domain or task keys differ: candidate domain={candidate.get('domain')!r}, "
                         f"incumbent={incumbent.get('domain')!r}; compare the same domain and complete TRAIN task set")
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
