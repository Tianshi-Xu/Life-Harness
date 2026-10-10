"""Public-only interchange format for native benchmark trajectories.

This module never reads raw evaluator-private results. Callers must export the
public conversation and the scalar training outcome explicitly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

DOMAINS = {"dbbench", "webshop", "alfworld", "airline", "retail", "telecom", "os"}
FIELDS = {"task_id", "trial", "success", "messages", "infrastructure_error",
          "status", "wall_seconds"}
MESSAGE_FIELDS = {"role", "content", "tool_calls", "tool_call_id", "name"}


def public_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("a nonempty public conversation is required")
    out = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
                "system", "user", "assistant", "tool"}:
            raise ValueError("unsupported public message")
        # Do not silently accept accidentally exported oracle fields.
        if set(message) - MESSAGE_FIELDS:
            raise ValueError("unexpected message metadata; sanitize explicitly")
        out.append(copy.deepcopy(message))
    return out


def validate_records(payload):
    if set(payload) != {"schema_version", "domain", "split", "records"}:
        raise ValueError("unexpected envelope fields")
    if payload["schema_version"] != 1 or payload["domain"] not in DOMAINS:
        raise ValueError("unsupported native record schema/domain")
    # Tests must never enter the optimizer's directory tree.
    if payload["split"] != "train":
        raise ValueError("only training evidence may be exported to AHE")
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("empty evidence")
    seen = set()
    for record in records:
        if set(record) - FIELDS or not {"task_id", "trial", "success", "messages"} <= set(record):
            raise ValueError("unexpected or missing public record fields")
        if not isinstance(record["task_id"], str) or not record["task_id"]:
            raise ValueError("task_id must be a nonempty string")
        if type(record["trial"]) is not int or record["trial"] < 0:
            raise ValueError("invalid trial")
        if type(record["success"]) is not int or record["success"] not in (0, 1):
            raise ValueError("success must be binary")
        if type(record.get("infrastructure_error", False)) is not bool:
            raise ValueError("infrastructure_error must be boolean")
        if record.get("infrastructure_error") and record["success"]:
            raise ValueError("infrastructure error cannot be a success")
        key = (record["task_id"], record["trial"])
        if key in seen:
            raise ValueError("duplicate trial")
        seen.add(key)
        public_messages(record["messages"])
    return copy.deepcopy(records)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def export_job(payload, job_dir, runtime_artifacts=None):
    """Emit the Harbor-shaped files consumed by UNCHANGED AHE readers.

    native_record.json is an explicit provenance record, not an invented NexAU
    span. Agent Debugger natively accepts {trace_id, messages}; no conversion to
    synthetic LLM spans or fabricated call counts is performed.
    """
    records = validate_records(payload)
    job_dir = Path(job_dir)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}(?:_[a-zA-Z0-9-]+)?", job_dir.name):
        raise ValueError("job directory must match upstream's timestamp discovery format")
    if job_dir.exists():
        raise FileExistsError("never overwrite an earlier evaluation job")
    job_dir.mkdir(parents=True)
    mapping = {}
    for record in records:
        task_id = record["task_id"]
        digest = hashlib.sha256(task_id.encode()).hexdigest()
        # Stable per-task names, including punctuation/Unicode, without path traversal.
        task_name = "native_" + digest
        suffix = hashlib.sha256(f"{task_id}:{record['trial']}".encode()).hexdigest()[:12]
        folder = job_dir / f"{task_name}__{suffix}"
        mapping[task_name] = task_id
        write_json(folder / "result.json", {"task_name": task_name,
                                             "trial": record["trial"]})
        # Preserve the native fixed denominator. Infrastructure failures remain
        # failures, rather than disappearing from AHE's pass@1 metric.
        reward = folder / "verifier/reward.txt"
        reward.parent.mkdir(parents=True)
        reward.write_text(str(record["success"]) + "\n")
        trace = {"trace_id": f"{task_name}:{record['trial']}",
                 "messages": public_messages(record["messages"])}
        write_json(folder / "agent/nexau_in_memory_tracer.cleaned.json", trace)
        write_json(folder / "native_record.json", {k: v for k, v in record.items()
                                                    if k != "messages"})
        (folder / "agent/nexau.txt").write_text(
            "Native benchmark transport; this is not a NexAU runtime log.\n"
            f"status={record.get('status', 'unknown')} "
            f"infrastructure_error={record.get('infrastructure_error', False)}\n")
        if runtime_artifacts:
            source = Path(runtime_artifacts[(record["task_id"], record["trial"])])
            for name in ("nexau_in_memory_tracer.json", "nexau_in_memory_tracer.cleaned.json", "nexau.txt"):
                if (source / name).exists():
                    shutil.copy2(source / name, folder / "agent" / name)
    write_json(job_dir / "native_receipt.json", {
        "schema_version": 1, "domain": payload["domain"], "split": "train",
        "task_mapping": mapping, "n_trials": len(records),
        "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "trace_format": "openai_messages", "synthetic_spans": False,
        "genuine_nexau_runtime": bool(runtime_artifacts)})
    # Publish completion LAST, exactly as upstream wait_for_job expects.
    write_json(job_dir / "result.json", {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "native_backend": True, "n_trials": len(records)})
    return job_dir
