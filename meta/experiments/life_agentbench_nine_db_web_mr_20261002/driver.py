"""Resume-safe six-round continuation, train-only AgentBench bundle-v2 driver."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from agentbench_chain import CompositeHarness, write_manifest
from candidate_contract import (SAFE_IMPORT_MEMBERS, filter_bundle_operations, prompt,
                                public_history, validate_bundle)
from protocol import (DOMAINS, HERE, ROOT, ROUNDS, atomic_json, bind_fingerprint,
                      digest, object_digest, task_ids, task_trial_keys,
                      validate_protocol)

CODE_ROOT = Path(__file__).resolve().parent


def score(summary: dict[str, Any]) -> float:
    if "per_task_trials" in summary:
        raise RuntimeError("AgentBench summaries must not contain per_task_trials")
    per_task = summary.get("per_task")
    if not isinstance(per_task, dict):
        raise RuntimeError("evaluation summary has no per-task outcomes")
    values = list(per_task.values())
    if not values or any(isinstance(value, bool) or value not in (0, 1) for value in values):
        raise RuntimeError("evaluation outcomes must be nonempty binary values")
    computed = sum(values) / len(values)
    supplied = float(summary.get("accuracy", computed))
    if not math.isfinite(supplied) or not math.isclose(supplied, computed, abs_tol=1e-12):
        raise RuntimeError("evaluation accuracy does not match per-task outcomes")
    return computed


def accept(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    if task_trial_keys(candidate) != task_trial_keys(incumbent):
        raise RuntimeError("Candidate and incumbent task keys differ")
    return score(candidate) > score(incumbent)


def next_batch(batch: int, accepted: bool) -> int:
    return batch if accepted else min(batch + 1, 5)


def snapshot_sha256(records: list[dict[str, Any]]) -> str:
    return object_digest(records)


def initial_state(domain: str) -> dict[str, Any]:
    import inheritance
    return inheritance.initialize(domain, HERE / domain)


def run_fingerprint(domain: str) -> dict[str, Any]:
    source_names = ("protocol.py", "candidate_contract.py", "driver.py",
                    "agentbench_chain.py", "agentbench_evaluate.py", "status.py", "inheritance.py", "runner.py")
    from inheritance import verified_seed
    seed = verified_seed(domain)
    return {"experiment": "life_agentbench_nine_db_web_mr_20261002",
            "protocol_sha256": digest(HERE / "protocol.json"), "domain": domain,
            "rounds": ROUNDS, "bundle_schema_version": 2,
            "proposer": "ModelRouter deepseek-v4-flash",
            "initial_active_snapshot": seed["active_snapshot"],
            "inherited_rounds": 6,
            "inherited_training_seal_sha256": __import__("protocol").protocol()["inherited_training_seal_sha256"],
            "prior_test_exposure": True,
            "test_policy": "rounds seven through nine train only; prior test exposure recorded; final tests after training seal",
            "source_sha256": {name: digest(CODE_ROOT / name) for name in source_names}}


def verify_active(run_dir: Path, state: dict[str, Any]) -> None:
    records = state["active_snapshot"]
    if snapshot_sha256(records) != state["active_snapshot_sha256"]:
        raise RuntimeError("active snapshot fingerprint mismatch")
    seen = set()
    for record in records:
        if record["mechanism_id"] in seen:
            raise RuntimeError("duplicate active mechanism")
        seen.add(record["mechanism_id"])
        relative = Path(record["artifact_path"])
        if (len(relative.parts) != 3 or relative.parts[0] != "versions"
                or relative.suffix != ".py"):
            raise RuntimeError("accepted artifact path is invalid")
        path = (run_dir / relative).resolve()
        if run_dir.resolve() not in path.parents or digest(path) != record["sha256"]:
            raise RuntimeError(f"accepted artifact tampered: {record['mechanism_id']}")
    if state.get("rounds"):
        logs = state.get("operation_log", [])
        if state.get("accepted_bundle_ids") != [entry.get("bundle_id") for entry in logs]:
            raise RuntimeError("accepted bundle audit log mismatch")
        replayed: list[dict[str, Any]] = []
        for entry in logs:
            if entry.get("parent_snapshot_sha256") != snapshot_sha256(replayed):
                raise RuntimeError("operation log parent snapshot mismatch")
            for operation in entry.get("operations", []):
                positions = {item["mechanism_id"]: index
                             for index, item in enumerate(replayed)}
                if operation.get("op") not in {"add", "modify", "delete"}:
                    raise RuntimeError("operation log contains an invalid operation")
                mechanism_id = operation["mechanism_id"]
                prior = replayed[positions[mechanism_id]] if mechanism_id in positions else None
                if operation["op"] in {"modify", "delete"} and (
                        prior is None or operation.get("expected_revision") != prior["revision"]
                        or operation.get("expected_sha256") != prior["sha256"]):
                    raise RuntimeError("operation log has a stale mechanism reference")
                if operation["op"] == "delete":
                    if operation["target_layer"] != prior["target_layer"]:
                        raise RuntimeError("operation log delete layer mismatch")
                    replayed.pop(positions[mechanism_id])
                    continue
                previous = prior if operation["op"] == "modify" else None
                if operation["op"] == "modify" and previous is None:
                    raise RuntimeError("operation log modifies a missing mechanism")
                if operation["op"] == "add" and mechanism_id in positions:
                    raise RuntimeError("operation log adds a duplicate mechanism")
                revision = previous["revision"] + 1 if previous else 1
                relative = Path("versions") / entry["bundle_id"] / f"{operation['operation_id']}.py"
                artifact_path = run_dir / relative
                if (not artifact_path.is_file()
                        or digest(artifact_path) != operation.get("artifact_sha256")):
                    raise RuntimeError("operation log artifact mismatch")
                record = {"mechanism_id": mechanism_id, "revision": revision,
                          "target_layer": operation["target_layer"],
                          "artifact_path": str(relative), "sha256": digest(artifact_path)}
                if previous:
                    replayed[positions[mechanism_id]] = record
                else:
                    replayed.append(record)
            if entry.get("committed_snapshot_sha256") != snapshot_sha256(replayed):
                raise RuntimeError("operation log committed snapshot mismatch")
        if replayed != records:
            raise RuntimeError("active snapshot does not match operation log")


def _apply(active: list[dict[str, Any]], bundle: dict[str, Any],
           artifact_path: Callable[[dict[str, Any], int], tuple[str, str]]) -> list[dict[str, Any]]:
    result = [dict(item) for item in active]
    for operation in bundle["operations"]:
        positions = {item["mechanism_id"]: index for index, item in enumerate(result)}
        op = operation["op"]
        if op == "delete":
            result.pop(positions[operation["mechanism_id"]])
            continue
        previous = result[positions[operation["mechanism_id"]]] if op == "modify" else None
        revision = previous["revision"] + 1 if previous else 1
        relative, sha256 = artifact_path(operation, revision)
        record = {"mechanism_id": operation["mechanism_id"], "revision": revision,
                  "target_layer": operation["target_layer"],
                  "artifact_path": relative, "sha256": sha256}
        if previous:
            result[positions[operation["mechanism_id"]]] = record
        else:
            result.append(record)
    return result


def stage_bundle(run_dir: Path, round_number: int, bundle_path: Path,
                 state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    verify_active(run_dir, state)
    root = bundle_path.parent
    bundle = validate_bundle(json.loads(bundle_path.read_text()), state["domain"],
                             state["active_snapshot"], state["active_snapshot_sha256"], root)
    if bundle["bundle_id"] in state.get("accepted_bundle_ids", []):
        raise ValueError("bundle_id was already accepted")
    stage_root = run_dir / "runtime" / f"staged_{round_number:02d}"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    artifacts = stage_root / "artifacts"
    artifacts.mkdir(parents=True)
    current_sources = {row["mechanism_id"]: run_dir / row["artifact_path"]
                       for row in state["active_snapshot"]}

    def staged_path(operation: dict[str, Any], revision: int) -> tuple[str, str]:
        target = artifacts / f"{operation['mechanism_id']}_r{revision}.py"
        shutil.copy2(operation["artifact_path"], target)
        return str(target.relative_to(stage_root)), digest(target)

    staged = _apply(state["active_snapshot"], bundle, staged_path)
    changed = {op["mechanism_id"] for op in bundle["operations"]}
    for row in staged:
        if row["mechanism_id"] not in changed:
            source = current_sources[row["mechanism_id"]]
            target = artifacts / f"{row['mechanism_id']}_r{row['revision']}.py"
            shutil.copy2(source, target)
            row["artifact_path"] = str(target.relative_to(stage_root))
    staged_sha = snapshot_sha256(staged)
    manifest = write_manifest(stage_root / "manifest.json", staged, stage_root, staged_sha)
    staged_by_mechanism = {row["mechanism_id"]: row for row in staged}
    for operation in bundle["operations"]:
        if operation["op"] != "delete":
            record = staged_by_mechanism[operation["mechanism_id"]]
            operation["artifact_path"] = (stage_root / record["artifact_path"]).resolve()
            operation["artifact_sha256"] = record["sha256"]
    return bundle, staged, manifest


def commit_bundle(run_dir: Path, bundle: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    verify_active(run_dir, state)
    final_dir = run_dir / "versions" / bundle["bundle_id"]
    temporary = final_dir.with_name(final_dir.name + ".tmp")
    if temporary.exists():
        expected = {f"{operation['operation_id']}.py": operation["artifact_sha256"]
                    for operation in bundle["operations"] if operation["op"] != "delete"}
        actual = {path.name: digest(path) for path in temporary.iterdir() if path.is_file()}
        if not final_dir.exists() and actual == expected:
            temporary.replace(final_dir)
        else:
            shutil.rmtree(temporary)

    def existing_path(operation: dict[str, Any], _revision: int) -> tuple[str, str]:
        target = final_dir / f"{operation['operation_id']}.py"
        if not target.is_file() or digest(target) != operation["artifact_sha256"]:
            raise RuntimeError("cached accepted bundle mismatch")
        relative = Path("versions") / bundle["bundle_id"] / target.name
        return str(relative), digest(target)

    if final_dir.exists():
        result = _apply(state["active_snapshot"], bundle, existing_path)
    else:
        temporary.mkdir(parents=True)

        def accepted_path(operation: dict[str, Any], _revision: int) -> tuple[str, str]:
            source = operation["artifact_path"]
            expected = operation["artifact_sha256"]
            if digest(source) != expected:
                raise RuntimeError("evaluated staged artifact changed before commit")
            target = temporary / f"{operation['operation_id']}.py"
            shutil.copy2(source, target)
            if digest(target) != expected:
                raise RuntimeError("accepted artifact copy mismatch")
            relative = Path("versions") / bundle["bundle_id"] / target.name
            return str(relative), expected

        try:
            result = _apply(state["active_snapshot"], bundle, accepted_path)
            temporary.replace(final_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    verify = {"active_snapshot": result, "active_snapshot_sha256": snapshot_sha256(result)}
    verify_active(run_dir, verify)
    return result


def write_active_chain(run_dir: Path, state: dict[str, Any], name: str = "incumbent") -> Path:
    verify_active(run_dir, state)
    directory = run_dir / "runtime" / name
    directory.mkdir(parents=True, exist_ok=True)
    return write_manifest(directory / "manifest.json", state["active_snapshot"], run_dir,
                          state["active_snapshot_sha256"])


def _phase_summary_path(domain: str, phase_name: str) -> Path:
    from inheritance import phase_summary_path
    return phase_summary_path(domain, phase_name)


def phase(domain: str, split: str, phase_name: str, chain: Path, batch: int) -> dict[str, Any]:
    python = {
        "alfworld": ROOT / "AgentBench/.native/alfworld-venv/bin/python",
        "dbbench": ROOT / "AgentBench/.native/core/bin/python",
        "webshop": ROOT / "AgentBench/.native/webshop-venv/bin/python",
    }[domain]
    command = [str(python), "-u", str(HERE / "agentbench_evaluate.py"), domain,
               split, phase_name, str(chain), "--batch", str(batch)]
    env = dict(os.environ)
    env["PATH"] = "/mnt/data/xts/tools/uv/bin:" + env.get("PATH", "")
    if domain == "webshop":
        java = ROOT / "AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64"
        env.update(JAVA_HOME=str(java), JVM_PATH=str(java / "lib/server/libjvm.so"),
                   PATH=str(java / "bin") + ":" + env["PATH"], _JAVA_OPTIONS="-Xmx4g",
                   PYTHONPATH=str(ROOT / "AgentBench/.native/webshop-src") + ":" + str(ROOT / "AgentBench"))
    log_path = HERE / domain / f"{phase_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=21600)
    if result.returncode:
        raise RuntimeError(f"evaluation failed; see {log_path}")
    return json.loads(_phase_summary_path(domain, phase_name).read_text())


def _export_evidence(workspace: Path, phase_name: str, summary: dict[str, Any],
                     output_dir: Path | None = None) -> None:
    safe = {key: value for key, value in summary.items()
            if "test" not in key.lower() and "meta" not in key.lower()
            and key not in {"save_dir", "cmd", "ids", "per_task", "per_task_trials"}}
    destination = workspace / "evidence" / phase_name
    atomic_json(destination / "summary.json", safe)
    source = Path(summary.get("output_dir", output_dir or
                              workspace.parents[1] / "evals" / phase_name))
    if not source.exists() and "output_dir" not in summary:
        source = _phase_summary_path(summary["domain"], phase_name).parent
    episodes = []
    for episode_path in sorted(source.glob("episode_*.json")):
        row = json.loads(episode_path.read_text())
        episodes.append({key: row.get(key) for key in
                         ("success", "status", "turns", "history", "calls", "rewards")})
    atomic_json(destination / "public_trajectories.json", episodes)


def _workspace_hashes(workspace: Path) -> dict[str, str]:
    result = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace)
        if relative == Path("bundle.json") or relative.parts[:1] == ("artifacts",):
            continue
        result[str(relative)] = digest(path)
    return result


def _prepare_workspace(domain: str, round_number: int, run_dir: Path,
                       state: dict[str, Any], destination: Path | None = None) -> Path:
    workspace = destination or run_dir / "proposer" / f"round_{round_number:02d}"
    workspace.mkdir(parents=True, exist_ok=True)
    active_dir = workspace / "active"
    active_dir.mkdir(exist_ok=True)
    for row in state["active_snapshot"]:
        target = active_dir / f"{row['mechanism_id']}_r{row['revision']}.py"
        source = run_dir / row["artifact_path"]
        if target.exists() and digest(target) != row["sha256"]:
            raise RuntimeError("proposer active mechanism snapshot changed")
        if not target.exists():
            shutil.copy2(source, target)
    atomic_json(workspace / "active_snapshot.json", {
        "sha256": state["active_snapshot_sha256"], "records": state["active_snapshot"]})
    atomic_json(workspace / "inventory.json", [
        {key: row[key] for key in ("mechanism_id", "revision", "target_layer", "sha256")}
        for row in state["active_snapshot"]])
    atomic_json(workspace / "history.json", public_history(state["rounds"]))
    source_view = workspace / "source_view"
    source_view.mkdir(exist_ok=True)
    allowed_imports = "\n".join(
        f"from {module} import {', '.join(sorted(members))}"
        for module, members in sorted(SAFE_IMPORT_MEMBERS.items()))
    (source_view / "contract.txt").write_text(
        "bundle.json must contain exactly these fields and no others:\n"
        '{"schema_version":2,"bundle_id":"lowercase_identifier","domain":"DOMAIN",'
        '"base_snapshot_sha256":"ACTIVE_SHA256","hypothesis":"nonempty",'
        '"operations":[OPERATION,...]}\n'
        "Identifiers match [a-z][a-z0-9_]{0,63}. Each operation_id and mechanism_id is "
        "unique within the bundle. Operations contain exactly the fields for their type:\n"
        'add: {"operation_id":"...","mechanism_id":"...","op":"add",'
        '"target_layer":"h2|h3|h4|h5","hypothesis":"nonempty",'
        '"artifact":"artifacts/name.py"}\n'
        'modify: the add fields with op="modify", plus "expected_revision":INTEGER and '
        '"expected_sha256":"INVENTORY_SHA256"\n'
        'delete: {"operation_id":"...","mechanism_id":"...","op":"delete",'
        '"target_layer":"CURRENT_LAYER","hypothesis":"nonempty",'
        '"expected_revision":INTEGER,"expected_sha256":"INVENTORY_SHA256"}\n'
        "Delete forbids artifact. Artifact paths are relative direct artifacts/*.py files. "
        "An invalid individual operation is rejected and logged without discarding valid sibling "
        "operations; all remaining operations are staged simultaneously as one bundle. "
        "Each artifact assigns TARGET_LAYER exactly once and defines class Harness with "
        "exactly one synchronous matching hook and no inherited or dynamic extra hooks. Exact "
        "signatures are h2(self,message,history), h3(self,tools), h4(self,messages,remaining), "
        "and h5(self,messages), without optional, variadic, or keyword-only parameters. H3 changes "
        "descriptions only; H2 returns an assistant dict; H4/H5 have no word-count limit. "
        "Direct import statements are forbidden. Only the exact from-imports below are allowed; "
        "aliases are permitted, but other members and relative imports are forbidden:\n"
        + allowed_imports + "\n")
    snapshot = ROOT / "meta/experiments/meta_from_scratch_six_20260915/task_snapshot/src/server/harness"
    for name in ("session.py", f"{domain}.py"):
        source = snapshot / name
        target = source_view / name
        if target.exists() and digest(target) != digest(source):
            raise RuntimeError("proposer public source view changed")
        if not target.exists():
            shutil.copy2(source, target)
    return workspace


MODEL_ROUTER_ENV = Path("/mnt/data/xts/skill/test_ap/env.sh")
MODEL_ROUTER_MODEL = "deepseek-v4-flash"
MODEL_ROUTER_DEFAULT_HOST = "https://routify-pub.alibaba-inc.com"


def _model_router_config() -> tuple[str, str]:
    api_key = os.environ.get("MR_API_KEY")
    host = os.environ.get("MR_HOST", MODEL_ROUTER_DEFAULT_HOST)
    if api_key:
        return api_key, host.rstrip("/")
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\0%s\\0" "$MR_API_KEY" '
         '"${MR_HOST:-https://routify-pub.alibaba-inc.com}"', "bash", str(MODEL_ROUTER_ENV)],
        capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("ModelRouter environment could not be loaded")
    fields = result.stdout.split(b"\0")
    if len(fields) < 3 or not fields[0]:
        raise RuntimeError("MR_API_KEY is unavailable")
    return fields[0].decode(), fields[1].decode().rstrip("/")


def _sandbox_path(sandbox: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("path must be relative to the proposer workspace")
    resolved = (sandbox / candidate).resolve()
    if sandbox.resolve() not in resolved.parents and resolved != sandbox.resolve():
        raise ValueError("path escapes the proposer workspace")
    return resolved


def _proposer_tools() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": "glob_files",
         "description": "List workspace files matching a relative glob.",
         "parameters": {"type": "object", "properties": {
             "pattern": {"type": "string"}}, "required": ["pattern"],
             "additionalProperties": False}}},
        {"type": "function", "function": {"name": "read_file",
         "description": "Read a byte range from a UTF-8 workspace file.",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string"}, "start_byte": {"type": "integer", "minimum": 0},
             "max_bytes": {"type": "integer", "minimum": 1, "maximum": 16000}},
             "required": ["path"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "search_text",
         "description": "Find literal text in one workspace file and return bounded snippets.",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string"}, "query": {"type": "string"},
             "max_matches": {"type": "integer", "minimum": 1, "maximum": 20},
             "context_chars": {"type": "integer", "minimum": 0, "maximum": 1000}},
             "required": ["path", "query"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "read_json",
         "description": "Inspect a JSON value or a slice of a top-level/list field.",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string"}, "index": {"type": "integer", "minimum": 0},
             "field": {"type": "string"}, "start": {"type": "integer"},
             "count": {"type": "integer", "minimum": 1, "maximum": 20},
             "max_chars": {"type": "integer", "minimum": 100, "maximum": 16000}},
             "required": ["path"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "write_file",
         "description": "Write bundle.json or one direct artifacts/*.py output file.",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string"}, "content": {"type": "string"}},
             "required": ["path", "content"], "additionalProperties": False}}},
    ]


def _execute_proposer_tool(sandbox: Path, name: str, arguments: dict[str, Any]) -> Any:
    if name == "glob_files":
        pattern = arguments["pattern"]
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("glob must stay inside the workspace")
        paths = [path for path in sorted(sandbox.glob(pattern)) if path.is_file()]
        return [{"path": str(path.relative_to(sandbox)), "bytes": path.stat().st_size}
                for path in paths[:200]]
    if name == "read_file":
        path = _sandbox_path(sandbox, arguments["path"])
        start = int(arguments.get("start_byte", 0))
        maximum = min(int(arguments.get("max_bytes", 12000)), 16000)
        with path.open("rb") as stream:
            stream.seek(start)
            data = stream.read(maximum)
        return {"path": arguments["path"], "start_byte": start,
                "next_byte": start + len(data), "eof": len(data) < maximum,
                "content": data.decode("utf-8", errors="replace")}
    if name == "search_text":
        path = _sandbox_path(sandbox, arguments["path"])
        query = arguments["query"]
        if not query or len(query) > 500:
            raise ValueError("query must contain 1 to 500 characters")
        text = path.read_text(errors="replace")
        folded = text.casefold()
        needle = query.casefold()
        maximum = min(int(arguments.get("max_matches", 10)), 20)
        context = min(int(arguments.get("context_chars", 300)), 1000)
        matches = []
        offset = 0
        while len(matches) < maximum:
            position = folded.find(needle, offset)
            if position < 0:
                break
            matches.append({"offset": position,
                            "snippet": text[max(0, position - context):
                                            position + len(query) + context]})
            offset = position + len(needle)
        return {"path": arguments["path"], "matches": matches,
                "truncated": len(matches) == maximum}
    if name == "read_json":
        path = _sandbox_path(sandbox, arguments["path"])
        value = json.loads(path.read_text())
        if "index" in arguments:
            if not isinstance(value, list):
                raise ValueError("index requires a top-level JSON list")
            value = value[int(arguments["index"])]
        if "field" in arguments:
            if not isinstance(value, dict) or arguments["field"] not in value:
                raise ValueError("requested JSON field is unavailable")
            value = value[arguments["field"]]
        if isinstance(value, list) and ("start" in arguments or "count" in arguments):
            start = int(arguments.get("start", 0))
            count = min(int(arguments.get("count", 10)), 20)
            if start < 0:
                start = max(0, len(value) + start)
            value = value[start:start + count]
        text = json.dumps(value, ensure_ascii=False)
        maximum = min(int(arguments.get("max_chars", 12000)), 16000)
        return {"content": text[:maximum], "truncated": len(text) > maximum,
                "total_chars": len(text)}
    if name == "write_file":
        relative = Path(arguments["path"])
        allowed = relative == Path("bundle.json") or (
            len(relative.parts) == 2 and relative.parts[0] == "artifacts"
            and relative.suffix == ".py")
        if not allowed:
            raise ValueError("writes are restricted to bundle.json and artifacts/*.py")
        path = _sandbox_path(sandbox, arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["content"])
        return {"path": arguments["path"], "bytes": path.stat().st_size}
    raise ValueError(f"unknown proposer tool: {name}")


def _model_router_completion(messages: list[dict[str, Any]],
                             tools: list[dict[str, Any]]) -> dict[str, Any]:
    api_key, host = _model_router_config()
    payload = {"model": MODEL_ROUTER_MODEL, "messages": messages,
               "tools": tools, "tool_choice": "auto", "stream": False,
               "max_tokens": 8192}
    request = urllib.request.Request(
        host + "/protocol/openai/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=900) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"ModelRouter HTTP {exc.code}: {detail}") from exc
    if result.get("error"):
        raise RuntimeError(f"ModelRouter error: {result['error']}")
    choices = result.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise RuntimeError("ModelRouter response has no assistant message")
    return {"message": choices[0]["message"], "finish_reason": choices[0].get("finish_reason"),
            "usage": result.get("usage", {})}


def _run_model_router_agent(sandbox: Path, instruction: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": (
            "You are the Life-Harness proposer. Inspect only the supplied workspace through tools. "
            "Never request hidden test or Meta information. Use at most 16 inspection turns. You "
            "must create bundle.json and every referenced artifacts/*.py file with write_file, then "
            "stop. Do not merely print files or spend the response only reasoning.")},
        {"role": "user", "content": instruction},
    ]
    calls = 0
    final_content = ""
    usage: dict[str, Any] = {}
    all_tools = _proposer_tools()
    write_tools = [tool for tool in all_tools if tool["function"]["name"] == "write_file"]
    for turn in range(32):
        response = _model_router_completion(messages, all_tools if turn < 16 else write_tools)
        message = response["message"]
        usage = response["usage"]
        tool_calls = message.get("tool_calls") or []
        assistant = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("reasoning_content") is not None:
            assistant["reasoning_content"] = message["reasoning_content"]
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if not tool_calls:
            final_content = assistant["content"]
            if (sandbox / "bundle.json").is_file():
                break
            messages.append({"role": "user", "content": (
                "Stop analyzing. Write every required artifacts/*.py file and bundle.json now "
                "using write_file. Do not respond without creating them.")})
            continue
        for call in tool_calls:
            calls += 1
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
                result = _execute_proposer_tool(sandbox, call["function"]["name"], arguments)
                content = json.dumps({"ok": True, "result": result}, ensure_ascii=False)
            except Exception as exc:
                content = json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    else:
        raise RuntimeError("ModelRouter proposer exceeded 32 tool turns")
    return {"tool_calls": calls, "final_content": final_content,
            "usage": usage, "message_count": len(messages)}


def propose(domain: str, round_number: int, workspace: Path,
            run_dir: Path, state: dict[str, Any]) -> Path:
    cache = run_dir / "proposals" / f"round_{round_number:02d}"
    frozen = _workspace_hashes(workspace)
    input_sha = object_digest(frozen)
    metadata_path = cache / "cache.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["input_sha256"] != input_sha:
            raise RuntimeError("cached proposal input mismatch")
        outputs = {str(path.relative_to(cache)): digest(path) for path in sorted(cache.rglob("*"))
                   if path.is_file() and path != metadata_path}
        if outputs != metadata["outputs"]:
            raise RuntimeError("cached proposal output mismatch")
        return cache / "bundle.json"
    bundle_path = workspace / "bundle.json"
    artifacts = workspace / "artifacts"
    bundle_path.unlink(missing_ok=True)
    shutil.rmtree(artifacts, ignore_errors=True)
    atomic_json(run_dir / "proposals" / f"inputs_{round_number:02d}.json", frozen)
    with tempfile.TemporaryDirectory(prefix=f"life-proposer-{domain}-{round_number:02d}-",
                                     dir="/tmp") as temporary:
        sandbox = Path(temporary) / "workspace"
        shutil.copytree(workspace, sandbox)
        text = prompt(domain, round_number, sandbox) + (
            "\nUse glob_files, read_file, search_text, and read_json to inspect the workspace. "
            "Use write_file for bundle.json and artifacts/*.py. The workspace path in this prompt "
            "is informational; every tool path must be relative to it.")
        prompt_path = run_dir / "proposals" / f"prompt_{round_number:02d}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(text)
        sandbox_bundle = sandbox / "bundle.json"
        sandbox_artifacts = sandbox / "artifacts"
        retry_feedback = None
        for attempt in range(1, 4):
            sandbox_bundle.unlink(missing_ok=True)
            shutil.rmtree(sandbox_artifacts, ignore_errors=True)
            sandbox_artifacts.mkdir()
            attempt_text = text
            previous_feedback = retry_feedback
            retry_feedback = None
            if previous_feedback:
                attempt_text += ("\n\nThe previous attempt was rejected by the local public contract "
                                 f"validator: {previous_feedback}\nCorrect that issue in this fresh attempt.")
            attempt_path = (run_dir / "proposals" /
                            f"model_router_{round_number:02d}_attempt_{attempt:02d}.json")
            attempt_result: dict[str, Any] = {
                "attempt": attempt, "model": MODEL_ROUTER_MODEL,
                "endpoint": "/protocol/openai/v1/chat/completions"}
            try:
                attempt_result.update(_run_model_router_agent(sandbox, attempt_text))
            except Exception as exc:
                retry_feedback = f"{type(exc).__name__}: {exc}"
                attempt_result["request_error"] = retry_feedback
            if _workspace_hashes(workspace) != frozen or _workspace_hashes(sandbox) != frozen:
                raise RuntimeError("proposer modified a frozen input")
            if retry_feedback is None and sandbox_bundle.exists():
                try:
                    filtered, rejected = filter_bundle_operations(
                        json.loads(sandbox_bundle.read_text()), domain,
                        state["active_snapshot"], state["active_snapshot_sha256"], sandbox)
                    attempt_result["rejected_operations"] = rejected
                    if not filtered["operations"]:
                        raise ValueError("all bundle operations were rejected")
                    document = {**filtered, "operations": [
                        {key: value for key, value in operation.items()
                         if key != "artifact_path"}
                        for operation in filtered["operations"]]}
                    atomic_json(sandbox_bundle, document)
                    validate_bundle(document, domain, state["active_snapshot"],
                                    state["active_snapshot_sha256"], sandbox)
                    shutil.copy2(sandbox_bundle, bundle_path)
                    if sandbox_artifacts.exists():
                        shutil.copytree(sandbox_artifacts, artifacts)
                    atomic_json(attempt_path, attempt_result)
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    retry_feedback = f"{type(exc).__name__}: {exc}"
            elif retry_feedback is None:
                retry_feedback = "bundle.json was not created"
            attempt_result["validation_error"] = retry_feedback
            atomic_json(attempt_path, attempt_result)
        else:
            raise RuntimeError("proposer failed to produce a valid bundle after 3 attempts")
    cache.mkdir(parents=True)
    shutil.copy2(bundle_path, cache / "bundle.json")
    if artifacts.exists():
        shutil.copytree(artifacts, cache / "artifacts")
    outputs = {str(path.relative_to(cache)): digest(path) for path in sorted(cache.rglob("*"))
               if path.is_file() and path != metadata_path}
    atomic_json(metadata_path, {"input_sha256": input_sha, "outputs": outputs})
    return cache / "bundle.json"


def preflight(manifest: Path) -> dict[str, Any]:
    root, records = __import__("agentbench_chain").manifest_records(manifest)
    CompositeHarness(records, root)
    for path in [root / row["artifact_path"] for row in records]:
        py_compile.compile(str(path), doraise=True)
    return {"manifest_sha256": digest(manifest), "snapshot_sha256": snapshot_sha256(records),
            "validation": "pass"}


def _assert_expected_keys(summary: dict[str, Any], expected_ids: list[Any]) -> None:
    expected = {(str(task), 0) for task in expected_ids}
    if task_trial_keys(summary) != expected:
        raise RuntimeError("evaluation task keys do not match frozen protocol")


def _verified_phase_summary(run_dir: Path, domain: str, phase_name: str, batch: int,
                            expected_ids: list[Any]) -> dict[str, Any]:
    path = _phase_summary_path(domain, phase_name)
    if not path.is_file():
        raise RuntimeError(f"training evidence is missing: {phase_name}")
    summary = json.loads(path.read_text())
    if (summary.get("domain") != domain or summary.get("split") != "train"
            or summary.get("phase") != phase_name or summary.get("batch") != batch):
        raise RuntimeError(f"training evidence metadata mismatch: {phase_name}")
    _assert_expected_keys(summary, expected_ids)
    score(summary)
    __import__("agentbench_evaluate").validate_cached_summary(path.parent, summary)
    return summary


def _assert_active_chain_config(run_dir: Path, phase_name: str,
                                active: list[dict[str, Any]], active_sha: str) -> None:
    runtime = run_dir / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-chain-", dir=runtime) as temporary:
        manifest = write_manifest(Path(temporary) / "manifest.json", active, run_dir, active_sha)
        expected_hashes = __import__("agentbench_evaluate").chain_hashes(manifest)
    config_path = run_dir / "evals" / phase_name / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"training evaluation configuration is missing: {phase_name}")
    config = json.loads(config_path.read_text())
    if (config.get("snapshot_sha256") != object_digest(active)
            or config.get("chain_hashes") != expected_hashes):
        raise RuntimeError(f"incumbent evaluation chain mismatch: {phase_name}")


def verify_training_history(run_dir: Path, state: dict[str, Any]) -> None:
    historical = validate_protocol()
    domain = state["domain"]
    spec = historical["domains"][domain]
    from inheritance import verified_seed
    seed = verified_seed(domain)
    if state["rounds"][:6] != seed["rounds"]:
        raise RuntimeError("inherited six-round decisions changed")
    active = [dict(item) for item in seed["active_snapshot"]]
    active_sha = snapshot_sha256(active)
    batch = seed["batch"]
    baseline_phase = seed["baseline_phase"] or f"batch_{batch:02d}_incumbent"
    replayed_rounds = list(seed["rounds"])
    operation_logs = {row["round"]: row for row in state.get("operation_log", [])}
    if len(state.get("rounds", [])) != state.get("round"):
        raise RuntimeError("training history round count mismatch")
    for round_number, row in enumerate(state["rounds"][6:], 7):
        if row.get("round") != round_number or row.get("train_batch") != batch:
            raise RuntimeError("training history order or batch mismatch")
        ids = task_ids(spec, batch, "train")
        baseline = _verified_phase_summary(run_dir, domain, baseline_phase, batch, ids)
        if baseline_phase.startswith("batch_"):
            _assert_active_chain_config(run_dir, baseline_phase, active, active_sha)
        candidate_phase = f"candidate_{round_number:02d}"
        candidate = _verified_phase_summary(run_dir, domain, candidate_phase, batch, ids)
        if (not math.isclose(score(baseline), row["incumbent_score"], abs_tol=1e-12)
                or not math.isclose(score(candidate), row["candidate_score"], abs_tol=1e-12)
                or accept(candidate, baseline) != row["accepted"]):
            raise RuntimeError("training history score or acceptance mismatch")
        cache = run_dir / "proposals" / f"round_{round_number:02d}"
        bundle_path = cache / "bundle.json"
        inputs_path = run_dir / "proposals" / f"inputs_{round_number:02d}.json"
        metadata_path = cache / "cache.json"
        if not bundle_path.is_file() or not inputs_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("training proposal cache is incomplete")
        metadata = json.loads(metadata_path.read_text())
        outputs = {str(path.relative_to(cache)): digest(path) for path in sorted(cache.rglob("*"))
                   if path.is_file() and path != metadata_path}
        replay_state = {"active_snapshot": active, "active_snapshot_sha256": active_sha,
                        "rounds": replayed_rounds}
        runtime = run_dir / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="verify-inputs-", dir=runtime) as temporary:
            workspace = _prepare_workspace(domain, round_number, run_dir, replay_state,
                                           Path(temporary) / "workspace")
            _export_evidence(workspace, baseline_phase, baseline,
                             run_dir / "evals" / baseline_phase)
            expected_inputs = _workspace_hashes(workspace)
        supplied_inputs = json.loads(inputs_path.read_text())
        if (supplied_inputs != expected_inputs
                or metadata.get("input_sha256") != object_digest(expected_inputs)
                or metadata.get("outputs") != outputs or digest(bundle_path) != row["bundle_sha256"]):
            raise RuntimeError("training proposal cache mismatch")
        parsed = validate_bundle(json.loads(bundle_path.read_text()), domain, active, active_sha, cache)
        if parsed["bundle_id"] != row["bundle_id"]:
            raise RuntimeError("training bundle identity mismatch")
        staged_manifest = run_dir / "runtime" / f"staged_{round_number:02d}" / "manifest.json"
        _root, staged = __import__("agentbench_chain").manifest_records(staged_manifest)
        staged_by_mechanism = {item["mechanism_id"]: item for item in staged}
        for operation in parsed["operations"]:
            if (operation["op"] != "delete"
                    and digest(operation["artifact_path"])
                    != staged_by_mechanism[operation["mechanism_id"]]["sha256"]):
                raise RuntimeError("proposal artifact differs from evaluated staged artifact")
        expected_staged = _apply(
            active, parsed,
            lambda operation, _revision: (
                staged_by_mechanism[operation["mechanism_id"]]["artifact_path"],
                staged_by_mechanism[operation["mechanism_id"]]["sha256"]))
        identity = lambda records: [
            {key: item[key] for key in ("mechanism_id", "revision", "target_layer", "sha256")}
            for item in records]
        if identity(expected_staged) != identity(staged):
            raise RuntimeError("staged snapshot does not match proposed operations")
        candidate_config = json.loads(
            (run_dir / "evals" / candidate_phase / "config.json").read_text())
        if (candidate_config.get("snapshot_sha256") != object_digest(staged)
                or candidate_config.get("chain_hashes")
                != __import__("agentbench_evaluate").chain_hashes(staged_manifest)):
            raise RuntimeError("candidate evaluation chain mismatch")
        if row["accepted"]:
            log = operation_logs.get(round_number)
            if not log or log.get("bundle_id") != parsed["bundle_id"]:
                raise RuntimeError("accepted operation log mismatch")

            def committed_path(operation: dict[str, Any], _revision: int) -> tuple[str, str]:
                relative = Path("versions") / parsed["bundle_id"] / f"{operation['operation_id']}.py"
                committed = run_dir / relative
                staged_sha = staged_by_mechanism[operation["mechanism_id"]]["sha256"]
                if not committed.is_file() or digest(committed) != staged_sha:
                    raise RuntimeError("accepted artifact differs from evaluated staged artifact")
                return str(relative), staged_sha

            active = _apply(active, parsed, committed_path)
            baseline_phase = candidate_phase
        else:
            batch = next_batch(batch, False)
            baseline_phase = f"batch_{batch:02d}_incumbent"
        active_sha = snapshot_sha256(active)
        if row.get("next_batch") != batch or row.get("active_snapshot_sha256") != active_sha:
            raise RuntimeError("training history transition mismatch")
        replayed_rounds.append(row)
    if active != state["active_snapshot"] or active_sha != state["active_snapshot_sha256"]:
        raise RuntimeError("training history does not reconstruct active snapshot")


def run_domain(domain: str, evaluate: Callable[..., dict[str, Any]] = phase,
               propose_bundle: Callable[..., Path] = propose,
               do_preflight: Callable[[Path], dict[str, Any]] = preflight) -> dict[str, Any]:
    historical = validate_protocol()
    if domain not in DOMAINS:
        raise ValueError("unsupported domain")
    spec = historical["domains"][domain]
    run_dir = HERE / domain
    run_dir.mkdir(parents=True, exist_ok=True)
    bind_fingerprint(run_dir / "run_config.json", run_fingerprint(domain))
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else initial_state(domain)
    if (state.get("domain") != domain or state.get("round") != len(state.get("rounds", []))
            or not 0 <= state.get("round", -1) <= ROUNDS):
        raise RuntimeError("persisted state structure is invalid")
    verify_active(run_dir, state)
    if evaluate is phase and state["round"]:
        verify_training_history(run_dir, state)
    if state["status"] == "complete":
        if state["round"] != ROUNDS:
            raise RuntimeError("completed state does not contain six rounds")
        return state
    for round_number in range(state["round"] + 1, ROUNDS + 1):
        batch = state["batch"]
        incumbent_chain = write_active_chain(run_dir, state)
        if state["baseline_summary"] is None:
            baseline_phase = f"batch_{batch:02d}_incumbent"
            baseline = evaluate(domain, "train", baseline_phase, incumbent_chain, batch)
            _assert_expected_keys(baseline, task_ids(spec, batch, "train"))
            score(baseline)
            state.update(baseline_phase=baseline_phase, baseline_summary=baseline)
            atomic_json(state_path, state)
        else:
            baseline = state["baseline_summary"]
            if (baseline.get("domain") != domain or baseline.get("split") != "train"
                    or baseline.get("phase") != state["baseline_phase"]
                    or baseline.get("batch") != batch):
                raise RuntimeError("resumed baseline summary does not match state")
            _assert_expected_keys(baseline, task_ids(spec, batch, "train"))
            score(baseline)
            if evaluate is phase:
                cached_path = _phase_summary_path(domain, state["baseline_phase"])
                if not cached_path.is_file() or json.loads(cached_path.read_text()) != baseline:
                    raise RuntimeError("resumed baseline differs from evaluator cache")
                __import__("agentbench_evaluate").validate_cached_summary(
                    cached_path.parent, baseline)
        workspace = _prepare_workspace(domain, round_number, run_dir, state)
        _export_evidence(workspace, state["baseline_phase"], baseline)
        bundle_path = propose_bundle(domain, round_number, workspace, run_dir, state)
        bundle, _staged, staged_chain = stage_bundle(run_dir, round_number, bundle_path, state)
        check = do_preflight(staged_chain)
        atomic_json(run_dir / "proposals" / f"preflight_{round_number:02d}.json", check)
        candidate_phase = f"candidate_{round_number:02d}"
        candidate = evaluate(domain, "train", candidate_phase, staged_chain, batch)
        _assert_expected_keys(candidate, task_ids(spec, batch, "train"))
        accepted = accept(candidate, baseline)
        if accepted:
            parent_snapshot_sha256 = state["active_snapshot_sha256"]
            active = commit_bundle(run_dir, bundle, state)
            committed_snapshot_sha256 = snapshot_sha256(active)
            operations = []
            for operation in bundle["operations"]:
                record = {key: value for key, value in operation.items()
                          if key not in {"artifact_path", "artifact_sha256"}}
                if operation["op"] != "delete":
                    record["artifact_sha256"] = operation["artifact_sha256"]
                operations.append(record)
            state.setdefault("accepted_bundle_ids", []).append(bundle["bundle_id"])
            state.setdefault("operation_log", []).append({
                "round": round_number, "bundle_id": bundle["bundle_id"],
                "hypothesis": bundle["hypothesis"],
                "parent_snapshot_sha256": parent_snapshot_sha256,
                "committed_snapshot_sha256": committed_snapshot_sha256,
                "operations": operations,
            })
            state.update(active_snapshot=active,
                         active_snapshot_sha256=committed_snapshot_sha256,
                         baseline_phase=candidate_phase, baseline_summary=candidate)
        else:
            state.update(batch=next_batch(batch, False), baseline_phase=None,
                         baseline_summary=None)
        row = {"round": round_number, "train_batch": batch,
               "incumbent_score": score(baseline), "candidate_score": score(candidate),
               "accepted": accepted, "next_batch": state["batch"],
               "batch_rotated": not accepted and state["batch"] != batch,
               "bundle_id": bundle["bundle_id"], "bundle_sha256": digest(bundle_path),
               "bundle_hypothesis": bundle["hypothesis"],
               "operations": [{"operation_id": item["operation_id"],
                                "mechanism_id": item["mechanism_id"],
                                "op": item["op"], "target_layer": item["target_layer"],
                                "hypothesis": item["hypothesis"]}
                               for item in bundle["operations"]],
               "active_snapshot_sha256": state["active_snapshot_sha256"]}
        state["rounds"].append(row)
        state["round"] = round_number
        atomic_json(state_path, state)
    state["status"] = "complete"
    atomic_json(state_path, state)
    if evaluate is phase:
        verify_training_history(run_dir, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=DOMAINS)
    args = parser.parse_args()
    run_dir = HERE / args.domain
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"{args.domain} already has a running driver") from exc
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        run_domain(args.domain)


if __name__ == "__main__":
    main()
