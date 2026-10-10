"""New Telecom-only Life run; shared MR gate, paired TRAIN, per-round AHE stop gate."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import json

from meta import harness_hooks as hooks
from meta.life_evolve import profile
from meta.life_evolve.context import ContextLimit, OUTPUT_LIMIT, verify_usage
from meta.tau_comparison.common import (ROOT, TAU, AHE_NATIVE, child_env, digest,
                                        load_module, read, save)
from meta.tau_comparison.solver_proxy import SolverProxy

REFERENCE = ROOT / "meta/experiments/meta_ahe_taubench_six_flash_tokens_20261004_repair8"
COMPOSITION = ROOT / "meta/experiments/life_agentbench_flash100k_repair_20261004"


def composition_module():
    # Reuse proven immutable staging/CAS/commit, not a historical optimizer/runtime.
    sys.path.insert(0, str(COMPOSITION))
    module = load_module("life_tau_composition", COMPOSITION / "composition.py")

    def validate(document, domain, records, snapshot, workspace):
        bundle, rejected = profile.filter_operations(document, domain, records, snapshot, workspace)
        if rejected:
            raise ValueError("qualified proposal changed before staging")
        for op in bundle["operations"]:
            if op["op"] != "delete":
                op["artifact_path"] = Path(workspace) / op["artifact"]
        return bundle

    module.validate_bundle = validate
    return module


def stop_reason(rows, same_batch=True):
    comparison = "same-batch AHE" if same_batch else "historical AHE (unpaired TRAIN batches)"
    if rows and rows[-1]["round"] >= 3 and rows[-1]["gap_pp"] >= 30 - 1e-9:
        return f"round >=3 trails {comparison} by at least 30 percentage points"
    if len(rows) >= 2 and all(r["gap_pp"] >= 20 - 1e-9 for r in rows[-2:]):
        return f"two consecutive rounds trail {comparison} by at least 20 percentage points"
    return None


def optimizer_usage(summary, experiment, number=None, inherited=None):
    groups = [g for g in summary["groups"].values() if g["experiment"] == experiment
              and g["domain"] == "telecom" and g["role"] == "life_optimizer"
              and (number is None or g["round"] == number)]
    fields = ("attempts", "known_usage_calls", "unknown_usage_calls", "pending_calls",
              "input_tokens", "output_tokens", "total_tokens", "rate_limit_rejections_without_usage")
    result = {k: sum(g.get(k, 0) for g in groups) for k in fields}
    if inherited is not None and number in (None, 1):
        for field in fields:
            result[field] += inherited.get(field, 0)
    result["accounting_complete"] = result["unknown_usage_calls"] == result["pending_calls"] == 0
    result["inference_accounting_complete"] = (result["pending_calls"] == 0
        and result["unknown_usage_calls"] == result["rate_limit_rejections_without_usage"])
    return result


def initialize(root):
    if root.exists():
        raise FileExistsError("new run required; paid sessions are never blindly restarted")
    original = read(REFERENCE / "protocol.json")
    spec = original["domains_spec"]["telecom"]
    split = read(TAU / "data/tau2/domains/telecom/split_tasks.json")
    if (set(split["train"]) & set(split["test"]) or set(spec["test_task_ids"]) != set(split["test"])
            or spec["train_trials"] != 1 or len(spec["batches"]) != 6
            or any(len(b["task_ids"]) != 50 or len(set(b["task_ids"])) != 50
                   or not set(b["task_ids"]) <= set(split["train"]) for b in spec["batches"])):
        raise ValueError("frozen Telecom TRAIN/TEST schedule mismatch")
    references = read(REFERENCE / "TRAINING_RESULTS.json")["methods"]["ahe"]["telecom"]
    if [r["round"] for r in references] != list(range(1, 7)):
        raise ValueError("six completed AHE reference rounds required")
    frozen = {str(ROOT / name): sha for name, sha in profile.source_hashes().items()}
    frozen.update({str(p): digest(p) for p in (
        Path(__file__), ROOT / "meta/tau_comparison/solver_proxy.py",
        ROOT / "meta/tau_comparison/common.py", TAU / "scripts/eval_harness.py",
        REFERENCE / "protocol.json", REFERENCE / "TRAINING_RESULTS.json",
        REFERENCE / "training_seal.json", REFERENCE / "FINAL_RESULTS.json")})
    for directory in (TAU / "src/tau2", TAU / "data/tau2/domains/telecom"):
        frozen.update({str(p): digest(p) for p in directory.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts})
    for name in ("api_broker.py", "metered_broker.py", "token_accounting.py", "connection.py"):
        p = AHE_NATIVE / "ahe_native" / name
        frozen[str(p)] = digest(p)
    for name in ("composition.py", "candidate_contract.py", "agentbench_chain.py", "protocol.py"):
        p = COMPOSITION / name
        frozen[str(p)] = digest(p)
    protocol = {"name": root.name, "domain": "telecom", "rounds": 6,
        "initialization": "empty H2-H5; no imported old/released/AHE strategies",
        "prior_benchmark_test_exposure": True, "spec": spec, "ahe_reference": references,
        "solver": {"model": "qwen3-4b", "total_context": 32768, "max_output": 4096, "workers": 50},
        "optimizer": {"model": "deepseek-v4-flash", "max_output": OUTPUT_LIMIT,
                      "interaction_limit": None, "token_limit": profile.MAX_TOTAL_TOKENS, "max_repairs": 3},
        "user_simulator": {"model": "deepseek-v4-flash", "max_output": 100000},
        "global_mr_concurrency": 8, "global_mr_min_interval": 1.25,
        "selection": "strict improvement on same complete 50-task TRAIN batch",
        "comparison": "retained Life snapshot measured on AHE same-round batch; not a component ablation",
        "stop": {"two_consecutive_gap_pp": 20, "single_gap_pp_from_round3": 30,
                 "invalid_or_unmeasured_candidate": "stop, do not advance round"},
        "test": "once after all six completed rounds and training seal; no TEST on early stop"}
    root.mkdir(parents=True)
    save(root / "protocol.json", protocol)
    save(root / "frozen_sources.json", frozen)
    save(root / "telecom/state.json", {"domain": "telecom", "round": 0, "rounds": [],
        "prior_test_exposure": True, "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
        "accepted_bundle_ids": [], "operation_log": []})
    save(root / "status.json", {"phase": "prepared", "completed_rounds": 0})
    return protocol


def prepare_continuation(root, parent):
    """Reuse only the audited, stopped first proposal; never reset its paid session."""
    sys.path.insert(0, str(AHE_NATIVE))
    from ahe_native.token_accounting import TokenLedger
    from meta.tau_hooks import training_case_id
    parent = Path(parent).resolve()
    status, state = read(parent / "status.json"), read(parent / "telecom/state.json")
    metrics = read(parent / "flash_metrics.json")
    if (status.get("phase") != "stopped_on_error" or state.get("round") != 0
            or state.get("rounds") or state.get("active_snapshot")
            or not metrics.get("closed") or metrics["gate"]["active"]):
        raise ValueError("continuation requires a stopped, empty-incumbent first proposal")
    session = parent / "telecom/optimizer_sessions/round_01"
    checkpoint = read(session / "agent_checkpoint.json")
    if checkpoint.get("inflight") or checkpoint.get("error") != "repair_limit":
        raise ValueError("first proposal is not a closed, exhausted preflight session")
    inputs = parent / "optimizer_inputs/round_01"
    receipt = read(inputs / "receipt.json")
    if checkpoint["input_sha256"] != hooks.object_digest(receipt):
        raise ValueError("paid session input binding changed")
    for name, expected in receipt["inputs"].items():
        if digest(profile.safe_path(inputs, name)) != expected:
            raise ValueError("paid proposal input changed: " + name)
    candidate = session / "workspace"
    files = {str(p.relative_to(candidate)): digest(p) for p in candidate.rglob("*") if p.is_file()}
    if files != checkpoint["workspace_hashes"]:
        raise ValueError("paid candidate changed after its final checkpoint")
    # Only the explicitly corrected current optimizer/launcher may differ. Native
    # data, scoring, runtime, interfaces and published comparison stay frozen.
    changed = {str(Path(profile.__file__).with_name(name)) for name in
               ("profile.py", "worker.py", "PROMPT.md", "SCHEMAS.md", "tau_run.py")}
    for name, expected in read(parent / "frozen_sources.json").items():
        if name not in changed and digest(name) != expected:
            raise ValueError("non-optimizer frozen source changed: " + name)
    old = read(parent / "protocol.json")
    prior = optimizer_usage(TokenLedger(parent / "ledgers/mr_requests.jsonl").summary(), parent.name)
    if (not prior["accounting_complete"] or prior["total_tokens"] <= 0
            or prior != read(parent / "OPTIMIZER_TOKEN_USAGE.json")
            or prior["total_tokens"] != checkpoint["usage"]["total_tokens"]
            or prior["attempts"] != checkpoint["requests"]):
        raise ValueError("paid first-proposal usage is incomplete or differs from its checkpoint")
    for name, ids in (("initial_batch_00", old["spec"]["batches"][0]["task_ids"]),
                      ("setup_smoke", old["spec"]["batches"][0]["task_ids"][:1])):
        directory = parent / "telecom/evals" / name
        saved, summary = read(directory / "evaluation_receipt.json"), read(directory / "life_train/summary.json")
        original_manifest = parent / "telecom/runtime/setup_empty/manifest.json"
        _, records = hooks.manifest_records(original_manifest)
        if (saved["split"] != "train" or saved["ids"] != ids
                or records or saved["manifest_sha256"] != digest(original_manifest)
                or digest(directory / "life_train/summary.json") != saved["summary_sha256"]
                or summary["n"] != len(ids) or summary["infra_errors"]
                or set(summary["per_task"]) != {training_case_id(t) for t in ids}
                or sum(summary["per_task"].values()) != summary["successes"]):
            raise ValueError("complete unchanged baseline/smoke required for reuse")
    protocol = initialize(root)
    if {k: v for k, v in old.items() if k != "name"} != {k: v for k, v in protocol.items() if k != "name"}:
        raise ValueError("continuation would change non-optimizer experimental settings")
    protocol["continuation"] = {"parent": str(parent), "optimizer_usage": prior,
        "proposer_requests": checkpoint["requests"], "repairs_granted": checkpoint["repairs_granted"],
        "note": "paid round-1 code reused after code-gate correction; rounds 2-6 use current Evolve"}
    save(root / "protocol.json", protocol)
    for name in ("initial_batch_00", "setup_smoke"):
        shutil.copytree(parent / "telecom/evals" / name, root / "telecom/evals" / name)
    work = root / "telecom"
    empty = composition_module().write_active_chain(work, read(work / "state.json"), "setup_empty")
    # The verified empty snapshot is unchanged; only its artifact_root moved.
    # Rebind copied cache receipts, retaining the original evaluation identity.
    for name in ("initial_batch_00", "setup_smoke"):
        path = work / "evals" / name / "evaluation_receipt.json"
        saved = read(path)
        save(path, {**saved, "original_manifest_sha256": saved["manifest_sha256"],
                    "manifest_sha256": digest(empty), "reused_from": str(parent / "telecom/evals" / name)})
    workspace = root / "optimizer_inputs/round_01"
    profile.prepare(work, work / "evals/initial_batch_00/life_train", workspace, seed=1)
    for name, expected in receipt["inputs"].items():
        if name.startswith("evidence/") and digest(workspace / name) != expected:
            raise ValueError("reused baseline differs from the paid proposal's public evidence")

    def no_paid_replay(*args, **kwargs):
        raise RuntimeError("paid candidate requires code repair; do not regenerate it during offline continuation setup")

    output = work / "proposals/round_01"
    profile.propose(workspace, output, completion=no_paid_replay,
        checkpoint_dir=work / "optimizer_sessions/round_01", recovery_outputs=candidate)
    validation = read(output / "validation.json")
    if validation["proposer_requests"] or validation["usage"] or validation.get("rejected_operations"):
        raise ValueError("continuation did not retain the complete candidate without new API calls")
    save(root / "CONTINUATION_AUDIT.json", {"parent": str(parent), "parent_candidate_files": files,
        "parent_checkpoint_sha256": digest(session / "agent_checkpoint.json"),
        "parent_ledger_sha256": digest(parent / "ledgers/mr_requests.jsonl"),
        "prior_optimizer_usage": prior, "new_optimizer_requests": 0,
        "cases_replayed": len(validation["cases"]), "calls_replayed": validation["calls_replayed"],
        "qualified_candidate_sha256": digest(output / "bundle.json")})
    save(root / "OPTIMIZER_TOKEN_USAGE.json", prior)
    return protocol


def prepare_accuracy_only_resume(root):
    """Audit the completed first candidate before cancelling only its comparison."""
    sys.path.insert(0, str(AHE_NATIVE))
    from ahe_native.token_accounting import TokenLedger
    from meta.tau_hooks import training_case_id
    with (root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status, protocol = read(root / "status.json"), read(root / "protocol.json")
        state, frozen = read(root / "telecom/state.json"), read(root / "frozen_sources.json")
        metrics = read(root / "flash_metrics.json")
        audit_path = root / "ACCURACY_ONLY_AMENDMENT.json"
        if (audit_path.exists() or status.get("evaluation") != "comparison_01"
                or status.get("phase") != "evaluating" or state["round"] or state["rounds"]
                or not metrics.get("closed") or Path(f"/proc/{status['pid']}/stat").exists()):
            raise ValueError("only a stopped first-round comparison with a completed candidate may resume")
        for name, sha in frozen.items():
            if name != str(Path(__file__).resolve()) and digest(name) != sha:
                raise ValueError("non-controller frozen source changed: " + name)
        composition_module().verify_active(root / "telecom", state)
        ids = protocol["spec"]["batches"][0]["task_ids"]
        summaries = []
        for name, chain in (("initial_batch_00", "incumbent_01"), ("candidate_01", "staged_01")):
            directory = root / "telecom/evals" / name
            receipt, summary = read(directory / "evaluation_receipt.json"), read(directory / "life_train/summary.json")
            if (receipt["split"] != "train" or receipt["ids"] != ids
                    or receipt["manifest_sha256"] != digest(root / "telecom/runtime" / chain / "manifest.json")
                    or receipt["summary_sha256"] != digest(directory / "life_train/summary.json")
                    or summary["n"] != len(ids) or summary["infra_errors"]
                    or set(summary["per_task"]) != {training_case_id(t) for t in ids}
                    or sum(summary["per_task"].values()) != summary["successes"]
                    or summary["accuracy"] != summary["successes"] / len(ids)):
                raise ValueError("complete unchanged baseline/candidate required")
            summaries.append(summary)
        incumbent, candidate = summaries
        bundle = read(root / "telecom/proposals/round_01/bundle.json")
        validation = read(root / "telecom/proposals/round_01/validation.json")
        if (not profile.strict_improvement(candidate, incumbent, expected_task_ids=[training_case_id(t) for t in ids])
                or state["accepted_bundle_ids"] != [bundle["bundle_id"]] or len(state["operation_log"]) != 1
                or state["operation_log"][0]["committed_snapshot_sha256"] != state["active_snapshot_sha256"]):
            raise ValueError("first-round accepted candidate/state does not match")
        inherited = protocol["continuation"]["optimizer_usage"]
        ledger_path = root / "ledgers/mr_requests.jsonl"
        usage = optimizer_usage(TokenLedger(ledger_path).summary(), root.name, 1, inherited)
        current = optimizer_usage(TokenLedger(ledger_path).summary(), root.name, 1)
        if (not usage["inference_accounting_complete"] or current["attempts"]
                or usage != read(root / "OPTIMIZER_TOKEN_USAGE.json")):
            raise ValueError("completed candidate expense changed or new optimizer calls exist")
        ref = protocol["ahe_reference"][0]
        row = {"round": 1, "train_batch": 0, "incumbent_score": incumbent["accuracy"],
            "candidate_score": candidate["accuracy"], "incumbent_successes": incumbent["successes"],
            "candidate_successes": candidate["successes"], "accepted": True, "bundle_id": bundle["bundle_id"],
            "bundle_hypothesis": bundle["hypothesis"], "operations": state["operation_log"][0]["operations"],
            "active_snapshot_sha256": state["active_snapshot_sha256"], "comparison_batch": 0,
            "comparison_score": candidate["accuracy"], "comparison_infra_errors": 0,
            "ahe_measured_batch": ref["measured_batch"], "same_batch_comparison": False,
            "ahe_score": ref["candidate_train_accuracy"], "gap_pp": 100 * (ref["candidate_train_accuracy"] - candidate["accuracy"]),
            "optimizer_usage": usage, "reused_paid_candidate": True,
            "new_proposer_requests": validation["proposer_requests"], "new_optimizer_tokens": 0,
            "proposer_requests": protocol["continuation"]["proposer_requests"],
            "repairs_granted": protocol["continuation"]["repairs_granted"]}
        save(audit_path, {"reason": "user requested accuracy only; no extra AHE batch alignment",
            "prior_protocol": protocol, "prior_frozen_sources": frozen, "prior_state": state,
            "prior_status": status, "shutdown_metrics": metrics,
            "cancelled_evaluation": "comparison_01 (partial; not an accuracy result)",
            "completed_round": row, "ledger_sha256": digest(ledger_path),
            "controller_sha256": digest(__file__)})
        state.update(round=1, rounds=[row])
        save(root / "telecom/state.json", state)
        protocol.update(accuracy_only=True, comparison="own selection-batch accuracy; historical AHE scores are unpaired",
            accuracy_only_resume={"round": 1, "state_sha256": digest(root / "telecom/state.json"),
                                  "ledger_sha256": digest(ledger_path)})
        save(root / "protocol.json", protocol)
        frozen[str(Path(__file__).resolve())] = digest(__file__)
        save(root / "frozen_sources.json", frozen)
        run = Run(root)
        run.report()
        run.progress("prepared", resume_from_round=2, comparison="accuracy_only")


def audit_branch_sources(parent, source_upgrade=None):
    """Permit only explicitly fingerprinted optimizer diagnostic upgrades."""
    allowed = {str(Path(__file__).resolve()), str(Path(profile.__file__).with_name("PROMPT.md"))}
    audit = read(source_upgrade) if source_upgrade is not None else {}
    archives = [] if source_upgrade is None else [
        {'path': str(Path(source_upgrade).parent / 'source_before'),
         'hashes': audit.get('source_files_before', {})}, *audit.get('earlier_source_archives', [])]
    changes = {}
    upgradeable = {'meta/life_evolve/profile.py', 'meta/life_evolve/worker.py', 'meta/life_evolve/SCHEMAS.md',
        'meta/life_optimizer_v25/bundle.py', 'meta/life_optimizer_v25/context.py', 'meta/life_optimizer_v25/worker.py',
        'meta/agentbench_hooks_v31/contract.py', 'meta/agentbench_hooks_v31/runtime.py',
        'meta/harness_hooks.py', 'meta/environment_state.py'}
    for name, sha in read(parent / "frozen_sources.json").items():
        current = digest(name)
        if current == sha or name in allowed:
            continue
        relative = str(Path(name).relative_to(ROOT))
        originals = [Path(entry['path']) / relative for entry in archives
                     if entry.get('hashes', {}).get(relative) == sha]
        if (relative not in upgradeable or audit.get('source_files_after', {}).get(relative) != current
                or not any(path.is_file() and digest(path) == sha for path in originals)):
            raise ValueError("non-prompt frozen source changed without matching before/after source audit: " + name)
        changes[relative] = {'before': sha, 'after': current}
    return changes


def prepare_round_two_branch(root, parent, wait_for=None, *, source_upgrade=None, history_from=None):
    """Fork the retained round-2 snapshot; reuse its exact round-3 TRAIN evidence."""
    parent = Path(parent).resolve()
    state = read(parent / "telecom/state.json")
    seal = read(parent / "training_seal.json")
    if (read(parent / "status.json")["phase"] != "complete"
            or digest(parent / "telecom/state.json") != seal["state_sha256"]
            or state["active_snapshot"] != seal["snapshot"]
            or [r["round"] for r in state["rounds"]] != list(range(1, 7))
            or not all(r["accepted"] for r in state["rounds"][:2])
            or any(r["accepted"] for r in state["rounds"][2:])
            or state["active_snapshot_sha256"] != state["rounds"][1]["active_snapshot_sha256"]):
        raise ValueError("sealed final snapshot must be the unchanged accepted round-2 snapshot")
    compose = composition_module()
    compose.verify_active(parent / "telecom", state)
    source_changes = audit_branch_sources(parent, source_upgrade)
    history_roots, history_usages = [], []
    sources = history_from if isinstance(history_from, (list, tuple)) else ([] if history_from is None else [history_from])
    for source in sources:
        history_root = Path(source).resolve()
        if history_root in history_roots or history_root == parent:
            raise ValueError('additional history must name distinct sealed branches; never double-count ancestor spend')
        past, past_seal = read(history_root / 'telecom/state.json'), read(history_root / 'training_seal.json')
        metrics, history_usage = read(history_root / 'flash_metrics.json'), read(history_root / 'OPTIMIZER_TOKEN_USAGE.json')
        origin = read(history_root / 'protocol.json').get('round_two_branch', {})
        if (read(history_root / 'status.json').get('phase') != 'complete'
                or digest(history_root / 'telecom/state.json') != past_seal['state_sha256']
                or past['active_snapshot'] != past_seal['snapshot']
                or past['active_snapshot'] != state['active_snapshot']
                or Path(origin.get('parent', '')).resolve() != parent
                or not metrics.get('closed') or metrics['gate']['active']
                or not history_usage.get('accounting_complete')
                or past_seal['optimizer_usage'] != history_usage):
            raise ValueError('additional history must be a closed sealed TRAIN branch with the same retained incumbent and complete usage')
        audit_branch_sources(history_root, source_upgrade)
        history_roots.append(history_root)
        history_usages.append(history_usage)
    old_protocol = read(parent / "protocol.json")
    if (not old_protocol.get("accuracy_only")
            or old_protocol["spec"] != read(REFERENCE / "protocol.json")["domains_spec"]["telecom"]):
        raise ValueError("accuracy-only TRAIN parent required")
    ids = old_protocol["spec"]["batches"][2]["task_ids"]
    baseline = parent / "telecom/evals/initial_batch_02"
    receipt, summary = read(baseline / "evaluation_receipt.json"), read(baseline / "life_train/summary.json")
    from meta.tau_hooks import training_case_id
    if (receipt["split"] != "train" or receipt["ids"] != ids
            or receipt["summary_sha256"] != digest(baseline / "life_train/summary.json")
            or receipt["manifest_sha256"] != digest(parent / "telecom/runtime/incumbent_03/manifest.json")
            or summary["n"] != 50 or summary["infra_errors"]
            or set(summary["per_task"]) != {training_case_id(t) for t in ids}
            or sum(summary["per_task"].values()) != summary["successes"]
            or summary["accuracy"] != summary["successes"] / 50
            or sorted(v for k, v in summary["chain_hashes"].items() if k != "manifest")
               != sorted(r["sha256"] for r in state["active_snapshot"])):
        raise ValueError("unchanged complete round-3 TRAIN baseline required")
    dependency = None
    if wait_for is not None:
        wait_for = Path(wait_for).resolve()
        status = read(wait_for / "status.json")
        if read(wait_for / "protocol.json").get("global_mr_concurrency") != 8:
            raise ValueError("expected existing eight-slot MR test run")
        dependency = {"root": str(wait_for), "pid": status["pid"],
                      "protocol_sha256": digest(wait_for / "protocol.json")}
    protocol = initialize(root)
    work = root / "telecom"
    shutil.copytree(parent / "telecom/versions", work / "versions")
    state.update(round=2, rounds=state["rounds"][:2])
    save(work / "state.json", state)
    protocol.update(accuracy_only=True, rounds=3, train_only=True,
        initialization="retained sealed round-2 Life snapshot; one new TRAIN proposal only",
        comparison="strict improvement against reused round-2 incumbent on the same 50 TRAIN tasks",
        test="none; this continuation does not select or evaluate on TEST",
        wait_for_run=dependency,
        round_two_branch={"parent": str(parent), "parent_seal_sha256": digest(parent / "training_seal.json"),
            "round": 2, "state_sha256": digest(work / "state.json"), "ledger_sha256": None,
            "seed_optimizer_tokens": sum(r["optimizer_usage"]["total_tokens"] for r in state["rounds"]),
            "accounting": "OPTIMIZER_TOKEN_USAGE records NEW spend only; rounds 1-2 are reused history",
            "history": "original rounds 3-6 remain visible as prior TRAIN attempts, not new branch rounds"})
    if source_upgrade is not None:
        save(root / 'SOURCE_UPGRADE_AUDIT.json', {'origin': str(Path(source_upgrade).resolve()),
            'origin_sha256': digest(source_upgrade), 'changed_sources': source_changes,
            'note': 'new source-bound workspace; old receipts and candidate code are not rewritten'})
        frozen = read(root / 'frozen_sources.json')
        frozen[str(root / 'SOURCE_UPGRADE_AUDIT.json')] = digest(root / 'SOURCE_UPGRADE_AUDIT.json')
        save(root / 'frozen_sources.json', frozen)
        protocol['source_upgrade'] = {'audit': 'SOURCE_UPGRADE_AUDIT.json', 'changed_sources': source_changes}
    if history_roots:
        protocol['additional_train_history'] = [
            {'root': str(path), 'seal_sha256': digest(path / 'training_seal.json'), 'optimizer_usage': usage}
            for path, usage in zip(history_roots, history_usages)]
    protocol['ancestor_optimizer_tokens'] = seal['optimizer_usage']['total_tokens'] + sum(
        usage['total_tokens'] for usage in history_usages)
    save(root / "protocol.json", protocol)
    manifest = compose.write_active_chain(work, state, "incumbent_03")
    copied = work / "evals/initial_batch_02"
    shutil.copytree(baseline / "life_train", copied / "life_train")
    save(copied / "evaluation_receipt.json", {**receipt,
        "manifest_sha256": digest(manifest), "reused_from": str(baseline),
        "original_manifest_sha256": receipt["manifest_sha256"]})
    # Prepare with the old state so rejected TRAIN attempts remain visible, while
    # active code is identical to this branch's round-2 incumbent. No TEST input.
    extra = {'additional_history': [path / 'telecom' for path in history_roots]} if history_roots else {}
    profile.prepare(parent / "telecom", baseline / "life_train", root / "optimizer_inputs/round_03", seed=3, **extra)
    run = Run(root)
    run.report()
    run.progress("prepared", resume_from_round=3, inherited_rounds=2, new_rounds=1)
    return protocol


def prepare_next_round_branch(root, parent, *, source_upgrade=None):
    """Continue one TRAIN round from a sealed incumbent, without copying bulk logs."""
    parent = Path(parent).resolve()
    prior = read(parent / 'protocol.json')
    paths = [parent]
    paths += [Path(path) for path in prior.get('train_branch', {}).get('history_roots', [])]
    if prior.get('round_two_branch'):
        paths.append(Path(prior['round_two_branch']['parent']))
    additional = prior.get('additional_train_history', [])
    if isinstance(additional, dict):
        additional = [additional]
    paths += [Path(row['root']) for row in additional]
    sources = list(dict.fromkeys(path.resolve() for path in paths))
    changes, lineage_tokens = {}, 0
    for path in sources:
        state, seal = read(path / 'telecom/state.json'), read(path / 'training_seal.json')
        metrics, usage = read(path / 'flash_metrics.json'), read(path / 'OPTIMIZER_TOKEN_USAGE.json')
        if (read(path / 'status.json').get('phase') != 'complete' or state.get('domain') != 'telecom'
                or digest(path / 'telecom/state.json') != seal['state_sha256']
                or state['active_snapshot'] != seal['snapshot'] or seal['optimizer_usage'] != usage
                or not usage.get('accounting_complete') or not metrics.get('closed') or metrics['gate']['active']):
            raise ValueError('continuation needs closed, sealed TRAIN history and complete optimizer usage: ' + str(path))
        composition_module().verify_active(path / 'telecom', state)
        changes.update(audit_branch_sources(path, source_upgrade))
        lineage_tokens += usage['total_tokens']
    state = read(parent / 'telecom/state.json')
    number = state['round'] + 1
    if (not prior.get('accuracy_only') or prior['spec'] != read(REFERENCE / 'protocol.json')['domains_spec']['telecom']
            or not 1 < number <= len(prior['spec']['batches'])
            or [row['round'] for row in state['rounds']] != list(range(1, number))):
        raise ValueError('next round must follow contiguous completed TRAIN rounds on the frozen six-round schedule')
    parent_usage = read(parent / 'OPTIMIZER_TOKEN_USAGE.json')
    if lineage_tokens != prior.get('ancestor_optimizer_tokens', 0) + parent_usage['total_tokens']:
        raise ValueError('parent lineage token total disagrees with registered sealed sources; audit missing/duplicate history, not the totals')
    protocol = initialize(root)
    shutil.copytree(parent / 'telecom/versions', root / 'telecom/versions')
    save(root / 'telecom/state.json', state)
    protocol.update(accuracy_only=True, train_only=True, rounds=number,
        initialization=f'sealed round-{number - 1} incumbent; one new rotated TRAIN round only',
        comparison='strict improvement on the same complete new 50-task TRAIN batch; historical AHE only',
        test='none; no TEST-based policy selection',
        ancestor_optimizer_tokens=lineage_tokens,
        train_branch={'parent': str(parent), 'parent_seal_sha256': digest(parent / 'training_seal.json'),
            'round': number - 1, 'state_sha256': digest(root / 'telecom/state.json'), 'ledger_sha256': None,
            'history_roots': [str(path) for path in sources],
            'accounting': 'parent complete lineage is inherited once; OPTIMIZER_TOKEN_USAGE records new spend only'})
    if source_upgrade is not None:
        save(root / 'SOURCE_UPGRADE_AUDIT.json', {'origin': str(Path(source_upgrade).resolve()),
            'origin_sha256': digest(source_upgrade), 'changed_sources': changes,
            'note': 'fresh source-bound continuation; historical paid receipts and code remain unchanged'})
        frozen = read(root / 'frozen_sources.json')
        frozen[str(root / 'SOURCE_UPGRADE_AUDIT.json')] = digest(root / 'SOURCE_UPGRADE_AUDIT.json')
        save(root / 'frozen_sources.json', frozen)
        protocol['source_upgrade'] = {'audit': 'SOURCE_UPGRADE_AUDIT.json', 'changed_sources': changes}
    frozen = read(root / 'frozen_sources.json')
    for path in sources:
        frozen.update({str(path / name): digest(path / name) for name in
            ('telecom/state.json', 'training_seal.json', 'protocol.json', 'OPTIMIZER_TOKEN_USAGE.json')})
    save(root / 'frozen_sources.json', frozen)
    save(root / 'protocol.json', protocol)
    run = Run(root)
    run.report()
    run.progress('prepared', resume_from_round=number, inherited_rounds=number - 1, new_rounds=1)
    return protocol


class Run:
    def __init__(self, root):
        self.root, self.work = root, root / "telecom"
        self.protocol, self.frozen = read(root / "protocol.json"), read(root / "frozen_sources.json")
        self.state, self.compose = read(self.work / "state.json"), composition_module()
        self.broker = self.solver = None
        self.verify()

    def verify(self):
        if any(not Path(p).is_file() or digest(p) != sha for p, sha in self.frozen.items()):
            raise RuntimeError("frozen source/data/reference changed; stop without paid replay")
        self.compose.verify_active(self.work, self.state)
        if "continuation" in self.protocol:
            audit = read(self.root / "CONTINUATION_AUDIT.json")
            if (audit["prior_optimizer_usage"] != self.protocol["continuation"]["optimizer_usage"]
                    or audit["qualified_candidate_sha256"] != digest(self.work / "proposals/round_01/bundle.json")):
                raise ValueError("audited continuation candidate or inherited usage changed")

    def progress(self, phase, **extra):
        value = {"phase": phase, "utc_unix": time.time(), "pid": os.getpid(),
                 "completed_rounds": self.state["round"], **extra}
        save(self.root / "status.json", value)
        print(json.dumps(value), flush=True)

    def usage(self, number=None):
        summary = self.broker.ledger.summary()
        inherited = self.protocol.get("continuation", {}).get("optimizer_usage")
        result = optimizer_usage(summary, self.root.name, number, inherited)
        save(self.root / "OPTIMIZER_TOKEN_USAGE.json", optimizer_usage(summary, self.root.name, inherited=inherited))
        return result

    def metrics(self):
        while not self.stopped.wait(30):
            save(self.root / "flash_metrics.json", {"utc_unix": time.time(),
                "gate": self.gate.snapshot(), "usage": self.broker.ledger.summary()["overall"]})

    def evaluate(self, split, name, manifest, ids, number):
        from ahe_native.token_accounting import scoped_url
        from meta.tau_hooks import training_case_id
        self.verify()
        directory = self.work / "evals" / name
        public = directory / "life_train"
        receipt = {"split": split, "ids": ids, "manifest_sha256": digest(manifest)}
        if (directory / "evaluation_receipt.json").exists():
            saved = read(directory / "evaluation_receipt.json")
            if any(saved.get(k) != v for k, v in receipt.items()):
                raise ValueError("completed evaluation differs from requested content/tasks")
            path = public / "summary.json" if split == "train" else directory / "harness_summary.json"
            if digest(path) != saved["summary_sha256"]:
                raise ValueError("completed evaluation changed")
            return read(path), public
        if directory.exists():
            raise RuntimeError("incomplete native evaluation; no automatic paid rerun")
        directory.mkdir(parents=True)
        task_file = directory / "task_ids.txt"
        task_file.write_text("\n".join(ids) + "\n")
        user = scoped_url(self.broker.url, experiment=self.root.name, domain="telecom",
                          round=number, role="user_simulator")
        command = [str(TAU / ".venv/bin/python"), "-u", "-m", "meta.tau_hooks",
            "--life-manifest", str(manifest), "--domain", "telecom", "--split", split,
            "--task-id-file", str(task_file), "--trials", "1", "--concurrency", "50",
            "--agent-llm", "openai/qwen3-4b", "--agent-api-base", self.solver.url,
            "--agent-max-tokens", "4096", "--user-llm", "openai/deepseek-v4-flash",
            "--user-api-base", user, "--user-api-key-env", "OPENAI_API_KEY",
            "--user-max-tokens", "100000", "--save-to", str(directory)]
        self.progress("evaluating", evaluation=name, split=split, task_count=len(ids))
        with (directory / "process.log").open("a") as log:
            result = subprocess.run(command, cwd=ROOT, env=child_env(PYTHONPATH=str(TAU / "src") + ":" + str(ROOT)),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=21600)
        if result.returncode:
            raise RuntimeError("native evaluation failed; inspect " + str(directory / "process.log"))
        path = public / "summary.json" if split == "train" else directory / "harness_summary.json"
        summary = read(path)
        expected = {training_case_id(t) for t in ids} if split == "train" else set(ids)
        if (summary["n"] != len(ids) or set(summary["per_task"]) != expected
                or summary.get("domain") != "telecom"):
            raise ValueError("native evaluation is not the complete fixed task batch")
        summary["accuracy"] = summary["successes"] / len(ids)
        if split == "train":
            if sum(summary["per_task"].values()) != summary["successes"]:
                raise ValueError("binary fixed-denominator score mismatch")
            save(path, summary)
        save(directory / "evaluation_receipt.json", {**receipt, "summary_sha256": digest(path)})
        return summary, public

    def completion(self, number):
        from ahe_native.token_accounting import scoped_url
        endpoint = scoped_url(self.broker.url, experiment=self.root.name, domain="telecom",
                              round=number, role="life_optimizer")

        def call(messages, tools):
            self.progress("evolving", round=number, gate=self.gate.snapshot())
            payload = {"model": "deepseek-v4-flash", "messages": messages, "tools": tools,
                       "tool_choice": "auto", "max_tokens": OUTPUT_LIMIT, "stream": False}
            request = urllib.request.Request(endpoint + "/chat/completions", method="POST",
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"})
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=2100) as response:
                    value = json.load(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                if exc.code == 400 and any(t in body.lower() for t in ("context", "maximum sequence", "too long")):
                    raise ContextLimit("provider context exhausted") from exc
                raise RuntimeError("Flash HTTP failure " + str(exc.code)) from exc
            verify_usage(value)
            if not value.get("choices") or value.get("error"):
                raise RuntimeError("Flash response has no assistant choice")
            return value
        return call

    def report(self):
        rows = self.state["rounds"]
        save(self.root / "TRAINING_RESULTS.json", {"domain": "telecom", "rounds": rows})
        lines = ["# Life / AHE Telecom: new Life six-round run", "",
            "TRAIN only; comparison uses each AHE round's exact measured batch. No component ablation.", "",
            "| Round | Selection batch | Incumbent | Candidate | Accepted | Comparison batch | Life retained | AHE | Gap pp | Evolve tokens |",
            "|---|---|---|---|---|---|---|---|---|---|"]
        if "continuation" in self.protocol:
            lines[0] = "# Life / AHE Telecom: audited code-gate continuation"
            lines[2] += " Paid round-1 code and historical failed-preflight expense are reused, not a clean new-method run."
        if self.protocol.get("accuracy_only"):
            lines[2] = "TRAIN only. Life reports its own complete selection batch; AHE historical scores use different batches. No extra AHE-alignment evaluation."
            if "continuation" in self.protocol:
                lines[2] += " Paid round-1 code and expense are reused."
            lines[4:6] = ["| Round | TRAIN batch | Incumbent | Candidate | Accepted | Retained | AHE historical | Evolve tokens |",
                          "|---|---|---|---|---|---|---|---|"]
        if self.protocol.get("round_two_branch"):
            lines[0] = "# Life Telecom: one-round continuation from the retained round-2 snapshot"
            lines[2] += " Rounds 1-2 and baseline evidence are reused; only round 3 is new. No TEST. New optimizer spend is recorded separately."
        if self.protocol.get('train_branch'):
            inherited = self.protocol['train_branch']['round']
            lines[0] = f'# Life Telecom: one-round continuation from sealed round {inherited}'
            lines[2] += f' Rounds 1-{inherited} are reused; only round {inherited + 1} is new. Rotated TRAIN batch, no TEST. New spend is separate.'
        for r in rows:
            if self.protocol.get("accuracy_only"):
                lines.append(f"| {r['round']} | {r['train_batch']} | {r['incumbent_score']:.1%} | {r['candidate_score']:.1%} | "
                    f"{r['accepted']} | {r['comparison_score']:.1%} | {r['ahe_score']:.1%} | {r['optimizer_usage']['total_tokens']} |")
            else:
                lines.append(f"| {r['round']} | {r['train_batch']} | {r['incumbent_score']:.1%} | {r['candidate_score']:.1%} | "
                f"{r['accepted']} | {r['comparison_batch']} | {r['comparison_score']:.1%} | {r['ahe_score']:.1%} | "
                f"{r['gap_pp']:.1f} | {r['optimizer_usage']['total_tokens']} |")
        (self.root / "COMPARISON.md").write_text("\n".join(lines) + "\n")

    def round(self, number):
        from meta.tau_hooks import training_case_id
        self.verify()
        batches = self.protocol["spec"]["batches"]
        ids = batches[number - 1]["task_ids"]
        incumbent_manifest = self.compose.write_active_chain(self.work, self.state, f"incumbent_{number:02d}")
        baseline_name = (f"initial_batch_{number - 1:02d}" if self.protocol.get("accuracy_only")
                         else "initial_batch_00" if number == 1 else f"comparison_{number - 1:02d}")
        incumbent, evidence_dir = self.evaluate("train", baseline_name, incumbent_manifest, ids, number)
        workspace, output = self.root / "optimizer_inputs" / f"round_{number:02d}", self.work / "proposals" / f"round_{number:02d}"
        if not workspace.exists():
            history = [Path(path) / 'telecom' for path in self.protocol.get('train_branch', {}).get('history_roots', [])]
            extra = {'additional_history': history} if history else {}
            profile.prepare(self.work, evidence_dir, workspace, seed=number, **extra)
        else:
            profile.verify_inputs(workspace)
        self.progress("evolving", round=number, incumbent_score=incumbent["accuracy"])
        bundle_path = profile.propose(workspace, output, completion=self.completion(number),
            checkpoint_dir=self.work / "optimizer_sessions" / f"round_{number:02d}")
        usage = self.usage(number)
        validation = read(output / "validation.json")
        current_usage = optimizer_usage(self.broker.ledger.summary(), self.root.name, number)
        if (not usage["inference_accounting_complete"]
                or current_usage["total_tokens"] != validation["usage"].get("total_tokens", 0) or not usage["total_tokens"]):
            raise RuntimeError("optimizer ledger/checkpoint accounting mismatch")
        bundle, staged, candidate_manifest = self.compose.stage_bundle(self.work, number, bundle_path, self.state)
        candidate, _ = self.evaluate("train", f"candidate_{number:02d}", candidate_manifest, ids, number)
        accepted = profile.strict_improvement(candidate, incumbent, expected_task_ids=[training_case_id(t) for t in ids])
        parent = self.state["active_snapshot_sha256"]
        operations = [{k: v for k, v in op.items() if k != "artifact_path"} for op in bundle["operations"]]
        if accepted:
            active = self.compose.commit_bundle(self.work, bundle, self.state)
            self.state["active_snapshot"], self.state["active_snapshot_sha256"] = active, hooks.object_digest(active)
            self.state["accepted_bundle_ids"].append(bundle["bundle_id"])
            self.state["operation_log"].append({"bundle_id": bundle["bundle_id"], "operations": operations,
                "parent_snapshot_sha256": parent, "committed_snapshot_sha256": hooks.object_digest(active)})
        save(self.work / "state.json", self.state)
        retained = self.compose.write_active_chain(self.work, self.state, f"retained_{number:02d}")
        ref = self.protocol["ahe_reference"][number - 1]
        comparison_batch = number - 1 if self.protocol.get("accuracy_only") else ref["measured_batch"]
        if comparison_batch == number - 1:
            comparison = candidate if accepted else incumbent
        else:
            comparison, _ = self.evaluate("train", f"comparison_{number:02d}", retained,
                batches[ref["measured_batch"]]["task_ids"], number)
        row = {"round": number, "train_batch": number - 1, "incumbent_score": incumbent["accuracy"],
            "candidate_score": candidate["accuracy"], "incumbent_successes": incumbent["successes"],
            "candidate_successes": candidate["successes"], "accepted": accepted, "bundle_id": bundle["bundle_id"],
            "bundle_hypothesis": bundle["hypothesis"], "operations": operations,
            "active_snapshot_sha256": self.state["active_snapshot_sha256"], "comparison_batch": comparison_batch,
            "ahe_measured_batch": ref["measured_batch"], "same_batch_comparison": comparison_batch == ref["measured_batch"],
            "comparison_score": comparison["accuracy"], "comparison_infra_errors": comparison["infra_errors"],
            "ahe_score": ref["candidate_train_accuracy"],
            "gap_pp": 100 * (ref["candidate_train_accuracy"] - comparison["accuracy"]), "optimizer_usage": usage,
            "proposer_requests": validation["proposer_requests"], "repairs_granted": validation["repairs_granted"]}
        if number == 1 and "continuation" in self.protocol:
            inherited = self.protocol["continuation"]
            row.update(reused_paid_candidate=True, new_proposer_requests=validation["proposer_requests"],
                new_optimizer_tokens=current_usage["total_tokens"],
                proposer_requests=inherited["proposer_requests"], repairs_granted=inherited["repairs_granted"])
        self.state["rounds"].append(row)
        self.state["round"] = number
        save(self.work / "state.json", self.state)
        self.verify()
        self.report()
        self.progress("round_complete", **{k: row[k] for k in ("round", "candidate_score", "accepted", "comparison_score", "ahe_score", "gap_pp")})
        return stop_reason(self.state["rounds"], same_batch=not self.protocol.get("accuracy_only", False))

    def run(self):
        sys.path.insert(0, str(AHE_NATIVE))
        from ahe_native.api_broker import RequestGate
        from ahe_native.connection import credentials
        from ahe_native.metered_broker import Broker
        with (self.root / ".runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            resume = self.protocol.get('train_branch', self.protocol.get("round_two_branch", self.protocol.get("accuracy_only_resume", {})))
            ledger_path = self.root / "ledgers/mr_requests.jsonl"
            if (read(self.root / "status.json")["phase"] != "prepared"
                    or (self.state["round"] and (not self.protocol.get("accuracy_only")
                        or self.state["round"] != resume.get("round")
                        or digest(self.work / "state.json") != resume.get("state_sha256")
                        or (digest(ledger_path) if ledger_path.exists() else None) != resume.get("ledger_sha256")))):
                raise RuntimeError("started run requires explicit audited continuation, not automatic replay")
            dependency = self.protocol.get("wait_for_run")
            if dependency:
                other = Path(dependency["root"])
                while True:
                    if digest(other / "protocol.json") != dependency["protocol_sha256"]:
                        raise RuntimeError("queued MR dependency changed")
                    stat = Path(f"/proc/{dependency['pid']}/stat")
                    try:
                        live = stat.read_text().split(") ", 1)[1].split()[0] != "Z"
                    except FileNotFoundError:
                        live = False
                    if not live:
                        if read(other / "status.json")["mr_gate"]["active"]:
                            raise RuntimeError("prior MR gate did not finish cleanly")
                        break
                    self.progress("queued_for_mr", waiting_for=str(other), round=self.state["round"] + 1)
                    time.sleep(30)
                self.verify()
            key, endpoint = credentials()
            self.gate = RequestGate(8, min_interval=1.25)
            self.broker = Broker(key, endpoint, self.gate, rate_limit_retries=3,
                ledger_path=self.root / "ledgers/mr_requests.jsonl", ledger_summary_path=self.root / "ledgers/MR_TOKEN_USAGE.json")
            del key
            self.stopped = threading.Event()
            try:
                self.solver = SolverProxy()
                thread = threading.Thread(target=self.metrics, daemon=True)
                thread.start()
                if not self.state["round"]:
                    empty = self.compose.write_active_chain(self.work, self.state, "setup_empty")
                    smoke, _ = self.evaluate("train", "setup_smoke", empty, self.protocol["spec"]["batches"][0]["task_ids"][:1], 1)
                    if smoke["infra_errors"]:
                        raise RuntimeError("one-task native integration smoke failed; no Evolve calls")
                for number in range(self.state["round"] + 1, self.protocol.get("rounds", 6) + 1):
                    reason = self.round(number)
                    if self.protocol.get("train_only"):
                        continue
                    if reason:
                        self.progress("stopped_for_performance", reason=reason)
                        return
                save(self.root / "training_seal.json", {"state_sha256": digest(self.work / "state.json"),
                    "optimizer_usage": self.usage(), "snapshot": self.state["active_snapshot"]})
                if self.protocol.get("train_only"):
                    self.progress("complete", train_only=True, new_rounds=len(self.state["rounds"]) - resume.get('round', 2))
                    return
                self.broker.blocked_roles.add("life_optimizer")
                final = self.compose.write_active_chain(self.work, self.state, "final")
                result, _ = self.evaluate("test", "final_test", final, self.protocol["spec"]["test_task_ids"], 6)
                save(self.root / "FINAL_RESULTS.json", {"life": result, "ahe": read(REFERENCE / "FINAL_RESULTS.json")["methods"]["ahe:telecom"],
                    "published_life_reference": 0.575, "note": "fixed final retained snapshot; not TEST-selected"})
                self.progress("complete", test_accuracy=result["accuracy"])
            except Exception as exc:
                self.usage()
                save(self.root / "failure.json", {"type": type(exc).__name__, "detail": str(exc)[:6000]})
                self.progress("stopped_on_error", error_type=type(exc).__name__)
                raise
            finally:
                self.stopped.set()
                if self.solver is not None:
                    self.solver.close()
                self.broker.close()
                save(self.root / "flash_metrics.json", {"utc_unix": time.time(), "gate": self.gate.snapshot(),
                    "usage": self.broker.ledger.summary()["overall"], "closed": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="life_taubench_telecom_six_20261006")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--continue-from", type=Path, help="offline audit/reuse of a stopped first proposal into a NEW directory")
    parser.add_argument("--resume-accuracy-only", action="store_true", help="offline audit of a cancelled first comparison; reuse its completed candidate")
    parser.add_argument("--branch-from-round-two", type=Path, help="new one-round TRAIN branch from a sealed round-2 incumbent")
    parser.add_argument('--next-round-from', type=Path, help='one rotated TRAIN round from a closed sealed incumbent in a NEW directory')
    parser.add_argument("--wait-for-run", type=Path, help="queue MR calls until the existing test runner exits")
    parser.add_argument("--source-upgrade-audit", type=Path, help="explicit before/after source audit for a NEW round-two branch")
    parser.add_argument("--history-from", type=Path, action="append", help="closed sealed TRAIN branch; repeat to retain several past attempts")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]+", args.name):
        parser.error("safe experiment name required")
    root = ROOT / "meta/experiments" / args.name
    setups = sum(bool(value) for value in (args.continue_from, args.resume_accuracy_only, args.branch_from_round_two, args.next_round_from))
    if ((args.run and setups) or setups > 1
            or ((args.wait_for_run or args.history_from) and not args.branch_from_round_two)
            or (args.source_upgrade_audit and not (args.branch_from_round_two or args.next_round_from))):
        parser.error("continuation setup is separate from the paid runner")
    if not args.run:
        if args.next_round_from:
            prepare_next_round_branch(root, args.next_round_from, source_upgrade=args.source_upgrade_audit)
        elif args.branch_from_round_two:
            prepare_round_two_branch(root, args.branch_from_round_two, args.wait_for_run,
                source_upgrade=args.source_upgrade_audit, history_from=args.history_from)
        elif args.resume_accuracy_only:
            prepare_accuracy_only_resume(root)
        elif args.continue_from:
            prepare_continuation(root, args.continue_from)
        else:
            initialize(root)
    if args.launch:
        with (root / "runner.log").open("a") as log:
            process = subprocess.Popen([str(TAU / ".venv/bin/python"), "-u", "-m", "meta.life_evolve.tau_run",
                "--name", args.name, "--run"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=child_env(PYTHONPATH=str(TAU / "src") + ":" + str(ROOT)), start_new_session=True)
        save(root / "launch.json", {"pid": process.pid, "started": time.time(), "domain": "telecom",
            "rounds": read(root / "protocol.json")["rounds"]})
        print(json.dumps({"pid": process.pid, "path": str(root)}))
    elif args.run:
        Run(root).run()
    else:
        print(root)


if __name__ == "__main__":
    main()
