"""Audited, zero-replay recovery of the native TRAIN trace-export failure."""
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from meta import harness_hooks as hooks
from meta.tau_comparison.common import AHE_NATIVE, ROOT, TAU, digest, read, save


def prepare(root, parent):
    from meta import life_tau_run as runner, tau_hooks
    sys.path.insert(0, str(AHE_NATIVE))
    from ahe_native.token_accounting import TokenLedger
    metrics, protocol = read(parent / "flash_metrics.json"), read(parent / "protocol.json")
    if (read(parent / "status.json")["phase"] != "stopped_on_error" or not metrics.get("closed")
            or metrics["gate"]["active"] or (parent / "training_seal.json").exists()):
        raise ValueError("closed, unsealed failed parent required")
    changed = str(Path(runner.__file__).resolve())
    for name, sha in read(parent / "frozen_sources.json").items():
        if name != changed and digest(name) != sha:
            raise ValueError("non-controller source/data changed: " + name)
    states = {d: read(parent / d / "state.json") for d in runner.DOMAINS}
    if (states["airline"]["round"] != 2 or states["retail"]["round"] != 6
            or "incumbent_03/process.log" not in read(parent / "airline/failure.json")["detail"]
            or (parent / "optimizer_inputs/airline/round_03").exists()
            or (parent / "airline/proposals/round_03").exists()):
        raise ValueError("only the audited pre-Evolve third-round export failure may resume")
    ledger_path = parent / "ledgers/mr_requests.jsonl"
    ledger = TokenLedger(ledger_path).summary()
    if not ledger["overall"]["accounting_complete"]:
        raise ValueError("complete measured inherited optimizer ledger required")
    for domain, state in states.items():
        runner.composition_module().verify_active(parent / domain, state)
        if [r["round"] for r in state["rounds"]] != list(range(1, state["round"] + 1)):
            raise ValueError("completed round history is incomplete")
        for row in state["rounds"]:
            actual = runner.optimizer_usage(ledger, parent.name, domain, row["round"])
            if actual != row["optimizer_usage"]:
                raise ValueError("inherited round expense does not match its API receipts")
        if runner.optimizer_usage(ledger, parent.name, domain)["total_tokens"] != sum(
                row["optimizer_usage"]["total_tokens"] for row in state["rounds"]):
            raise ValueError("unexpected paid calls in unfinished round")
    source = parent / "airline/evals/incumbent_03"
    raw_path = source / "results.json"
    raw = read(raw_path)
    ids, trials = protocol["domains_spec"]["airline"]["batches"][2]["task_ids"], 2
    expected = {(task, trial) for task in ids for trial in range(trials)}
    seen = [(str(s["task_id"]), s.get("trial")) for s in raw["simulations"]]
    _, records = hooks.manifest_records(parent / "airline/runtime/incumbent_03/manifest.json")
    if (len(seen) != len(expected) or set(seen) != expected or raw["info"]["num_trials"] != trials
            or raw["info"]["environment_info"]["domain_name"] != "airline"
            or records != states["airline"]["active_snapshot"]):
        raise ValueError("native result is not the complete exact incumbent batch")
    simulations = [SimpleNamespace(**{**s, "reward_info":
        SimpleNamespace(**s["reward_info"]) if s.get("reward_info") else None}) for s in raw["simulations"]]
    normalized = runner.export_results(SimpleNamespace(simulations=simulations), trials)
    if any(not (s.info or {}).get("life_public_episode") for s in normalized.simulations):
        raise ValueError("missing public trace is not a genuine native failure placeholder")
    runner.initialize(root)
    current = read(root / "protocol.json")
    ignore = {"name", "continuation"}
    if {k: v for k, v in current.items() if k not in ignore} != {k: v for k, v in protocol.items() if k not in ignore}:
        raise ValueError("continuation changes model, schedule or benchmark settings")
    current["continuation"] = {"parent": str(parent), "start_rounds": {"airline": 3, "retail": 7},
        "optimizer_calls_replayed": 0, "solver_or_simulator_calls_replayed": 0,
        "correction": "native infrastructure failure has an explicitly unavailable public trace and zero reward"}
    save(root / "protocol.json", current)
    save(root / "inherited_optimizer_ledger.json", ledger)
    for domain, state in states.items():
        shutil.copy2(parent / domain / "state.json", root / domain / "state.json")
        shutil.copytree(parent / domain / "versions", root / domain / "versions")
        save(root / domain / "status.json", {"phase": "prepared_resume", "completed_rounds": state["round"]})
        for name in ("TRAINING_RESULTS.json", "OPTIMIZER_TOKEN_USAGE.json"):
            shutil.copy2(parent / domain / name, root / domain / name)
    save(root / "OPTIMIZER_TOKEN_USAGE.json", {"domains": {d: runner.optimizer_usage(ledger, parent.name, d) for d in runner.DOMAINS},
        "total": runner.optimizer_usage(ledger, parent.name), "excludes": ["local_solver", "user_simulator", "NL_judge"]})
    r = runner.Run(root, "airline")
    manifest = r.compose.write_active_chain(r.work, r.state, "incumbent_03")
    _, receipt = tau_hooks.install("airline", manifest)
    receipt["native_cli_sha256"] = digest(TAU / "scripts/eval_harness.py")
    target = root / "airline/evals/incumbent_03"
    summary = tau_hooks.export_training(normalized, "airline", manifest, target / "life_train", receipt)
    summary.update(accuracy=summary["successes"] / len(expected), recovered_from=str(raw_path),
                   source_results_sha256=digest(raw_path))
    save(target / "life_train/summary.json", summary)
    runner.strict_improvement(summary, summary, [runner.case_key(t, i, trials) for t, i in expected])
    save(target / "evaluation_receipt.json", {"domain": "airline", "split": "train", "ids": ids,
        "trials": trials, "manifest_sha256": digest(manifest),
        "summary_sha256": digest(target / "life_train/summary.json"), "recovered_from": str(source)})
    frozen = read(root / "frozen_sources.json")
    audit_files = [root / "protocol.json", root / "inherited_optimizer_ledger.json", ledger_path,
        raw_path, parent / "protocol.json", parent / "frozen_sources.json",
        *[parent / d / "state.json" for d in runner.DOMAINS]]
    frozen.update({str(p): digest(p) for p in audit_files})
    save(root / "frozen_sources.json", frozen)
    save(root / "CONTINUATION_AUDIT.json", {"parent": str(parent), "source_files": {str(p): digest(p) for p in audit_files},
        "state_hashes_unchanged": {d: digest(root / d / "state.json") == digest(parent / d / "state.json") for d in runner.DOMAINS},
        "recovered_incumbent": {k: summary[k] for k in ("n", "successes", "infra_errors", "accuracy")},
        "inherited_optimizer_tokens": runner.optimizer_usage(ledger, parent.name)["total_tokens"],
        "optimizer_calls_replayed": 0, "solver_or_simulator_calls_replayed": 0})
    for domain in runner.DOMAINS:
        runner.Run(root, domain).verify()
