"""Audit/reuse the completed first round; repair native evidence binding only."""
import argparse
import fcntl
from pathlib import Path
import re
import shutil
import sys
import threading
import time

from meta import harness_hooks as hooks
from meta.life_agentbench_run import Run as BaseRun, evaluation_config, native_module
from meta.tau_comparison.common import ROOT, TAU, AHE_NATIVE, child_env, digest, read, save

PARENT = ROOT / "meta/experiments/life_webshop_six_current_20261006_envfix"


def bind_summary(directory, manifest):
    """Bind public evidence to the content actually installed by this evaluation."""
    root, records = hooks.manifest_records(manifest)
    summary = read(directory / "summary.json")
    if summary["snapshot_sha256"] != hooks.object_digest(records):
        raise ValueError("evaluation snapshot does not match installed content")
    summary["chain_hashes"] = {"manifest": digest(manifest),
        **{r["artifact_path"]: digest(root / r["artifact_path"]) for r in records}}
    save(directory / "summary.json", summary)
    native_module().validate_cached_summary(directory, summary)
    return summary


def prepare(root, parent=PARENT):
    sys.path.insert(0, str(AHE_NATIVE))
    from ahe_native.token_accounting import TokenLedger
    if root.exists():
        raise FileExistsError("new audited continuation directory required")
    old = BaseRun(parent)
    old.verify()
    status, metrics = read(parent / "status.json"), read(parent / "flash_metrics.json")
    if (status["phase"] != "stopped_on_error" or not metrics.get("closed") or metrics["gate"]["active"]
            or old.state["round"] != 1 or len(old.state["rounds"]) != 1
            or read(parent / "failure.json")["detail"] != "evidence is not from the exact incumbent content"
            or (parent / "optimizer_inputs/round_02").exists()
            or (old.work / "optimizer_sessions/round_02").exists()):
        raise ValueError("only the verified pre-API second-round evidence-binding failure may resume")
    from meta.life_agentbench_run import optimizer_usage
    usage = optimizer_usage(TokenLedger(parent / "ledgers/mr_requests.jsonl").summary(), parent.name)
    if (not usage["accounting_complete"] or usage != old.state["rounds"][0]["optimizer_usage"]
            or usage != read(parent / "OPTIMIZER_TOKEN_USAGE.json") or usage["total_tokens"] <= 0):
        raise ValueError("complete exactly-attributed first-round cost required")
    source = old.work / "evals/incumbent_02"
    summary = read(source / "summary.json")
    native_module().validate_cached_summary(source, summary)
    original_manifest = old.work / "runtime/incumbent_02/manifest.json"
    _, records = hooks.manifest_records(original_manifest)
    ids = old.protocol["batches"][1]
    if (records != old.state["active_snapshot"] or summary["ids"] != ids or summary["n"] != 50
            or summary["snapshot_sha256"] != old.state["active_snapshot_sha256"]
            or summary["manifest_sha256"] != digest(original_manifest) or summary.get("chain_hashes")):
        raise ValueError("complete unchanged, missing-binding-only baseline required")
    protocol = {**old.protocol, "name": root.name,
        "continuation": {"parent": str(parent), "optimizer_usage": usage,
            "state_sha256": digest(old.work / "state.json"),
            "note": "reuse paid first-round code and complete second-round baseline; only missing evidence binding repaired"}}
    root.mkdir(parents=True)
    save(root / "protocol.json", protocol)
    frozen = {**old.frozen, str(Path(__file__).resolve()): digest(__file__),
              str(root / "protocol.json"): digest(root / "protocol.json")}
    save(root / "frozen_sources.json", frozen)
    work = root / "webshop"
    work.mkdir()
    shutil.copy2(old.work / "state.json", work / "state.json")
    shutil.copytree(old.work / "versions", work / "versions", ignore=shutil.ignore_patterns("__pycache__"))
    resumed = Run(root)
    manifest = resumed.compose.write_active_chain(work, resumed.state, "incumbent_02")
    config = evaluation_config(root, "train", "incumbent_02", manifest, ids)
    old_config = read(source / "config.json")
    changed = {"manifest_sha256", "protocol_sha256", "config_sha256"}
    if {k: v for k, v in old_config.items() if k not in changed} != {k: v for k, v in config.items() if k not in changed}:
        raise ValueError("continuation would change native benchmark or solver settings")
    target = work / "evals/incumbent_02"
    save(target / "config.json", config)
    original_hashes = {"summary.json": digest(source / "summary.json"), "config.json": digest(source / "config.json")}
    for index in ids:
        name = f"episode_{index}.json"
        original_hashes[name] = digest(source / name)
        row = read(source / name)
        save(target / name, {**row, "config_sha256": config["config_sha256"],
            "source_config_sha256": row["config_sha256"], "reused_from": str(source / name)})
    save(target / "summary.json", {**summary, **config, "reused_from": str(source),
        "source_summary_sha256": original_hashes["summary.json"]})
    bind_summary(target, manifest)
    save(root / "CONTINUATION_AUDIT.json", {"parent": str(parent), "original_baseline_hashes": original_hashes,
        "parent_ledger_sha256": digest(parent / "ledgers/mr_requests.jsonl"), "inherited_optimizer_usage": usage,
        "parent_state_sha256": digest(old.work / "state.json"), "new_state_sha256": digest(work / "state.json"),
        "new_baseline_summary_sha256": digest(target / "summary.json"), "replayed_model_calls": 0,
        "configuration_rebinding": sorted(changed), "selection_or_strategy_changed": False})
    for name in ("COMPARISON_REFERENCE_CORRECTION.json", "BENCH_DATA_AUDIT.json"):
        if (parent / name).is_file():
            shutil.copy2(parent / name, root / name)
    save(root / "TRAINING_RESULTS.json", {"domain": "webshop", "rounds": resumed.state["rounds"]})
    save(root / "OPTIMIZER_TOKEN_USAGE.json", usage)
    save(root / "status.json", {"phase": "prepared", "completed_rounds": 1})
    resumed.verify()


class Run(BaseRun):
    def evaluate(self, split, name, manifest, ids):
        summary, directory = super().evaluate(split, name, manifest, ids)
        return bind_summary(directory, manifest), directory

    def usage(self, number=None):
        current = super().usage(number)
        inherited = self.protocol["continuation"]["optimizer_usage"]
        if number in (None, 1):
            for k, v in inherited.items():
                if type(v) is int:
                    current[k] += v
        if number is not None:
            self.usage()
        else:
            save(self.root / "OPTIMIZER_TOKEN_USAGE.json", current)
        return current

    def run(self):
        sys.path.insert(0, str(AHE_NATIVE))
        from ahe_native.api_broker import RequestGate
        from ahe_native.connection import credentials
        from ahe_native.metered_broker import Broker
        with (self.root / ".runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (read(self.root / "status.json")["phase"] != "prepared" or self.state["round"] != 1
                    or digest(self.work / "state.json") != self.protocol["continuation"]["state_sha256"]):
                raise RuntimeError("only the audited first-round continuation may start")
            self.verify()
            key, endpoint = credentials()
            self.gate = RequestGate(1, min_interval=1.25)  # 1 + new Tau shared gate 7 <= 8
            self.broker = Broker(key, endpoint, self.gate, rate_limit_retries=3,
                ledger_path=self.root / "ledgers/mr_requests.jsonl", ledger_summary_path=self.root / "ledgers/MR_TOKEN_USAGE.json")
            del key
            self.stopped = threading.Event()
            threading.Thread(target=self.metrics, daemon=True).start()
            try:
                for number in range(2, 7):
                    self.round(number)
                save(self.root / "training_seal.json", {"state_sha256": digest(self.work / "state.json"),
                    "optimizer_usage": self.usage(), "snapshot": self.state["active_snapshot"]})
                self.broker.blocked_roles.add("life_optimizer")
                final = self.compose.write_active_chain(self.work, self.state, "final")
                result, _ = self.evaluate("test", "final_test", final, self.protocol["test_ids"])
                save(self.root / "FINAL_RESULTS.json", {"life": result, "published_life_reference": .455,
                    "note": "fixed TRAIN-selected incumbent; inherited first-round expense included"})
                self.progress("complete", test_accuracy=result["accuracy"])
            except Exception as exc:
                self.usage()
                save(self.root / "failure.json", {"type": type(exc).__name__, "detail": str(exc)[:6000]})
                self.progress("stopped_on_error", error_type=type(exc).__name__)
                raise
            finally:
                self.stopped.set()
                self.broker.close()
                save(self.root / "flash_metrics.json", {"gate": self.gate.snapshot(), "closed": True,
                    "usage": self.broker.ledger.summary()["overall"], "reserved_concurrency": 1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="life_webshop_six_current_20261007_binding_resume")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]+", args.name):
        parser.error("safe experiment name required")
    root = ROOT / "meta/experiments" / args.name
    if args.run:
        Run(root).run()
    else:
        prepare(root)
        print(root)


if __name__ == "__main__":
    main()
