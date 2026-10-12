"""Generic, identity-starting Life-Harness evolution and frozen evaluation."""
import fcntl
import inspect
import json
import math
import os
from pathlib import Path

from evolution import harness_hooks as hooks
from evolution.life_evolve import profile
from evolution.life_evolve.tau_run import composition_module
from .adapter import BenchmarkAdapter
from .records import Ledger, save
from .runtime import artifacts, case_id, json_copy, run_episode


def definition(adapter):
    if not isinstance(adapter, BenchmarkAdapter):
        raise TypeError("adapter must implement BenchmarkAdapter")
    if adapter.name in hooks.BUILTIN_DOMAINS:
        raise ValueError("use a unique custom adapter name; built-in benchmark drivers retain their native contracts")
    hooks.register_domain(adapter.name, adapter.public_context_fields)
    source = inspect.getsourcefile(type(adapter))
    if not source or not Path(source).is_file():
        raise ValueError("adapter class requires a source file for fingerprinting")
    return {"name": adapter.name, "public_context_fields": sorted(adapter.public_context_fields),
            "configuration": json_copy(adapter.fingerprint()), "adapter_source_sha256": hooks.digest(Path(source))}


def task_ids(adapter, split):
    ids = list(adapter.task_ids(split))
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("task_ids must be a nonempty collection of unique strings")
    return ids


def strict_improvement(candidate, incumbent, expected):
    """Full native TRAIN set, with no built-in benchmark-specific batch size."""
    expected = {case_id(task) for task in expected}
    for score in (candidate, incumbent):
        outcomes = score.get("per_task", {})
        if (not score.get("complete") or score.get("n") != len(expected) or set(outcomes) != expected
                or any(type(v) is not int or v not in (0, 1) for v in outcomes.values())
                or score.get("successes") != sum(outcomes.values())
                or not math.isclose(score.get("accuracy", -1), sum(outcomes.values()) / len(expected), abs_tol=1e-12)):
            raise ValueError("strict selection requires both complete matching native TRAIN evaluations")
    return candidate["successes"] > incumbent["successes"]


def evaluate(adapter, manifest, output, *, split="test", ledger=None, round_number=0, phase="evaluation"):
    """Evaluate a frozen bundle; invokes the solver/native grader, never an optimizer."""
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("fresh evaluation directory required")
    config = definition(adapter)
    ids = task_ids(adapter, split)
    pairs, hashes = artifacts(manifest)
    output.mkdir(parents=True)
    ledger = ledger or Ledger(output / "requests.jsonl")
    rows = []
    save(output / "config.json", {"adapter": config, "split": split, "task_ids": ids,
        "chain_hashes": hashes, "round": round_number, "phase": phase})
    for task in ids:
        if definition(adapter) != config or artifacts(manifest)[1] != hashes or task_ids(adapter, split) != ids:
            raise RuntimeError("adapter, task set or frozen harness changed during evaluation")
        row = run_episode(adapter, task, split, pairs, ledger,
            {"split": split, "round": round_number, "phase": phase})
        rows.append(row)
        save(output / f"episode_{row['index']}.json", row)
        outcomes = {r["index"]: r["success"] for r in rows}
        summary = {"domain": adapter.name, "split": split, "n": len(ids), "evaluated": len(rows),
            "complete": len(rows) == len(ids), "successes": sum(outcomes.values()),
            "accuracy": sum(outcomes.values()) / len(ids), "per_task": outcomes,
            "infra_errors": sum(r["infrastructure_error"] for r in rows), "chain_hashes": hashes,
            "solver_usage": ledger.summary("solver", round_number, phase),
            "other_model_usage": {role: ledger.summary(role, round_number, phase)
                for role in ledger.roles() if role not in {"solver", "life_optimizer"}}}
        save(output / "summary.json", summary)
    if definition(adapter) != config or artifacts(manifest)[1] != hashes or task_ids(adapter, split) != ids:
        raise RuntimeError("adapter, task set or frozen harness changed during evaluation")
    return summary


class ExtensionRunner:
    def __init__(self, adapter, output, *, rounds=6, completion=None, offline_proposer=None):
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be positive")
        if completion is not None and offline_proposer is not None:
            raise ValueError("choose a model completion or a non-API offline proposer")
        self.adapter, self.root, self.rounds = adapter, Path(output).resolve(), rounds
        self.config = definition(adapter)
        self.ids = {s: task_ids(adapter, s) for s in ("train", "test")}
        if set(self.ids["train"]) & set(self.ids["test"]):
            raise ValueError("TRAIN and TEST task identities must be disjoint")
        self.completion, self.offline_proposer = completion, offline_proposer
        self.work = self.root / adapter.name
        self.compose = composition_module()

    def verify(self):
        if definition(self.adapter) != self.config or any(task_ids(self.adapter, s) != self.ids[s] for s in self.ids):
            raise RuntimeError("benchmark or solver configuration changed")
        if self.sources != profile.source_hashes() or self.extension_sources != extension_hashes():
            raise RuntimeError("evolution/runtime code changed")
        self.compose.verify_active(self.work, self.state)

    def publish(self):
        save(self.root / "TOKEN_USAGE.json", {"optimizer": self.ledger.summary("life_optimizer"),
            "solver": self.ledger.summary("solver"),
            "other_model_roles": {role: self.ledger.summary(role) for role in self.ledger.roles()
                if role not in {"solver", "life_optimizer"}},
            "optimizer_backend": "offline_no_api" if self.offline_proposer else "life_evolve"})
        save(self.root / "TRAINING_RESULTS.json", {"domain": self.adapter.name, "rounds": self.state["rounds"]})

    def run(self):
        if self.root.exists():
            raise FileExistsError("fresh run required; started/failed paid work is never automatically replayed")
        if self.offline_proposer is None and self.completion is None:
            if not os.environ.get("MR_API_KEY") or not os.environ.get("MR_HOST"):
                raise ValueError("set MR_API_KEY and MR_HOST before launching model evolution")
            profile.flash_counter()  # Verify the pinned tokenizer before spending on solver calls.
        self.work.mkdir(parents=True)
        self.ledger = Ledger(self.root / "requests.jsonl")
        self.sources, self.extension_sources = profile.source_hashes(), extension_hashes()
        self.state = {"domain": self.adapter.name, "public_context_fields": self.config["public_context_fields"],
            "round": 0, "rounds": [], "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
            "accepted_bundle_ids": [], "operation_log": [], "prior_test_exposure": False}
        save(self.work / "state.json", self.state)
        save(self.root / "protocol.json", {"adapter": self.config, "rounds": self.rounds,
            "splits": self.ids, "source_hashes": self.sources, "extension_source_hashes": self.extension_sources,
            "initialization": "empty H2-H5", "selection": "strict improvement on the same complete TRAIN task set",
            "test_policy": "one fixed final TEST after TRAIN selection is sealed",
            "optimizer_backend": "offline_no_api" if self.offline_proposer else "life_evolve"})
        with (self.root / ".runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                for number in range(1, self.rounds + 1):
                    save(self.root / "status.json", {"phase": "training", "round": number})
                    self.round(number)
                self.verify()
                save(self.root / "training_seal.json", {"state_sha256": hooks.digest(self.work / "state.json"),
                    "snapshot": self.state["active_snapshot"], "adapter": self.config})
                final = self.compose.write_active_chain(self.work, self.state, "final")
                save(self.root / "status.json", {"phase": "final_test"})
                score = self.score(final, "test", "final_test", self.rounds)
                if hooks.digest(self.work / "state.json") != json.loads((self.root / "training_seal.json").read_text())["state_sha256"]:
                    raise RuntimeError("TRAIN state changed during final TEST")
                save(self.root / "FINAL_RESULTS.json", {"life": score, "manifest": str(final),
                    "training_seal_sha256": hooks.digest(self.root / "training_seal.json"), "used_for_selection": False})
                save(self.root / "status.json", {"phase": "complete"})
            except Exception as exc:
                save(self.root / "status.json", {"phase": "stopped", "error_type": type(exc).__name__})
                raise
            finally:
                self.publish()
        return self.root

    def score(self, manifest, split, phase, number):
        self.verify()
        summary = evaluate(self.adapter, manifest, self.work / "evals" / phase,
            split=split, ledger=self.ledger, round_number=number, phase=phase)
        self.verify()
        if summary["infra_errors"]:
            raise RuntimeError("native infrastructure errors recorded; stop before candidate selection")
        return summary

    def round(self, number):
        incumbent_manifest = self.compose.write_active_chain(self.work, self.state, f"incumbent_{number:02d}")
        evidence = self.work / "evals" / f"incumbent_{number:02d}"
        incumbent = self.score(incumbent_manifest, "train", evidence.name, number)
        workspace, candidate = self.root / f"optimizer_inputs/round_{number:02d}", self.work / f"proposals/round_{number:02d}"
        profile.prepare(self.work, evidence, workspace, seed=number)

        def completion(messages, tools):
            request_id = self.ledger.start("life_optimizer", split="train", round=number, phase="propose")
            try:
                response = (self.completion or profile.model_router_completion)(messages, tools)
            except Exception as exc:
                self.ledger.finish(request_id, None, error_type=type(exc).__name__)
                self.publish()
                raise
            self.ledger.finish(request_id, response.get("usage"))
            self.publish()
            return response

        if self.offline_proposer:
            bundle_path = Path(self.offline_proposer(workspace, candidate, number)).resolve()
            if bundle_path != candidate / "bundle.json":
                raise ValueError("offline proposer must write candidate/bundle.json")
        else:
            bundle_path = profile.propose(workspace, candidate, completion=completion,
                checkpoint_dir=self.work / f"optimizer_sessions/round_{number:02d}")
        # Bind host-known CAS fields without rewriting paid proposal/checkpoint outputs.
        _, bound, rejected, _ = profile.candidate_chain(workspace, candidate)
        qualified = self.work / "qualified" / f"round_{number:02d}"
        qualified.mkdir(parents=True)
        for operation in bound["operations"]:
            if operation["op"] != "delete":
                source = profile.safe_path(candidate, operation["artifact"])
                target = profile.safe_path(qualified, operation["artifact"])
                target.parent.mkdir(parents=True, exist_ok=True)
                self.compose.copy_verified(source, target)
        bundle_path = qualified / "bundle.json"
        save(bundle_path, bound)
        report = profile.check_candidate(workspace, qualified)
        save(qualified / "validation.json", {**report, "original_rejected_operations": rejected})
        if not report["passed"]:
            raise ValueError("candidate failed shared safety/interface replay")
        usage = self.ledger.summary("life_optimizer", number)
        if not usage["accounting_complete"]:
            raise RuntimeError("optimizer usage is unknown; do not promote or replay paid work")
        bundle, _, staged = self.compose.stage_bundle(self.work, number, bundle_path, self.state)
        challenger = self.score(staged, "train", f"candidate_{number:02d}", number)
        accepted = strict_improvement(challenger, incumbent, self.ids["train"])
        parent = self.state["active_snapshot_sha256"]
        operations = [{k: v for k, v in op.items() if k != "artifact_path"} for op in bundle["operations"]]
        if accepted:
            self.state["active_snapshot"] = self.compose.commit_bundle(self.work, bundle, self.state)
            self.state["active_snapshot_sha256"] = hooks.object_digest(self.state["active_snapshot"])
            self.state["accepted_bundle_ids"].append(bundle["bundle_id"])
            self.state["operation_log"].append({"bundle_id": bundle["bundle_id"], "operations": operations,
                "parent_snapshot_sha256": parent, "committed_snapshot_sha256": self.state["active_snapshot_sha256"]})
        self.state["rounds"].append({"round": number, "train_batch": 0, "accepted": accepted,
            "incumbent_score": incumbent["accuracy"], "candidate_score": challenger["accuracy"],
            "incumbent_successes": incumbent["successes"], "candidate_successes": challenger["successes"],
            "bundle_id": bundle["bundle_id"], "bundle_hypothesis": bundle["hypothesis"], "operations": operations,
            "incumbent_eval": str(evidence), "candidate_eval": str(self.work / "evals" / f"candidate_{number:02d}"),
            "active_snapshot_sha256": self.state["active_snapshot_sha256"], "optimizer_usage": usage})
        self.state["round"] = number
        save(self.work / "state.json", self.state)
        self.verify()
        self.publish()


def extension_hashes():
    return {p.name: hooks.digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
