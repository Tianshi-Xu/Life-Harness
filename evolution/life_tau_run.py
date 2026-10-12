"""Fresh airline/retail Life six-round runs with separately metered optimization."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

from evolution import harness_hooks as hooks
from evolution.life_agentbench_run import Run as BaseRun
from evolution.life_evolve import profile
from evolution.life_evolve.context import ContextLimit, OUTPUT_LIMIT, verify_usage
from evolution.life_evolve.tau_run import COMPOSITION, composition_module
from evolution.tau_comparison.common import (ROOT, TAU, AHE_NATIVE, ALI_URL, aliyun_key,
                                        child_env, digest, read, save)
from evolution.tau_comparison.solver_proxy import SolverProxy

DOMAINS = ("airline", "retail", "telecom")

def case_key(task_id, trial=0, trials=1):
    # Keep the existing single-trial identity; avoid collisions for Airline's two trials.
    value = str(task_id) if trials == 1 else json.dumps([str(task_id), trial])
    return "native_" + hashlib.sha256(value.encode()).hexdigest()


def optimizer_usage(summary, experiment, domain=None, number=None, inherited=None):
    groups = [g for g in summary["groups"].values() if g["experiment"] == experiment]
    groups += list((inherited or {"groups": {}})["groups"].values())
    groups = [g for g in groups if g["role"] == "life_optimizer" and g["domain"] in DOMAINS
        and (domain is None or g["domain"] == domain) and (number is None or g["round"] == number)]
    fields = ("attempts", "known_usage_calls", "unknown_usage_calls", "pending_calls",
              "input_tokens", "output_tokens", "total_tokens", "rate_limit_rejections_without_usage")
    result = {k: sum(g.get(k, 0) for g in groups) for k in fields}
    result["accounting_complete"] = result["unknown_usage_calls"] == result["pending_calls"] == 0
    result["inference_accounting_complete"] = (result["pending_calls"] == 0
        and result["unknown_usage_calls"] == result["rate_limit_rejections_without_usage"])
    return result


def strict_improvement(candidate, incumbent, expected):
    """Same strict binary complete-batch rule, including all native task/trial pairs."""
    expected = set(expected)
    for summary in (candidate, incumbent):
        scores = summary.get("per_task")
        if (summary.get("split") != "train" or not expected or not isinstance(scores, dict)
                or set(scores) != expected or summary.get("n") != len(expected)
                or any(type(v) is not int or v not in (0, 1) for v in scores.values())
                or type(summary.get("successes")) is not int or summary["successes"] != sum(scores.values())):
            raise ValueError("complete binary task/trial TRAIN batch required")
        accuracy = float(summary.get("accuracy", sum(scores.values()) / len(expected)))
        if not math.isfinite(accuracy) or not math.isclose(accuracy, sum(scores.values()) / len(expected), abs_tol=1e-12):
            raise ValueError("fixed-denominator accuracy mismatch")
    if candidate.get("domain") not in DOMAINS or candidate["domain"] != incumbent.get("domain"):
        raise ValueError("candidate/incumbent domain mismatch")
    return candidate["successes"] > incumbent["successes"]


def initialize(root):
    if root.exists():
        raise FileExistsError("new experiment required; never replay a paid session blindly")
    reference = ROOT / "evolution/reproduction/reference"
    previous = read(reference / "manifest.json")
    specs = {d: previous["domains"][d] for d in DOMAINS}
    for d, spec in specs.items():
        split = read(TAU / f"data/tau2/domains/{d}/split_tasks.json")
        expected_n, expected_trials = (30, 2) if d == "airline" else (50, 1)
        if (set(split["train"]) & set(split["test"]) or set(spec["test_task_ids"]) != set(split["test"])
                or spec["train_trials"] != expected_trials or len(spec["batches"]) != 6
                or any(len(b["task_ids"]) != expected_n or len(set(b["task_ids"])) != expected_n
                   or not set(b["task_ids"]) <= set(split["train"]) for b in spec["batches"])):
            raise ValueError("frozen native TRAIN/TEST schedule mismatch: " + d)
    frozen = {str(ROOT / name): sha for name, sha in profile.source_hashes().items()}
    files = [Path(__file__), ROOT / "evolution/life_agentbench_run.py",
             ROOT / "evolution/tau_comparison/solver_proxy.py", ROOT / "evolution/tau_comparison/common.py",
             reference / "manifest.json", TAU / "scripts/eval_harness.py"]
    files += list(COMPOSITION.glob("*.py"))
    files += [AHE_NATIVE / "ahe_native" / name for name in
              ("api_broker.py", "metered_broker.py", "token_accounting.py", "connection.py")]
    for directory in [TAU / "src/tau2", *[TAU / f"data/tau2/domains/{d}" for d in DOMAINS]]:
        files += [p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    frozen.update({str(p.resolve()): digest(p) for p in files})
    protocol = {"name": root.name, "domains": list(DOMAINS), "rounds": 6, "domains_spec": specs,
        "initialization": "empty H2-H5; no imported historical strategies",
        "solver": {"model": "qwen3-4b", "total_context": 32768, "max_output": 4096},
        "user": {"airline": "qwen3.8-flash", "retail": "qwen3.8-flash", "telecom": "deepseek-v4-flash"},
        "optimizer": {"model": "deepseek-v4-flash", "max_output": OUTPUT_LIMIT,
            "max_repairs": 3, "interaction_limit": None, "token_limit": profile.MAX_TOTAL_TOKENS},
        "global_mr_concurrency": 8,
        "selection": "strict improvement on complete same task/trial TRAIN batch",
        "test": "one fixed final TEST per domain after all six-round states are sealed"}
    root.mkdir(parents=True)
    save(root / "protocol.json", protocol)
    frozen[str(root / "protocol.json")] = digest(root / "protocol.json")
    save(root / "frozen_sources.json", frozen)
    for domain in DOMAINS:
        save(root / domain / "state.json", {"domain": domain, "round": 0, "rounds": [],
            "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
            "accepted_bundle_ids": [], "operation_log": []})
    save(root / "status.json", {"phase": "prepared", "completed_rounds": 0})
    return protocol


def export_results(results, trials):
    """Keep genuine native failure placeholders, without inventing their history."""
    simulations = []
    for original in results.simulations:
        simulation = copy.copy(original)
        info = getattr(original, "info", None) or {}
        if (info.get("life_public_episode") is None
                and getattr(original, "reward_info", None) is None
                and getattr(original, "termination_reason", None) == "infrastructure_error"
                and info.get("error_type") and type(info.get("failed_after_attempts")) is int
                and info["failed_after_attempts"] > 0):
            # Native retry exhaustion has no finalized Life session. Do not copy
            # its traceback, simulator instructions, task answers or credentials.
            simulation.info = {**info, "life_public_episode": {
                "index": case_key(original.task_id, original.trial, trials),
                "success": 0, "status": "infrastructure_error", "infrastructure_error": True,
                "trace_unavailable": True, "error_type": info["error_type"],
                "turns": 0, "history": [], "calls": [], "harness_trace": []}}
        if trials > 1:
            simulation.task_id = json.dumps([str(original.task_id), original.trial])
        simulations.append(simulation)
    return SimpleNamespace(simulations=simulations)


def fixed_test_summary(summary, simulations, ids):
    """Canonical single-trial TEST: an infrastructure failure remains in n."""
    outcomes, errors = {}, 0
    for simulation in simulations:
        task = str(simulation["task_id"])
        if task not in ids or task in outcomes or simulation.get("trial") != 0:
            raise ValueError("unexpected or duplicate TEST task/trial")
        reward = simulation.get("reward_info")
        error = reward is None or simulation["termination_reason"] in {
            "agent_error", "user_error", "infrastructure_error"}
        value = (reward or {}).get("reward", 0)
        if value not in (0, 1):
            raise ValueError("binary native TEST reward required")
        outcomes[task] = int(not error and value == 1)
        errors += error
    if set(outcomes) != set(ids):
        raise ValueError("complete native TEST task set required")
    return {**summary, "split": "test", "n": len(ids), "per_task": outcomes,
            "successes": sum(outcomes.values()), "infra_errors": errors,
            "accuracy": sum(outcomes.values()) / len(ids)}


def evaluation_cli(domain, native_args):
    """Opt-in trial-safe public export, without editing the sealed Tau adapter."""
    from evolution import tau_hooks
    original = tau_hooks.export_training
    trial_count = int(native_args[native_args.index("--trials") + 1])

    def export(results, domain, manifest, destination, interface_receipt=None):
        return original(export_results(results, trial_count), domain, manifest, destination, interface_receipt)

    tau_hooks.export_training = export
    try:
        tau_hooks.evaluation_cli(native_args)
    finally:
        tau_hooks.export_training = original


class Run(BaseRun):
    def __init__(self, master, domain, broker=None, ali=None, solver=None):
        self.master, self.domain = master, domain
        self.root = self.work = master / domain
        self.protocol, self.frozen = read(master / "protocol.json"), read(master / "frozen_sources.json")
        self.state, self.compose = read(self.work / "state.json"), composition_module()
        self.spec = self.protocol["domains_spec"][domain]
        self.broker, self.ali, self.solver = broker, ali, solver
        inherited = master / "inherited_optimizer_ledger.json"
        self.inherited = read(inherited) if inherited.exists() else None

    def usage(self, number=None):
        summary = self.broker.ledger.summary()
        save(self.root / "OPTIMIZER_TOKEN_USAGE.json", optimizer_usage(summary, self.master.name, self.domain, inherited=self.inherited))
        return optimizer_usage(summary, self.master.name, self.domain, number, inherited=self.inherited)

    def evaluate(self, split, name, manifest, ids, number, trials=None):
        from ahe_native.token_accounting import scoped_url
        self.verify()
        trials = trials or (self.spec["train_trials"] if split == "train" else 1)
        _, records = hooks.manifest_records(manifest)
        if split == "test":
            seal = read(self.master / "training_seal.json")
            if (name != "final_test" or ids != self.spec["test_task_ids"] or trials != 1
                    or self.state["round"] != 6 or digest(self.work / "state.json") != seal["states"][self.domain]
                    or records != self.state["active_snapshot"]):
                raise ValueError("TEST requires complete sealed TRAIN-selected final snapshot")
        elif split != "train" or not set(ids) <= set(sum([b["task_ids"] for b in self.spec["batches"]], [])):
            raise ValueError("unknown TRAIN task set")
        directory, public = self.work / "evals" / name, self.work / "evals" / name / "life_train"
        receipt = {"domain": self.domain, "split": split, "ids": ids, "trials": trials,
                   "manifest_sha256": digest(manifest)}
        path = public / "summary.json" if split == "train" else directory / "harness_summary.json"
        if (directory / "evaluation_receipt.json").exists():
            saved = read(directory / "evaluation_receipt.json")
            if any(saved.get(k) != v for k, v in receipt.items()) or digest(path) != saved["summary_sha256"]:
                raise ValueError("completed evaluation identity changed")
            return read(path), public
        if directory.exists():
            raise RuntimeError("incomplete native evaluation; no automatic paid replay")
        directory.mkdir(parents=True)
        task_file = directory / "task_ids.txt"
        task_file.write_text("\n".join(ids) + "\n")
        user = scoped_url((self.broker if self.domain == "telecom" else self.ali).url, experiment=self.master.name, domain=self.domain,
                          round=number, role="user_simulator")
        command = [str(TAU / ".venv/bin/python"), "-u", "-m", "evolution.life_tau_run", "--evaluate", self.domain,
            "--life-manifest", str(manifest), "--domain", self.domain, "--split", split,
            "--task-id-file", str(task_file), "--trials", str(trials), "--concurrency", str(max(50, len(ids) * trials)),
            "--agent-llm", "openai/qwen3-4b", "--agent-api-base", self.solver.url,
            "--agent-max-tokens", "4096", "--user-llm", ("openai/deepseek-v4-flash" if self.domain == "telecom" else "openai/qwen3.8-flash"), "--user-api-base", user,
            "--user-api-key-env", "OPENAI_API_KEY", "--user-max-tokens", ("100000" if self.domain == "telecom" else "4096"), "--user-enable-thinking",
            "--save-to", str(directory)]
        if self.domain == "retail":
            command.append("--nl")
        self.progress("evaluating", evaluation=name, split=split, tasks=len(ids), episodes=len(ids) * trials)
        with (directory / "process.log").open("a") as log:
            result = subprocess.run(command, cwd=ROOT, env=child_env(PYTHONPATH=str(TAU / "src") + ":" + str(ROOT)),
                stdout=log, stderr=subprocess.STDOUT, timeout=21600)
        if result.returncode:
            raise RuntimeError("native evaluation failed: " + str(directory / "process.log"))
        summary = read(path)
        if split == "test":
            summary = fixed_test_summary(summary, read(directory / "results.json")["simulations"], ids)
            save(path, summary)
        expected = {case_key(t, i, trials) for t in ids for i in range(trials)} if split == "train" else set(ids)
        if summary["n"] != len(expected) or set(summary["per_task"]) != expected or summary.get("domain") != self.domain:
            raise ValueError("native evaluation is not the complete task/trial batch")
        if split == "train":
            summary["accuracy"] = summary["successes"] / len(expected)
            strict_improvement(summary, summary, expected)
            save(path, summary)
        save(directory / "evaluation_receipt.json", {**receipt, "summary_sha256": digest(path)})
        return summary, public

    def completion(self, number):
        from ahe_native.token_accounting import scoped_url
        endpoint = scoped_url(self.broker.url, experiment=self.master.name, domain=self.domain,
                              round=number, role="life_optimizer")

        def call(messages, tools):
            if (self.master / "training_seal.json").exists():
                raise RuntimeError("training is sealed")
            self.progress("evolving", round=number)
            request = urllib.request.Request(endpoint + "/chat/completions", method="POST",
                data=json.dumps({"model": "deepseek-v4-flash", "messages": messages, "tools": tools,
                    "tool_choice": "auto", "max_tokens": OUTPUT_LIMIT, "stream": False}).encode(),
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
                raise RuntimeError("Flash response missing assistant choice")
            return value
        return call

    def round(self, number):
        ids, trials = self.spec["batches"][number - 1]["task_ids"], self.spec["train_trials"]
        incumbent_manifest = self.compose.write_active_chain(self.work, self.state, f"incumbent_{number:02d}")
        incumbent, evidence = self.evaluate("train", f"incumbent_{number:02d}", incumbent_manifest, ids, number)
        workspace = self.master / "optimizer_inputs" / self.domain / f"round_{number:02d}"
        output = self.work / f"proposals/round_{number:02d}"
        profile.prepare(self.work, evidence, workspace, seed=number)
        bundle_path = profile.propose(workspace, output, completion=self.completion(number),
            checkpoint_dir=self.work / f"optimizer_sessions/round_{number:02d}")
        validation, usage = read(output / "validation.json"), self.usage(number)
        if (not usage["inference_accounting_complete"] or not usage["total_tokens"]
                or usage["total_tokens"] != validation["usage"].get("total_tokens", 0)):
            raise RuntimeError("optimizer ledger/checkpoint mismatch")
        bundle, _, candidate_manifest = self.compose.stage_bundle(self.work, number, bundle_path, self.state)
        candidate, _ = self.evaluate("train", f"candidate_{number:02d}", candidate_manifest, ids, number)
        accepted = strict_improvement(candidate, incumbent, [case_key(t, i, trials) for t in ids for i in range(trials)])
        parent = self.state["active_snapshot_sha256"]
        operations = [{k: v for k, v in op.items() if k != "artifact_path"} for op in bundle["operations"]]
        if accepted:
            active = self.compose.commit_bundle(self.work, bundle, self.state)
            self.state["active_snapshot"], self.state["active_snapshot_sha256"] = active, hooks.object_digest(active)
            self.state["accepted_bundle_ids"].append(bundle["bundle_id"])
            self.state["operation_log"].append({"bundle_id": bundle["bundle_id"], "operations": operations,
                "parent_snapshot_sha256": parent, "committed_snapshot_sha256": hooks.object_digest(active)})
        row = {"round": number, "train_batch": number - 1, "train_n": len(ids), "train_trials": trials,
            "train_episodes": len(ids) * trials, "incumbent_score": incumbent["accuracy"],
            "candidate_score": candidate["accuracy"], "retained_score": (candidate if accepted else incumbent)["accuracy"],
            "incumbent_successes": incumbent["successes"], "candidate_successes": candidate["successes"],
            "incumbent_infra_errors": incumbent["infra_errors"], "candidate_infra_errors": candidate["infra_errors"],
            "accepted": accepted, "bundle_id": bundle["bundle_id"], "bundle_hypothesis": bundle["hypothesis"],
            "operations": operations, "active_snapshot_sha256": self.state["active_snapshot_sha256"],
            "optimizer_usage": usage, "proposer_requests": validation["proposer_requests"], "repairs_granted": validation["repairs_granted"]}
        self.state["rounds"].append(row)
        self.state["round"] = number
        save(self.work / "state.json", self.state)
        self.verify()
        save(self.root / "TRAINING_RESULTS.json", {"domain": self.domain, "rounds": self.state["rounds"]})
        self.progress("round_complete", **{k: row[k] for k in ("round", "candidate_score", "retained_score", "accepted")})

    def train(self):
        try:
            for number in range(self.state["round"] + 1, 7):
                self.round(number)
            self.progress("training_complete")
        except Exception as exc:
            self.usage()
            save(self.root / "failure.json", {"type": type(exc).__name__, "detail": str(exc)[:6000]})
            self.progress("stopped_on_error", error_type=type(exc).__name__)
            raise


def run(root):
    sys.path.insert(0, str(AHE_NATIVE))
    import httpx
    from ahe_native.api_broker import RequestGate
    from ahe_native.connection import credentials
    from ahe_native.metered_broker import Broker
    with (root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(root / "status.json")["phase"] != "prepared":
            raise RuntimeError("started run requires explicit audited continuation")
        key, endpoint = credentials()
        gate = RequestGate(8, min_interval=1.25)
        mr = Broker(key, endpoint, gate, rate_limit_retries=3, ledger_path=root / "ledgers/mr_requests.jsonl",
                    ledger_summary_path=root / "ledgers/MR_TOKEN_USAGE.json")
        ali = Broker(aliyun_key(), ALI_URL, rate_limit_retries=3, allowed_model="qwen3.8-flash",
                    ledger_path=root / "ledgers/aliyun_requests.jsonl", ledger_summary_path=root / "ledgers/ALIYUN_TOKEN_USAGE.json")
        del key
        ali.client.close()
        ali.client = httpx.Client(timeout=600, trust_env=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=200))
        solver, stopped = SolverProxy(), threading.Event()
        runs = {d: Run(root, d, mr, ali, solver) for d in DOMAINS}
        publish_lock = threading.Lock()

        def publish():
            with publish_lock:
                summary = mr.ledger.summary()
                inherited = runs["airline"].inherited
                save(root / "OPTIMIZER_TOKEN_USAGE.json", {"domains": {d: optimizer_usage(summary, root.name, d, inherited=inherited) for d in DOMAINS},
                    "total": optimizer_usage(summary, root.name, inherited=inherited), "excludes": ["local_solver", "user_simulator", "NL_judge"]})
                save(root / "flash_metrics.json", {"gate": gate.snapshot(), "usage": summary["overall"],
                    "reserved_webshop_slots": 0, "global_concurrency_upper_bound": 8, "closed": stopped.is_set()})

        def monitor():
            while not stopped.wait(30):
                publish()

        threading.Thread(target=monitor, daemon=True).start()
        save(root / "status.json", {"phase": "setup_smokes", "pid": os.getpid(), "utc_unix": time.time()})
        try:
            for r in runs.values():
                r.verify()
            with ThreadPoolExecutor(max_workers=len(DOMAINS)) as pool:
                smokes = [pool.submit(r.evaluate, "train", "setup_smoke",
                    r.compose.write_active_chain(r.work, r.state, "setup_empty"),
                    r.spec["batches"][0]["task_ids"][:1], 1) for r in runs.values() if not r.state["round"]]
                if any(f.result()[0]["infra_errors"] for f in smokes):
                    raise RuntimeError("native setup smoke failed; no optimizer requests")
                save(root / "status.json", {"phase": "training", "pid": os.getpid(), "utc_unix": time.time()})
                jobs = [pool.submit(r.train) for r in runs.values()]
                for f in jobs:
                    f.result()
                publish()
                usage = optimizer_usage(mr.ledger.summary(), root.name, inherited=runs["airline"].inherited)
                if not usage["inference_accounting_complete"] or any(r.state["round"] != 6 for r in runs.values()):
                    raise RuntimeError("six-round training or optimizer accounting incomplete")
                save(root / "training_seal.json", {"states": {d: digest(r.work / "state.json") for d, r in runs.items()},
                    "optimizer_usage": usage, "snapshots": {d: r.state["active_snapshot"] for d, r in runs.items()}})
                mr.blocked_roles.add("life_optimizer")
                save(root / "status.json", {"phase": "final_tests", "pid": os.getpid(), "utc_unix": time.time()})
                tests = {d: pool.submit(r.evaluate, "test", "final_test",
                    r.compose.write_active_chain(r.work, r.state, "final"), r.spec["test_task_ids"], 6) for d, r in runs.items()}
                final = {}
                for d, f in tests.items():
                    result = {"life": f.result()[0],
                        "note": "fixed TRAIN-selected final snapshot, not TEST-selected"}
                    save(root / d / "FINAL_RESULTS.json", result)
                    runs[d].progress("complete", test_accuracy=result["life"]["accuracy"])
                    final[d] = result
                save(root / "FINAL_RESULTS.json", {"domains": final})
                save(root / "status.json", {"phase": "complete", "pid": os.getpid(), "utc_unix": time.time()})
        except Exception as exc:
            save(root / "failure.json", {"type": type(exc).__name__, "detail": str(exc)[:6000]})
            save(root / "status.json", {"phase": "stopped_on_error", "pid": os.getpid(), "error_type": type(exc).__name__})
            raise
        finally:
            stopped.set()
            solver.close()
            mr.close()
            ali.close()
            publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--evaluate", choices=DOMAINS)
    args, native_args = parser.parse_known_args()
    if args.evaluate:
        evaluation_cli(args.evaluate, native_args)
        return
    if native_args or not re.fullmatch(r"[a-z][a-z0-9_]+", args.name):
        parser.error("safe experiment name/no unknown arguments required")
    root = ROOT / "evolution/experiments" / args.name
    if args.run:
        run(root)
    else:
        initialize(root)
        if args.launch:
            with (root / "runner.log").open("a") as log:
                p = subprocess.Popen([str(TAU / ".venv/bin/python"), "-u", "-m", "evolution.life_tau_run", "--name", args.name, "--run"],
                    cwd=ROOT, env=child_env(PYTHONPATH=str(TAU / "src") + ":" + str(ROOT)), stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            save(root / "launch.json", {"pid": p.pid, "started": time.time(), "domains": list(DOMAINS), "rounds": 6})
            print(json.dumps({"pid": p.pid, "path": str(root)}))
        else:
            print(root)


if __name__ == "__main__":
    main()
